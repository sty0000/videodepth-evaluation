import math
import torch
import torch.nn.functional as F
import torchvision.transforms as tvf
import time

from typing import List, Optional, Tuple
from omegaconf import DictConfig
from PIL import Image

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from pi3.models.pi3 import Pi3
from pi3.utils.geometry import se3_inverse


def load_images_cut3r(filelist: List[str], PIXEL_LIMIT: int = 255000, new_width: Optional[int] = None, verbose: bool = False, patch_size: int = 16):
    """
    Loads images, resizes them, and strictly aligns their dimensions to multiples of patch_size.
    """
    sources = [] 
    for img_path in filelist:
        try:
            sources.append(Image.open(img_path).convert('RGB'))
        except Exception as e:
            print(f"Could not load image {img_path}: {e}")

    if not sources:
        return torch.empty(0)

    first_img = sources[0]
    W_orig, H_orig = first_img.size
    
    # --- 核心修改：对齐到 patch_size 而不是 14 ---
    if new_width is None:
        scale = math.sqrt(PIXEL_LIMIT / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
        W_target, H_target = W_orig * scale, H_orig * scale
        k, m = round(W_target / patch_size), round(H_target / patch_size)
        while (k * patch_size) * (m * patch_size) > PIXEL_LIMIT:
            if k / m > W_target / H_target: k -= 1
            else: m -= 1
        TARGET_W, TARGET_H = max(1, k) * patch_size, max(1, m) * patch_size
    else:
        TARGET_W, TARGET_H = new_width, round(H_orig * (new_width / W_orig) / patch_size) * patch_size
        
    if verbose:
        print(f"Images resized to: ({TARGET_W}, {TARGET_H}), aligned to patch_size {patch_size}")

    tensor_list = []
    to_tensor_transform = tvf.ToTensor()
    for img_pil in sources:
        try:
            resized_img = img_pil.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
            tensor_list.append(to_tensor_transform(resized_img))
        except Exception as e:
            print(f"Error processing an image: {e}")

    return torch.stack(tensor_list, dim=0)

def load_and_resize_cut3r(filelist: List[str], new_width: int, device: str, verbose: bool, patch_size: int = 16):
    imgs = load_images_cut3r(filelist, new_width=new_width, verbose=verbose, patch_size=patch_size).to(device)
    if imgs.numel() == 0:
        raise ValueError("Failed to load any valid images.")
        
    ori_h, ori_w = imgs.shape[-2:]
    # 再次确保张量维度被 patch_size 整除
    patch_h, patch_w = ori_h // patch_size, ori_w // patch_size
    imgs = F.interpolate(imgs, (patch_h * patch_size, patch_w * patch_size), mode="bilinear", align_corners=False, antialias=True).unsqueeze(0)
    return imgs

def _wrap_imgs_to_views_cut3r(imgs: torch.Tensor):
    """
    Convert imgs shaped (B=1, N, C, H, W) to list of dicts expected by ARCroco3DStereo.
    必须包含时序模型 (AR) 所需的内存控制键值。
    """
    if imgs.ndim != 5:
        raise ValueError("Expected imgs with 5 dims (1, N, C, H, W)")
    
    B, N, C, H, W = imgs.shape
    if B != 1:
        raise NotImplementedError("Batch size > 1 is not supported in this wrapper.")

    device = imgs.device
    dtype = imgs.dtype
    views = []
    
    for i in range(N):
        view = {
            "img": imgs[:, i],  # (1, 3, H, W)
            # Cut3r 强依赖 true_shape 进行位置编码截断
            "true_shape": torch.tensor([[H, W]], device=device),
            # 标记有效图像区域
            "img_mask": torch.tensor([True], device=device, dtype=torch.bool),
            # 提供空的 ray_map 占位符 (B, H, W, 6)
            "ray_map": torch.zeros((1, H, W, 6), device=device, dtype=dtype),
            # 将 ray_mask 设为 False，明确告诉模型这部分信息缺失，防止其将 0 编码进特征
            "ray_mask": torch.tensor([False], device=device, dtype=torch.bool),
            # 核心逻辑：第一帧必须 reset=True 以清空 LocalMemory，后续帧 reset=False 继承记忆
            "reset": torch.tensor([True], device=device, dtype=torch.bool) if i == 0 else torch.tensor([False], device=device, dtype=torch.bool),
            # 允许模型更新状态
            "update": torch.tensor([True], device=device, dtype=torch.bool)
        }
        views.append(view)
        
    return views

def infer_monodepth(file: str, model: Pi3, hydra_cfg: DictConfig):

    imgs = load_and_resize14([file], new_width=hydra_cfg.load_img_size, device=hydra_cfg.device, verbose=hydra_cfg.verbose)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.amp.autocast(hydra_cfg.device, dtype=dtype):
            pred = model(imgs)

    points = pred['local_points'][0]         # (1, h_14, w_14, 3)
    depth_map = points[0, ..., -1].detach()  # (h_14, w_14)
    return depth_map  # torch.Tensor


def infer_videodepth(filelist: str, model: Pi3, hydra_cfg: DictConfig):
    imgs = load_and_resize_cut3r(filelist, new_width=hydra_cfg.load_img_size, device=hydra_cfg.device, verbose=hydra_cfg.verbose)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    start = time.time()
    with torch.no_grad():
        with torch.amp.autocast(hydra_cfg.device, dtype=dtype):
            views = _wrap_imgs_to_views_cut3r(imgs)
            pred = model(views)
    end = time.time()

    # === 开始 Cut3r 的输出解析逻辑 ===
    # 此时 pred 是 ARCroco3DStereoOutput 对象，pred.ress 是一个长度为 N 的列表
    if not hasattr(pred, 'ress') or not isinstance(pred.ress, list):
        raise RuntimeError(f"Unexpected model output type: {type(pred)}. Expected ARCroco3DStereoOutput.")
    
    depth_maps_list = []
    
    # 遍历每一帧的输出字典
    for i, res in enumerate(pred.ress):
        if not isinstance(res, dict):
            raise TypeError(f"Expected dict in pred.ress[{i}], got {type(res)}")
        
        # 针对 Cut3r 的下游 Head 进行精准提取
        if 'pts3d_in_self_view' in res:
            # 维度通常为 (1, H, W, 3)，提取 Z 轴 (深度)
            pts = res['pts3d_in_self_view']
            depth = pts[0, ..., -1]  # (H, W)
        elif 'pts3d' in res:
            pts = res['pts3d']
            depth = pts[0, ..., -1]  # (H, W)
        elif 'depth' in res:
            depth = res['depth'][0]  # (H, W)
        else:
            raise KeyError(f"Frame {i}: Could not extract depth. Available keys: {res.keys()}")
            
        depth_maps_list.append(depth)

    # 堆叠成 (N, H, W) 的连续内存张量
    depth_maps = torch.stack(depth_maps_list, dim=0).detach().cpu()
    # === 解析逻辑结束 ===
    # === 深度值域清洗与安全检查 ===
    # 1. 过滤所有病态的负深度值 (将其截断为非常小的正数)
    filtered_depth_maps = torch.clamp(depth_maps, min=1e-3)
    
    # 2. 打印当前张量的值域分布，打破你的盲点
    print(f"Depth tensor stats - Min: {filtered_depth_maps.min().item():.4f}, Max: {filtered_depth_maps.max().item():.4f}, Mean: {filtered_depth_maps.mean().item():.4f}")

    return end - start, depth_maps


def infer_cameras_w2c(filelist: str, model: Pi3, hydra_cfg: DictConfig):

    imgs = load_and_resize14(filelist, new_width=hydra_cfg.load_img_size, device=hydra_cfg.device, verbose=hydra_cfg.verbose)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.amp.autocast(hydra_cfg.device, dtype=dtype):
            pred = model(imgs)

    poses_c2w_all = pred['camera_poses'].cpu()
    extrinsics = se3_inverse(poses_c2w_all[0])

    return extrinsics, None


def infer_cameras_c2w(filelist: str, model: Pi3, hydra_cfg: DictConfig):

    imgs = load_and_resize14(filelist, new_width=hydra_cfg.load_img_size, device=hydra_cfg.device, verbose=hydra_cfg.verbose)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.amp.autocast(hydra_cfg.device, dtype=dtype):
            pred = model(imgs)

    poses_c2w_all = pred['camera_poses'].cpu()

    return poses_c2w_all[0], None

def infer_mv_pointclouds(filelist: str, model: Pi3, hydra_cfg: DictConfig, data_size: Tuple[int, int]):

    imgs = load_and_resize14(filelist, new_width=hydra_cfg.load_img_size, device=hydra_cfg.device, verbose=hydra_cfg.verbose)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.amp.autocast(hydra_cfg.device, dtype=dtype):
            pred = model(imgs)
    
    global_points = pred['points'][0]  # (N, h, w, 3)
    global_points = F.interpolate(
        global_points.permute(0, 3, 1, 2), data_size,
        mode="bilinear", align_corners=False, antialias=True
    ).permute(0, 2, 3, 1)  # align to gt

    return global_points.cpu().numpy()
def infer_restoration(file: str, model: Pi3, hydra_cfg: DictConfig):
    """Run restoration inference on a single image and return the clean image (numpy, RGB, uint8)."""
    imgs = load_and_resize14([file], new_width=hydra_cfg.load_img_size, device=hydra_cfg.device, verbose=hydra_cfg.verbose)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.amp.autocast(hydra_cfg.device, dtype=dtype):
            pred = model(imgs)

    if 'clean_image' not in pred:
        return None

    # pred['clean_image']: (B, N, C, H, W) in [0, 1]
    clean_img = pred['clean_image'][0, 0].detach().float().cpu()
    import numpy as np
    clean_np = clean_img.permute(1, 2, 0).numpy().clip(0, 1)
    clean_np = (clean_np * 255).astype(np.uint8)

    # Resize back to original image size
    from PIL import Image as PILImage
    orig_img = PILImage.open(file).convert('RGB')
    orig_w, orig_h = orig_img.size
    if clean_np.shape[0] != orig_h or clean_np.shape[1] != orig_w:
        import cv2
        clean_np = cv2.resize(clean_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    return clean_np  # (H, W, 3), RGB, uint8
