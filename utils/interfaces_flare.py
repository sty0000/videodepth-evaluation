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
        # 强制将传入的 new_width 也对齐到 patch_size 的倍数
        TARGET_W = round(new_width / patch_size) * patch_size
        TARGET_H = round(H_orig * (TARGET_W / W_orig) / patch_size) * patch_size
        
        # 防止因四舍五入导致尺寸变成 0
        TARGET_W = max(patch_size, TARGET_W)
        TARGET_H = max(patch_size, TARGET_H)
        
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

def _wrap_imgs_to_views_flare(imgs: torch.Tensor):
    """
    将 imgs (1, N, C, H, W) 转换为 FLARE 所需的 view1 和 view2。
    使用 Batch 维度并行处理 N 帧 (Self-pairing 单目模式)。
    """
    if imgs.ndim != 5:
        raise ValueError("Expected imgs with 5 dims (1, N, C, H, W)")
    
    B, N, C, H, W = imgs.shape
    device = imgs.device
    
    # 提取 N 帧作为 Batch 维度 -> (N, C, H, W)
    batch_imgs = imgs[0] 
    
    # 构造孪生网络的双输入
    view1 = {
        "img": batch_imgs,
        "true_shape": torch.tensor([[H, W]], device=device).repeat(N, 1),
        "idx": torch.arange(N, device=device)  # 帮助模型定位的辅助索引
    }
    
    # 将 view2 设为与 view1 完全一致，迫使模型进行单目退化推理
    view2 = {
        "img": batch_imgs,
        "true_shape": torch.tensor([[H, W]], device=device).repeat(N, 1),
        "idx": torch.arange(N, device=device)
    }
    
    return view1, view2

def infer_monodepth(file: str, model: Pi3, hydra_cfg: DictConfig):

    imgs = load_and_resize14([file], new_width=hydra_cfg.load_img_size, device=hydra_cfg.device, verbose=hydra_cfg.verbose)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.amp.autocast(hydra_cfg.device, dtype=dtype):
            pred = model(imgs)

    points = pred['local_points'][0]         # (1, h_14, w_14, 3)
    depth_map = points[0, ..., -1].detach()  # (h_14, w_14)
    return depth_map  # torch.Tensor


def infer_videodepth(filelist: List[str], model: any, hydra_cfg: DictConfig):
    # 安全的 Batch Size。
    batch_size = 2 
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    all_depth_maps = []
    start = time.time()
    
    # 按照 batch_size 切分长视频序列，分块推断
    for i in range(0, len(filelist), batch_size):
        chunk_files = filelist[i:i+batch_size]
        
        # 强制按 64 对齐加载图片
        imgs = load_and_resize_cut3r(chunk_files, new_width=hydra_cfg.load_img_size, device=hydra_cfg.device, verbose=hydra_cfg.verbose, patch_size=64)

        with torch.no_grad():
            with torch.amp.autocast(hydra_cfg.device, dtype=dtype):
                view1, view2 = _wrap_imgs_to_views_flare(imgs)
                
                # --- 终极物理对齐 ---
                # 无论之前怎么处理的，直接读取当前张量的绝对物理尺寸
                actual_h, actual_w = view1['img'].shape[-2], view1['img'].shape[-1]
                
                # 强行覆写 true_shape，杜绝任何小数或非 64 倍数的可能
                view1['true_shape'] = torch.tensor([[actual_h, actual_w]], device=hydra_cfg.device).repeat(view1['img'].shape[0], 1)
                view2['true_shape'] = torch.tensor([[actual_h, actual_w]], device=hydra_cfg.device).repeat(view2['img'].shape[0], 1)
                
                # 推理当前小块
                res1, res2, pred_cameras = model([view1], [view2])
                debug()  # 在这里设置断点，检查 res1、res2 和 pred_cameras 的内容，确保它们包含预期的键和值
                
        # === 解析原生输出 ===
        if 'pts3d' in res1:
            #pts = res1['pts3d'][:, 0]  # (B, H, W, 3)
            #depth_chunk = pts[..., -1].detach().cpu()  
            # 1. pts_world 是世界坐标系下的点云 (B, H, W, 3)
            # 关键：必须临时转换为 float32 进行几何运算，防止 bfloat16 的低精度导致空间旋转产生数值畸变
            pts_world = res1['pts3d'][:, 0].to(torch.float32) 
            B, H, W, _ = pts_world.shape
            
            # 2. 提取最后一次迭代收敛的相机位姿
            final_cameras = pred_cameras[-1]
            
            # 3. 提取 view1 的 C2W 旋转 R 和平移 T
            # final_cameras['R'] 的 shape 是 (B, 2, 3, 3)，[:, 0] 取出 view1
            R_c2w = final_cameras['R'][:, 0].to(torch.float32)  # (B, 3, 3)
            T_c2w = final_cameras['T'][:, 0].to(torch.float32)  # (B, 3)
            
            # 4. 执行数学求逆，构建 W2C 位姿
            R_w2c = R_c2w.transpose(1, 2)  # (B, 3, 3)
            # bmm 处理批量矩阵向量乘法: (B, 3, 3) @ (B, 3, 1) -> (B, 3)
            T_w2c = -torch.bmm(R_w2c, T_c2w.unsqueeze(-1)).squeeze(-1)  
            
            # 5. 高性能空间投影： P_c = R_w2c @ P_w + T_w2c
            # einsum 'bij,bhwj->bhwi' 等价于在每个像素的 3D 向量上做矩阵乘法
            pts_cam = torch.einsum('bij,bhwj->bhwi', R_w2c, pts_world) + T_w2c.view(-1, 1, 1, 3)
            
            # 6. 提取相机坐标系的真实深度 Z，并还原到原始精度 (bfloat16/float16)
            pts_cam = pts_cam.to(res1['pts3d'].dtype)
            depth_chunk = pts_cam[..., -1].detach().cpu()
            
            # 7. 物理防御性 Mask：剔除因网络回归失效导致在相机背后的噪点 (Z <= 0)
            valid_mask = depth_chunk > 0
            depth_chunk = depth_chunk * valid_mask
            all_depth_maps.append(depth_chunk)
        else:
            raise KeyError(f"Could not extract depth. Available keys: {res1.keys()}")
            
        del imgs, view1, view2, res1, res2, pred_cameras, pts
        torch.cuda.empty_cache()

    end = time.time()

    depth_maps = torch.cat(all_depth_maps, dim=0)

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
