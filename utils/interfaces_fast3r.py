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


def load_images(filelist: List[str], PIXEL_LIMIT: int = 255000, new_width: Optional[int] = None, verbose: bool = False):
    """
    Loads images from a directory or video, resizes them to a uniform size,
    then converts and stacks them into a single [N, 3, H, W] PyTorch tensor.
    """
    sources = [] 
    
    # --- 1. Load image paths or video frames ---
    for img_path in filelist:
        try:
            sources.append(Image.open(img_path).convert('RGB'))
        except Exception as e:
            print(f"Could not load image {img_path}: {e}")

    if not sources:
        print("No images found or loaded.")
        return torch.empty(0)

    if verbose:
        print(f"Found {len(sources)} images/frames. Processing...")

    # --- 2. Determine a uniform target size for all images based on the first image ---
    # This is necessary to ensure all tensors have the same dimensions for stacking.
    first_img = sources[0]
    W_orig, H_orig = first_img.size
    if new_width is None:
        scale = math.sqrt(PIXEL_LIMIT / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
        W_target, H_target = W_orig * scale, H_orig * scale
        k, m = round(W_target / 14), round(H_target / 14)
        while (k * 14) * (m * 14) > PIXEL_LIMIT:
            if k / m > W_target / H_target: k -= 1
            else: m -= 1
        TARGET_W, TARGET_H = max(1, k) * 14, max(1, m) * 14
    else:
        TARGET_W, TARGET_H = new_width, round(H_orig * (new_width / W_orig) / 14) * 14
    try:
        lcm = math.lcm(14,16)
    except AttributeError:
        def _gcd(a, b):
            while b:
                a, b = b, a % b
            return a
        lcm = 14 * 16 // _gcd(14, 16)
    def _round_to_multiple(x, m):
        x = max(int(m), int(round(x/m) * m))
        return x
    TARGET_W = _round_to_multiple(TARGET_W, lcm)
    TARGET_H = _round_to_multiple(TARGET_H, lcm)
    if verbose:
        print(f"All images will be resized to a uniform size: ({TARGET_W}, {TARGET_H})")

    # --- 3. Resize images and convert them to tensors in the [0, 1] range ---
    tensor_list = []
    # Define a transform to convert a PIL Image to a CxHxW tensor and normalize to [0,1]
    to_tensor_transform = tvf.ToTensor()
    
    for img_pil in sources:
        try:
            # Resize to the uniform target size
            resized_img = img_pil.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
            # Convert to tensor
            img_tensor = to_tensor_transform(resized_img)
            tensor_list.append(img_tensor)
        except Exception as e:
            print(f"Error processing an image: {e}")

    if not tensor_list:
        print("No images were successfully processed.")
        return torch.empty(0)

    # --- 4. Stack the list of tensors into a single [N, C, H, W] batch tensor ---
    return torch.stack(tensor_list, dim=0)


def load_and_resize14(filelist: List[str], new_width: int, device: str, verbose: bool):
    imgs = load_images(filelist, new_width=new_width, verbose=verbose).to(device)

    ori_h, ori_w = imgs.shape[-2:]
    patch_h, patch_w = ori_h // 14, ori_w // 14
    # (N, 3, h, w) -> (1, N, 3, h_14, w_14)
    imgs = F.interpolate(imgs, (patch_h * 14, patch_w * 14), mode="bilinear", align_corners=False, antialias=True).unsqueeze(0)
    return imgs

def _wrap_imgs_to_views(imgs: torch.Tensor):
    """
    Convert imgs shaped (B=1, N, C, H, W) to list of dicts expected by Fast3R:
    [{"img": tensor_of_shape_(B, C, H, W)}, ...]
    """
    if imgs.ndim != 5:
        raise ValueError("Expected imgs with 5 dims (1, N, C, H, W)")
    B, N = imgs.shape[0], imgs.shape[1]
    views = [{"img": imgs[:, i]} for i in range(N)]
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

    imgs = load_and_resize14(filelist, new_width=hydra_cfg.load_img_size, device=hydra_cfg.device, verbose=hydra_cfg.verbose)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    start = time.time()
    with torch.no_grad():
        with torch.amp.autocast(hydra_cfg.device, dtype=dtype):
            views = _wrap_imgs_to_views(imgs) if isinstance(imgs, torch.Tensor) and imgs.ndim == 5 else imgs
            pred = model(views)
    end = time.time()

    # === 强契约提取：拒绝一切模糊猜测 ===
    # 明确 Fast3R/Pi3 模型的相机坐标系点云键名 (请确认你的模型使用的是 pts3d_local 还是 local_points)
    target_key = 'pts3d_local' 
    
    if target_key not in pred:
        raise KeyError(f"Fatal Error: Model output missing strictly required key '{target_key}'. Available keys: {pred.keys()}")
    
    # 获取相机坐标系 3D 点云
    pts_cam = pred[target_key]
    
    # 明确维度提取 (假设输出是 B=1, 提取 batch 0)
    # 此时的 pts_cam[0] shape 应当严格是 (N, H, W, 3) 或 (H, W, 3)
    pts_cam_squeeze = pts_cam[0] 
    
    # 提取相机坐标系深度 Z
    depth_map = pts_cam_squeeze[..., -1].detach().cpu()
    
    # --- 不要忘了上一回合确认的物理防御 ---
    # 剔除相机背后 (Z <= 0) 的无效网络噪点
    valid_mask = depth_map > 0
    depth_map = depth_map * valid_mask

    return end - start, depth_map


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
