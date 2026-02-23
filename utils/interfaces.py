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

# ==============================================================================
# 1. Fast3R / Pi3 专用的图像加载与预处理
# ==============================================================================
def load_images(filelist: List[str], PIXEL_LIMIT: int = 255000, new_width: Optional[int] = None, verbose: bool = False):
    sources = [] 
    for img_path in filelist:
        try:
            sources.append(Image.open(img_path).convert('RGB'))
        except Exception as e:
            print(f"Could not load image {img_path}: {e}")

    if not sources:
        print("No images found or loaded.")
        return torch.empty(0)

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

    tensor_list = []
    to_tensor_transform = tvf.ToTensor()
    for img_pil in sources:
        try:
            resized_img = img_pil.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
            tensor_list.append(to_tensor_transform(resized_img))
        except Exception as e:
            print(f"Error processing an image: {e}")

    if not tensor_list:
        return torch.empty(0)
    return torch.stack(tensor_list, dim=0)

def load_and_resize14(filelist: List[str], new_width: int, device: str, verbose: bool):
    imgs = load_images(filelist, new_width=new_width, verbose=verbose).to(device)
    ori_h, ori_w = imgs.shape[-2:]
    patch_h, patch_w = ori_h // 14, ori_w // 14
    imgs = F.interpolate(imgs, (patch_h * 14, patch_w * 14), mode="bilinear", align_corners=False, antialias=True).unsqueeze(0)
    return imgs

def _wrap_imgs_to_views(imgs: torch.Tensor):
    if imgs.ndim != 5:
        raise ValueError("Expected imgs with 5 dims (1, N, C, H, W)")
    B, N = imgs.shape[0], imgs.shape[1]
    views = [{"img": imgs[:, i]} for i in range(N)]
    return views


# ==============================================================================
# 2. CUT3R & FLARE 专用的图像加载与预处理
# ==============================================================================
def load_images_cut3r(filelist: List[str], PIXEL_LIMIT: int = 255000, new_width: Optional[int] = None, verbose: bool = False, patch_size: int = 16):
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
    
    if new_width is None:
        scale = math.sqrt(PIXEL_LIMIT / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
        W_target, H_target = W_orig * scale, H_orig * scale
        k, m = round(W_target / patch_size), round(H_target / patch_size)
        while (k * patch_size) * (m * patch_size) > PIXEL_LIMIT:
            if k / m > W_target / H_target: k -= 1
            else: m -= 1
        TARGET_W, TARGET_H = max(1, k) * patch_size, max(1, m) * patch_size
    else:
        TARGET_W = round(new_width / patch_size) * patch_size
        TARGET_H = round(H_orig * (TARGET_W / W_orig) / patch_size) * patch_size
        TARGET_W = max(patch_size, TARGET_W)
        TARGET_H = max(patch_size, TARGET_H)
        
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
    patch_h, patch_w = ori_h // patch_size, ori_w // patch_size
    imgs = F.interpolate(imgs, (patch_h * patch_size, patch_w * patch_size), mode="bilinear", align_corners=False, antialias=True).unsqueeze(0)
    return imgs

def _wrap_imgs_to_views_cut3r(imgs: torch.Tensor):
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
            "img": imgs[:, i],
            "true_shape": torch.tensor([[H, W]], device=device),
            "img_mask": torch.tensor([True], device=device, dtype=torch.bool),
            "ray_map": torch.zeros((1, H, W, 6), device=device, dtype=dtype),
            "ray_mask": torch.tensor([False], device=device, dtype=torch.bool),
            "reset": torch.tensor([True], device=device, dtype=torch.bool) if i == 0 else torch.tensor([False], device=device, dtype=torch.bool),
            "update": torch.tensor([True], device=device, dtype=torch.bool)
        }
        views.append(view)
        
    return views

def _wrap_imgs_to_views_flare(imgs: torch.Tensor):
    if imgs.ndim != 5:
        raise ValueError("Expected imgs with 5 dims (1, N, C, H, W)")
    
    B, N, C, H, W = imgs.shape
    device = imgs.device
    batch_imgs = imgs[0] 
    
    view1 = {
        "img": batch_imgs,
        "true_shape": torch.tensor([[H, W]], device=device).repeat(N, 1),
        "idx": torch.arange(N, device=device) 
    }
    
    view2 = {
        "img": batch_imgs,
        "true_shape": torch.tensor([[H, W]], device=device).repeat(N, 1),
        "idx": torch.arange(N, device=device)
    }
    
    return view1, view2


# ==============================================================================
# 3. 核心统一接口：推理路由分发中心
# ==============================================================================
def infer_videodepth(filelist: List[str], model: any, hydra_cfg: DictConfig):
    """
    统一推理入口，内部根据 hydra_cfg.model_type 或 model 类名进行物理切割分流。
    """
    # 动态探测当前运行的模型身份
    model_type = hydra_cfg.get("model_type", "unknown").lower()
    
    # 如果 cfg 中没写，强制通过类名进行推断 (防御性探测)
    if model_type == "unknown":
        class_name = model.__class__.__name__.lower()
        if "mast3r" in class_name or "flare" in class_name:
            model_type = "flare"
        elif "croco" in class_name or "cut3r" in class_name:
            model_type = "cut3r"
        else:
            model_type = "fast3r" # 默认 fallback 为 Pi3/Fast3R

    # ---------------------------------------------------------
    # 分支 1: FLARE 逻辑 (已包含 Chunking 和 W2C 对齐)
    # ---------------------------------------------------------
    if model_type == "flare":
        batch_size = 2 
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        all_depth_maps = []
        start = time.time()
        
        for i in range(0, len(filelist), batch_size):
            chunk_files = filelist[i:i+batch_size]
            imgs = load_and_resize_cut3r(chunk_files, new_width=hydra_cfg.load_img_size, device=hydra_cfg.device, verbose=hydra_cfg.verbose, patch_size=64)

            with torch.no_grad(), torch.amp.autocast(hydra_cfg.device, dtype=dtype):
                view1, view2 = _wrap_imgs_to_views_flare(imgs)
                actual_h, actual_w = view1['img'].shape[-2], view1['img'].shape[-1]
                view1['true_shape'] = torch.tensor([[actual_h, actual_w]], device=hydra_cfg.device).repeat(view1['img'].shape[0], 1)
                view2['true_shape'] = torch.tensor([[actual_h, actual_w]], device=hydra_cfg.device).repeat(view2['img'].shape[0], 1)
                res1, res2, pred_cameras = model([view1], [view2])
                
            if 'pts3d' in res1:
                pts_world = res1['pts3d'][:, 0].to(torch.float32) 
                B, H, W, _ = pts_world.shape
                final_cameras = pred_cameras[-1]
                R_c2w = final_cameras['R'][:, 0].to(torch.float32) 
                T_c2w = final_cameras['T'][:, 0].to(torch.float32) 
                R_w2c = R_c2w.transpose(1, 2) 
                T_w2c = -torch.bmm(R_w2c, T_c2w.unsqueeze(-1)).squeeze(-1)  
                pts_cam = torch.einsum('bij,bhwj->bhwi', R_w2c, pts_world) + T_w2c.view(-1, 1, 1, 3)
                pts_cam = pts_cam.to(res1['pts3d'].dtype)
                depth_chunk = pts_cam[..., -1].detach().cpu()
                
                valid_mask = depth_chunk > 0
                depth_chunk = depth_chunk * valid_mask
                all_depth_maps.append(depth_chunk)
            else:
                raise KeyError(f"Could not extract depth. Available keys: {res1.keys()}")
                
            del imgs, view1, view2, res1, res2, pred_cameras, pts_world, pts_cam
            torch.cuda.empty_cache()

        end = time.time()
        depth_maps = torch.cat(all_depth_maps, dim=0)
        # ==========================================================
        # DEBUG: 抽取图像与深度图进行伪彩色拼接可视化 (仅取首帧和末帧)
        # ==========================================================
        try:
            import matplotlib.cm as cm
            import numpy as np
            
            # 我们抽取视频的第 1 帧(idx=0) 和 最后 1 帧(idx=-1) 来检查
            for debug_idx, frame_name in zip([0, -1], ["first_frame", "last_frame"]):
                if abs(debug_idx) >= len(filelist) and debug_idx != 0:
                    continue
                    
                debug_img_path = filelist[debug_idx]
                debug_img = Image.open(debug_img_path).convert('RGB')
                
                # 获取对应的深度张量并转为 numpy
                # depth_maps shape is (N, H, W)
                debug_depth = depth_maps[debug_idx].clone().numpy()
                
                # 将原图 Resize 到深度图的分辨率以便于物理对齐拼接
                H, W = debug_depth.shape
                debug_img = debug_img.resize((W, H), Image.Resampling.LANCZOS)
                debug_img_np = np.array(debug_img)
                
                # 深度图归一化与伪彩色化 (使用 inferno 色带，高对比度)
                # 使用 2% 和 98% 分位数截断，防止极端噪点拉低整体对比度
                d_min, d_max = np.percentile(debug_depth, [2, 98])
                # 防止除以 0
                depth_norm = np.clip((debug_depth - d_min) / (d_max - d_min + 1e-5), 0, 1)
                depth_color = (cm.inferno(depth_norm)[:, :, :3] * 255).astype(np.uint8)
                
                # 水平拼接：左边 RGB，右边 Depth
                concat_img = np.concatenate([debug_img_np, depth_color], axis=1)
                
                # 保存图像 (加上模型名字避免覆盖)
                save_name = f"debug_vis_{model_type}_{frame_name}.png"
                Image.fromarray(concat_img).save(save_name)
                print(f"[DEBUG Probe] Saved visualization to {save_name}")
                
        except Exception as e:
            print(f"[DEBUG Probe] Visualization failed: {e}")

        # === 原有的返回语句 ===
        return end - start, depth_maps

    # ---------------------------------------------------------
    # 分支 2: FAST3R 逻辑
    # ---------------------------------------------------------
    elif model_type in ["fast3r", "pi3"]:
        imgs = load_and_resize14(filelist, new_width=hydra_cfg.load_img_size, device=hydra_cfg.device, verbose=hydra_cfg.verbose)
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        start = time.time()
        with torch.no_grad(), torch.amp.autocast(hydra_cfg.device, dtype=dtype):
            views = _wrap_imgs_to_views(imgs) if isinstance(imgs, torch.Tensor) and imgs.ndim == 5 else imgs
            pred = model(views)  # pred 是一个 list[dict]
        end = time.time()

        # === 基于源码的强契约提取 ===
        if not isinstance(pred, list):
            raise TypeError(f"Expected Fast3R output to be a list, got {type(pred)}")

        depth_maps_list = []
        target_key = 'pts3d_in_other_view'  # 源码铁证指定的键名
        
        for i, res in enumerate(pred):
            if target_key not in res:
                # 兼容性回退：以防你加载的是单帧退化的 AsymmetricCroCo3DStereo
                if 'pts3d' in res:
                    target_key = 'pts3d'
                else:
                    raise KeyError(f"Frame {i}: Fast3R missing '{target_key}' or 'pts3d'. Keys: {res.keys()}")
            
            # 源码中 img_result 的 shape 是 (B, H, W, 3)。我们提取 B=0 的 Z 轴。
            depth = res[target_key][0, ..., -1]  # (H, W)
            depth_maps_list.append(depth)
            
        depth_map = torch.stack(depth_maps_list, dim=0).detach().cpu()  # (N, H, W)
        
        # 物理防御：剔除相机背后的无效噪点
        valid_mask = depth_map > 0
        depth_map = depth_map * valid_mask

        # ==========================================================
        # DEBUG: 抽取图像与深度图进行伪彩色拼接可视化 (仅取首帧和末帧)
        # ==========================================================
        try:
            import matplotlib.cm as cm
            import numpy as np
            
            # 我们抽取视频的第 1 帧(idx=0) 和 最后 1 帧(idx=-1) 来检查
            for debug_idx, frame_name in zip([0, -1], ["first_frame", "last_frame"]):
                if abs(debug_idx) >= len(filelist) and debug_idx != 0:
                    continue
                    
                debug_img_path = filelist[debug_idx]
                debug_img = Image.open(debug_img_path).convert('RGB')
                
                # 获取对应的深度张量并转为 numpy
                # depth_maps shape is (N, H, W)
                debug_depth = depth_maps[debug_idx].clone().numpy()
                
                # 将原图 Resize 到深度图的分辨率以便于物理对齐拼接
                H, W = debug_depth.shape
                debug_img = debug_img.resize((W, H), Image.Resampling.LANCZOS)
                debug_img_np = np.array(debug_img)
                
                # 深度图归一化与伪彩色化 (使用 inferno 色带，高对比度)
                # 使用 2% 和 98% 分位数截断，防止极端噪点拉低整体对比度
                d_min, d_max = np.percentile(debug_depth, [2, 98])
                # 防止除以 0
                depth_norm = np.clip((debug_depth - d_min) / (d_max - d_min + 1e-5), 0, 1)
                depth_color = (cm.inferno(depth_norm)[:, :, :3] * 255).astype(np.uint8)
                
                # 水平拼接：左边 RGB，右边 Depth
                concat_img = np.concatenate([debug_img_np, depth_color], axis=1)
                
                # 保存图像 (加上模型名字避免覆盖)
                save_name = f"debug_vis_{model_type}_{frame_name}.png"
                Image.fromarray(concat_img).save(save_name)
                print(f"[DEBUG Probe] Saved visualization to {save_name}")
                
        except Exception as e:
            print(f"[DEBUG Probe] Visualization failed: {e}")

        # === 原有的返回语句 ===
        return end - start, depth_maps

    # ---------------------------------------------------------
    # 分支 3: PI3 逻辑
    # ---------------------------------------------------------
    elif model_type == "pi3":
        imgs = load_and_resize14(filelist, new_width=hydra_cfg.load_img_size, device=hydra_cfg.device, verbose=hydra_cfg.verbose)
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        start = time.time()
        with torch.no_grad(), torch.amp.autocast(hydra_cfg.device, dtype=dtype):
            views = _wrap_imgs_to_views(imgs) if isinstance(imgs, torch.Tensor) and imgs.ndim == 5 else imgs
            pred = model(views)
        end = time.time()

        target_key = 'pts3d_local' 
        if target_key not in pred:
            raise KeyError(f"Fatal Error: Model output missing strictly required key '{target_key}'. Available keys: {pred.keys()}")
        
        pts_cam = pred[target_key]
        pts_cam_squeeze = pts_cam[0] 
        depth_map = pts_cam_squeeze[..., -1].detach().cpu()
        
        valid_mask = depth_map > 0
        depth_map = depth_map * valid_mask

        # ==========================================================
        # DEBUG: 抽取图像与深度图进行伪彩色拼接可视化 (仅取首帧和末帧)
        # ==========================================================
        try:
            import matplotlib.cm as cm
            import numpy as np
            
            # 我们抽取视频的第 1 帧(idx=0) 和 最后 1 帧(idx=-1) 来检查
            for debug_idx, frame_name in zip([0, -1], ["first_frame", "last_frame"]):
                if abs(debug_idx) >= len(filelist) and debug_idx != 0:
                    continue
                    
                debug_img_path = filelist[debug_idx]
                debug_img = Image.open(debug_img_path).convert('RGB')
                
                # 获取对应的深度张量并转为 numpy
                # depth_maps shape is (N, H, W)
                debug_depth = depth_maps[debug_idx].clone().numpy()
                
                # 将原图 Resize 到深度图的分辨率以便于物理对齐拼接
                H, W = debug_depth.shape
                debug_img = debug_img.resize((W, H), Image.Resampling.LANCZOS)
                debug_img_np = np.array(debug_img)
                
                # 深度图归一化与伪彩色化 (使用 inferno 色带，高对比度)
                # 使用 2% 和 98% 分位数截断，防止极端噪点拉低整体对比度
                d_min, d_max = np.percentile(debug_depth, [2, 98])
                # 防止除以 0
                depth_norm = np.clip((debug_depth - d_min) / (d_max - d_min + 1e-5), 0, 1)
                depth_color = (cm.inferno(depth_norm)[:, :, :3] * 255).astype(np.uint8)
                
                # 水平拼接：左边 RGB，右边 Depth
                concat_img = np.concatenate([debug_img_np, depth_color], axis=1)
                
                # 保存图像 (加上模型名字避免覆盖)
                save_name = f"debug_vis_{model_type}_{frame_name}.png"
                Image.fromarray(concat_img).save(save_name)
                print(f"[DEBUG Probe] Saved visualization to {save_name}")
                
        except Exception as e:
            print(f"[DEBUG Probe] Visualization failed: {e}")

        # === 原有的返回语句 ===
        return end - start, depth_maps

    # ---------------------------------------------------------
    # 分支 4: CUT3R 逻辑
    # ---------------------------------------------------------
    elif model_type == "cut3r":
        imgs = load_and_resize_cut3r(filelist, new_width=hydra_cfg.load_img_size, device=hydra_cfg.device, verbose=hydra_cfg.verbose)
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        start = time.time()
        with torch.no_grad(), torch.amp.autocast(hydra_cfg.device, dtype=dtype):
            views = _wrap_imgs_to_views_cut3r(imgs)
            pred = model(views)
        end = time.time()

        if not hasattr(pred, 'ress') or not isinstance(pred.ress, list):
            raise RuntimeError(f"Unexpected model output type: {type(pred)}. Expected ARCroco3DStereoOutput.")
        
        depth_maps_list = []
        for i, res in enumerate(pred.ress):
            if not isinstance(res, dict):
                raise TypeError(f"Expected dict in pred.ress[{i}], got {type(res)}")
            
            if 'pts3d_in_self_view' in res:
                depth = res['pts3d_in_self_view'][0, ..., -1]  
            elif 'pts3d' in res:
                depth = res['pts3d'][0, ..., -1]  
            elif 'depth' in res:
                depth = res['depth'][0]  
            else:
                raise KeyError(f"Frame {i}: Could not extract depth. Available keys: {res.keys()}")
            depth_maps_list.append(depth)

        depth_maps = torch.stack(depth_maps_list, dim=0).detach().cpu()
        depth_maps = torch.clamp(depth_maps, min=1e-3)

        # ==========================================================
        # DEBUG: 抽取图像与深度图进行伪彩色拼接可视化 (仅取首帧和末帧)
        # ==========================================================
        try:
            import matplotlib.cm as cm
            import numpy as np
            
            # 我们抽取视频的第 1 帧(idx=0) 和 最后 1 帧(idx=-1) 来检查
            for debug_idx, frame_name in zip([0, -1], ["first_frame", "last_frame"]):
                if abs(debug_idx) >= len(filelist) and debug_idx != 0:
                    continue
                    
                debug_img_path = filelist[debug_idx]
                debug_img = Image.open(debug_img_path).convert('RGB')
                
                # 获取对应的深度张量并转为 numpy
                # depth_maps shape is (N, H, W)
                debug_depth = depth_maps[debug_idx].clone().numpy()
                
                # 将原图 Resize 到深度图的分辨率以便于物理对齐拼接
                H, W = debug_depth.shape
                debug_img = debug_img.resize((W, H), Image.Resampling.LANCZOS)
                debug_img_np = np.array(debug_img)
                
                # 深度图归一化与伪彩色化 (使用 inferno 色带，高对比度)
                # 使用 2% 和 98% 分位数截断，防止极端噪点拉低整体对比度
                d_min, d_max = np.percentile(debug_depth, [2, 98])
                # 防止除以 0
                depth_norm = np.clip((debug_depth - d_min) / (d_max - d_min + 1e-5), 0, 1)
                depth_color = (cm.inferno(depth_norm)[:, :, :3] * 255).astype(np.uint8)
                
                # 水平拼接：左边 RGB，右边 Depth
                concat_img = np.concatenate([debug_img_np, depth_color], axis=1)
                
                # 保存图像 (加上模型名字避免覆盖)
                save_name = f"debug_vis_{model_type}_{frame_name}.png"
                Image.fromarray(concat_img).save(save_name)
                print(f"[DEBUG Probe] Saved visualization to {save_name}")
                
        except Exception as e:
            print(f"[DEBUG Probe] Visualization failed: {e}")

        # === 原有的返回语句 ===
        return end - start, depth_maps

    else:
        raise ValueError(f"系统路由错误: 未知的模型类型标识 '{model_type}'")


# ==============================================================================
# 4. 未被调用但要求保留的尾部接口函数
# ==============================================================================
def infer_monodepth(file: str, model: Pi3, hydra_cfg: DictConfig):
    imgs = load_and_resize14([file], new_width=hydra_cfg.load_img_size, device=hydra_cfg.device, verbose=hydra_cfg.verbose)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad(), torch.amp.autocast(hydra_cfg.device, dtype=dtype):
        pred = model(imgs)
    points = pred['local_points'][0]        
    depth_map = points[0, ..., -1].detach() 
    return depth_map

def infer_cameras_w2c(filelist: str, model: Pi3, hydra_cfg: DictConfig):
    imgs = load_and_resize14(filelist, new_width=hydra_cfg.load_img_size, device=hydra_cfg.device, verbose=hydra_cfg.verbose)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad(), torch.amp.autocast(hydra_cfg.device, dtype=dtype):
        pred = model(imgs)
    poses_c2w_all = pred['camera_poses'].cpu()
    extrinsics = se3_inverse(poses_c2w_all[0])
    return extrinsics, None

def infer_cameras_c2w(filelist: str, model: Pi3, hydra_cfg: DictConfig):
    imgs = load_and_resize14(filelist, new_width=hydra_cfg.load_img_size, device=hydra_cfg.device, verbose=hydra_cfg.verbose)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad(), torch.amp.autocast(hydra_cfg.device, dtype=dtype):
        pred = model(imgs)
    poses_c2w_all = pred['camera_poses'].cpu()
    return poses_c2w_all[0], None

def infer_mv_pointclouds(filelist: str, model: Pi3, hydra_cfg: DictConfig, data_size: Tuple[int, int]):
    imgs = load_and_resize14(filelist, new_width=hydra_cfg.load_img_size, device=hydra_cfg.device, verbose=hydra_cfg.verbose)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad(), torch.amp.autocast(hydra_cfg.device, dtype=dtype):
        pred = model(imgs)
    global_points = pred['points'][0]  
    global_points = F.interpolate(
        global_points.permute(0, 3, 1, 2), data_size,
        mode="bilinear", align_corners=False, antialias=True
    ).permute(0, 2, 3, 1) 
    return global_points.cpu().numpy()

def infer_restoration(file: str, model: Pi3, hydra_cfg: DictConfig):
    imgs = load_and_resize14([file], new_width=hydra_cfg.load_img_size, device=hydra_cfg.device, verbose=hydra_cfg.verbose)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad(), torch.amp.autocast(hydra_cfg.device, dtype=dtype):
        pred = model(imgs)
    if 'clean_image' not in pred:
        return None
    clean_img = pred['clean_image'][0, 0].detach().float().cpu()
    import numpy as np
    clean_np = clean_img.permute(1, 2, 0).numpy().clip(0, 1)
    clean_np = (clean_np * 255).astype(np.uint8)
    from PIL import Image as PILImage
    orig_img = PILImage.open(file).convert('RGB')
    orig_w, orig_h = orig_img.size
    if clean_np.shape[0] != orig_h or clean_np.shape[1] != orig_w:
        import cv2
        clean_np = cv2.resize(clean_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return clean_np