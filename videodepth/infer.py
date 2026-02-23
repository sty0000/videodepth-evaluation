import hydra
import os
import os.path as osp
import torch
import logging
import json
import sys
import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf

# ==============================================================================
# 1. 路径与环境变量初始化
# ==============================================================================
import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# 导入上一回合统一后的接口
from utils.interfaces import infer_videodepth
from utils.files import get_all_sequences, list_imgs_a_sequence
from utils.messages import set_default_arg
from videodepth.utils import save_depth_maps

# ==============================================================================
# 2. 模型构建工厂 (附带绝对路径隔离与动态注入)
# ==============================================================================
def build_model(hydra_cfg: DictConfig, logger: logging.Logger):
    import sys
    model_type = hydra_cfg.get("model_type", "unknown").lower()
    logger.info(f"Initializing model factory for type: {model_type.upper()}")

    # ---------------------------------------------------------
    # FLARE 构建逻辑
    # ---------------------------------------------------------
    if model_type == "flare":
        # 1. 独占式注入：将 FLARE 及其依赖置于搜索路径最高优先级
        sys.path.insert(0, r"/data/users/yihao/codes/FLARE")
        sys.path.insert(0, r"/data/users/yihao/codes/FLARE/dust3r")
        
        # 2. 此时导入，绝对 100% 是 FLARE 目录下的包
        from mast3r.model import AsymmetricMASt3R
        
        pretrained_path = hydra_cfg.get("flare", {}).get("pretrained_model_name_or_path")
        if not pretrained_path:
            raise ValueError("FLARE config missing 'pretrained_model_name_or_path'")
            
        inf = float('inf')
        model = AsymmetricMASt3R(
            wpose=False, pos_embed='RoPE100', patch_embed_cls='PatchEmbedDust3R', 
            img_size=(512, 512), head_type='catmlp+dpt', output_mode='pts3d+desc24', 
            depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), 
            enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, 
            dec_embed_dim=768, dec_depth=12, dec_num_heads=12, 
            two_confs=True, desc_conf_mode=('exp', 0, inf)
        )
        
        ckpt = torch.load(pretrained_path, map_location='cpu')
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        logger.info(f"FLARE loaded. Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")
        
        model = model.to(hydra_cfg.device).eval()

        # 动态属性与 AMP 防御
        for name, module in model.named_modules():
            if hasattr(module, 'proj') and hasattr(module.proj, 'out_channels'):
                module.embed_dim = module.proj.out_channels
        if hasattr(model, 'patch_embed_coarse1'): model.patch_embed_coarse1.embed_dim = 1024
        if hasattr(model, 'patch_embed_coarse2'): model.patch_embed_coarse2.embed_dim = 1024

        def force_float32_output(module, input, output):
            return output.to(torch.float32)
        for module in model.modules():
            if module.__class__.__name__ == 'PatchEmbedDust3R' and hasattr(module, 'proj'):
                module.proj.register_forward_hook(force_float32_output)
                
        return model

    # ---------------------------------------------------------
    # CUT3R 构建逻辑
    # ---------------------------------------------------------
    elif model_type == "cut3r":
        # 1. 独占式注入：将 CUT3R 及其依赖置于搜索路径最高优先级
        sys.path.insert(0, r"/data/users/yihao/codes/CUT3R")
        sys.path.insert(0, r"/data/users/yihao/codes/CUT3R/src")
        
        # 2. 此时导入的 dust3r，绝对是 CUT3R 自己修改过的版本
        from dust3r.model import ARCroco3DStereo
        
        pretrained_path = hydra_cfg.get("cut3r", {}).get("pretrained_model_name_or_path")
        if not pretrained_path:
            raise ValueError("CUT3R config missing 'pretrained_model_name_or_path'")
            
        model = ARCroco3DStereo.from_pretrained(pretrained_path).to(hydra_cfg.device).eval()
        
        # RoPE 负索引越界热修复
        if hasattr(model, 'rope') and model.rope is not None:
            original_rope_forward = model.rope.forward
            def safe_rope_forward(*args, **kwargs):
                new_args = list(args)
                if len(new_args) >= 2 and new_args[1] is not None:
                    pos = new_args[1]
                    new_args[1] = torch.where(pos < 0, torch.zeros_like(pos), pos)
                elif 'positions' in kwargs and kwargs['positions'] is not None:
                    pos = kwargs['positions']
                    kwargs['positions'] = torch.where(pos < 0, torch.zeros_like(pos), pos)
                return original_rope_forward(*new_args, **kwargs)
            model.rope.forward = safe_rope_forward
            logger.info("Applied Hotfix: CUT3R RoPE negative index interceptor.")
            
        return model

    # ---------------------------------------------------------
    # FAST3R 构建逻辑
    # ---------------------------------------------------------
    elif model_type == "fast3r":
        # 1. 独占式注入：将 FAST3R 置于搜索路径最高优先级
        sys.path.insert(0, r"/data/users/yihao/codes/fast3r")
        sys.path.insert(0, r"/data/users/yihao/codes/Pi3") # 如果依赖 Pi3 库
        
        # 2. 此时导入的，绝对是 fast3r 项目下的 FlashDUSt3R
        from fast3r.dust3r.model import FlashDUSt3R, AsymmetricCroCo3DStereo
        
        pretrained_path = hydra_cfg.get("fast3r", {}).get("pretrained_model_name_or_path")
        if not pretrained_path:
            raise ValueError("Fast3R config missing 'pretrained_model_name_or_path'")
            
        try:
            # 优先尝试从预训练加载 FlashDUSt3R
            model = FlashDUSt3R.from_pretrained(pretrained_path).to(hydra_cfg.device).eval()
            logger.info("Successfully loaded FlashDUSt3R for fast3r.")
        except Exception as e:
            logger.warning(f"FlashDUSt3R load failed: {e}. Attempting AsymmetricCroCo3DStereo fallback...")
            from fast3r.dust3r.model import AsymmetricCroCo3DStereo
            model = AsymmetricCroCo3DStereo.from_pretrained(pretrained_path).to(hydra_cfg.device).eval()
            
        return model

    else:
        raise ValueError(f"未配置构建逻辑的模型类型: {model_type}")

# ==============================================================================
# 3. 核心评估主循环 (The Invariant)
# ==============================================================================
@hydra.main(version_base="1.2", config_path="../configs", config_name="eval")
def main(hydra_cfg: DictConfig):
    logger = logging.getLogger("videodepth-infer")
    
    # 必须在配置文件中或命令行指定你要跑的模型
    if "model_type" not in hydra_cfg:
        raise ValueError("严重错误: 请在 yaml 或命令行中设置 model_type (例如: model_type=flare)")
        
    all_eval_datasets: ListConfig = hydra_cfg.eval_datasets 
    all_data_info: DictConfig = hydra_cfg.data 

    # --- 调用工厂创建模型 ---
    try:
        model = build_model(hydra_cfg, logger)
    except Exception as e:
        logger.exception(f"Failed to build model {hydra_cfg.model_type}. Aborting.")
        raise e

    # --- 数据集级循环 ---
    for idx_dataset, dataset_name in enumerate(all_eval_datasets, start=1):
        if dataset_name not in all_data_info:
            raise ValueError(f"Unknown dataset in global data information: {dataset_name}")
        dataset_info = all_data_info[dataset_name]
        
        output_root = osp.join(hydra_cfg.output_dir, dataset_name)
        logger.info(f"[{idx_dataset}/{len(all_eval_datasets)}] Infering videodepth on {dataset_name} dataset... Output: {osp.relpath(output_root, hydra_cfg.work_dir)}")

        # --- SQUID 数据集处理 ---
        if dataset_info.type == "squid":
            from utils.files import get_squid_image_sets, list_squid_imgs_an_image_set
            total_sets, processed = 0, 0
            
            for scene in dataset_info.ls_all_seqs:
                image_sets = get_squid_image_sets(dataset_info, scene)
                total_sets += len(image_sets)
                
                for image_set in image_sets:
                    filelist = list_squid_imgs_an_image_set(dataset_info, scene, image_set)
                    if filelist is None: continue
                    
                    save_dir = osp.join(output_root, scene, image_set)
                    if not hydra_cfg.overwrite and osp.isdir(save_dir) and len(os.listdir(save_dir)) >= 2:
                        processed += 1
                        continue
                    
                    # === 唯一不变的接口调用 ===
                    time_used, depth_maps = infer_videodepth(filelist, model, hydra_cfg)
                    logger.info(f"[{scene}/{image_set}] processed, time: {time_used}")
                    
                    os.makedirs(save_dir, exist_ok=True)
                    save_depth_maps(depth_maps, save_dir)
                    with open(osp.join(save_dir, "_time.json"), "w") as f:
                        json.dump({"time": time_used, "frames": len(filelist)}, f, indent=4)
                    processed += 1
            logger.info(f"SQUID: processed {processed}/{total_sets} image_sets")
            continue 
            
        # --- 常规 Video 数据集处理 ---
        elif dataset_info.type == "video":
            seq_list = get_all_sequences(dataset_info)
        elif dataset_info.type == "mono":
            raise ValueError("dataset type `mono` is not supported for videodepth evaluation")
        else:
            raise ValueError(f"Unknown dataset type: {dataset_info.type}")

        for seq_idx, seq in enumerate(seq_list, start=1):
            filelist = list_imgs_a_sequence(dataset_info, seq)
            
            max_frames = hydra_cfg.get("max_frames", None)
            if max_frames is not None and len(filelist) > max_frames:
                indices = np.linspace(0, len(filelist) - 1, max_frames, dtype=int)
                filelist = [filelist[i] for i in indices]
                
            save_dir = osp.join(output_root, seq)

            if not hydra_cfg.overwrite and (osp.isdir(save_dir) and len(os.listdir(save_dir)) == 2 * len(filelist) + 1):
                logger.info(f"[{seq_idx}/{len(seq_list)}] Sequence {seq} already processed, skipping.")
                continue
            
            # === 唯一不变的接口调用 ===
            time_used, depth_maps = infer_videodepth(filelist, model, hydra_cfg)
            logger.info(f"[{seq_idx}/{len(seq_list)}] Sequence {seq} processed, time: {time_used}, saving depth maps...")

            os.makedirs(save_dir, exist_ok=True)
            save_depth_maps(depth_maps, save_dir)
            with open(osp.join(save_dir, "_time.json"), "w") as f:
                json.dump({"time": time_used, "frames": len(filelist)}, f, indent=4)
                
    # 释放显存资源
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    set_default_arg("evaluation", "videodepth")
    os.environ["HYDRA_FULL_ERROR"] = '1'
    with torch.no_grad():
        main()