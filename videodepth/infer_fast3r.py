import hydra
import sys
import os
import os.path as osp
import torch
import logging
import json
from omegaconf import DictConfig, ListConfig

append_paths = [r"/data/users/yihao/codes/Pi3", r"/data/users/yihao/codes/fast3r", r"/data/users/yihao/codes/CUT3R/src"]
for path in append_paths:
    sys.path.append(path)

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
#from pi3.models.pi3 import Pi3
#from depth_anything_3.model.da3_training import DepthAnything3Net
from dust3r.model import ARCroco3DStereo
from utils.interfaces import infer_videodepth
from utils.files import get_all_sequences, list_imgs_a_sequence
from utils.messages import set_default_arg
from videodepth.utils import save_depth_maps
from omegaconf import OmegaConf
import numpy as np

@hydra.main(version_base="1.2", config_path="../configs", config_name="eval")
def main(hydra_cfg: DictConfig):
    all_eval_datasets: ListConfig      = hydra_cfg.eval_datasets  # see configs/evaluation/videodepth.yaml
    all_data_info: DictConfig          = hydra_cfg.data           # see configs/data/depth.yaml
    pretrained_model_name_or_path: str = hydra_cfg.cut3r.pretrained_model_name_or_path  # see configs/evaluation/videodepth.yaml

    # 0. create model
    
    #cfg = OmegaConf.load("configs/model/da3.yaml")
    #model_cfg = OmegaConf.to_container(cfg["model"], resolve=True)
    #model_cfg.pop("_target_", None)
    #model_cfg.pop("ckpt", None)

    #model = DepthAnything3Net(**model_cfg)
    #model.to(hydra_cfg.device).eval()

    # model = Pi3().to(hydra_cfg.device).eval()
    #state_dict = torch.load(pretrained_model_name_or_path, map_location=hydra_cfg.device,weights_only=True)
    #model.load_state_dict(state_dict, strict=False)
    
    # model = Pi3.from_pretrained(pretrained_model_name_or_path).to(hydra_cfg.device).eval()

    # 使用 CUT3R 的 ARCroco3DStereo.from_pretrained 加载
    model = ARCroco3DStereo.from_pretrained(pretrained_model_name_or_path).to(hydra_cfg.device).eval()
    logger = logging.getLogger("videodepth-infer")
    logger.info(f"Loaded CUT3R (ARCroco3DStereo) from {pretrained_model_name_or_path}")

    for idx_dataset, dataset_name in enumerate(all_eval_datasets, start=1):
        if dataset_name not in all_data_info:
            raise ValueError(f"Unknown dataset in global data information: {dataset_name}")
        dataset_info = all_data_info[dataset_name]
        
        output_root = osp.join(hydra_cfg.output_dir, dataset_name)  # 移到前面
        logger.info(f"[{idx_dataset}/{len(all_eval_datasets)}] Infering videodepth on {dataset_name} dataset...")

        if dataset_info.type == "squid":
            # SQUID 特殊处理：场景 -> image_set -> 两张图
            from utils.files import get_squid_image_sets, list_squid_imgs_an_image_set
            
            total_sets = 0
            processed = 0
            for scene in dataset_info.ls_all_seqs:
                image_sets = get_squid_image_sets(dataset_info, scene)
                total_sets += len(image_sets)
                
                for image_set in image_sets:
                    filelist = list_squid_imgs_an_image_set(dataset_info, scene, image_set)
                    if filelist is None:
                        continue
                    
                    save_dir = osp.join(output_root, scene, image_set)
                    
                    # 跳过已处理的
                    if not hydra_cfg.overwrite and osp.isdir(save_dir) and len(os.listdir(save_dir)) >= 2:
                        processed += 1
                        continue
                    
                    # 推理
                    time_used, depth_maps = infer_videodepth(filelist, model, hydra_cfg)
                    logger.info(f"[{scene}/{image_set}] processed, time: {time_used}")
                    
                    # 保存
                    os.makedirs(save_dir, exist_ok=True)
                    save_depth_maps(depth_maps, save_dir)
                    with open(osp.join(save_dir, "_time.json"), "w") as f:
                        json.dump({"time": time_used, "frames": len(filelist)}, f, indent=4)
                    processed += 1
            
            logger.info(f"SQUID: processed {processed}/{total_sets} image_sets")
            continue  # 跳过后面的 video 类型处理
        
        elif dataset_info.type == "video":
            seq_list = get_all_sequences(dataset_info)
        elif dataset_info.type == "mono":
            raise ValueError("dataset type `mono` is not supported for videodepth evaluation")
        else:
            raise ValueError(f"Unknown dataset type: {dataset_info.type}")

        model = model.eval()
        output_root = osp.join(hydra_cfg.output_dir, dataset_name)
        logger.info(f"[{idx_dataset}/{len(all_eval_datasets)}] Infering videodepth on {dataset_name} dataset..., output to {osp.relpath(output_root, hydra_cfg.work_dir)}")

        # 3. infer for each sequence (video)
        for seq_idx, seq in enumerate(seq_list, start=1):
            filelist = list_imgs_a_sequence(dataset_info, seq)
            # uniformly sample frames if max_frames is set
            max_frames = hydra_cfg.get("max_frames", None)
            if max_frames is not None and len(filelist) > max_frames:
                indices = np.linspace(0, len(filelist) - 1, max_frames, dtype=int)
                filelist = [filelist[i] for i in indices]
            save_dir = osp.join(output_root, seq)

            if not hydra_cfg.overwrite and (osp.isdir(save_dir) and len(os.listdir(save_dir)) == 2 * len(filelist) + 1):
                logger.info(f"[{seq_idx}/{len(seq_list)}] Sequence {seq} already processed, skipping.")
                continue
            
            # time_used: float, or List[float] (len = 2)
            # depth_maps: (N, H, W), torch.Tensor
            # conf_self: (N, H, W) torch.Tensor, or just None is ok
            time_used, depth_maps= infer_videodepth(filelist, model, hydra_cfg)
            logger.info(f"[{seq_idx}/{len(seq_list)}] Sequence {seq} processed, time: {time_used}, saving depth maps...")

            os.makedirs(save_dir, exist_ok=True)
            save_depth_maps(depth_maps, save_dir)
            # save time
            with open(osp.join(save_dir, "_time.json"), "w") as f:
                json.dump({
                    "time": time_used,
                    "frames": len(filelist),
                }, f, indent=4)
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    set_default_arg("evaluation", "videodepth")
    os.environ["HYDRA_FULL_ERROR"] = '1'
    with torch.no_grad():
        main()