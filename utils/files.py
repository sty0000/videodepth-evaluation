import os
import os.path as osp
import glob
from typing import Optional
from omegaconf import DictConfig, ListConfig


def get_all_sequences(dataset_cfg: DictConfig, sort_by_seq_name: bool = True):
    if isinstance(dataset_cfg.ls_all_seqs, str):
        # if ls_all_seqs is a string, it is the root path of sequences
        seq_list = [d for d in os.listdir(dataset_cfg.ls_all_seqs) if osp.isdir(osp.join(dataset_cfg.ls_all_seqs, d))]
    elif isinstance(dataset_cfg.ls_all_seqs, ListConfig):
        # if ls_all_seqs is a ListConfig, it is the ListConfig of sequence names
        seq_list = dataset_cfg.ls_all_seqs
    else:
        raise ValueError(f"Unknown ls_all_seqs type: {type(dataset_cfg.ls_all_seqs)}, ls_all_seqs is {dataset_cfg.ls_all_seqs}, which should be a string or a ListConfig")
    return sorted(seq_list) if sort_by_seq_name else seq_list

def list_imgs_a_sequence(dataset_cfg: DictConfig, seq: Optional[str] = None):
    subdir = dataset_cfg.img.path.format(seq=seq)  # string include {seq}
    ext = dataset_cfg.img.ext
    filelist = sorted(glob.glob(f"{subdir}/*.{ext}"))
    return filelist

def list_depths_a_sequence(dataset_cfg: DictConfig, seq: Optional[str] = None):
    subdir = dataset_cfg.depth.path.format(seq=seq)  # string include {seq}
    ext = dataset_cfg.depth.ext
    filelist = sorted(glob.glob(f"{subdir}/*.{ext}"))
    return filelist

# ---------------------- SQUID-specific functions ----------------------

def get_squid_image_sets(dataset_cfg: DictConfig, scene: str):
    """Get all image_set folders for a SQUID scene"""
    scene_path = osp.join(dataset_cfg.root_path, scene)
    image_sets = []
    for item in sorted(os.listdir(scene_path)):
        if item.startswith('image_set_'):
            full_path = osp.join(scene_path, item)
            if osp.isdir(full_path):
                image_sets.append(item)
    return image_sets


def list_squid_imgs_an_image_set(dataset_cfg: DictConfig, scene: str, image_set: str):
    """List left and right images for a SQUID image_set"""
    image_set_path = osp.join(dataset_cfg.root_path, scene, image_set)
    files = os.listdir(image_set_path)
    left_img = right_img = None
    
    for f in files:
        if 'resizedUndistort.tif' in f:
            if f.startswith('LFT_'):
                left_img = osp.join(image_set_path, f)
            elif f.startswith('RGT_'):
                right_img = osp.join(image_set_path, f)
    
    if left_img and right_img:
        return [left_img, right_img]
    return None


def list_squid_depth_an_image_set(dataset_cfg: DictConfig, scene: str, image_set: str):
    """Get depth GT path for a SQUID image_set"""
    depth_path = osp.join(dataset_cfg.root_path, scene, image_set, 'distanceFromCamera.mat')
    return depth_path if osp.exists(depth_path) else None