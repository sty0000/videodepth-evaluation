# %%
"""
Process FLSea dataset: sample 100 frames from each scene
- Start from frame 0
- Stride of 5
- Output to /fs/jiayi/flsea_100
"""

import glob
import os
import shutil

# Source and destination paths
src_root = "/fs/jiayi/flsea"  
dst_root = "/data/users/jiayi/datasets/flsea_100"

# Process both canyons and red_sea subsets
subsets = ["canyons", "red_sea"]

for subset in subsets:
    src_subset = os.path.join(src_root, subset)
    dst_subset = os.path.join(dst_root, subset)
    
    if not os.path.exists(src_subset):
        print(f"Skipping {subset}: {src_subset} not found")
        continue
    
    # Get all scene directories (exclude calibration)
    scene_dirs = sorted([
        d for d in os.listdir(src_subset)
        if os.path.isdir(os.path.join(src_subset, d)) and d.lower() != 'calibration'
    ])
    
    print(f"\n{'='*50}")
    print(f"Processing {subset}: {len(scene_dirs)} scenes")
    print(f"{'='*50}")
    
    for scene in scene_dirs:
        src_scene = os.path.join(src_subset, scene)
        dst_scene = os.path.join(dst_subset, scene)
        
        # --- Process RGB images ---
        src_img_dir = os.path.join(src_scene, "imgs")
        dst_img_dir = os.path.join(dst_scene, "imgs")
        
        if os.path.exists(src_img_dir):
            # Get all tiff images
            img_frames = glob.glob(os.path.join(src_img_dir, "*.tiff"))
            img_frames = sorted(img_frames)
            
            # Sample: start from 0, stride 5, take 100 frames
            img_frames_sampled = img_frames[0::5][:100]
            
            os.makedirs(dst_img_dir, exist_ok=True)
            for frame in img_frames_sampled:
                shutil.copy(frame, dst_img_dir)
            
            print(f"{scene}: {len(img_frames)} imgs -> {len(img_frames_sampled)} sampled")
        else:
            print(f"{scene}: imgs folder not found")
            continue
        
        # --- Process depth maps ---
        src_depth_dir = os.path.join(src_scene, "depth")
        dst_depth_dir = os.path.join(dst_scene, "depth")
        
        if os.path.exists(src_depth_dir):
            # Get all tif depth maps
            depth_frames = glob.glob(os.path.join(src_depth_dir, "*.tif"))
            depth_frames = sorted(depth_frames)
            
            # Sample: start from 0, stride 5, take 100 frames
            depth_frames_sampled = depth_frames[0::5][:100]
            
            os.makedirs(dst_depth_dir, exist_ok=True)
            for frame in depth_frames_sampled:
                shutil.copy(frame, dst_depth_dir)
            
            print(f"{scene}: {len(depth_frames)} depths -> {len(depth_frames_sampled)} sampled")
        else:
            print(f"{scene}: depth folder not found")

print(f"\n{'='*50}")
print(f"Done! Output saved to: {dst_root}")
print(f"{'='*50}")
