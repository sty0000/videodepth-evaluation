import torch
import sys

# 替换为你真实的 pth 路径
ckpt_path = r"/data/users/yihao/codes/checkpoints/flare/geometry_pose.pth"

try:
    print(f"Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    # 判断它到底是裸状态字典，还是包裹了 metadata 的字典
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    
    print("\n--- 顶层 Keys ---")
    if isinstance(ckpt, dict):
        print(list(ckpt.keys()))
    else:
        print("Not a dictionary.")

    print("\n--- 关键张量形状诊断 ---")
    keys_to_check = [
        'patch_embed.proj.weight', # 决定 patch_size 和 enc_embed_dim
        'decoder_embed.weight',    # 决定 dec_embed_dim
        'enc_blocks.0.norm1.weight', # 确认 encoder 维度
        'dec_blocks.0.norm1.weight'  # 确认 decoder 维度
    ]
    
    for k in keys_to_check:
        if k in state_dict:
            print(f"{k}: {state_dict[k].shape}")
        else:
            print(f"{k}: 缺失 (Missing)")
            
    # 统计层数
    enc_layers = sum(1 for k in state_dict.keys() if k.endswith('.norm1.weight') and 'enc_blocks' in k)
    dec_layers = sum(1 for k in state_dict.keys() if k.endswith('.norm1.weight') and 'dec_blocks' in k)
    print(f"\n推断的 Encoder 层数: {enc_layers}")
    print(f"推断的 Decoder 层数: {dec_layers}")

except Exception as e:
    print(f"解析失败: {e}")