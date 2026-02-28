import torch
import json
from safetensors import safe_open
import sys

# 打开文件用于写入
with open('./test_model_view.txt', 'w', encoding='utf-8') as f:
    # 创建一个函数同时打印到屏幕和文件
    def print_both(*args, **kwargs):
        print(*args, **kwargs)
        print(*args, **kwargs, file=f)
    
    # 加载权重
    weights = {}
    with safe_open("./checkpoints/tpi05_libero/model.safetensors", framework="pt", device="cpu") as sf:
        for key in sf.keys():
            weights[key] = sf.get_tensor(key)
    
    print_both("\n🔍 完整模型结构：")
    print_both("="*60)
    
    # 排序并打印所有层
    sorted_keys = sorted(weights.keys())
    for i, key in enumerate(sorted_keys):
        shape = tuple(weights[key].shape)
        # 计算参数量
        params = 1
        for dim in shape:
            params *= dim
        print_both(f"{i+1:3d}. {key:50s} {str(shape):20s} ({params:,} params)")
    
    total_params = sum(w.numel() for w in weights.values())
    print_both("="*60)
    print_both(f"📊 总参数量: {total_params:,} ({total_params/1e6:.2f}M)")

print("✅ 结果已保存到 ./test_model_view.txt")