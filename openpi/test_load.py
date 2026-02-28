import torch
import json
from safetensors import safe_open
import os

# 加载配置
with open("./checkpoints/tpi05_libero/config.json", 'r') as f:
    config = json.load(f)

# 加载权重
weights = {}
with safe_open("./checkpoints/tpi05_libero/model.safetensors", framework="pt", device="cpu") as f:
    for key in f.keys():
        weights[key] = f.get_tensor(key)

print(f"✅ 模型加载成功！")
print(f"配置: {config['type']}")
print(f"权重数量: {len(weights)}")
print(f"输入特征: {list(config['input_features'].keys())}")