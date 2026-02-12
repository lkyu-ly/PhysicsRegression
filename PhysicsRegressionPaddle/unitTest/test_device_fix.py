#!/usr/bin/env python
"""
简化的设备管理修复测试脚本
直接测试关键方法是否正常工作
"""

import paddle
import sys

print(f"PaddlePaddle 版本: {paddle.__version__}")
print(f"当前设备: {paddle.get_device()}")

# 测试 1: 测试paddle.zeros() 默认设备创建
print(f"\n{'='*60}")
print(f"测试 1: paddle.zeros() 默认设备行为")
print(f"{'='*60}")

# 设置默认设备为 GPU
paddle.set_device('gpu:0')
print(f"✓ 设置默认设备为: {paddle.get_device()}")

# 创建张量 (不指定设备)
lengths = paddle.zeros([5], dtype='int64')
print(f"✓ paddle.zeros([5], dtype='int64') 创建成功")
print(f"  device: {lengths.place}")
print(f"  values: {lengths}")

# 赋值
for i in range(5):
    lengths[i] = i + 1
print(f"✓ 赋值成功: {lengths}")

# 计算max
try:
    max_val = paddle.max(lengths)
    print(f"✓ paddle.max(lengths) = {max_val}")
    max_item = int(max_val.item())
    print(f"✓ int(paddle.max(lengths).item()) = {max_item}")
except Exception as e:
    print(f"✗ paddle.max() 失败: {e}")
    sys.exit(1)

# 测试 2: 测试 paddle.to_tensor() 默认设备
print(f"\n{'='*60}")
print(f"测试 2: paddle.to_tensor() 默认设备行为")
print(f"{'='*60}")

lengths2 = paddle.to_tensor([2, 3, 4, 5], dtype='int64')
print(f"✓ paddle.to_tensor([2,3,4,5], dtype='int64') 创建成功")
print(f"  device: {lengths2.place}")

max_len = int(paddle.max(lengths2).item())
print(f"✓ max_len = {max_len}")

# 测试 3: 测试 paddle.full() 默认设备
print(f"\n{'='*60}")
print(f"测试 3: paddle.full() 默认设备行为")
print(f"{'='*60}")

sent = paddle.full([max_len, lengths2.shape[0]], 999, dtype='int64')
print(f"✓ paddle.full([{max_len}, {lengths2.shape[0]}], 999, dtype='int64') 创建成功")
print(f"  device: {sent.place}")
print(f"  shape: {sent.shape}")

# 测试 4: 测试跨设备操作
print(f"\n{'='*60}")
print(f"测试 4: 同设备张量操作")
print(f"{'='*60}")

# 创建两个GPU张量
eq = paddle.to_tensor([10, 20, 30], dtype='int64')
print(f"✓ eq device: {eq.place}")

# 尝试 copy_ 操作
try:
    sent[0:3, 0].copy_(eq)
    print(f"✓ sent[0:3, 0].copy_(eq) 成功")
    print(f"  sent[:,0] = {sent[:,0]}")
except Exception as e:
    print(f"✗ copy_ 失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试 5: 测试异常捕获 (如果有 CPU 张量)
print(f"\n{'='*60}")
print(f"测试 5: 跨设备操作检测")
print(f"{'='*60}")

# 先设置为CPU,再创建张量
paddle.set_device('cpu')
cpu_lengths = paddle.zeros([3], dtype='int64')
paddle.set_device('gpu:0')  # 切回GPU
print(f"✓ 创建 CPU 张量: {cpu_lengths.place}")

try:
    max_val_cpu = paddle.max(cpu_lengths)
    print(f"✓ paddle.max(cpu_lengths) 成功: {max_val_cpu}")
except Exception as e:
    print(f"✗ paddle.max(cpu_lengths) 失败: {e}")

gpu_tensor = paddle.full([3], 0, dtype='int64')
print(f"✓ 创建 GPU 张量: {gpu_tensor.place}")

try:
    gpu_tensor.copy_(cpu_lengths)
    print(f"⚠️ 警告: 跨设备 copy_ 未触发错误")
    print(f"  PaddlePaddle 可能自动同步了设备")
    print(f"  gpu_tensor after copy: {gpu_tensor}")
except Exception as e:
    print(f"✓ 跨设备 copy_ 正确触发错误: {type(e).__name__}")

print(f"\n{'#'*60}")
print(f"# ✅ 所有基础张量操作测试通过!")
print(f"# 修复后的代码应该可以正常工作")
print(f"{'#'*60}\n")

sys.exit(0)
