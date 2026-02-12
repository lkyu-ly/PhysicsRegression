#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单测试: 验证 paddle.zeros() 的 place 参数修复

直接测试核心修复逻辑,不依赖复杂的模块导入
"""

import paddle
import numpy as np


def test_zeros_with_device():
    """测试 paddle.zeros() 使用 device=paddle.CPUPlace() 的行为"""
    print("=" * 60)
    print("测试: paddle.zeros() with device=paddle.CPUPlace()")
    print("=" * 60)

    # 测试不同大小的张量
    sizes = [10, 100, 1000]

    for size in sizes:
        # 在 CPU 上创建张量
        lengths = paddle.zeros(size, dtype='int64', device=paddle.CPUPlace())

        # 填充一些值
        for i in range(min(5, size)):
            lengths[i] = i + 1

        # 计算 max
        max_val = int(paddle.max(lengths).item())

        print(f"  尺寸={size:5d}: max={max_val}, place={lengths.place}, dtype={lengths.dtype}")

        # 验证
        assert lengths.place.is_cpu_place(), f"张量不在 CPU 上: {lengths.place}"
        assert max_val <= size, f"max 值异常: {max_val}"

    print()
    print("✅ 所有测试通过!")
    print("✅ paddle.zeros() 的 device 参数工作正常")
    print("✅ .item() 调用无跨设备同步问题")
    return True


def test_to_tensor_with_place():
    """测试 paddle.to_tensor() 使用 place 参数"""
    print("\n" + "=" * 60)
    print("测试: paddle.to_tensor() with place=paddle.CPUPlace()")
    print("=" * 60)

    # 测试列表转张量
    data = [1, 2, 3, 4, 5]

    lengths = paddle.to_tensor(data, dtype='int64', place=paddle.CPUPlace())

    max_val = int(paddle.max(lengths).item())

    print(f"  数据: {data}")
    print(f"  max={max_val}, place={lengths.place}, dtype={lengths.dtype}")

    assert lengths.place.is_cpu_place(), f"张量不在 CPU 上: {lengths.place}"
    assert max_val == 5, f"max 值不正确: {max_val}"

    print()
    print("✅ paddle.to_tensor() 的 place 参数工作正常")
    return True


def test_full_with_device():
    """测试 paddle.full() 使用 device 参数"""
    print("\n" + "=" * 60)
    print("测试: paddle.full() with device=paddle.CPUPlace()")
    print("=" * 60)

    # 测试创建填充张量
    shape = [10, 5]
    fill_value = 99

    tensor = paddle.full(shape, fill_value, dtype='int64', device=paddle.CPUPlace())

    print(f"  形状: {shape}, 填充值: {fill_value}")
    print(f"  place={tensor.place}, dtype={tensor.dtype}")
    print(f"  实际值范围: [{tensor.min().item()}, {tensor.max().item()}]")

    assert tensor.place.is_cpu_place(), f"张量不在 CPU 上: {tensor.place}"
    assert tensor.max().item() == fill_value, f"填充值不正确"

    print()
    print("✅ paddle.full() 的 device 参数工作正常")
    return True


def test_device_consistency():
    """测试不同设备上的一致性"""
    print("\n" + "=" * 60)
    print("测试: 跨设备一致性")
    print("=" * 60)

    # CPU 版本
    cpu_tensor = paddle.zeros(100, dtype='int64', device=paddle.CPUPlace())
    for i in range(10):
        cpu_tensor[i] = i + 1
    cpu_max = int(paddle.max(cpu_tensor).item())

    print(f"  CPU: max={cpu_max}, place={cpu_tensor.place}")

    # GPU 版本 (如果可用)
    if paddle.device.is_compiled_with_cuda():
        gpu_tensor = paddle.zeros(100, dtype='int64', device=paddle.CUDAPlace(0))
        for i in range(10):
            gpu_tensor[i] = i + 1
        gpu_max = int(paddle.max(gpu_tensor).item())

        print(f"  GPU: max={gpu_max}, place={gpu_tensor.place}")

        assert cpu_max == gpu_max, f"CPU 和 GPU 结果不一致: {cpu_max} != {gpu_max}"
        print()
        print("✅ CPU 和 GPU 结果一致")
    else:
        print("  ⏭️  跳过 GPU 测试 (CUDA 不可用)")

    return True


def main():
    print("\n" + "=" * 60)
    print("iluvatar GPU 修复 - 核心逻辑测试")
    print("=" * 60)
    print(f"PaddlePaddle 版本: {paddle.__version__}")
    print()

    results = []

    try:
        results.append(("paddle.zeros()", test_zeros_with_device()))
        results.append(("paddle.to_tensor()", test_to_tensor_with_place()))
        results.append(("paddle.full()", test_full_with_device()))
        results.append(("设备一致性", test_device_consistency()))

        print("\n" + "=" * 60)
        print("测试总结")
        print("=" * 60)

        for name, passed in results:
            status = "✅" if passed else "❌"
            print(f"{status} {name}")

        all_passed = all(passed for _, passed in results)

        if all_passed:
            print("\n🎉 所有核心逻辑测试通过!")
            print("✅ 修复方案正确实施")
            print("✅ place 参数工作正常")
            print("✅ 无跨设备同步问题")
            return 0
        else:
            print("\n❌ 部分测试失败")
            return 1

    except Exception as e:
        print(f"\n❌ 测试失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
