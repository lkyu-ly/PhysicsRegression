#!/usr/bin/env python3
"""
LinearPointEmbedder性能测试脚本

测试第一阶段优化的效果:
1. batch方法优化
2. 预计算token ID
3. 浮点数编码缓存
4. hint_encode简化
"""

import sys
import time
import numpy as np
import paddle

sys.path.append("/home/lkyu/baidu/PhyE2E/PhysicsRegressionPaddle")

from symbolicregression.envs import build_env
from symbolicregression.model import build_modules
from parsers import get_parser


def create_test_data(batch_size=256, n_points=100, n_vars=2):
    """创建测试数据"""
    sequences = []
    for _ in range(batch_size):
        seq = []
        for _ in range(n_points):
            x = np.random.randn(n_vars).astype(np.float64)
            y = np.random.randn(1).astype(np.float64)
            seq.append((x, y))
        sequences.append(seq)
    return sequences


def create_test_hints(batch_size=256, n_vars=2):
    """创建测试提示"""
    # 单位提示
    units = []
    for _ in range(batch_size):
        seq_units = []
        for _ in range(n_vars + 1):  # n_vars个输入 + 1个输出
            seq_units.append(np.array([0, 0, 0, 0, 0]))  # 无量纲
        units.append(seq_units)

    # 复杂度提示 (使用字符串: simple, middle, hard)
    complexity = [["simple"] for _ in range(batch_size)]

    # 一元运算提示
    unarys = [["sin", "cos", "exp"] for _ in range(batch_size)]

    return [units, complexity, unarys]


def test_embedder_performance(n_iterations=10):
    """测试embedder性能"""
    print("=" * 80)
    print("LinearPointEmbedder 性能测试")
    print("=" * 80)

    # 1. 初始化环境和模型
    print("\n[1/4] 初始化环境和模型...")
    parser = get_parser()
    params = parser.parse_args([
        "--max_len", "200",
        "--batch_size", "256",
        "--use_hints", "units,complexity,unarys",
        "--cpu", "True",  # 使用CPU测试避免GPU传输影响
    ])

    env = build_env(params)
    modules = build_modules(env, params)
    embedder = modules["embedder"]
    embedder.eval()

    print(f"   ✓ 环境初始化完成")
    print(f"   ✓ Embedder类型: {type(embedder).__name__}")

    # 2. 创建测试数据
    print("\n[2/4] 创建测试数据...")
    batch_size = 256
    n_points = 100
    n_vars = 2

    sequences = create_test_data(batch_size, n_points, n_vars)
    hints = create_test_hints(batch_size, n_vars)

    print(f"   ✓ Batch size: {batch_size}")
    print(f"   ✓ Points per sequence: {n_points}")
    print(f"   ✓ Variables: {n_vars}")
    print(f"   ✓ Total data points: {batch_size * n_points}")

    # 3. 预热
    print("\n[3/4] 预热运行...")
    for _ in range(3):
        _ = embedder(sequences, hints)
    print(f"   ✓ 预热完成")

    # 4. 性能测试
    print(f"\n[4/4] 性能测试 (迭代次数: {n_iterations})...")
    times = []

    for i in range(n_iterations):
        start = time.time()
        output, lengths = embedder(sequences, hints)
        elapsed = (time.time() - start) * 1000  # 转换为毫秒
        times.append(elapsed)
        print(f"   迭代 {i+1}/{n_iterations}: {elapsed:.2f}ms")

    # 5. 统计结果
    print("\n" + "=" * 80)
    print("性能统计")
    print("=" * 80)

    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)

    print(f"\n平均时间: {avg_time:.2f}ms")
    print(f"标准差:   {std_time:.2f}ms")
    print(f"最小时间: {min_time:.2f}ms")
    print(f"最大时间: {max_time:.2f}ms")

    # 6. 输出形状验证
    print("\n" + "=" * 80)
    print("输出验证")
    print("=" * 80)
    print(f"\n输出形状: {output.shape}")
    print(f"长度形状: {lengths.shape}")

    # 7. 性能目标对比
    print("\n" + "=" * 80)
    print("性能目标对比")
    print("=" * 80)

    # 假设原始版本的时间（根据计划中的估计）
    # 原始LinearPointEmbedder: 5168ms (总共5860ms中的88%)
    # 目标: 减少40-50% -> 2584-3101ms

    # 但这里我们测试的是单次forward的时间，不是整个训练步骤
    # 所以我们只能看相对改进

    print(f"\n当前平均时间: {avg_time:.2f}ms")
    print(f"\n优化项:")
    print(f"  ✓ batch方法优化 (paddle.full)")
    print(f"  ✓ 预计算token ID")
    print(f"  ✓ hint_encode简化")
    print(f"  ✓ 批量编码优化 (encode_batch向量化)")
    print(f"  ✗ 浮点数编码缓存 (已移除 - 命中率低导致性能恶化)")

    print("\n" + "=" * 80)
    print("测试完成!")
    print("=" * 80)

    return {
        'avg_time': avg_time,
        'std_time': std_time,
        'output_shape': output.shape,
    }


if __name__ == "__main__":
    results = test_embedder_performance(n_iterations=10)
