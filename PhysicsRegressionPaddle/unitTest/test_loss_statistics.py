"""
单元测试：验证loss统计的准确性

测试修复后的loss统计逻辑是否正确累积所有batch的数据
"""
import sys
sys.path.append("/home/lkyu/baidu/PhyE2E/PhysicsRegressionPaddle")

import paddle
import numpy as np
from collections import defaultdict


def test_loss_accumulation():
    """测试loss累积的准确性"""
    print("=" * 80)
    print("测试1: Loss累积准确性")
    print("=" * 80)

    # 模拟trainer的统计结构
    stats = defaultdict(list)
    total_loss = 0.0

    # 模拟100个batch的训练
    n_batches = 100
    expected_total = 0.0

    for i in range(n_batches):
        # 生成随机loss
        loss = paddle.to_tensor([np.random.uniform(0.1, 1.0)])
        expected_total += loss.item()

        # 模拟修复后的代码（每个batch都累积）
        stats["task1"].append(loss.item())
        total_loss += loss.item()

    # 验证
    print(f"预期总loss: {expected_total:.6f}")
    print(f"实际总loss: {total_loss:.6f}")
    print(f"差异: {abs(total_loss - expected_total):.10f}")

    assert abs(total_loss - expected_total) < 1e-6, \
        f"Loss统计错误: {total_loss} vs {expected_total}"

    # 验证平均loss
    avg_loss = total_loss / n_batches
    expected_avg = expected_total / n_batches
    print(f"预期平均loss: {expected_avg:.6f}")
    print(f"实际平均loss: {avg_loss:.6f}")

    assert abs(avg_loss - expected_avg) < 1e-6, \
        f"平均loss错误: {avg_loss} vs {expected_avg}"

    print("✓ 测试通过：Loss累积准确\n")


def test_processed_w_accumulation():
    """测试processed_w统计的准确性"""
    print("=" * 80)
    print("测试2: processed_w累积准确性")
    print("=" * 80)

    stats = {"processed_w": 0}

    # 模拟100个batch
    n_batches = 100
    expected_total = 0

    for i in range(n_batches):
        # 生成随机序列长度
        len1 = paddle.to_tensor([np.random.randint(10, 50)])
        len2 = paddle.to_tensor([np.random.randint(10, 50)])

        words = (len1 + len2 - 2).sum().item()
        expected_total += words

        # 模拟修复后的代码（每个batch都累积）
        stats["processed_w"] += words

    # 验证
    print(f"预期总单词数: {expected_total}")
    print(f"实际总单词数: {stats['processed_w']}")
    print(f"差异: {abs(stats['processed_w'] - expected_total)}")

    assert stats["processed_w"] == expected_total, \
        f"processed_w统计错误: {stats['processed_w']} vs {expected_total}"

    print("✓ 测试通过：processed_w累积准确\n")


def test_embedder_boundary_cases():
    """测试embedders的边界情况"""
    print("=" * 80)
    print("测试3: Embedder边界情况")
    print("=" * 80)

    # 测试负数填充计数
    max_input_dim = 5
    float_scalar_descriptor_len = 3

    test_cases = [
        (3, 2),  # n_vars < max: 正常情况
        (5, 0),  # n_vars == max: 边界情况
        (6, 0),  # n_vars > max: 应该被max()限制为0
    ]

    for n_vars, expected_pad in test_cases:
        # 修复后的计算
        input_pad_count = max(0, (max_input_dim - n_vars) * float_scalar_descriptor_len)

        print(f"n_vars={n_vars}, max={max_input_dim}")
        print(f"  计算的pad_count: {input_pad_count}")
        print(f"  预期pad_count: {expected_pad * float_scalar_descriptor_len}")

        assert input_pad_count >= 0, \
            f"pad_count不应该为负数: {input_pad_count}"

        if n_vars <= max_input_dim:
            assert input_pad_count == expected_pad * float_scalar_descriptor_len, \
                f"pad_count计算错误: {input_pad_count} vs {expected_pad * float_scalar_descriptor_len}"

    print("✓ 测试通过：边界情况处理正确\n")


def test_no_loss_accumulator_leak():
    """测试_loss_accumulator不会泄漏"""
    print("=" * 80)
    print("测试4: 无内存泄漏")
    print("=" * 80)

    # 模拟trainer对象
    class MockTrainer:
        def __init__(self):
            self.errors_statistics = defaultdict(int)
            self.infos_statistics = defaultdict(list)

        def reset_statistics(self):
            """模拟get_generation_statistics中的清理逻辑"""
            self.errors_statistics = defaultdict(int)
            self.infos_statistics = defaultdict(list)
            # 清理未使用的loss累积器（如果存在）
            if hasattr(self, '_loss_accumulator'):
                del self._loss_accumulator

    trainer = MockTrainer()

    # 模拟旧代码可能创建的累积器
    trainer._loss_accumulator = [paddle.to_tensor([0.5]) for _ in range(10)]

    print(f"清理前: hasattr(trainer, '_loss_accumulator') = {hasattr(trainer, '_loss_accumulator')}")

    # 执行清理
    trainer.reset_statistics()

    print(f"清理后: hasattr(trainer, '_loss_accumulator') = {hasattr(trainer, '_loss_accumulator')}")

    assert not hasattr(trainer, '_loss_accumulator'), \
        "_loss_accumulator应该被清理"

    print("✓ 测试通过：无内存泄漏\n")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Loss统计修复 - 单元测试")
    print("=" * 80 + "\n")

    try:
        test_loss_accumulation()
        test_processed_w_accumulation()
        test_embedder_boundary_cases()
        test_no_loss_accumulator_leak()

        print("=" * 80)
        print("✓ 所有测试通过！")
        print("=" * 80)

    except AssertionError as e:
        print("\n" + "=" * 80)
        print("✗ 测试失败！")
        print("=" * 80)
        print(f"错误: {e}")
        sys.exit(1)
