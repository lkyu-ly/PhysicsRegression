#!/usr/bin/env python
"""
iluvatar GPU 异常数值诊断脚本

问题描述：
  在 get_length_after_batching() 中出现异常大的序列长度值
  错误值: 4603318688058332089 (正常应为 1-200)

测试目标：
  1. paddle.zeros() 初始化是否正确
  2. 张量索引赋值是否正常
  3. paddle.max() 计算是否正确
  4. .item() 转换是否正常
  5. 设备同步问题
  6. 内存布局问题
"""

import paddle
import numpy as np
import sys
import traceback

print("="*80)
print("iluvatar GPU 异常数值诊断")
print("="*80)
print(f"PaddlePaddle 版本: {paddle.__version__}")
print(f"当前设备: {paddle.get_device()}")
# 注意: paddle 没有 sys 属性,跳过平台检测
print("="*80)


# ============= 测试 1: paddle.zeros() 初始化 =============
print("\n" + "="*80)
print("测试 1: paddle.zeros() 初始化验证")
print("="*80)

def test_zeros_initialization(device, test_name):
    """测试 zeros() 是否正确初始化为 0"""
    print(f"\n[{test_name}] 测试设备: {device}")

    try:
        paddle.set_device(device)

        # 创建张量
        lengths = paddle.zeros([5], dtype='int64')
        print(f"  ✓ 创建成功: shape={lengths.shape}, dtype={lengths.dtype}")
        print(f"  ✓ 设备位置: {lengths.place}")

        # 立即检查值
        print(f"  ✓ 初始值 (numpy转换): {lengths.numpy()}")
        print(f"  ✓ 初始值 (python列表): {lengths.tolist()}")

        # 检查是否全为0
        values = lengths.numpy()
        if not np.all(values == 0):
            print(f"  ❌ 错误: zeros() 未正确初始化!")
            print(f"     期望: [0, 0, 0, 0, 0]")
            print(f"     实际: {values}")
            return False
        else:
            print(f"  ✅ zeros() 初始化正确")
            return True

    except Exception as e:
        print(f"  ❌ 异常: {e}")
        traceback.print_exc()
        return False


# ============= 测试 2: 索引赋值 =============
print("\n" + "="*80)
print("测试 2: 张量索引赋值验证")
print("="*80)

def test_index_assignment(device, test_name):
    """测试索引赋值是否正常工作"""
    print(f"\n[{test_name}] 测试设备: {device}")

    try:
        paddle.set_device(device)

        # 创建张量
        lengths = paddle.zeros([5], dtype='int64')
        print(f"  ✓ 初始值: {lengths.tolist()}")

        # 逐个赋值
        test_values = [3, 5, 2, 8, 1]
        for i, val in enumerate(test_values):
            lengths[i] = val
            print(f"  ✓ lengths[{i}] = {val}")

        # 读取值验证
        result = lengths.tolist()
        print(f"  ✓ 赋值后: {result}")

        if result != test_values:
            print(f"  ❌ 错误: 索引赋值失败!")
            print(f"     期望: {test_values}")
            print(f"     实际: {result}")
            return False
        else:
            print(f"  ✅ 索引赋值正确")
            return True

    except Exception as e:
        print(f"  ❌ 异常: {e}")
        traceback.print_exc()
        return False


# ============= 测试 3: paddle.max() 计算 =============
print("\n" + "="*80)
print("测试 3: paddle.max() 计算验证")
print("="*80)

def test_max_calculation(device, test_name):
    """测试 paddle.max() 是否正确计算"""
    print(f"\n[{test_name}] 测试设备: {device}")

    try:
        paddle.set_device(device)

        # 创建已知值的张量
        test_values = [3, 5, 2, 8, 1]
        lengths = paddle.to_tensor(test_values, dtype='int64')
        print(f"  ✓ 输入值: {lengths.tolist()}")

        # 计算 max
        max_tensor = paddle.max(lengths)
        print(f"  ✓ paddle.max() 返回类型: {type(max_tensor)}")
        print(f"  ✓ paddle.max() 返回值: {max_tensor}")
        print(f"  ✓ paddle.max() shape: {max_tensor.shape}")
        print(f"  ✓ paddle.max() dtype: {max_tensor.dtype}")

        # 转换为标量
        max_value = max_tensor.item()
        print(f"  ✓ .item() 返回类型: {type(max_value)}")
        print(f"  ✓ .item() 返回值: {max_value}")

        # 转换为 int
        max_int = int(max_value)
        print(f"  ✓ int() 转换结果: {max_int}")

        if max_int != 8:
            print(f"  ❌ 错误: paddle.max() 计算错误!")
            print(f"     期望: 8")
            print(f"     实际: {max_int}")
            return False
        else:
            print(f"  ✅ paddle.max() 计算正确")
            return True

    except Exception as e:
        print(f"  ❌ 异常: {e}")
        traceback.print_exc()
        return False


# ============= 测试 4: 完整流程模拟 =============
print("\n" + "="*80)
print("测试 4: get_length_after_batching() 完整流程模拟")
print("="*80)

def test_full_workflow(device, test_name):
    """完整模拟 get_length_after_batching() 的流程"""
    print(f"\n[{test_name}] 测试设备: {device}")

    try:
        paddle.set_device(device)

        # 模拟序列列表
        class FakeSeq:
            def __init__(self, length):
                self._len = length
            def __len__(self):
                return self._len

        seqs = [FakeSeq(3), FakeSeq(5), FakeSeq(2), FakeSeq(8), FakeSeq(1)]
        expected_lengths = [3, 5, 2, 8, 1]
        max_seq_len = 200

        print(f"  ✓ 输入序列长度: {expected_lengths}")

        # 步骤 1: 创建张量
        lengths = paddle.zeros(len(seqs), dtype='int64')
        print(f"  ✓ 步骤1 - zeros(): {lengths.tolist()}")

        # 步骤 2: 赋值
        for i, seq in enumerate(seqs):
            lengths[i] = len(seq)
        print(f"  ✓ 步骤2 - 赋值后: {lengths.tolist()}")

        # 步骤 3: 检查值
        current_values = lengths.tolist()
        if current_values != expected_lengths:
            print(f"  ❌ 错误: 赋值后值不匹配!")
            print(f"     期望: {expected_lengths}")
            print(f"     实际: {current_values}")
            return False

        # 步骤 4: 计算 max
        max_tensor = paddle.max(lengths)
        print(f"  ✓ 步骤3 - paddle.max(): {max_tensor}")

        # 步骤 5: 转换为标量
        max_value = max_tensor.item()
        print(f"  ✓ 步骤4 - .item(): {max_value} (type: {type(max_value)})")

        # 步骤 6: 转换为 int
        max_length = int(max_value)
        print(f"  ✓ 步骤5 - int(): {max_length}")

        # 步骤 7: 断言
        if max_length > max_seq_len:
            print(f"  ❌ 错误: 最终值异常!")
            print(f"     计算 max_length = {max_length}")
            print(f"     超过最大限制 {max_seq_len}")
            return False
        else:
            print(f"  ✅ 完整流程正确, max_length = {max_length}")
            return True

    except Exception as e:
        print(f"  ❌ 异常: {e}")
        traceback.print_exc()
        return False


# ============= 测试 5: 压力测试（多次重复） =============
print("\n" + "="*80)
print("测试 5: 压力测试（重复100次检测稳定性）")
print("="*80)

def test_stress(device, test_name, n_iterations=100):
    """压力测试，检测是否偶发错误"""
    print(f"\n[{test_name}] 测试设备: {device}")
    print(f"  迭代次数: {n_iterations}")

    paddle.set_device(device)
    errors = []

    for i in range(n_iterations):
        try:
            # 创建随机长度的序列
            n_seqs = np.random.randint(1, 10)
            expected_lengths = [np.random.randint(1, 100) for _ in range(n_seqs)]

            # 执行流程
            lengths = paddle.zeros(n_seqs, dtype='int64')
            for j, length in enumerate(expected_lengths):
                lengths[j] = length

            # 验证
            result = lengths.tolist()
            if result != expected_lengths:
                errors.append({
                    'iteration': i,
                    'expected': expected_lengths,
                    'actual': result,
                    'stage': 'assignment'
                })
                continue

            # 计算 max
            max_length = int(paddle.max(lengths).item())
            expected_max = max(expected_lengths)

            if max_length != expected_max:
                errors.append({
                    'iteration': i,
                    'expected_max': expected_max,
                    'actual_max': max_length,
                    'stage': 'max_calculation'
                })

        except Exception as e:
            errors.append({
                'iteration': i,
                'exception': str(e),
                'stage': 'exception'
            })

    # 报告结果
    if errors:
        print(f"  ❌ 发现 {len(errors)} 个错误:")
        for err in errors[:5]:  # 只显示前5个
            print(f"     迭代 {err['iteration']}: {err}")
        if len(errors) > 5:
            print(f"     ... (共 {len(errors)} 个错误)")
        return False
    else:
        print(f"  ✅ 全部 {n_iterations} 次测试通过")
        return True


# ============= 测试 6: 内存内容检查 =============
print("\n" + "="*80)
print("测试 6: 内存内容详细检查")
print("="*80)

def test_memory_inspection(device, test_name):
    """检查张量的实际内存内容"""
    print(f"\n[{test_name}] 测试设备: {device}")

    try:
        paddle.set_device(device)

        # 创建张量
        lengths = paddle.zeros([5], dtype='int64')
        print(f"  ✓ 创建后立即检查:")
        print(f"    - numpy(): {lengths.numpy()}")
        print(f"    - tolist(): {lengths.tolist()}")
        print(f"    - 内存地址: {lengths.data_ptr() if hasattr(lengths, 'data_ptr') else 'N/A'}")

        # 赋值
        test_values = [3, 5, 2, 8, 1]
        for i, val in enumerate(test_values):
            lengths[i] = val

        print(f"  ✓ 赋值后检查:")
        print(f"    - numpy(): {lengths.numpy()}")
        print(f"    - tolist(): {lengths.tolist()}")

        # 强制同步（如果可能）
        if hasattr(paddle, 'device'):
            try:
                paddle.device.synchronize()
                print(f"  ✓ 设备同步成功")
            except:
                print(f"  ⚠️ 设备同步不支持")

        # 再次检查
        print(f"  ✓ 同步后检查:")
        print(f"    - numpy(): {lengths.numpy()}")
        print(f"    - tolist(): {lengths.tolist()}")

        # 计算 max
        max_val = paddle.max(lengths)
        print(f"  ✓ paddle.max() 结果:")
        print(f"    - 类型: {type(max_val)}")
        print(f"    - 值: {max_val}")
        print(f"    - numpy(): {max_val.numpy()}")
        print(f"    - item(): {max_val.item()}")

        return True

    except Exception as e:
        print(f"  ❌ 异常: {e}")
        traceback.print_exc()
        return False


# ============= 主测试流程 =============
def main():
    # 检测可用设备
    devices_to_test = ['cpu']

    # 检查是否有 GPU
    try:
        paddle.set_device('gpu:0')
        devices_to_test.append('gpu:0')
        print("✓ 检测到 NVIDIA GPU")
    except:
        print("⚠️ 未检测到 NVIDIA GPU")

    # 检查是否有 iluvatar GPU
    try:
        paddle.set_device('iluvatar_gpu:0')
        devices_to_test.append('iluvatar_gpu:0')
        print("✓ 检测到 iluvatar GPU")
    except:
        print("⚠️ 未检测到 iluvatar GPU")

    print(f"\n将测试设备: {devices_to_test}")

    # 运行所有测试
    all_results = {}

    for device in devices_to_test:
        print(f"\n{'#'*80}")
        print(f"# 开始测试设备: {device}")
        print(f"{'#'*80}")

        device_results = {}

        # 测试 1
        device_results['test1_zeros'] = test_zeros_initialization(
            device, f"Test1-{device}"
        )

        # 测试 2
        device_results['test2_assignment'] = test_index_assignment(
            device, f"Test2-{device}"
        )

        # 测试 3
        device_results['test3_max'] = test_max_calculation(
            device, f"Test3-{device}"
        )

        # 测试 4
        device_results['test4_workflow'] = test_full_workflow(
            device, f"Test4-{device}"
        )

        # 测试 5
        device_results['test5_stress'] = test_stress(
            device, f"Test5-{device}", n_iterations=50
        )

        # 测试 6
        device_results['test6_memory'] = test_memory_inspection(
            device, f"Test6-{device}"
        )

        all_results[device] = device_results

    # 打印总结
    print(f"\n{'='*80}")
    print("测试总结")
    print(f"{'='*80}")

    for device, results in all_results.items():
        print(f"\n设备: {device}")
        for test_name, passed in results.items():
            status = "✅ 通过" if passed else "❌ 失败"
            print(f"  {test_name}: {status}")

    print(f"\n{'='*80}")
    print("诊断完成")
    print(f"{'='*80}")

    # 返回是否有失败
    has_failure = any(not passed for results in all_results.values() for passed in results.values())
    return 0 if not has_failure else 1


if __name__ == "__main__":
    sys.exit(main())
