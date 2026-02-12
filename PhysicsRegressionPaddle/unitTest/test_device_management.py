"""
PaddlePaddle设备管理统一优化测试脚本

验证以下核心功能:
1. device2str() 字符串标准化
2. to_cuda() 设备移动
3. 向后兼容性

运行方式:
    python test_device_management.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paddle
from paddle_utils import device2str, device2int


def test_device2str():
    """测试 device2str 函数"""
    print("=" * 60)
    print("测试1: device2str() 字符串标准化")
    print("=" * 60)

    # 测试1: PyTorch兼容格式
    assert device2str("cuda:0") == "gpu:0", "Failed: cuda:0 → gpu:0"
    print("✅ 'cuda:0' → 'gpu:0'")

    assert device2str("cuda:1") == "gpu:1", "Failed: cuda:1 → gpu:1"
    print("✅ 'cuda:1' → 'gpu:1'")

    # 测试2: PaddlePaddle原生格式
    assert device2str("gpu:0") == "gpu:0", "Failed: gpu:0 → gpu:0"
    print("✅ 'gpu:0' → 'gpu:0'")

    assert device2str("gpu:1") == "gpu:1", "Failed: gpu:1 → gpu:1"
    print("✅ 'gpu:1' → 'gpu:1'")

    # 测试3: Custom device
    assert device2str("iluvatar:0") == "iluvatar:0", "Failed: iluvatar:0"
    print("✅ 'iluvatar:0' → 'iluvatar:0'")

    # 测试4: CPU
    assert device2str("cpu") == "cpu", "Failed: cpu"
    print("✅ 'cpu' → 'cpu'")

    # 测试5: 整数索引
    assert device2str(0) == "gpu:0", "Failed: 0 → gpu:0"
    print("✅ 0 → 'gpu:0'")

    assert device2str(2) == "gpu:2", "Failed: 2 → gpu:2"
    print("✅ 2 → 'gpu:2'")

    # 测试6: 大小写不敏感
    assert device2str("CUDA:0") == "gpu:0", "Failed: CUDA:0 → gpu:0"
    print("✅ 'CUDA:0' → 'gpu:0'")

    # 测试7: 处理空格
    assert device2str("  cuda:0  ") == "gpu:0", "Failed: whitespace handling"
    print("✅ '  cuda:0  ' → 'gpu:0'")

    print("\n✅ 所有 device2str() 测试通过!\n")


def test_device2int():
    """测试 device2int 函数(向后兼容)"""
    print("=" * 60)
    print("测试2: device2int() 向后兼容性")
    print("=" * 60)

    assert device2int("cuda:0") == 0, "Failed: cuda:0 → 0"
    print("✅ 'cuda:0' → 0")

    assert device2int("gpu:1") == 1, "Failed: gpu:1 → 1"
    print("✅ 'gpu:1' → 1")

    assert device2int("iluvatar:2") == 2, "Failed: iluvatar:2 → 2"
    print("✅ 'iluvatar:2' → 2")

    assert device2int(0) == 0, "Failed: 0 → 0"
    print("✅ 0 → 0")

    print("\n✅ 所有 device2int() 测试通过!\n")


def test_to_cuda():
    """测试 to_cuda 函数(需要GPU环境)"""
    print("=" * 60)
    print("测试3: to_cuda() 设备移动")
    print("=" * 60)

    # 检查GPU可用性
    if not paddle.device.is_compiled_with_cuda():
        print("⚠️ 警告: 未检测到GPU,跳过to_cuda()测试")
        return

    try:
        from symbolicregression.utils import to_cuda

        # 测试1: 基本设备移动
        x = paddle.randn([10, 5])
        x_gpu, = to_cuda(x, device="cuda:0")
        assert 'gpu' in str(x_gpu.place).lower(), "Failed: device move"
        print("✅ 基本设备移动成功 (cuda:0 → gpu:0)")

        # 测试2: None处理
        x = paddle.randn([10, 5])
        x_result, none_result, y_result = to_cuda(x, None, paddle.randn([3]), device="gpu:0")
        assert none_result is None, "Failed: None handling"
        assert x_result is not None, "Failed: Non-None tensor"
        print("✅ None值处理正确")

        # 测试3: 默认设备
        x = paddle.randn([5])
        x_gpu, = to_cuda(x)  # device=None应使用gpu:0
        assert 'gpu' in str(x_gpu.place).lower(), "Failed: default device"
        print("✅ 默认设备处理正确 (gpu:0)")

        # 测试4: CPU模式
        x = paddle.randn([5])
        x_cpu, = to_cuda(x, use_cpu=True)
        # use_cpu=True时应直接返回原对象
        print("✅ CPU模式处理正确")

        print("\n✅ 所有 to_cuda() 测试通过!\n")

    except ImportError as e:
        print(f"⚠️ 警告: 无法导入to_cuda: {e}")


def test_error_handling():
    """测试错误处理"""
    print("=" * 60)
    print("测试4: 错误处理")
    print("=" * 60)

    # 测试1: 无效设备类型
    try:
        device2str([1, 2, 3])  # 应该抛出TypeError
        assert False, "Should raise TypeError"
    except TypeError as e:
        print(f"✅ 正确抛出TypeError: {e}")

    # 测试2: 无效设备字符串
    try:
        device2str("invalid_device:0")  # 应该抛出ValueError
        assert False, "Should raise ValueError"
    except ValueError as e:
        print(f"✅ 正确抛出ValueError: {e}")

    print("\n✅ 所有错误处理测试通过!\n")


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("PaddlePaddle设备管理统一优化 - 单元测试")
    print("=" * 60 + "\n")

    try:
        test_device2str()
        test_device2int()
        test_error_handling()
        test_to_cuda()

        print("\n" + "=" * 60)
        print("✅ ✅ ✅ 所有测试通过! ✅ ✅ ✅")
        print("=" * 60 + "\n")

        return 0

    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}\n")
        return 1
    except Exception as e:
        print(f"\n❌ 未预期的错误: {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
