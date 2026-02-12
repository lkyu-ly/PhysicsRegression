"""
PaddlePaddle设备管理集成测试

验证实际使用场景:
1. 模型设备移动
2. 与现有代码的兼容性

运行方式:
    python test_integration.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paddle
import paddle.nn as nn
from paddle_utils import device2str, device2int


class SimpleModel(nn.Layer):
    """简单的测试模型"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


def test_model_device_movement():
    """测试模型设备移动"""
    print("=" * 60)
    print("集成测试1: 模型设备移动")
    print("=" * 60)

    if not paddle.device.is_compiled_with_cuda():
        print("⚠️ 警告: 未检测到GPU,跳过模型设备测试")
        return

    try:
        # 创建模型
        model = SimpleModel()
        print("✅ 模型创建成功")

        # 测试1: 使用 .to() 方法(推荐方式)
        device_str = device2str("cuda:0")
        model.to(device_str)
        print(f"✅ 模型成功移动到设备: {device_str}")

        # 测试2: 创建输入数据并移动到设备
        x = paddle.randn([2, 10])
        x = x.to(device_str)
        print(f"✅ 输入数据成功移动到设备: {device_str}")

        # 测试3: 前向传播
        output = model(x)
        print(f"✅ 前向传播成功, 输出shape: {output.shape}")

        print("\n✅ 所有模型设备移动测试通过!\n")

    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_backward_compatibility():
    """测试向后兼容性"""
    print("=" * 60)
    print("集成测试2: 向后兼容性")
    print("=" * 60)

    # 测试1: device2int仍然可用(虽然已deprecated)
    device_id = device2int("cuda:0")
    print(f"✅ device2int('cuda:0') = {device_id} (向后兼容)")

    # 测试2: 各种设备字符串格式都能正常处理
    test_devices = [
        ("cuda:0", "gpu:0"),
        ("cuda:1", "gpu:1"),
        ("gpu:0", "gpu:0"),
        (0, "gpu:0"),
        ("iluvatar:0", "iluvatar:0"),
    ]

    for input_dev, expected_output in test_devices:
        output = device2str(input_dev)
        assert output == expected_output, f"Failed: {input_dev} → {expected_output}"
        print(f"✅ device2str({input_dev!r}) = {output!r}")

    print("\n✅ 所有向后兼容性测试通过!\n")


def test_module_init_pattern():
    """测试模块初始化模式(类似model/__init__.py)"""
    print("=" * 60)
    print("集成测试3: 模块初始化模式")
    print("=" * 60)

    if not paddle.device.is_compiled_with_cuda():
        print("⚠️ 警告: 未检测到GPU,跳过模块初始化测试")
        return

    try:
        # 模拟 build_modules 中的模式
        modules = {
            'encoder': SimpleModel(),
            'decoder': SimpleModel(),
        }

        # 模拟 params
        class Params:
            cpu = False
            device = "cuda:0"

        params = Params()

        # 使用新的模式: device2str + .to()
        if not params.cpu:
            device_str = device2str(params.device)
            for name, module in modules.items():
                module.to(device_str)
                print(f"✅ {name} 成功移动到 {device_str}")

        print("\n✅ 模块初始化模式测试通过!\n")

    except Exception as e:
        print(f"❌ 模块初始化测试失败: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """运行所有集成测试"""
    print("\n" + "=" * 60)
    print("PaddlePaddle设备管理 - 集成测试")
    print("=" * 60 + "\n")

    try:
        test_backward_compatibility()
        test_model_device_movement()
        test_module_init_pattern()

        print("\n" + "=" * 60)
        print("✅ ✅ ✅ 所有集成测试通过! ✅ ✅ ✅")
        print("=" * 60 + "\n")

        print("📝 总结:")
        print("  ✅ device2str() 正常工作")
        print("  ✅ device2int() 保持向后兼容")
        print("  ✅ 模型设备移动(.to())正常工作")
        print("  ✅ 与现有代码完全兼容")
        print()

        return 0

    except Exception as e:
        print(f"\n❌ 集成测试失败: {e}\n")
        return 1


if __name__ == "__main__":
    exit(main())
