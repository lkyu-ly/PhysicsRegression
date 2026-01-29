#!/usr/bin/env python3
"""测试 Hessian 修复"""
import paddle
import numpy as np


def test_hessian_extraction():
    """测试 Hessian 对象的正确提取"""
    print("=" * 60)
    print("测试 Hessian 提取修复")
    print("=" * 60)

    # 简单模型: f(x) = x0^2 + x1^2
    class SimpleModel(paddle.nn.Layer):
        def forward(self, x):
            return paddle.sum(x**2)

    model = SimpleModel()
    x = paddle.to_tensor([1.0, 2.0])
    x.stop_gradient = False  # ← 必须设置

    # 创建 Hessian 对象
    h_obj = paddle.incubate.autograd.Hessian(func=model, xs=x, is_batched=False)

    print(f"\n1. Hessian 对象类型: {type(h_obj)}")
    print(f"   是否是 Tensor: {isinstance(h_obj, paddle.Tensor)}")

    # 使用切片提取张量
    h_matrix = h_obj[:]

    print(f"\n2. 提取后张量类型: {type(h_matrix)}")
    print(f"   是否是 Tensor: {isinstance(h_matrix, paddle.Tensor)}")
    print(f"   形状: {h_matrix.shape}")

    # 验证可以调用张量方法
    try:
        h_detach = h_matrix.detach()
        h_cpu = h_matrix.cpu()
        print(f"\n3. ✅ 张量方法测试通过")
        print(f"   .detach() 成功")
        print(f"   .cpu() 成功")
    except Exception as e:
        print(f"\n3. ❌ 张量方法测试失败: {e}")
        return False

    # 验证数值正确性
    # 理论 Hessian: [[2, 0], [0, 2]]
    expected = paddle.to_tensor([[2.0, 0.0], [0, 2.0]])
    diff = paddle.abs(h_matrix - expected).max().item()

    print(f"\n4. 数值验证:")
    print(f"   计算结果:\n{h_matrix.numpy()}")
    print(f"   理论值:\n{expected.numpy()}")
    print(f"   最大误差: {diff:.2e}")

    if diff < 1e-5:
        print(f"   ✅ 数值正确")
    else:
        print(f"   ❌ 数值误差过大")
        return False

    print("\n" + "=" * 60)
    print("✅ 所有测试通过！")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_hessian_extraction()
    exit(0 if success else 1)
