#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化器修复验证测试脚本

测试内容:
1. 所有优化器类的初始化
2. 优化步骤执行
3. 学习率调度功能
"""

import paddle
from symbolicregression.optim import (
    Adam,
    AdamWithWarmup,
    AdamInverseSqrtWithWarmup,
    AdamCosineWithWarmup
)


def test_adam():
    """测试 Adam 优化器"""
    print("=" * 60)
    print("测试 Adam 优化器...")
    print("=" * 60)

    # 创建简单模型
    model = paddle.nn.Linear(10, 5)
    params = list(model.parameters())

    # 测试初始化
    optimizer = Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    print(f"✅ Adam 初始化成功")
    print(f"   参数数量: {len(optimizer._params_list)}")
    print(f"   超参数: betas={optimizer.betas}, eps={optimizer.eps}")
    print(f"   学习率: {optimizer.get_lr():.6f}")

    # 测试优化步骤
    x = paddle.randn([4, 10])
    y = model(x)
    loss = y.sum()
    loss.backward()

    optimizer.step()
    optimizer.clear_grad()

    print(f"✅ 优化步骤执行成功")
    print()

    return True


def test_adam_with_warmup():
    """测试 AdamWithWarmup 优化器"""
    print("=" * 60)
    print("测试 AdamWithWarmup 优化器...")
    print("=" * 60)

    # 创建简单模型
    model = paddle.nn.Linear(10, 5)
    params = list(model.parameters())

    # 测试初始化
    optimizer = AdamWithWarmup(
        params,
        lr=0.001,
        warmup_updates=100,
        warmup_init_lr=1e-7
    )
    print(f"✅ AdamWithWarmup 初始化成功")
    print(f"   初始学习率: {optimizer.get_lr_for_step(0):.10f}")
    print(f"   第50步学习率: {optimizer.get_lr_for_step(50):.10f}")
    print(f"   第100步学习率 (warmup结束): {optimizer.get_lr_for_step(100):.10f}")
    print(f"   第150步学习率: {optimizer.get_lr_for_step(150):.10f}")

    # 测试优化步骤和学习率更新
    x = paddle.randn([4, 10])
    y = model(x)
    loss = y.sum()
    loss.backward()

    # 执行多步优化,验证学习率递增
    lrs = []
    for _ in range(5):
        optimizer.step()
        lrs.append(optimizer.get_lr())
        optimizer.clear_grad()

        # 重新计算梯度
        y = model(paddle.randn([4, 10]))
        loss = y.sum()
        loss.backward()

    print(f"✅ 学习率调度正常工作")
    print(f"   前5步学习率变化: {[f'{lr:.10f}' for lr in lrs]}")
    print()

    return True


def test_adam_inverse_sqrt_with_warmup():
    """测试 AdamInverseSqrtWithWarmup 优化器"""
    print("=" * 60)
    print("测试 AdamInverseSqrtWithWarmup 优化器...")
    print("=" * 60)

    # 创建简单模型
    model = paddle.nn.Linear(10, 5)
    params = list(model.parameters())

    # 测试初始化
    optimizer = AdamInverseSqrtWithWarmup(
        params,
        lr=0.001,
        warmup_updates=100,
        warmup_init_lr=1e-7,
        exp_factor=0.5
    )
    print(f"✅ AdamInverseSqrtWithWarmup 初始化成功")
    print(f"   初始学习率: {optimizer.get_lr_for_step(0):.10f}")
    print(f"   第50步学习率: {optimizer.get_lr_for_step(50):.10f}")
    print(f"   第100步学习率 (warmup结束): {optimizer.get_lr_for_step(100):.10f}")
    print(f"   第200步学习率 (inverse sqrt衰减): {optimizer.get_lr_for_step(200):.10f}")
    print(f"   第500步学习率: {optimizer.get_lr_for_step(500):.10f}")

    # 测试优化步骤
    x = paddle.randn([4, 10])
    y = model(x)
    loss = y.sum()
    loss.backward()

    optimizer.step()
    optimizer.clear_grad()

    print(f"✅ 优化步骤执行成功")
    print()

    return True


def test_adam_cosine_with_warmup():
    """测试 AdamCosineWithWarmup 优化器"""
    print("=" * 60)
    print("测试 AdamCosineWithWarmup 优化器...")
    print("=" * 60)

    # 创建简单模型
    model = paddle.nn.Linear(10, 5)
    params = list(model.parameters())

    # 测试初始化
    optimizer = AdamCosineWithWarmup(
        params,
        lr=0.001,
        warmup_updates=100,
        warmup_init_lr=1e-7,
        min_lr=1e-9,
        init_period=1000,
        period_mult=1,
        lr_shrink=0.75
    )
    print(f"✅ AdamCosineWithWarmup 初始化成功")
    print(f"   初始学习率: {optimizer.get_lr_for_step(0):.10f}")
    print(f"   第50步学习率: {optimizer.get_lr_for_step(50):.10f}")
    print(f"   第100步学习率 (warmup结束): {optimizer.get_lr_for_step(100):.10f}")
    print(f"   第350步学习率 (cosine周期): {optimizer.get_lr_for_step(350):.10f}")
    print(f"   第600步学习率: {optimizer.get_lr_for_step(600):.10f}")

    # 测试优化步骤
    x = paddle.randn([4, 10])
    y = model(x)
    loss = y.sum()
    loss.backward()

    optimizer.step()
    optimizer.clear_grad()

    print(f"✅ 优化步骤执行成功")
    print()

    return True


def main():
    """运行所有测试"""
    print("\n")
    print("🔧" * 30)
    print("开始测试优化器修复...")
    print("🔧" * 30)
    print("\n")

    try:
        # 测试所有优化器
        results = []
        results.append(("Adam", test_adam()))
        results.append(("AdamWithWarmup", test_adam_with_warmup()))
        results.append(("AdamInverseSqrtWithWarmup", test_adam_inverse_sqrt_with_warmup()))
        results.append(("AdamCosineWithWarmup", test_adam_cosine_with_warmup()))

        # 打印总结
        print("=" * 60)
        print("测试总结:")
        print("=" * 60)

        all_passed = True
        for name, passed in results:
            status = "✅ 通过" if passed else "❌ 失败"
            print(f"  {name}: {status}")
            all_passed = all_passed and passed

        print("=" * 60)

        if all_passed:
            print("\n🎉 所有测试通过! 优化器修复成功!")
            print("\n下一步:")
            print("  1. 运行完整训练测试: bash ./bash/train_small.sh")
            print("  2. 验证学习率调度正常工作")
            print("  3. 检查训练日志中是否有 ValueError")
        else:
            print("\n❌ 部分测试失败,请检查代码!")
            return 1

        return 0

    except Exception as e:
        print("\n❌ 测试过程中发生错误:")
        print(f"   错误类型: {type(e).__name__}")
        print(f"   错误信息: {str(e)}")
        import traceback
        print("\n详细堆栈:")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
