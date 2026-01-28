#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
迁移修复验证测试脚本

测试内容:
1. to_cuda() 函数的设备管理修复 (问题6)
2. transformer.py 中 .new() 方法的修复 (问题7)
3. 优化器初始化修复 (问题5)
"""

import sys
import paddle
from symbolicregression.utils import to_cuda
from paddle_utils import device2int


def test_device_management():
    """测试问题6: tensor.cuda(device=) 参数修复"""
    print("=" * 60)
    print("测试 1: to_cuda 函数 (问题6修复)")
    print("=" * 60)

    try:
        # 测试基本张量移动
        x = paddle.randn([3, 4])
        print(f"✓ 创建张量成功，原始设备: {x.place}")

        # 测试 to_cuda 函数
        y, = to_cuda(x, device=0)
        print(f"✓ to_cuda(device=0) 成功，目标设备: {y.place}")

        # 测试字符串设备参数
        a = paddle.randn([2, 3])
        b, = to_cuda(a, device="cuda:0")
        print(f"✓ to_cuda(device='cuda:0') 成功，目标设备: {b.place}")

        # 测试 None 值处理
        x1 = paddle.randn([2, 2])
        x2 = None
        x3 = paddle.randn([3, 3])
        result = to_cuda(x1, x2, x3, device=0)
        assert result[1] is None, "None值应该保持为None"
        print(f"✓ None 值处理正确")

        # 测试 use_cpu 标志
        x = paddle.randn([2, 2])
        y, = to_cuda(x, use_cpu=True, device=0)
        print(f"✓ use_cpu=True 模式正常")

        print("\n✅ 问题6修复验证通过: to_cuda 函数工作正常\n")
        return True

    except Exception as e:
        print(f"\n❌ 问题6修复验证失败:")
        print(f"   错误类型: {type(e).__name__}")
        print(f"   错误信息: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_transformer_tensor_creation():
    """测试问题7: tensor.new() 方法修复"""
    print("=" * 60)
    print("测试 2: Transformer 张量创建 (问题7修复)")
    print("=" * 60)

    try:
        # 导入 transformer 模块
        from symbolicregression.model.transformer import TransformerModel
        from symbolicregression import envs
        import argparse

        # 创建最小化参数
        params = argparse.Namespace(
            # 环境基本参数（必需）
            env_name='char_sp',
            env_base_seed=-1,
            # 编码器参数
            enc_emb_dim=128,
            n_enc_layers=1,
            n_enc_heads=4,
            n_enc_hidden_layers=1,
            enc_attention_dropout=0.1,
            dropout=0.1,
            attention_dropout=0.1,
            # 解码器参数
            dec_emb_dim=128,
            n_dec_layers=1,
            n_dec_heads=4,
            n_dec_hidden_layers=1,
            dec_attention_dropout=0.1,
            # 其他
            sinusoidal_embeddings=False,
            share_inout_emb=False,
            reload_emb='',
            emb_dim=128,
            # 环境参数
            operators='add:10,mul:10,sub:5,div:5',
            max_ops=5,
            int_base=10,
            balanced_ops=True,
            positive=True,
            nonnegative=True,
            max_len=50,
            precision=3,
            double_seq=True,
            # 数据生成
            n_points=10,
            n_output=1,
            n_output_units=1,
            use_hints='units,complexity',
            max_hint_complexity=5,
            # 词表参数
            max_int=100,
            # 额外必需参数
            max_number_bags=None,
            skip_zero_gradient=True,
            use_controller=False
        )

        print("✓ 参数初始化成功")

        # 创建环境
        env = envs.build_env(params)
        print("✓ 环境创建成功")

        # 创建简单的 id2word 词表
        id2word = {i: str(i) for i in range(100)}
        id2word[0] = '<PAD>'
        id2word[1] = '<EOS>'
        id2word[2] = '<BOS>'

        # 测试编码器初始化（不会触发 .new() 调用）
        encoder = TransformerModel(
            params=params,
            id2word=id2word,
            is_encoder=True,
            is_decoder=False,
            with_output=False
        )
        print("✓ Transformer编码器初始化成功")

        # 测试解码器初始化
        decoder = TransformerModel(
            params=params,
            id2word=id2word,
            is_encoder=False,
            is_decoder=True,
            with_output=True
        )
        print("✓ Transformer解码器初始化成功")

        # 测试前向传播（会触发内部张量创建）
        batch_size = 2
        seq_len = 10

        # 创建输入张量
        x = paddle.randn([seq_len, batch_size, params.enc_emb_dim])
        lengths = paddle.to_tensor([seq_len, seq_len], dtype='int64')

        # 编码器前向
        encoded = encoder.fwd(
            mode='fwd',
            x=x,
            lengths=lengths,
            causal=False
        )
        print(f"✓ 编码器前向传播成功，输出形状: {encoded.shape}")

        # 解码器前向（需要目标序列）
        y = paddle.randint(0, 100, [seq_len, batch_size], dtype='int64')
        decoded = decoder.fwd(
            mode='fwd',
            x=y,
            lengths=lengths,
            causal=True,
            src_enc=encoded,
            src_len=lengths
        )
        print(f"✓ 解码器前向传播成功，输出形状: {decoded.shape}")

        print("\n✅ 问题7修复验证通过: Transformer张量创建正常\n")
        return True

    except Exception as e:
        print(f"\n❌ 问题7修复验证失败:")
        print(f"   错误类型: {type(e).__name__}")
        print(f"   错误信息: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_optimizer_initialization():
    """测试问题5: 优化器初始化修复"""
    print("=" * 60)
    print("测试 3: 优化器初始化 (问题5修复)")
    print("=" * 60)

    try:
        from symbolicregression.optim import (
            Adam,
            AdamWithWarmup,
            AdamInverseSqrtWithWarmup,
            AdamCosineWithWarmup
        )

        # 创建简单模型
        model = paddle.nn.Linear(10, 5)
        params = list(model.parameters())

        # 测试 Adam
        optimizer1 = Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
        print(f"✓ Adam 初始化成功")

        # 测试 AdamWithWarmup
        optimizer2 = AdamWithWarmup(
            params,
            lr=0.001,
            warmup_updates=100,
            warmup_init_lr=1e-7
        )
        print(f"✓ AdamWithWarmup 初始化成功")

        # 测试 AdamInverseSqrtWithWarmup
        optimizer3 = AdamInverseSqrtWithWarmup(
            params,
            lr=0.001,
            warmup_updates=100,
            warmup_init_lr=1e-7,
            exp_factor=0.5
        )
        print(f"✓ AdamInverseSqrtWithWarmup 初始化成功")

        # 测试 AdamCosineWithWarmup
        optimizer4 = AdamCosineWithWarmup(
            params,
            lr=0.001,
            warmup_updates=100,
            warmup_init_lr=1e-7,
            min_lr=1e-9,
            init_period=1000,
            period_mult=1,
            lr_shrink=0.75
        )
        print(f"✓ AdamCosineWithWarmup 初始化成功")

        # 测试优化步骤
        x = paddle.randn([4, 10])
        y = model(x)
        loss = y.sum()
        loss.backward()

        optimizer1.step()
        optimizer1.clear_grad()
        print(f"✓ 优化器step执行成功")

        print("\n✅ 问题5修复验证通过: 所有优化器初始化正常\n")
        return True

    except Exception as e:
        print(f"\n❌ 问题5修复验证失败:")
        print(f"   错误类型: {type(e).__name__}")
        print(f"   错误信息: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n")
    print("🔧" * 30)
    print("开始验证PaddlePaddle迁移修复...")
    print("🔧" * 30)
    print("\n")

    results = []

    # 测试1: 设备管理
    results.append(("to_cuda函数 (问题6)", test_device_management()))

    # 测试2: Transformer张量创建
    results.append(("Transformer张量创建 (问题7)", test_transformer_tensor_creation()))

    # 测试3: 优化器初始化
    results.append(("优化器初始化 (问题5)", test_optimizer_initialization()))

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
        print("\n🎉 所有测试通过! PaddlePaddle迁移修复成功!")
        print("\n修复总结:")
        print("  问题5: ✅ 优化器基类初始化签名已修复 (optim.py)")
        print("  问题6: ✅ tensor.cuda(device=) 参数已修复 (utils.py)")
        print("  问题7: ✅ tensor.new() 方法已替换 (transformer.py, 15处)")
        print("\n所有修复已记录到: PADDLE_MIGRATION.md")
        print("\n下一步:")
        print("  1. 运行完整训练: bash ./bash/train_small.sh")
        print("  2. 验证训练稳定性和收敛性")
        print("  3. 对比PyTorch版本的数值精度")
        return 0
    else:
        print("\n❌ 部分测试失败,请检查修复!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
