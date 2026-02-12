#!/usr/bin/env python
"""
GPU训练简单测试 - 验证修复后的train.py能否正常运行
"""
import subprocess
import sys

print("="*80)
print("GPU 训练修复验证测试")
print("="*80)

# 测试参数
cmd = [
    "python", "train.py",
    "--device", "gpu:0",
    "--max_epoch", "1",
    "--n_steps_per_epoch", "3",  # 极小步数,快速测试
    "--dump_path", "./",
    "--exp_name", "device_fix_test",
    "--exp_id", "0",
    "--collate_queue_size", "1000",
    "--batch_size", "32",  # 极小批次
    "--save_periodic", "-1",  # 不保存
    "--eval_size", "10",
    "--batch_size_eval", "10",
    "--num_workers", "0",
    "--max_len", "100",
    "--max_number_bags", "-1",
    "--max_input_points", "50",
    "--tokens_per_batch", "5000",  # 极小tokens
    "--add_consts", "1",
    "--use_exprs", "1000",
    "--use_dimension_mask", "0",
    "--expr_train_data_path", "./data/exprs_train.json",
    "--expr_valid_data_path", "./data/exprs_valid.json",
    "--sub_expr_train_path", "./data/exprs_seperated_train.json",
    "--sub_expr_valid_path", "./data/exprs_seperated_valid.json",
    "--decode_physical_units", "single-seq",
    "--use_hints", "units,complexity,unarys,consts",
    "--random_variables_sequence", "0",
    "--max_trials", "5",
    "--generate_datapoints_distribution", "positive,multi",
    "--rescale", "0",
]

print("\n运行命令:")
print(" ".join(cmd))
print("\n" + "="*80)

try:
    result = subprocess.run(cmd, timeout=180, capture_output=True, text=True)

    print("\n标准输出:")
    print(result.stdout)

    if result.stderr:
        print("\n标准错误:")
        print(result.stderr)

    if result.returncode == 0:
        print("\n" + "="*80)
        print("✅ 训练测试成功完成!")
        print("="*80)
        sys.exit(0)
    else:
        print("\n" + "="*80)
        print(f"❌ 训练测试失败,退出码: {result.returncode}")
        print("="*80)

        # 检查特定错误
        full_output = result.stdout + result.stderr
        if "AssertionError" in full_output and "y == self.pad_index" in full_output:
            print("⚠️ 检测到 AssertionError (问题1) - 修复可能不完整")
        elif "CUDA error 700" in full_output or "illegal memory access" in full_output:
            print("⚠️ 检测到 CUDA error 700 (问题2) - 修复可能不完整")

        sys.exit(1)

except subprocess.TimeoutExpired:
    print("\n" + "="*80)
    print("⏱️ 训练超时 (>180秒) - 可能有其他问题")
    print("="*80)
    sys.exit(1)
except Exception as e:
    print("\n" + "="*80)
    print(f"❌ 运行错误: {e}")
    print("="*80)
    sys.exit(1)
