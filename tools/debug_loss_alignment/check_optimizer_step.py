#!/usr/bin/env python3
import importlib.util
import math
import numpy as np
from pathlib import Path


def reference_adam_step(param, grad, exp_avg, exp_avg_sq, step, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    exp_avg = beta1 * exp_avg + (1 - beta1) * grad
    exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad * grad
    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step
    step_size = lr * math.sqrt(bias_correction2) / bias_correction1
    param = param - step_size * exp_avg / (np.sqrt(exp_avg_sq) + eps)
    return param, exp_avg, exp_avg_sq


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def check_torch_adam(repo_root: Path, param_np, grad_np, expected_param, lr):
    import torch

    optim_mod = load_module(
        repo_root / "PhysicsRegression" / "symbolicregression" / "optim.py",
        "torch_phy_optim",
    )
    param = torch.nn.Parameter(torch.tensor(param_np.copy(), dtype=torch.float32))
    opt = optim_mod.Adam([param], lr=lr)
    loss = (param * torch.tensor(grad_np.copy(), dtype=torch.float32)).sum()
    loss.backward()
    opt.step()
    actual = param.detach().cpu().numpy()
    max_abs = np.max(np.abs(actual - expected_param))
    print(f"torch_adam_max_abs={max_abs:.12e}")
    assert max_abs < 1e-7, max_abs


def check_paddle_adam(repo_root: Path, param_np, grad_np, expected_param, lr):
    import paddle

    optim_mod = load_module(
        repo_root / "PhysicsRegressionPaddle" / "symbolicregression" / "optim.py",
        "paddle_phy_optim",
    )
    param = paddle.create_parameter(shape=param_np.shape, dtype="float32")
    param.set_value(paddle.to_tensor(param_np.copy(), dtype="float32"))
    opt = optim_mod.Adam([param], lr=lr)
    loss = paddle.sum(param * paddle.to_tensor(grad_np.copy(), dtype="float32"))
    loss.backward()
    opt.step()
    actual = param.numpy()
    max_abs = np.max(np.abs(actual - expected_param))
    print(f"paddle_adam_max_abs={max_abs:.12e}")
    assert max_abs < 1e-7, max_abs


def check_warmup_schedule(repo_root: Path):
    torch_mod = load_module(
        repo_root / "PhysicsRegression" / "symbolicregression" / "optim.py",
        "torch_phy_optim_warmup",
    )
    paddle_mod = load_module(
        repo_root / "PhysicsRegressionPaddle" / "symbolicregression" / "optim.py",
        "paddle_phy_optim_warmup",
    )
    for step in [0, 1, 50, 100, 200, 500]:
        # Use an uninitialized instance only to call the pure formula method.
        torch_obj = object.__new__(torch_mod.AdamInverseSqrtWithWarmup)
        torch_obj.warmup_updates = 100
        torch_obj.warmup_init_lr = 1e-7
        torch_obj.lr_step = (1e-4 - 1e-7) / 100
        torch_obj.exp_factor = 0.5
        torch_obj.decay_factor = 1e-4 * 100 ** 0.5

        paddle_obj = object.__new__(paddle_mod.AdamInverseSqrtWithWarmup)
        paddle_obj.warmup_updates = 100
        paddle_obj.warmup_init_lr = 1e-7
        paddle_obj.lr_step = (1e-4 - 1e-7) / 100
        paddle_obj.exp_factor = 0.5
        paddle_obj.decay_factor = 1e-4 * 100 ** 0.5

        torch_lr = torch_obj.get_lr_for_step(step)
        paddle_lr = paddle_obj.get_lr_for_step(step)
        err = abs(torch_lr - paddle_lr)
        print(f"warmup_step={step} torch_lr={torch_lr:.12e} paddle_lr={paddle_lr:.12e} abs={err:.12e}")
        assert err < 1e-15, err


def main():
    repo_root = Path(__file__).resolve().parents[2]
    rng = np.random.default_rng(0)
    param = rng.normal(size=(4, 5)).astype("float32")
    grad = rng.normal(size=(4, 5)).astype("float32")
    exp_avg = np.zeros_like(param)
    exp_avg_sq = np.zeros_like(param)
    lr = 1e-4

    expected_param, expected_exp_avg, expected_exp_avg_sq = reference_adam_step(
        param.copy(), grad, exp_avg.copy(), exp_avg_sq.copy(), step=1, lr=lr
    )

    print("reference_param_hash_input_shape=(4,5)")
    print(f"expected_param_mean={expected_param.mean():.12e}")
    print(f"expected_param_std={expected_param.std():.12e}")
    print(f"expected_exp_avg_mean={expected_exp_avg.mean():.12e}")
    print(f"expected_exp_avg_sq_mean={expected_exp_avg_sq.mean():.12e}")
    check_torch_adam(repo_root, param, grad, expected_param, lr)
    check_paddle_adam(repo_root, param, grad, expected_param, lr)
    check_warmup_schedule(repo_root)


if __name__ == "__main__":
    main()
