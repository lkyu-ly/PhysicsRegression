#!/usr/bin/env python3
import argparse
import contextlib
import os
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import paddle
import torch


@contextlib.contextmanager
def project_context(project_dir: Path):
    old_cwd = Path.cwd()
    old_sys_path = list(sys.path)
    os.chdir(project_dir)
    sys.path.insert(0, str(project_dir))
    try:
        yield
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_sys_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch checkpoint to PaddlePaddle format"
    )
    parser.add_argument("--torch-model", default="models/model.pt")
    parser.add_argument("--paddle-model", default="models/model.pdparams")
    return parser.parse_args()


def normalize_params(raw_params):
    if isinstance(raw_params, Namespace):
        return raw_params
    if isinstance(raw_params, dict):
        return Namespace(**raw_params)
    raise TypeError(f"Unsupported params type: {type(raw_params)!r}")


def build_paddle_modules(repo_root: Path, params: Namespace):
    paddle_root = repo_root / "PhysicsRegressionPaddle"
    with project_context(paddle_root):
        from symbolicregression.envs import build_env
        from symbolicregression.model import build_modules

        params.cpu = True
        env = build_env(params)
        modules = build_modules(env, params)
        return env, modules


def validate_state_dict(torch_state_dict, paddle_state_dict, module_name: str):
    torch_keys = list(torch_state_dict.keys())
    paddle_keys = list(paddle_state_dict.keys())
    if torch_keys != paddle_keys:
        raise ValueError(
            f"{module_name} key mismatch:\n"
            f"torch={torch_keys[:5]}...\n"
            f"paddle={paddle_keys[:5]}..."
        )
    for key in torch_keys:
        torch_shape = tuple(torch_state_dict[key].shape)
        paddle_shape = tuple(paddle_state_dict[key].shape)
        if torch_shape != paddle_shape:
            raise ValueError(
                f"{module_name}.{key} shape mismatch: "
                f"{torch_shape} != {paddle_shape}"
            )


def convert_torch_state_dict(torch_state_dict):
    converted = {}
    for key, tensor in torch_state_dict.items():
        array = tensor.detach().cpu().numpy()
        converted[key] = paddle.to_tensor(array)
    return converted


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    torch_model_path = repo_root / args.torch_model
    paddle_model_path = repo_root / args.paddle_model
    paddle_model_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(torch_model_path, map_location="cpu", weights_only=False)
    params = normalize_params(checkpoint["params"])
    _, modules = build_paddle_modules(repo_root, params)

    save_dict = {"params": vars(params).copy()}
    for module_name in ("embedder", "encoder", "decoder"):
        paddle_state_dict = modules[module_name].state_dict()
        torch_state_dict = checkpoint[module_name]
        validate_state_dict(torch_state_dict, paddle_state_dict, module_name)
        converted = convert_torch_state_dict(torch_state_dict)
        modules[module_name].set_state_dict(converted)
        save_dict[module_name] = modules[module_name].state_dict()
        print(f"[OK] {module_name}: {len(converted)} tensors verified and loaded")

    paddle.save(save_dict, str(paddle_model_path))
    print(f"[DONE] Saved converted checkpoint to: {paddle_model_path}")


if __name__ == "__main__":
    main()
