#!/usr/bin/env python3
import argparse
import contextlib
import os
import subprocess
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np


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
        description="Compare Torch and Paddle forward precision"
    )
    parser.add_argument("--torch-model", default="models/model.pt")
    parser.add_argument("--paddle-model", default="models/model.pdparams")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-points", type=int, default=8)
    parser.add_argument("--input-dim", type=int, default=1)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--worker", choices=["torch", "paddle"])
    parser.add_argument("--artifact")
    return parser.parse_args()


def normalize_params(raw_params):
    if isinstance(raw_params, Namespace):
        return raw_params
    if isinstance(raw_params, dict):
        return Namespace(**raw_params)
    raise TypeError(f"Unsupported params type: {type(raw_params)!r}")


def build_random_inputs(seed: int, num_points: int, input_dim: int):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((num_points, input_dim), dtype=np.float32)
    y = rng.standard_normal((num_points, 1), dtype=np.float32)
    inputs = [[[x[i], y[i]] for i in range(num_points)]]
    return x, y, inputs


def build_minimal_hints(use_hints: str, input_dim: int):
    hints = []
    for hint_name in use_hints.split(","):
        if hint_name == "units":
            units = [np.array([0, 0, 0, 0, 0]) for _ in range(input_dim)] + [
                np.array([0, 0, 0, 0, 0])
            ]
            hints.append([units])
        elif hint_name == "complexity":
            hints.append([["simple"]])
        elif hint_name == "unarys":
            hints.append([[]])
        elif hint_name == "consts":
            hints.append([[]])
        elif hint_name == "add_structure":
            hints.append([[]])
        elif hint_name == "mul_structure":
            hints.append([[]])
        else:
            raise ValueError(f"Unsupported hint type: {hint_name}")
    return hints


def run_torch_worker(
    repo_root: Path, torch_model: Path, seed: int, num_points: int, input_dim: int
):
    import torch

    torch_root = repo_root / "PhysicsRegression"
    with project_context(torch_root):
        from symbolicregression.envs import build_env
        from symbolicregression.envs.node import Node
        from symbolicregression.model import build_modules

        checkpoint = torch.load(torch_model, map_location="cpu", weights_only=False)
        params = normalize_params(checkpoint["params"])
        params.cpu = True
        env = build_env(params)
        modules = build_modules(env, params)
        for name in ("embedder", "encoder", "decoder"):
            modules[name].load_state_dict(checkpoint[name])
            modules[name].eval()

        _, _, inputs = build_random_inputs(seed, num_points, input_dim)
        hints = build_minimal_hints(params.use_hints, input_dim)

        with torch.no_grad():
            x1, len1 = modules["embedder"](inputs, hints)
            root = Node("x_0", params)
            root.unit = np.array([0, 0, 0, 0, 0])
            tree_encoded, _ = env.equation_encoder.encode(root)
            x2, len2, units = env.batch_equations(
                env.word_to_idx([tree_encoded], float_input=False),
                decode_physical_units=params.decode_physical_units,
            )
            alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
            pred_mask = alen[:, None] < len2[None] - 1
            y_tgt = x2[1:].masked_select(pred_mask[:-1])
            encoded = modules["encoder"]("fwd", x=x1, lengths=len1, causal=False)
            decoded = modules["decoder"](
                "fwd",
                x=x2,
                lengths=len2,
                causal=True,
                src_enc=encoded.transpose(0, 1),
                src_len=len1,
                units=units,
            )
            scores, _ = modules["decoder"](
                "predict",
                tensor=decoded,
                pred_mask=pred_mask,
                y=y_tgt,
                get_scores=False,
            )
            return scores.detach().cpu().numpy()


def run_paddle_worker(
    repo_root: Path, paddle_model: Path, seed: int, num_points: int, input_dim: int
):
    import paddle

    paddle_root = repo_root / "PhysicsRegressionPaddle"
    with project_context(paddle_root):
        from symbolicregression.envs import build_env
        from symbolicregression.envs.node import Node
        from symbolicregression.model import build_modules

        checkpoint = paddle.load(str(paddle_model))
        params = normalize_params(checkpoint["params"])
        params.cpu = True
        env = build_env(params)
        modules = build_modules(env, params)
        for name in ("embedder", "encoder", "decoder"):
            modules[name].set_state_dict(checkpoint[name])
            modules[name].eval()

        _, _, inputs = build_random_inputs(seed, num_points, input_dim)
        hints = build_minimal_hints(params.use_hints, input_dim)

        with paddle.no_grad():
            x1, len1 = modules["embedder"](inputs, hints)
            root = Node("x_0", params)
            root.unit = np.array([0, 0, 0, 0, 0])
            tree_encoded, _ = env.equation_encoder.encode(root)
            x2, len2, units = env.batch_equations(
                env.word_to_idx([tree_encoded], float_input=False),
                decode_physical_units=params.decode_physical_units,
            )
            alen = paddle.arange(int(paddle.max(len2).item()), dtype="int64")
            pred_mask = alen[:, None] < len2[None] - 1
            y_tgt = x2[1:].masked_select(pred_mask[:-1])
            encoded = modules["encoder"]("fwd", x=x1, lengths=len1, causal=False)
            decoded = modules["decoder"](
                "fwd",
                x=x2,
                lengths=len2,
                causal=True,
                src_enc=encoded.transpose([1, 0, 2]),
                src_len=len1,
                units=units,
            )
            scores, _ = modules["decoder"](
                "predict",
                tensor=decoded,
                pred_mask=pred_mask,
                y=y_tgt,
                get_scores=False,
            )
            return scores.cpu().numpy()


def parent_main(args):
    repo_root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="phye2e_compare_") as temp_dir:
        temp_dir = Path(temp_dir)
        torch_out = temp_dir / "torch_logits.npy"
        paddle_out = temp_dir / "paddle_logits.npy"

        base_cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--torch-model",
            args.torch_model,
            "--paddle-model",
            args.paddle_model,
            "--seed",
            str(args.seed),
            "--num-points",
            str(args.num_points),
            "--input-dim",
            str(args.input_dim),
        ]
        subprocess.run(
            base_cmd + ["--worker", "torch", "--artifact", str(torch_out)], check=True
        )
        subprocess.run(
            base_cmd
            + ["--worker", "paddle", "--artifact", str(paddle_out)],
            check=True,
        )

        torch_logits = np.load(torch_out)
        paddle_logits = np.load(paddle_out)
        abs_err = np.abs(paddle_logits - torch_logits)
        rel_err = abs_err / np.maximum(np.abs(torch_logits), args.eps)

        print(f"logits_shape: {torch_logits.shape}")
        print(f"torch_dtype: {torch_logits.dtype}")
        print(f"paddle_dtype: {paddle_logits.dtype}")
        print(f"mean_abs_error: {abs_err.mean():.10e}")
        print(f"max_abs_error: {abs_err.max():.10e}")
        print(f"mean_rel_error: {rel_err.mean():.10e}")
        print(f"max_rel_error: {rel_err.max():.10e}")


def worker_main(args):
    repo_root = Path(__file__).resolve().parent
    artifact = Path(args.artifact)
    if args.worker == "torch":
        logits = run_torch_worker(
            repo_root,
            repo_root / args.torch_model,
            args.seed,
            args.num_points,
            args.input_dim,
        )
    else:
        logits = run_paddle_worker(
            repo_root,
            repo_root / args.paddle_model,
            args.seed,
            args.num_points,
            args.input_dim,
        )
    np.save(artifact, logits)


if __name__ == "__main__":
    args = parse_args()
    if args.worker is None:
        parent_main(args)
    else:
        worker_main(args)
