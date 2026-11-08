#!/usr/bin/env python3
import argparse
import contextlib
import os
import subprocess
import sys
import tempfile
import time
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
    parser.add_argument("--worker", choices=["torch", "paddle", "paddle-bench"])
    parser.add_argument("--artifact")
    # ---- CINN on/off 前向延迟基准（仅 paddle 侧）----
    parser.add_argument(
        "--bench",
        action="store_true",
        help="Benchmark paddle single-forward latency with CINN off/on",
    )
    parser.add_argument(
        "--num-seqs", type=int, default=1, help="Batch size (replicated sequences)"
    )
    parser.add_argument(
        "--warmup", type=int, default=3, help="Warmup runs excluded from timing"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Timed runs to average (B=1 噪声 ±20%，3 次不够)",
    )
    parser.add_argument("--cpu", action="store_true", help="Run benchmark on CPU")
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


def run_paddle_bench_worker(args):
    """Paddle 单次前向延迟基准：CINN 开关由 PHYE2E_USE_CINN 环境变量控制。

    与 trainer.py / train_small_cinn.sh 的既有语义保持一致：
    - 开启时仅对 encoder/decoder 做 paddle.jit.to_static（embedder 保持动态）；
    - FLAGS_*（prim/CINN）由父进程在子进程环境中注入，需在 import paddle 前生效。
    """
    import paddle

    repo_root = Path(__file__).resolve().parent
    use_cinn = os.environ.get("PHYE2E_USE_CINN", "").lower() in ("1", "true", "yes")
    paddle_root = repo_root / "PhysicsRegressionPaddle"
    with project_context(paddle_root):
        from symbolicregression.envs import build_env
        from symbolicregression.envs.node import Node
        from symbolicregression.model import build_modules

        checkpoint = paddle.load(str(repo_root / args.paddle_model))
        params = normalize_params(checkpoint["params"])
        params.cpu = args.cpu
        env = build_env(params)
        modules = build_modules(env, params)
        for name in ("embedder", "encoder", "decoder"):
            modules[name].set_state_dict(checkpoint[name])
            modules[name].eval()

        if use_cinn:
            for name in ("encoder", "decoder"):
                modules[name] = paddle.jit.to_static(modules[name])

        _, _, inputs = build_random_inputs(args.seed, args.num_points, args.input_dim)
        inputs = inputs * args.num_seqs
        hints = [h * args.num_seqs for h in build_minimal_hints(params.use_hints, args.input_dim)]
        root = Node("x_0", params)
        root.unit = np.array([0, 0, 0, 0, 0])
        tree_encoded, _ = env.equation_encoder.encode(root)
        x2, len2, units = env.batch_equations(
            env.word_to_idx([tree_encoded] * args.num_seqs, float_input=False),
            decode_physical_units=params.decode_physical_units,
        )

        # 计时口径：仅主模型本体三模块（embedder -> encoder -> decoder 的 fwd）。
        # decoder "predict" 分支整体不计入——它把输出投影与 mask 选择、
        # cross_entropy、topk 等训练/解码外围逻辑耦合在一起，无法干净切分。
        # embedder 与训练侧约定一致保持动态图；to_static 仅包裹 encoder/decoder。
        def forward_once():
            x1, len1 = modules["embedder"](inputs, hints)
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
            return decoded

        sync = (lambda: None) if args.cpu else paddle.device.synchronize

        with paddle.no_grad():
            # 预热运行：触发 CINN 编译 / 算子缓存，不计入计时
            for _ in range(args.warmup):
                forward_once()
            sync()

            times_ms = []
            for _ in range(args.runs):
                t0 = time.perf_counter()
                forward_once()
                sync()
                times_ms.append((time.perf_counter() - t0) * 1000.0)
        mean_ms = sum(times_ms) / len(times_ms)

        print(
            f"[paddle][CINN={'on' if use_cinn else 'off'}] "
            f"single forward: {mean_ms:.2f} ms "
            f"(mean of {args.runs} runs, {args.warmup} warmup excluded; "
            f"batch={args.num_seqs}, device={'cpu' if args.cpu else 'gpu'})"
        )
        # 机器可读行，供父进程汇总；附逐次耗时，便于核查是否仍有编译尾巴
        print(f"BENCH_RESULT cinn={'on' if use_cinn else 'off'} {mean_ms:.3f} "
              f"runs=[{', '.join(f'{t:.2f}' for t in times_ms)}]")
        return mean_ms


CINN_BENCH_ENV_KEYS = (
    "PHYE2E_USE_CINN",
    "FLAGS_prim_enable_dynamic",
    "FLAGS_prim_all",
    "FLAGS_use_cinn",
)


def bench_main(args):
    """依次以 CINN off/on 两个独立子进程运行 paddle 前向基准并汇总。"""
    script = Path(__file__).resolve()
    base_cmd = [
        sys.executable,
        str(script),
        "--paddle-model",
        args.paddle_model,
        "--seed",
        str(args.seed),
        "--num-points",
        str(args.num_points),
        "--input-dim",
        str(args.input_dim),
        "--num-seqs",
        str(args.num_seqs),
        "--warmup",
        str(args.warmup),
        "--runs",
        str(args.runs),
    ]
    if args.cpu:
        base_cmd.append("--cpu")

    # OFF：显式剥离父 shell 可能残留的 CINN 相关环境变量，保证纯动态图
    env_off = {k: v for k, v in os.environ.items() if k not in CINN_BENCH_ENV_KEYS}
    # ON：与 bash/train_small_cinn.sh 完全一致的 flag 组合
    env_on = {
        **env_off,
        "PHYE2E_USE_CINN": "true",
        "FLAGS_prim_enable_dynamic": "true",
        "FLAGS_prim_all": "true",
        "FLAGS_use_cinn": "true",
    }

    results = {}
    for label, env in (("off", env_off), ("on", env_on)):
        proc = subprocess.run(
            base_cmd + ["--worker", "paddle-bench"],
            env=env,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print(proc.stdout[-3000:])
            print(proc.stderr[-3000:], file=sys.stderr)
            raise SystemExit(f"[CINN {label}] benchmark worker failed")
        for line in proc.stdout.splitlines():
            if line.startswith("BENCH_RESULT"):
                results[label] = float(line.split()[2])
        if proc.stderr.strip():
            print(f"[CINN {label}] stderr (tail):\n{proc.stderr[-1500:]}")

    print("\n=== Paddle forward latency (mean of {} runs, warmup excluded) ===".format(args.runs))
    print(f"CINN off: {results['off']:.2f} ms")
    print(f"CINN on : {results['on']:.2f} ms")
    print(f"speedup (t_off / t_on): {results['off'] / results['on']:.2f}x")


def parent_main(args):
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
    if args.worker == "paddle-bench":
        run_paddle_bench_worker(args)
        return
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
        if args.bench:
            bench_main(args)
        else:
            parent_main(args)
    else:
        worker_main(args)
