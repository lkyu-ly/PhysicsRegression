#!/usr/bin/env python3
import argparse
import contextlib
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np


TRAIN_SMALL_ARGS = [
    "--max_epoch", "2",
    "--dump_path", "./",
    "--exp_name", "test",
    "--exp_id", "0",
    "--n_steps_per_epoch", "500",
    "--print_freq", "50",
    "--optimizer", "adam_inverse_sqrt,warmup_updates=100",
    "--collate_queue_size", "1000",
    "--batch_size", "512",
    "--save_periodic", "-1",
    "--save_periodic_from", "40",
    "--eval_size", "200",
    "--batch_size_eval", "64",
    "--num_workers", "0",
    "--max_len", "200",
    "--max_number_bags", "-1",
    "--max_input_points", "200",
    "--tokens_per_batch", "5000",
    "--add_consts", "1",
    "--device", "cuda:0",
    "--use_exprs", "200000",
    "--use_dimension_mask", "0",
    "--expr_train_data_path", "./data/exprs_train.json",
    "--expr_valid_data_path", "./data/exprs_valid.json",
    "--sub_expr_train_path", "./data/exprs_seperated_train.json",
    "--sub_expr_valid_path", "./data/exprs_seperated_valid.json",
    "--decode_physical_units", "single-seq",
    "--use_hints", "units,complexity,unarys,consts",
    "--random_variables_sequence", "0",
    "--max_trials", "10",
    "--generate_datapoints_distribution", "positive,multi",
    "--rescale", "0",
    "--cpu", "True",
]


@contextlib.contextmanager
def backend_context(repo_root: Path, backend: str):
    project = repo_root / ("PhysicsRegression" if backend == "torch" else "PhysicsRegressionPaddle")
    old_cwd = Path.cwd()
    old_sys_path = list(sys.path)
    os.chdir(project)
    sys.path.insert(0, str(project))
    try:
        yield project
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_sys_path


def to_builtin(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    return value


def stable_hash(value) -> str:
    payload = json.dumps(to_builtin(value), sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def seed_backend(backend: str, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if backend == "torch":
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    else:
        import paddle
        paddle.seed(seed)


def normalize_params(params):
    params.cpu = True
    params.num_workers = 0
    params.is_slurm_job = False
    params.debug_slurm = True
    params.n_nodes = 1
    params.node_id = 0
    params.local_rank = 0
    params.global_rank = 0
    params.world_size = 1
    params.n_gpu_per_node = 1
    params.multi_node = False
    params.multi_gpu = False
    params.is_master = True
    return params


def summarize_batch(step: int, batch, errors):
    infos = to_builtin(batch["infos"])
    tree_encoded = to_builtin(batch["tree_encoded"])
    dim_encoded = to_builtin(batch["dim_encoded"])
    x_to_fit = to_builtin(batch["x_to_fit"])
    y_to_fit = to_builtin(batch["y_to_fit"])
    input_lengths = infos.get("input_sequence_length", [])
    tree_lengths = [len(x) for x in tree_encoded]
    return {
        "step": step,
        "batch_size": len(tree_encoded),
        "input_length_min": min(input_lengths) if input_lengths else None,
        "input_length_max": max(input_lengths) if input_lengths else None,
        "input_length_sum": sum(input_lengths) if input_lengths else None,
        "tree_length_min": min(tree_lengths) if tree_lengths else None,
        "tree_length_max": max(tree_lengths) if tree_lengths else None,
        "tree_length_sum": sum(tree_lengths),
        "first_tree": tree_encoded[0] if tree_encoded else [],
        "first_dim": dim_encoded[0] if dim_encoded else [],
        "x_hash": stable_hash(x_to_fit),
        "y_hash": stable_hash(y_to_fit),
        "tree_hash": stable_hash(tree_encoded),
        "dim_hash": stable_hash(dim_encoded),
        "infos_hash": stable_hash(infos),
        "errors": to_builtin(errors),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["torch", "paddle"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    # Resolve output path to absolute BEFORE chdir (backend_context changes cwd)
    out_path_abs = Path(args.out).resolve()
    seed_backend(args.backend, args.seed)

    with backend_context(repo_root, args.backend):
        from parsers import get_parser
        from symbolicregression.envs import build_env
        from symbolicregression.model import build_modules

        params = get_parser().parse_args(TRAIN_SMALL_ARGS)
        params = normalize_params(params)
        env = build_env(params)
        # build_modules binds env.get_length_after_batching (needed by generate_sample)
        build_modules(env, params)
        iterator = iter(env.create_train_iterator(params.tasks[0], None, params))
        out_path = out_path_abs
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for step in range(args.steps):
                batch, errors = next(iterator)
                f.write(json.dumps(summarize_batch(step, batch, errors), sort_keys=True, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
