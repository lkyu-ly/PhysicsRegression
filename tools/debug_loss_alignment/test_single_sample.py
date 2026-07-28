#!/usr/bin/env python3
"""Test if a single generate_sample produces identical output across frameworks."""
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
    "--max_epoch", "2", "--dump_path", "./", "--exp_name", "test", "--exp_id", "0",
    "--n_steps_per_epoch", "500", "--print_freq", "50",
    "--optimizer", "adam_inverse_sqrt,warmup_updates=100",
    "--collate_queue_size", "1000", "--batch_size", "512",
    "--save_periodic", "-1", "--save_periodic_from", "40",
    "--eval_size", "200", "--batch_size_eval", "64",
    "--num_workers", "0", "--max_len", "200", "--max_number_bags", "-1",
    "--max_input_points", "200", "--tokens_per_batch", "5000", "--add_consts", "1",
    "--device", "cuda:0", "--use_exprs", "200000", "--use_dimension_mask", "0",
    "--expr_train_data_path", "./data/exprs_train.json",
    "--expr_valid_data_path", "./data/exprs_valid.json",
    "--sub_expr_train_path", "./data/exprs_seperated_train.json",
    "--sub_expr_valid_path", "./data/exprs_seperated_valid.json",
    "--decode_physical_units", "single-seq",
    "--use_hints", "units,complexity,unarys,consts",
    "--random_variables_sequence", "0", "--max_trials", "10",
    "--generate_datapoints_distribution", "positive,multi", "--rescale", "0",
    "--cpu", "True",
]


def stable_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


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


def rng_state_hash(rng):
    if rng is None:
        return (-1, "None")
    state = rng.get_state()
    pos = state[2]
    h = hashlib.sha256(str(state).encode()).hexdigest()[:12]
    return (pos, h)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["torch", "paddle"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=5)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    seed_backend(args.backend, args.seed)

    with backend_context(repo_root, args.backend):
        from parsers import get_parser
        from symbolicregression.envs import build_env
        from symbolicregression.model import build_modules

        params = get_parser().parse_args(TRAIN_SMALL_ARGS)
        params = normalize_params(params)
        env = build_env(params)
        build_modules(env, params)

        # Create dataset to get init_rng called
        from symbolicregression.envs.environment import EnvDataset
        dataset = EnvDataset(
            env, params.tasks[0], train=True, skip=True,
            params=params, path=None,
        )
        # init_rng is now called in __init__

        pos_before, hash_before = rng_state_hash(env.rng)
        print(f"[{args.backend}] env.rng before generate_sample: pos={pos_before}, hash={hash_before}")

        samples = []
        for i in range(args.num_samples):
            pos_before, _ = rng_state_hash(env.rng)
            sample = dataset.generate_sample()
            pos_after, _ = rng_state_hash(env.rng)

            # Hash key fields
            tree_encoded = sample.get("tree_encoded", [])
            x_key = stable_hash(sample.get("x_to_fit", []))
            y_key = stable_hash(sample.get("y_to_fit", []))
            tree_key = stable_hash(tree_encoded)
            gen_error = 0  # no errors accessible from sample

            print(f"  sample[{i}]: pos_before={pos_before}, pos_after={pos_after}, "
                  f"tree_len={len(tree_encoded)}, tree_hash={tree_key}, "
                  f"x_hash={x_key}, first_4_tokens={tree_encoded[:4]}")

        pos_after, hash_after = rng_state_hash(env.rng)
        print(f"[{args.backend}] env.rng after {args.num_samples} samples: pos={pos_after}, hash={hash_after}")


if __name__ == "__main__":
    main()
