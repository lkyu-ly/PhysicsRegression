#!/usr/bin/env python3
"""Trace env.rng state at key lifecycle points to find cross-framework divergence."""
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


def rng_state_hash(rng):
    """Return (position, state_hash) for a numpy RandomState."""
    if rng is None:
        return (-1, "None")
    state = rng.get_state()
    pos = state[2]
    h = hashlib.sha256(str(state).encode()).hexdigest()[:12]
    return (pos, h)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["torch", "paddle"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    out_path = Path(args.out).resolve()
    seed_backend(args.backend, args.seed)

    trace_log = []

    def trace(label, env_rng):
        pos, h = rng_state_hash(env_rng)
        entry = {"label": label, "rng_pos": pos, "rng_hash": h}
        trace_log.append(entry)
        print(f"[TRACE] {label}: rng_pos={pos}, rng_hash={h}")

    # Trace point 0: after seed
    trace("00_after_seed", None)

    with backend_context(repo_root, args.backend):
        from parsers import get_parser
        from symbolicregression.envs import build_env
        from symbolicregression.model import build_modules

        params = get_parser().parse_args(TRAIN_SMALL_ARGS)
        params = normalize_params(params)

        # Print key params
        print(f"[PARAMS] env_base_seed={params.env_base_seed}, use_controller={params.use_controller}")
        print(f"[PARAMS] p_add={params.p_add}, p_mul={params.p_mul}")
        print(f"[PARAMS] max_len={params.max_len}, tokens_per_batch={params.tokens_per_batch}")
        trace("01_params", None)

        env = build_env(params)
        trace("02_after_build_env", env.rng)

        modules = build_modules(env, params)
        trace("03_after_build_modules", env.rng)

        dataset_class = env.create_train_iterator(params.tasks[0], None, params).dataset.__class__

        # Monkey-patch init_rng to trace env.rng creation
        original_init_rng = dataset_class.init_rng
        def traced_init_rng(self_inner):
            trace("04a_init_rng_entry", self_inner.env.rng)
            result = original_init_rng(self_inner)
            trace("04b_init_rng_exit", self_inner.env.rng)
            return result
        dataset_class.init_rng = traced_init_rng

        # Monkey-patch generate_sample to trace each call
        original_generate_sample = dataset_class.generate_sample
        gen_sample_count = [0]
        def traced_generate_sample(self_inner):
            gen_sample_count[0] += 1
            trace(f"05a_gen_sample_entry_{gen_sample_count[0]}", self_inner.env.rng)
            result = original_generate_sample(self_inner)
            trace(f"05b_gen_sample_exit_{gen_sample_count[0]}", self_inner.env.rng)
            return result
        dataset_class.generate_sample = traced_generate_sample

        # Monkey-patch _fill_queue
        original_fill_queue = dataset_class._fill_queue
        fill_count = [0]
        def traced_fill_queue(self_inner, n, key_fn):
            fill_count[0] += 1
            trace(f"06a_fill_queue_entry_{fill_count[0]}", self_inner.env.rng)
            result = original_fill_queue(self_inner, n, key_fn)
            trace(f"06b_fill_queue_exit_{fill_count[0]}", self_inner.env.rng)
            return result
        dataset_class._fill_queue = traced_fill_queue

        iterator = iter(env.create_train_iterator(params.tasks[0], None, params))
        trace("07_after_create_iterator", env.rng)

        # Iterate batches
        for step in range(args.steps):
            trace(f"08a_step_{step}_before_next", env.rng)
            batch, errors = next(iterator)
            trace(f"08b_step_{step}_after_next", env.rng)
            print(f"[STEP {step}] batch_size={len(batch['tree_encoded'])}, gen_sample_calls={gen_sample_count[0]}, fill_queue_calls={fill_count[0]}")

    # Write trace log
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(trace_log, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
