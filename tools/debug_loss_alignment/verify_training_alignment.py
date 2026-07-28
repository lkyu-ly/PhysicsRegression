#!/usr/bin/env python3
"""Verify training alignment by running frameworks in separate subprocesses."""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


INNER_SCRIPT = r"""
import json, hashlib, os, random, sys
from pathlib import Path
import numpy as np

BACKEND = __BACKEND__
repo_root = Path(__REPO_ROOT__)
project = repo_root / ("PhysicsRegression" if BACKEND == "torch" else "PhysicsRegressionPaddle")
os.chdir(str(project))
sys.path.insert(0, str(project))

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

use_converted = __USE_CONVERTED_WEIGHTS__

random.seed(0)
np.random.seed(0)
if BACKEND == "torch":
    import torch
    torch.manual_seed(0)
else:
    import paddle
    paddle.seed(0)

from parsers import get_parser
from symbolicregression.envs import build_env
from symbolicregression.model import build_modules

params = get_parser().parse_args(TRAIN_SMALL_ARGS)
params.cpu = True; params.num_workers = 0
params.is_slurm_job = False; params.debug_slurm = True
params.n_nodes = 1; params.node_id = 0; params.local_rank = 0; params.global_rank = 0
params.world_size = 1; params.n_gpu_per_node = 1
params.multi_node = False; params.multi_gpu = False; params.is_master = True

env = build_env(params)
modules = build_modules(env, params)
embedder = modules["embedder"]
encoder = modules["encoder"]
decoder = modules["decoder"]

if use_converted:
    model_path = repo_root / "models" / ("model.pt" if BACKEND == "torch" else "model.pdparams")
    if BACKEND == "torch":
        data = torch.load(str(model_path), map_location="cpu", weights_only=False)
        embedder.load_state_dict(data["embedder"])
        encoder.load_state_dict(data["encoder"])
        decoder.load_state_dict(data["decoder"])
    else:
        data = paddle.load(str(model_path))
        embedder.set_state_dict(data["embedder"])
        encoder.set_state_dict(data["encoder"])
        decoder.set_state_dict(data["decoder"])

iterator = iter(env.create_train_iterator(params.tasks[0], None, params))

def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:16]

results = []
for step in range(__STEPS__):
    batch, errors = next(iterator)
    tree_encoded = batch["tree_encoded"]
    x_to_fit = batch["x_to_fit"]
    y_to_fit = batch["y_to_fit"]
    infos = batch["infos"]

    tree_hash = stable_hash(tree_encoded)
    x_hash = stable_hash(x_to_fit)

    x1 = []
    for seq_id in range(len(x_to_fit)):
        x1.append([])
        for seq_l in range(len(x_to_fit[seq_id])):
            x1[seq_id].append([x_to_fit[seq_id][seq_l], y_to_fit[seq_id][seq_l]])

    hints = []
    for used_hints in params.use_hints.split(","):
        hints.append(batch[used_hints])

    tree_ids = env.word_to_idx(tree_encoded, float_input=False)
    x2, len2, units = env.batch_equations(tree_ids)

    if BACKEND == "torch":
        import torch
        with torch.no_grad():
            x1_emb, len1 = embedder(x1, hints)
            encoded = encoder("fwd", x=x1_emb, lengths=len1, causal=False)
            decoded = decoder("fwd", x=x2, lengths=len2, causal=True,
                             src_enc=encoded.transpose(0, 1), src_len=len1, units=units)
            alen = torch.arange(torch.max(len2), dtype=torch.long)
            pred_mask = alen[:, None] < len2[None] - 1
            y = x2[1:].masked_select(pred_mask[:-1])
            _, loss = decoder("predict", tensor=decoded, pred_mask=pred_mask, y=y, get_scores=False)
        loss_val = loss.item()
    else:
        import paddle
        with paddle.no_grad():
            x1_emb, len1 = embedder(x1, hints)
            encoded = encoder("fwd", x=x1_emb, lengths=len1, causal=False)
            decoded = decoder("fwd", x=x2, lengths=len2, causal=True,
                             src_enc=encoded.transpose(0, 1), src_len=len1, units=units)
            alen = paddle.arange(paddle.max(len2), dtype="int64")
            pred_mask = alen[:, None] < len2[None] - 1
            y = x2[1:].masked_select(pred_mask[:-1])
            _, loss = decoder("predict", tensor=decoded, pred_mask=pred_mask, y=y, get_scores=False)
        loss_val = loss.item()

    results.append(dict(step=step, batch_size=len(tree_encoded),
                        tree_hash=tree_hash, x_hash=x_hash, loss=loss_val))

with open("__OUTPUT_FILE__", "w") as f:
    json.dump(results, f)
"""


def run_framework(backend: str, steps: int, repo_root: Path, use_converted: bool) -> list:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        output_file = tmp.name

    script = (INNER_SCRIPT
              .replace("__BACKEND__", repr(backend))
              .replace("__STEPS__", str(steps))
              .replace("__REPO_ROOT__", repr(str(repo_root)))
              .replace("__USE_CONVERTED_WEIGHTS__", "True" if use_converted else "False")
              .replace("__OUTPUT_FILE__", output_file))

    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=600,
    )

    if proc.returncode != 0:
        print(f"[{backend}] FAILED (rc={proc.returncode})", file=sys.stderr)
        print(f"[{backend}] stdout (last 2000):", proc.stdout[-2000:], file=sys.stderr)
        print(f"[{backend}] stderr (last 2000):", proc.stderr[-2000:], file=sys.stderr)
        return []

    for line in proc.stderr.splitlines():
        if any(kw in line for kw in ("Traceback", "Error:", "Exception")):
            print(f"[{backend}] {line}", file=sys.stderr)

    with open(output_file) as f:
        results = json.load(f)
    Path(output_file).unlink(missing_ok=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--use-converted-weights", action="store_true",
                        help="Load model.pt / model.pdparams before forward pass")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]

    mode = "with converted weights" if args.use_converted_weights else "random init"

    print(f"=== Torch Forward ({mode}) ===")
    torch_results = run_framework("torch", args.steps, repo_root, args.use_converted_weights)
    for r in torch_results:
        print(f"[torch] step={r['step']}: batch_size={r['batch_size']}, "
              f"tree_hash={r['tree_hash']}, loss={r['loss']:.6f}")

    print(f"\n=== Paddle Forward ({mode}) ===")
    paddle_results = run_framework("paddle", args.steps, repo_root, args.use_converted_weights)
    for r in paddle_results:
        print(f"[paddle] step={r['step']}: batch_size={r['batch_size']}, "
              f"tree_hash={r['tree_hash']}, loss={r['loss']:.6f}")

    if not torch_results or not paddle_results:
        print("\nOne or both frameworks failed. See stderr above.")
        return

    print("\n=== Comparison ===")
    if len(torch_results) != len(paddle_results):
        print(f"Step count mismatch: torch={len(torch_results)}, paddle={len(paddle_results)}")
        return

    all_match = True
    for tr, pr in zip(torch_results, paddle_results):
        tree_ok = tr["tree_hash"] == pr["tree_hash"]
        x_ok = tr["x_hash"] == pr["x_hash"]
        loss_diff = abs(tr["loss"] - pr["loss"])
        print(f"step={tr['step']}: tree_hash={'OK' if tree_ok else 'DIFF'}, "
              f"x_hash={'OK' if x_ok else 'DIFF'}, "
              f"torch_loss={tr['loss']:.6f}, paddle_loss={pr['loss']:.6f}, "
              f"diff={loss_diff:.2e}")
        if not tree_ok or not x_ok or loss_diff > 1e-4:
            all_match = False

    if all_match:
        print("\nALL MATCHED")
    else:
        print("\nMISMATCH DETECTED")


if __name__ == "__main__":
    main()
