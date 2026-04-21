# PhyE2E Torch 转 Paddle 权重转换与前向精度对比 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增两个根目录脚本，完成 `models/model.pt` 到 `models/model.pdparams` 的权重转换，并在固定随机输入下对比 Torch / Paddle 核心模型 teacher-forcing logits 的平均/最大绝对误差与相对误差。

**Architecture:** 转换脚本直接加载 Torch checkpoint，并在 Paddle 侧构建同构模块，按同名参数校验后保存为 Paddle 原生 checkpoint。对比脚本采用“父进程 + 自调用子进程 worker”结构分别执行 Torch / Paddle 前向，规避两侧工程相同包名造成的导入冲突。目标前向只覆盖 `embedder + encoder + decoder`，不进入公式搜索或物理应用流程。

**Tech Stack:** Python 3.10, PyTorch 2.x, PaddlePaddle 3.x, NumPy, 项目现有 `PhysicsRegression/` 与 `PhysicsRegressionPaddle/` 核心模块

---

## File Structure

- Create: `convert_torch_to_paddle.py`
  单一职责：加载 Torch checkpoint、构建 Paddle 模块、校验参数对齐、保存 `models/model.pdparams`
- Create: `compare_torch_paddle_forward.py`
  单一职责：在父进程中调度 Torch/Paddle worker 子进程、收集 logits、计算并打印误差
- Read: `PhysicsRegression/symbolicregression/model/__init__.py`
  用于确认 Torch 侧模块构建入口
- Read: `PhysicsRegressionPaddle/symbolicregression/model/__init__.py`
  用于确认 Paddle 侧模块构建入口
- Read: `PhysicsRegression/symbolicregression/envs/environment.py`
  用于复用 `word_to_idx()`、`batch_equations()` 等目标序列辅助逻辑
- Read: `PhysicsRegressionPaddle/symbolicregression/envs/environment.py`
  用于复用 Paddle 侧同构逻辑
- Read: `PhysicsRegression/symbolicregression/envs/node.py`
  用于构造最小合法目标树 `x_0`
- Read: `PhysicsRegressionPaddle/symbolicregression/envs/node.py`
  用于 Paddle worker 构造相同目标树

不新增共享 helper 模块。两个脚本各自保留极小的上下文管理辅助函数，避免为了消除少量重复而引入第三个文件，符合 YAGNI。

---

### Task 1: 实现 Torch 到 Paddle 的权重转换脚本

**Files:**

- Create: `convert_torch_to_paddle.py`
- Read: `PhysicsRegressionPaddle/symbolicregression/model/__init__.py`
- Read: `PhysicsRegressionPaddle/symbolicregression/envs/__init__.py`
- Output: `models/model.pdparams`

- [ ] **Step 1: 先运行不存在的脚本，确认冒烟命令当前失败**

Run:

```bash
python "convert_torch_to_paddle.py" --torch-model "models/model.pt" --paddle-model "models/model.pdparams"
```

Expected: FAIL with `can't open file` or `No such file or directory`

- [ ] **Step 2: 写最小可用的转换脚本**

```python
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
    parser = argparse.ArgumentParser()
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
            f"{module_name} key mismatch:\\n"
            f"torch={torch_keys[:5]}...\\n"
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
```

- [ ] **Step 3: 运行脚本，验证可以生成 Paddle checkpoint**

Run:

```bash
python "convert_torch_to_paddle.py" --torch-model "models/model.pt" --paddle-model "models/model.pdparams"
```

Expected: PASS and terminal contains lines similar to:

```text
[OK] embedder: 5 tensors verified and loaded
[OK] encoder: 34 tensors verified and loaded
[OK] decoder: 422 tensors verified and loaded
[DONE] Saved converted checkpoint to: /home/lkyu/baidu/PhyE2E/models/model.pdparams
```

- [ ] **Step 4: 用独立命令验证输出文件结构**

Run:

```bash
python - <<'PY'
import paddle
data = paddle.load("models/model.pdparams")
print(sorted(data.keys()))
print(type(data["params"]).__name__)
print(len(data["embedder"]), len(data["encoder"]), len(data["decoder"]))
PY
```

Expected: PASS and output includes:

```text
['decoder', 'embedder', 'encoder', 'params']
dict
5 34 422
```

- [ ] **Step 5: Commit**

```bash
git add convert_torch_to_paddle.py models/model.pdparams
git commit -m "feat: add torch to paddle checkpoint converter"
```

---

### Task 2: 实现双侧前向精度对比脚本

**Files:**

- Create: `compare_torch_paddle_forward.py`
- Read: `PhysicsRegression/symbolicregression/envs/environment.py`
- Read: `PhysicsRegressionPaddle/symbolicregression/envs/environment.py`
- Read: `PhysicsRegression/symbolicregression/envs/node.py`
- Read: `PhysicsRegressionPaddle/symbolicregression/envs/node.py`
- Depends on: `models/model.pt`, `models/model.pdparams`

- [ ] **Step 1: 先运行不存在的脚本，确认对比命令当前失败**

Run:

```bash
python "compare_torch_paddle_forward.py" --torch-model "models/model.pt" --paddle-model "models/model.pdparams" --seed 0 --num-points 8 --input-dim 1
```

Expected: FAIL with `can't open file` or `No such file or directory`

- [ ] **Step 2: 写最小可用的对比脚本，采用父进程 + worker 子进程结构**

```python
#!/usr/bin/env python3
import argparse
import contextlib
import json
import os
import subprocess
import sys
import tempfile
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
    parser = argparse.ArgumentParser()
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
            units = [np.array([0, 0, 0, 0, 0]) for _ in range(input_dim)] + [np.array([0, 0, 0, 0, 0])]
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


def run_torch_worker(repo_root: Path, torch_model: Path, seed: int, num_points: int, input_dim: int):
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
                "predict", tensor=decoded, pred_mask=pred_mask, y=y_tgt, get_scores=False
            )
            return scores.detach().cpu().numpy()


def run_paddle_worker(repo_root: Path, paddle_model: Path, seed: int, num_points: int, input_dim: int):
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
                "predict", tensor=decoded, pred_mask=pred_mask, y=y_tgt, get_scores=False
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
            "--torch-model", args.torch_model,
            "--paddle-model", args.paddle_model,
            "--seed", str(args.seed),
            "--num-points", str(args.num_points),
            "--input-dim", str(args.input_dim),
        ]
        subprocess.run(base_cmd + ["--worker", "torch", "--artifact", str(torch_out)], check=True)
        subprocess.run(base_cmd + ["--worker", "paddle", "--artifact", str(paddle_out)], check=True)

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
```

- [ ] **Step 3: 运行脚本，确认终端打印四项误差指标**

Run:

```bash
python "compare_torch_paddle_forward.py" --torch-model "models/model.pt" --paddle-model "models/model.pdparams" --seed 0 --num-points 8 --input-dim 1
```

Expected: PASS and terminal contains lines similar to:

```text
logits_shape: (7, 10570)
torch_dtype: float32
paddle_dtype: float32
mean_abs_error: 7.4210640000e-07
max_abs_error: 8.5830690000e-06
mean_rel_error: 2.1830747000e-06
max_rel_error: 9.1673850000e-03
```

- [ ] **Step 4: 用脚本化断言验证报告字段稳定输出**

Run:

```bash
python - <<'PY'
import subprocess
cmd = [
    "python",
    "compare_torch_paddle_forward.py",
    "--torch-model", "models/model.pt",
    "--paddle-model", "models/model.pdparams",
    "--seed", "0",
    "--num-points", "8",
    "--input-dim", "1",
]
result = subprocess.run(cmd, check=True, capture_output=True, text=True)
text = result.stdout
for key in [
    "logits_shape:",
    "torch_dtype:",
    "paddle_dtype:",
    "mean_abs_error:",
    "max_abs_error:",
    "mean_rel_error:",
    "max_rel_error:",
]:
    assert key in text, key
print("report keys verified")
PY
```

Expected: PASS and output is:

```text
report keys verified
```

- [ ] **Step 5: Commit**

```bash
git add compare_torch_paddle_forward.py
git commit -m "feat: add torch and paddle forward precision comparison"
```

---

### Task 3: 端到端复核最终交付物

**Files:**

- Verify: `convert_torch_to_paddle.py`
- Verify: `compare_torch_paddle_forward.py`
- Verify: `models/model.pdparams`

- [ ] **Step 1: 从根目录重新执行完整闭环**

Run:

```bash
python "convert_torch_to_paddle.py" --torch-model "models/model.pt" --paddle-model "models/model.pdparams"
python "compare_torch_paddle_forward.py" --torch-model "models/model.pt" --paddle-model "models/model.pdparams" --seed 0 --num-points 8 --input-dim 1
```

Expected: 两条命令都 PASS，第二条命令输出四项误差指标

- [ ] **Step 2: 确认只产生预期文件**

Run:

```bash
find "models" -maxdepth 1 -type f | sort
```

Expected: output includes:

```text
models/model.pdparams
models/model.pt
```

- [ ] **Step 3: 检查脚本帮助信息可读且默认路径正确**

Run:

```bash
python "convert_torch_to_paddle.py" --help
python "compare_torch_paddle_forward.py" --help
```

Expected: PASS and help text includes:

```text
--torch-model
--paddle-model
```

- [ ] **Step 4: 检查没有意外修改现有库代码**

Run:

```bash
git status --short -- convert_torch_to_paddle.py compare_torch_paddle_forward.py models/model.pdparams
```

Expected: 仅显示这三个路径；不依赖工作区是否已经存在其他无关改动

- [ ] **Step 5: Commit**

```bash
git add convert_torch_to_paddle.py compare_torch_paddle_forward.py models/model.pdparams
git commit -m "feat: add local torch to paddle precision validation scripts"
```

---

## Self-Review

### Spec coverage

- 权重转换脚本：Task 1 覆盖
- 双侧前向推理脚本：Task 2 覆盖
- 仅测试核心模块前向精度：Task 2 使用最小 core forward，未接入 Oracle/MCTS/GP
- 两个代码放根目录：Task 1 / Task 2 文件路径已固定
- 转换权重保存到 `models/`：Task 1 / Task 3 覆盖
- 实现最简优先：File Structure 明确只新增两个脚本，不新增共享模块

### Placeholder scan

- 无占位项、无延后实现描述
- 所有命令、路径、文件名、验证方式均已具体化

### Type consistency

- 转换脚本输出固定为 `models/model.pdparams`
- 对比脚本输入固定消费 `models/model.pt` 与 `models/model.pdparams`
- 对比输出固定为 logits 张量误差，不混入离散生成结果
