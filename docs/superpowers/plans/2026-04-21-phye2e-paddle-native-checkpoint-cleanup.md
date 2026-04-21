# PhyE2E Paddle 原生模型 I/O 收敛 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 删除 Paddle 侧旧 `.pkl/.pt` 兼容路径和旧转换脚本，把 `PhysicsRegressionPaddle` 的模型加载、保存、评估入口统一收敛到 Paddle 原生 `model.pdparams` 与 `checkpoint.pth` 两条路径，并用真实模型补齐回归测试。

**Architecture:** 新增一个极小的 `checkpoint_io.py` 统一做 Paddle 原生 bundle/checkpoint 校验与状态恢复；`PhysicsRegression.py`、`build_modules()`、`Trainer.reload_checkpoint()` 都改为依赖这套原生 helper。评估脚本改用 `--reload_model ../models/model.pdparams`，删除把 `model.pt` 当 checkpoint 的旧 hack；文档和 notebook 全部指向根目录官方脚本与 `models/` 下的 Paddle 文件。

**Tech Stack:** Python 3.10, PaddlePaddle 3.3, PyTorch 2.x, pytest, NumPy, 仓库根目录 `convert_torch_to_paddle.py` / `compare_torch_paddle_forward.py`

---

## File Structure

- Create: `PhysicsRegressionPaddle/symbolicregression/checkpoint_io.py`
  单一职责：加载 Paddle 原生模型 bundle / checkpoint，规范化 `params`，恢复 Layer / Optimizer / GradScaler 状态
- Create: `PhysicsRegressionPaddle/unitTest/test_checkpoint_io.py`
  单一职责：覆盖 helper 的结构校验、`.pt/.pkl` 拒绝策略、状态恢复逻辑
- Create: `PhysicsRegressionPaddle/unitTest/test_phyreg_native_policy.py`
  单一职责：验证 `PhyReg` 只接受 Paddle 原生模型，拒绝真实 `models/model.pt` 和临时生成的旧 `.pkl`
- Create: `PhysicsRegressionPaddle/unitTest/test_training_entrypoints_native_only.py`
  单一职责：验证 `Trainer.reload_checkpoint()` 不再接受推理模型，shell/eval 入口不再保留 `model.pt` 特判
- Create: `PhysicsRegressionPaddle/unitTest/test_root_model_scripts.py`
  单一职责：用根目录真实模型验证 `convert_torch_to_paddle.py` 与 `compare_torch_paddle_forward.py`
- Modify: `PhysicsRegressionPaddle/PhysicsRegression.py`
  删除多格式 `_load_model()`，改为调用 Paddle 原生 helper
- Modify: `PhysicsRegressionPaddle/symbolicregression/model/__init__.py`
  让 `reload_model` 只接受 Paddle 原生推理模型，并统一使用 `set_state_dict()`
- Modify: `PhysicsRegressionPaddle/symbolicregression/trainer.py`
  让 `reload_checkpoint` 只接受训练 checkpoint，删除模型 bundle fallback
- Modify: `PhysicsRegressionPaddle/bash/eval_bash.py`
  删除 `model.pt` 特判，让 Trainer 正常初始化
- Modify: `PhysicsRegressionPaddle/bash/eval_synthetic.sh`
  改成 `--reload_model ../models/model.pdparams`
- Modify: `PhysicsRegressionPaddle/bash/eval_feynman.sh`
  改成 `--reload_model ../models/model.pdparams`
- Delete: `PhysicsRegressionPaddle/tools/convert_model.py`
  旧 `.pkl` 转换脚本彻底移除
- Delete: `PhysicsRegressionPaddle/tools/__init__.py`
  若目录仅为旧转换脚本服务，则一并删除
- Modify: `PhysicsRegressionPaddle/README.md`
- Modify: `PhysicsRegressionPaddle/CLAUDE.md`
- Modify: `PhysicsRegressionPaddle/PADDLE_MIGRATION.md`
- Modify: `PhysicsRegressionPaddle/symbolicregression/CLAUDE.md`
- Modify: `PhysicsRegressionPaddle/Oracle/CLAUDE.md`
- Modify: `PhysicsRegressionPaddle/physical/CLAUDE.md`
  统一 Paddle 用户文档为根目录脚本 + `models/model.pdparams`
- Modify: `PhysicsRegressionPaddle/example.ipynb`
- Modify: `PhysicsRegressionPaddle/physical/case1_SSN.ipynb`
- Modify: `PhysicsRegressionPaddle/physical/case2_Plasma.ipynb`
- Modify: `PhysicsRegressionPaddle/physical/case3_DifferentialRotation.ipynb`
- Modify: `PhysicsRegressionPaddle/physical/case4_ContributionFunction.ipynb`
- Modify: `PhysicsRegressionPaddle/physical/case5_LunarTide.ipynb`
  统一 notebook 模型路径

---

### Task 1: 新增 Paddle 原生 checkpoint helper

**Files:**
- Create: `PhysicsRegressionPaddle/symbolicregression/checkpoint_io.py`
- Create: `PhysicsRegressionPaddle/unitTest/test_checkpoint_io.py`
- Test: `PhysicsRegressionPaddle/unitTest/test_checkpoint_io.py`

- [ ] **Step 1: 写 helper 的失败测试**

```python
import pickle
import sys
from argparse import Namespace
from pathlib import Path

import paddle
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PADDLE_ROOT = PROJECT_ROOT / "PhysicsRegressionPaddle"
if str(PADDLE_ROOT) not in sys.path:
    sys.path.insert(0, str(PADDLE_ROOT))

from symbolicregression.checkpoint_io import (
    load_paddle_model_bundle,
    normalize_params,
    set_grad_scaler_state,
    set_layer_state,
    set_optimizer_state,
)

MODEL_PT = PROJECT_ROOT / "models/model.pt"
MODEL_PDPARAMS = PROJECT_ROOT / "models/model.pdparams"


def make_legacy_pickle(path: Path):
    payload = torch.load(MODEL_PT, map_location="cpu", weights_only=False)
    legacy = {"params": vars(payload["params"]).copy()}
    for name in ("embedder", "encoder", "decoder"):
        legacy[name] = {
            key: value.detach().cpu().numpy() for key, value in payload[name].items()
        }
    with open(path, "wb") as handle:
        pickle.dump(legacy, handle, protocol=4)


def test_normalize_params_accepts_dict_and_namespace():
    assert isinstance(normalize_params({"a": 1}), Namespace)
    assert isinstance(normalize_params(Namespace(a=1)), Namespace)


def test_load_paddle_model_bundle_accepts_native_model():
    data = load_paddle_model_bundle(MODEL_PDPARAMS)
    assert set(data) >= {"embedder", "encoder", "decoder", "params"}
    assert isinstance(data["params"], Namespace)


def test_load_paddle_model_bundle_rejects_torch_model():
    with pytest.raises(ValueError, match="convert_torch_to_paddle.py"):
        load_paddle_model_bundle(MODEL_PT)


def test_load_paddle_model_bundle_rejects_legacy_pickle(tmp_path):
    legacy_path = tmp_path / "legacy.pkl"
    make_legacy_pickle(legacy_path)
    with pytest.raises(ValueError, match="仅支持 Paddle 原生"):
        load_paddle_model_bundle(legacy_path)


def test_state_helpers_restore_layer_optimizer_and_scaler():
    source = paddle.nn.Linear(4, 3)
    target = paddle.nn.Linear(4, 3)
    set_layer_state(target, source.state_dict(), "linear")
    for key, value in source.state_dict().items():
        assert key in target.state_dict()
        assert (target.state_dict()[key] == value).all()

    optimizer = paddle.optimizer.Adam(
        learning_rate=0.001, parameters=target.parameters()
    )
    optimizer_state = optimizer.state_dict()
    set_optimizer_state(optimizer, optimizer_state)

    scaler = paddle.amp.GradScaler(enable=False)
    scaler_state = scaler.state_dict()
    set_grad_scaler_state(scaler, scaler_state)
```

- [ ] **Step 2: 运行测试，确认当前失败**

Run:

```bash
pytest "PhysicsRegressionPaddle/unitTest/test_checkpoint_io.py" -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'symbolicregression.checkpoint_io'`

- [ ] **Step 3: 实现最小 helper**

```python
from argparse import Namespace
from pathlib import Path

import paddle

MODEL_BUNDLE_KEYS = ("embedder", "encoder", "decoder", "params")


def normalize_params(raw_params):
    if isinstance(raw_params, Namespace):
        return raw_params
    if isinstance(raw_params, dict):
        return Namespace(**raw_params)
    raise TypeError(f"Unsupported params type: {type(raw_params)!r}")


def load_paddle_payload(path):
    try:
        data = paddle.load(path=str(path))
    except Exception as exc:
        raise ValueError(
            f"仅支持 Paddle 原生模型文件: {path}。"
            "Torch 权重请先运行根目录 convert_torch_to_paddle.py。"
        ) from exc
    if not isinstance(data, dict):
        raise ValueError(f"模型文件结构非法: {path}")
    return dict(data)


def require_keys(data, required_keys, path):
    missing = [key for key in required_keys if key not in data]
    if missing:
        raise ValueError(f"文件缺少必要字段 {missing}: {path}")


def load_paddle_model_bundle(path):
    path = Path(path)
    if path.suffix in {".pt", ".pkl"}:
        raise ValueError(
            f"仅支持 Paddle 原生模型文件: {path}。"
            "Torch 权重请先运行根目录 convert_torch_to_paddle.py。"
        )
    data = load_paddle_payload(path)
    require_keys(data, MODEL_BUNDLE_KEYS, path)
    data["params"] = normalize_params(data["params"])
    return data


def set_layer_state(layer, state_dict, layer_name):
    try:
        layer.set_state_dict(state_dict)
    except Exception as exc:
        raise ValueError(f"{layer_name} 状态恢复失败") from exc


def set_modules_state(modules, payload, module_names=("embedder", "encoder", "decoder")):
    for module_name in module_names:
        set_layer_state(modules[module_name], payload[module_name], module_name)


def set_optimizer_state(optimizer, state_dict):
    optimizer.set_state_dict(state_dict)


def set_grad_scaler_state(scaler, state_dict):
    scaler.load_state_dict(state_dict)
```

- [ ] **Step 4: 重新运行 helper 测试**

Run:

```bash
pytest "PhysicsRegressionPaddle/unitTest/test_checkpoint_io.py" -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add \
  "PhysicsRegressionPaddle/symbolicregression/checkpoint_io.py" \
  "PhysicsRegressionPaddle/unitTest/test_checkpoint_io.py"
git commit -m "refactor: add paddle-native checkpoint io helpers"
```

---

### Task 2: 收敛 `PhyReg` 与 `reload_model` 到 Paddle 原生模型

**Files:**
- Modify: `PhysicsRegressionPaddle/PhysicsRegression.py`
- Modify: `PhysicsRegressionPaddle/symbolicregression/model/__init__.py`
- Create: `PhysicsRegressionPaddle/unitTest/test_phyreg_native_policy.py`
- Test: `PhysicsRegressionPaddle/unitTest/test_phyreg_native_policy.py`

- [ ] **Step 1: 写 public loader 的失败测试**

```python
import pickle
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PADDLE_ROOT = PROJECT_ROOT / "PhysicsRegressionPaddle"
if str(PADDLE_ROOT) not in sys.path:
    sys.path.insert(0, str(PADDLE_ROOT))

from PhysicsRegression import PhyReg

MODEL_PT = PROJECT_ROOT / "models/model.pt"
MODEL_PDPARAMS = PROJECT_ROOT / "models/model.pdparams"


def make_legacy_pickle(path: Path):
    payload = torch.load(MODEL_PT, map_location="cpu", weights_only=False)
    legacy = {"params": vars(payload["params"]).copy()}
    for name in ("embedder", "encoder", "decoder"):
        legacy[name] = {
            key: value.detach().cpu().numpy() for key, value in payload[name].items()
        }
    with open(path, "wb") as handle:
        pickle.dump(legacy, handle, protocol=4)


def test_phyreg_loads_native_model():
    model = PhyReg(MODEL_PDPARAMS, device="cpu")
    assert set(model.modules) == {"embedder", "encoder", "decoder"}


def test_phyreg_rejects_torch_pt():
    with pytest.raises(ValueError, match="convert_torch_to_paddle.py"):
        PhyReg(MODEL_PT, device="cpu")


def test_phyreg_rejects_legacy_pickle(tmp_path):
    legacy_path = tmp_path / "legacy.pkl"
    make_legacy_pickle(legacy_path)
    with pytest.raises(ValueError, match="仅支持 Paddle 原生"):
        PhyReg(legacy_path, device="cpu")
```

- [ ] **Step 2: 运行测试，确认旧兼容逻辑现在会失败**

Run:

```bash
pytest "PhysicsRegressionPaddle/unitTest/test_phyreg_native_policy.py" -q
```

Expected: FAIL because:
- 当前错误文案仍引用 `tools/convert_model.py`
- 当前 `PhyReg` 仍接受 `.pkl`

- [ ] **Step 3: 重写 `PhysicsRegression.py` 的加载入口，并让 `build_modules()` 只吃 Paddle bundle**

```python
# PhysicsRegressionPaddle/PhysicsRegression.py
from symbolicregression.checkpoint_io import (
    load_paddle_model_bundle,
    set_modules_state,
)


class PhyReg:
    def __init__(self, path, max_len=None, refinement_strategy=None, device=None):
        model = load_paddle_model_bundle(str(path))
        params = model["params"]
        params.rescale = False
        if max_len is not None:
            assert isinstance(max_len, int) and max_len > 0
            params.max_len = max_len
            params.max_input_points = max_len
        if refinement_strategy is not None:
            assert isinstance(refinement_strategy, str)
            params.refinement_strategy = refinement_strategy
        if device is not None:
            assert isinstance(device, str)
            params.device = device

        env = build_env(params)
        modules = build_modules(env, params)
        oracle = Oracle(env, env.generator, params)

        self.params = params
        self.env = env
        self.modules = modules
        self.oracle = oracle

        set_modules_state(modules, model)
        for module in modules.values():
            module.eval()

        self.mw = ModelWrapper(
            env=env,
            embedder=modules["embedder"],
            encoder=modules["encoder"],
            decoder=modules["decoder"],
            beam_length_penalty=params.beam_length_penalty,
            beam_size=params.beam_size,
            max_generated_output_len=params.max_generated_output_len,
            beam_early_stopping=params.beam_early_stopping,
            beam_temperature=params.beam_temperature,
            beam_type=params.beam_type,
        )

    def save(self, path):
        save_dict = {
            "embedder": self.mw.embedder.state_dict(),
            "encoder": self.mw.encoder.state_dict(),
            "decoder": self.mw.decoder.state_dict(),
            "params": vars(self.params).copy(),
        }
        paddle.save(obj=save_dict, path=path)
```

```python
# PhysicsRegressionPaddle/symbolicregression/model/__init__.py
from .checkpoint_io import load_paddle_model_bundle, set_modules_state


def build_modules(env, params):
    modules = {}
    modules["embedder"] = LinearPointEmbedder(params, env)
    env.get_length_after_batching = modules["embedder"].get_length_after_batching
    modules["encoder"] = TransformerModel(
        params,
        env.float_id2word,
        is_encoder=True,
        with_output=False,
        use_prior_embeddings=True,
        positional_embeddings=params.enc_positional_embeddings,
    )
    modules["decoder"] = TransformerModel(
        params,
        env.equation_id2word,
        is_encoder=False,
        with_output=True,
        use_prior_embeddings=False,
        positional_embeddings=params.dec_positional_embeddings,
    )
    if params.reload_model != "":
        logger.info(f"Reloading modules from {params.reload_model} ...")
        reloaded = load_paddle_model_bundle(params.reload_model)
        set_modules_state(modules, reloaded)
```

- [ ] **Step 4: 重新运行 public loader 测试**

Run:

```bash
pytest "PhysicsRegressionPaddle/unitTest/test_phyreg_native_policy.py" -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add \
  "PhysicsRegressionPaddle/PhysicsRegression.py" \
  "PhysicsRegressionPaddle/symbolicregression/model/__init__.py" \
  "PhysicsRegressionPaddle/unitTest/test_phyreg_native_policy.py"
git commit -m "refactor: make paddle model loading native-only"
```

---

### Task 3: 让 Trainer 和评估入口只接受各自正确的文件类型

**Files:**
- Modify: `PhysicsRegressionPaddle/symbolicregression/trainer.py`
- Modify: `PhysicsRegressionPaddle/bash/eval_bash.py`
- Modify: `PhysicsRegressionPaddle/bash/eval_synthetic.sh`
- Modify: `PhysicsRegressionPaddle/bash/eval_feynman.sh`
- Delete: `PhysicsRegressionPaddle/tools/convert_model.py`
- Delete: `PhysicsRegressionPaddle/tools/__init__.py`
- Create: `PhysicsRegressionPaddle/unitTest/test_training_entrypoints_native_only.py`
- Test: `PhysicsRegressionPaddle/unitTest/test_training_entrypoints_native_only.py`

- [ ] **Step 1: 写训练恢复与入口脚本的失败测试**

```python
import sys
from pathlib import Path
from types import SimpleNamespace

import paddle
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PADDLE_ROOT = PROJECT_ROOT / "PhysicsRegressionPaddle"
if str(PADDLE_ROOT) not in sys.path:
    sys.path.insert(0, str(PADDLE_ROOT))

from symbolicregression.trainer import Trainer

MODEL_PDPARAMS = PROJECT_ROOT / "models/model.pdparams"
EVAL_BASH = PROJECT_ROOT / "PhysicsRegressionPaddle/bash/eval_bash.py"
EVAL_SYNTHETIC = PROJECT_ROOT / "PhysicsRegressionPaddle/bash/eval_synthetic.sh"
EVAL_FEYNMAN = PROJECT_ROOT / "PhysicsRegressionPaddle/bash/eval_feynman.sh"


def make_fake_trainer(reload_checkpoint):
    trainer = Trainer.__new__(Trainer)
    trainer.params = SimpleNamespace(
        reload_checkpoint=str(reload_checkpoint),
        dump_path=str(PROJECT_ROOT / "tmp"),
        amp=-1,
        nvidia_apex=False,
        fp16=False,
    )
    trainer.modules = {
        "embedder": paddle.nn.Linear(2, 2),
        "encoder": paddle.nn.Linear(2, 2),
        "decoder": paddle.nn.Linear(2, 2),
    }
    params = []
    for module in trainer.modules.values():
        params.extend(module.parameters())
    trainer.optimizer = paddle.optimizer.Adam(learning_rate=0.001, parameters=params)
    trainer.scaler = None
    trainer.epoch = 0
    trainer.n_total_iter = 0
    trainer.best_metrics = {}
    trainer.best_stopping_criterion = None
    return trainer


def test_reload_checkpoint_rejects_inference_bundle():
    trainer = make_fake_trainer(MODEL_PDPARAMS)
    with pytest.raises(ValueError, match="reload_model"):
        Trainer.reload_checkpoint(trainer)


def test_eval_entrypoints_no_longer_reference_model_pt():
    eval_bash_source = EVAL_BASH.read_text(encoding="utf-8")
    assert 'if "model.pt" in params.reload_checkpoint' not in eval_bash_source
    assert 'Trainer(modules, env, params, path="model.pt", root="./")' not in eval_bash_source

    for script_path in (EVAL_SYNTHETIC, EVAL_FEYNMAN):
        script_text = script_path.read_text(encoding="utf-8")
        assert "model.pt" not in script_text
        assert "model.pdparams" in script_text
```

- [ ] **Step 2: 运行测试，确认当前失败**

Run:

```bash
pytest "PhysicsRegressionPaddle/unitTest/test_training_entrypoints_native_only.py" -q
```

Expected: FAIL because:
- 当前 `reload_checkpoint()` 仍接受推理模型 bundle
- 当前评估入口仍包含 `model.pt` 特判和默认路径

- [ ] **Step 3: 收敛 Trainer checkpoint 语义，并移除旧转换脚本**

```python
# PhysicsRegressionPaddle/symbolicregression/trainer.py
from symbolicregression.checkpoint_io import (
    load_paddle_payload,
    require_keys,
    set_grad_scaler_state,
    set_modules_state,
    set_optimizer_state,
)


def reload_checkpoint(self, path=None, root=None, requires_grad=True):
    if path is None:
        path = "checkpoint.pth"
    if os.path.isfile(self.params.reload_checkpoint):
        checkpoint_path = self.params.reload_checkpoint
    elif self.params.reload_checkpoint != "":
        checkpoint_path = os.path.join(self.params.reload_checkpoint, path)
        assert os.path.isfile(checkpoint_path)
    else:
        if root is not None:
            checkpoint_path = os.path.join(root, path)
        else:
            checkpoint_path = os.path.join(self.params.dump_path, path)
        if not os.path.isfile(checkpoint_path):
            logger.warning("Checkpoint path does not exist, {}".format(checkpoint_path))
            return

    logger.warning(f"Reloading checkpoint from {checkpoint_path} ...")
    data = load_paddle_payload(checkpoint_path)
    required_keys = (
        "epoch",
        "n_total_iter",
        "best_metrics",
        "best_stopping_criterion",
        "params",
        "embedder",
        "encoder",
        "decoder",
        "optimizer",
    )
    try:
        require_keys(data, required_keys, checkpoint_path)
    except ValueError as exc:
        raise ValueError(
            f"{checkpoint_path} 不是训练 checkpoint。"
            "如果要加载预训练模型，请使用 --reload_model。"
        ) from exc

    set_modules_state(self.modules, data)
    for module in self.modules.values():
        module.stop_gradient = not requires_grad

    if self.params.amp == -1 or not self.params.nvidia_apex:
        set_optimizer_state(self.optimizer, data["optimizer"])
        if "optimizer_num_updates" in data and hasattr(self.optimizer, "num_updates"):
            self.optimizer.num_updates = data["optimizer_num_updates"]
            if hasattr(self.optimizer, "get_lr_for_step"):
                self.optimizer._learning_rate = self.optimizer.get_lr_for_step(
                    self.optimizer.num_updates
                )

    if self.params.fp16 and not self.params.nvidia_apex:
        set_grad_scaler_state(self.scaler, data["scaler"])
    else:
        assert self.scaler is None and "scaler" not in data

    self.epoch = data["epoch"] + 1
    self.n_total_iter = data["n_total_iter"]
    self.best_metrics = data["best_metrics"]
    self.best_stopping_criterion = data["best_stopping_criterion"]
```

```python
# PhysicsRegressionPaddle/bash/eval_bash.py
def init_eval(params):
    init_distributed_mode(params)
    logger = initialize_exp(params, write_dump_path=False)
    if params.is_slurm_job:
        init_signal_handler()
    if not params.cpu:
        assert paddle.cuda.is_available()
    params.eval_only = True
    symbolicregression.utils.CUDA = not params.cpu
    if params.batch_size_eval is None:
        params.batch_size_eval = int(1.5 * params.batch_size)
    env = build_env(params)
    env.rng = np.random.RandomState()
    modules = build_modules(env, params)
    trainer = Trainer(modules, env, params)
    evaluator = Evaluator(trainer)
    return evaluator, logger
```

```bash
# PhysicsRegressionPaddle/bash/eval_synthetic.sh
--reload_model ../models/model.pdparams \
```

```bash
# PhysicsRegressionPaddle/bash/eval_feynman.sh
--reload_model ../models/model.pdparams \
```

```diff
*** Delete File: PhysicsRegressionPaddle/tools/convert_model.py
*** Delete File: PhysicsRegressionPaddle/tools/__init__.py
```

- [ ] **Step 4: 重新运行训练恢复与入口测试**

Run:

```bash
pytest "PhysicsRegressionPaddle/unitTest/test_training_entrypoints_native_only.py" -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add \
  "PhysicsRegressionPaddle/symbolicregression/trainer.py" \
  "PhysicsRegressionPaddle/bash/eval_bash.py" \
  "PhysicsRegressionPaddle/bash/eval_synthetic.sh" \
  "PhysicsRegressionPaddle/bash/eval_feynman.sh" \
  "PhysicsRegressionPaddle/unitTest/test_training_entrypoints_native_only.py"
git rm "PhysicsRegressionPaddle/tools/convert_model.py" "PhysicsRegressionPaddle/tools/__init__.py"
git commit -m "refactor: remove legacy paddle conversion compatibility"
```

---

### Task 4: 清理用户文档与 notebook 的旧模型路径

**Files:**
- Modify: `PhysicsRegressionPaddle/README.md`
- Modify: `PhysicsRegressionPaddle/CLAUDE.md`
- Modify: `PhysicsRegressionPaddle/PADDLE_MIGRATION.md`
- Modify: `PhysicsRegressionPaddle/symbolicregression/CLAUDE.md`
- Modify: `PhysicsRegressionPaddle/Oracle/CLAUDE.md`
- Modify: `PhysicsRegressionPaddle/physical/CLAUDE.md`
- Modify: `PhysicsRegressionPaddle/example.ipynb`
- Modify: `PhysicsRegressionPaddle/physical/case1_SSN.ipynb`
- Modify: `PhysicsRegressionPaddle/physical/case2_Plasma.ipynb`
- Modify: `PhysicsRegressionPaddle/physical/case3_DifferentialRotation.ipynb`
- Modify: `PhysicsRegressionPaddle/physical/case4_ContributionFunction.ipynb`
- Modify: `PhysicsRegressionPaddle/physical/case5_LunarTide.ipynb`

- [ ] **Step 1: 先用 grep 记录当前旧引用**

Run:

```bash
rg -n "model\.pt|convert_model\.py|\.pkl" \
  "PhysicsRegressionPaddle/README.md" \
  "PhysicsRegressionPaddle/CLAUDE.md" \
  "PhysicsRegressionPaddle/PADDLE_MIGRATION.md" \
  "PhysicsRegressionPaddle/symbolicregression/CLAUDE.md" \
  "PhysicsRegressionPaddle/Oracle/CLAUDE.md" \
  "PhysicsRegressionPaddle/physical/CLAUDE.md" \
  "PhysicsRegressionPaddle/example.ipynb" \
  "PhysicsRegressionPaddle/physical"
```

Expected: MATCHES found for `model.pt`

- [ ] **Step 2: 按最终路径规则改文档**

```diff
# PhysicsRegressionPaddle/README.md
- After downoading and replace it with the empty `model.pt` file, you can play with `example.ipynb` as a demo example.
+ Put the Torch checkpoint at `../models/model.pt`, run `python ../convert_torch_to_paddle.py`, then load `../models/model.pdparams` in Paddle examples.

- **`reload_checkpoint`**: The path to reload model or checkpoint (default to `model.pt`).
+ **`reload_model`**: Paddle inference model path, recommended `../models/model.pdparams`.
+ **`reload_checkpoint`**: Training checkpoint path such as `checkpoint.pth`, only for resuming training.
```

```diff
# PhysicsRegressionPaddle/CLAUDE.md / symbolicregression/CLAUDE.md / Oracle/CLAUDE.md
- model = PhyReg("model.pt")
+ model = PhyReg("../models/model.pdparams")
```

```diff
# PhysicsRegressionPaddle/physical/CLAUDE.md
- ls ../model.pdparams
+ ls ../../models/model.pdparams

- model = PhyReg("../model.pt")
+ model = PhyReg("../../models/model.pdparams")
```

```diff
# PhysicsRegressionPaddle/PADDLE_MIGRATION.md
- **示例文件**: `model.pt`（预训练模型）
+ **示例文件**: `../models/model.pdparams`（Paddle 推理模型）

- model = torch.load(path)
+ Paddle 运行时不再直接读取 torch.save() 文件；请先运行根目录 convert_torch_to_paddle.py。
```

- [ ] **Step 3: 改 notebook JSON 中的模型路径**

```diff
# PhysicsRegressionPaddle/example.ipynb
- "    path = \"./model.pt\"\n",
+ "    path = \"../models/model.pdparams\"\n",
```

```diff
# PhysicsRegressionPaddle/physical/case1_SSN.ipynb 等 5 个物理 notebook
- "    path = \"./model.pt\",\n",
+ "    path = \"../../models/model.pdparams\",\n",
```

- [ ] **Step 4: 重新 grep，确认目标文档不再引导用户走旧路径**

Run:

```bash
rg -n "model\.pt|convert_model\.py|\.pkl" \
  "PhysicsRegressionPaddle/README.md" \
  "PhysicsRegressionPaddle/CLAUDE.md" \
  "PhysicsRegressionPaddle/PADDLE_MIGRATION.md" \
  "PhysicsRegressionPaddle/symbolicregression/CLAUDE.md" \
  "PhysicsRegressionPaddle/Oracle/CLAUDE.md" \
  "PhysicsRegressionPaddle/physical/CLAUDE.md" \
  "PhysicsRegressionPaddle/example.ipynb" \
  "PhysicsRegressionPaddle/physical"
```

Expected: no matches in the targeted user-facing files for `model.pt`, `convert_model.py`, or `.pkl`

- [ ] **Step 5: Commit**

```bash
git add \
  "PhysicsRegressionPaddle/README.md" \
  "PhysicsRegressionPaddle/CLAUDE.md" \
  "PhysicsRegressionPaddle/PADDLE_MIGRATION.md" \
  "PhysicsRegressionPaddle/symbolicregression/CLAUDE.md" \
  "PhysicsRegressionPaddle/Oracle/CLAUDE.md" \
  "PhysicsRegressionPaddle/physical/CLAUDE.md" \
  "PhysicsRegressionPaddle/example.ipynb" \
  "PhysicsRegressionPaddle/physical/case1_SSN.ipynb" \
  "PhysicsRegressionPaddle/physical/case2_Plasma.ipynb" \
  "PhysicsRegressionPaddle/physical/case3_DifferentialRotation.ipynb" \
  "PhysicsRegressionPaddle/physical/case4_ContributionFunction.ipynb" \
  "PhysicsRegressionPaddle/physical/case5_LunarTide.ipynb"
git commit -m "docs: switch paddle usage to native model artifacts"
```

---

### Task 5: 用真实模型补齐根目录脚本回归测试并做总验证

**Files:**
- Create: `PhysicsRegressionPaddle/unitTest/test_root_model_scripts.py`
- Verify: `convert_torch_to_paddle.py`
- Verify: `compare_torch_paddle_forward.py`
- Verify: `models/model.pt`
- Verify: `models/model.pdparams`

- [ ] **Step 1: 写真实模型烟雾测试**

```python
import subprocess
import sys
from pathlib import Path

import paddle

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONVERT_SCRIPT = PROJECT_ROOT / "convert_torch_to_paddle.py"
COMPARE_SCRIPT = PROJECT_ROOT / "compare_torch_paddle_forward.py"
MODEL_PT = PROJECT_ROOT / "models/model.pt"
MODEL_PDPARAMS = PROJECT_ROOT / "models/model.pdparams"


def test_convert_script_writes_paddle_native_model():
    output_path = PROJECT_ROOT / "models/model.test.pdparams"
    if output_path.exists():
        output_path.unlink()
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(CONVERT_SCRIPT),
                "--torch-model",
                str(MODEL_PT),
                "--paddle-model",
                str(output_path),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "[DONE]" in result.stdout
        data = paddle.load(str(output_path))
        assert set(data) == {"embedder", "encoder", "decoder", "params"}
    finally:
        if output_path.exists():
            output_path.unlink()


def test_compare_script_reports_forward_metrics():
    result = subprocess.run(
        [
            sys.executable,
            str(COMPARE_SCRIPT),
            "--torch-model",
            str(MODEL_PT),
            "--paddle-model",
            str(MODEL_PDPARAMS),
            "--seed",
            "0",
            "--num-points",
            "8",
            "--input-dim",
            "1",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    for key in (
        "logits_shape:",
        "torch_dtype:",
        "paddle_dtype:",
        "mean_abs_error:",
        "max_abs_error:",
        "mean_rel_error:",
        "max_rel_error:",
    ):
        assert key in result.stdout
```

- [ ] **Step 2: 运行真实模型脚本测试，确认当前行为已被覆盖**

Run:

```bash
pytest "PhysicsRegressionPaddle/unitTest/test_root_model_scripts.py" -q
```

Expected: PASS

- [ ] **Step 3: 运行本次相关测试全集**

Run:

```bash
pytest \
  "PhysicsRegressionPaddle/unitTest/test_checkpoint_io.py" \
  "PhysicsRegressionPaddle/unitTest/test_phyreg_native_policy.py" \
  "PhysicsRegressionPaddle/unitTest/test_training_entrypoints_native_only.py" \
  "PhysicsRegressionPaddle/unitTest/test_root_model_scripts.py" \
  -q
```

Expected: PASS

- [ ] **Step 4: 再跑一次最终命令级验证**

Run:

```bash
python "convert_torch_to_paddle.py" --torch-model "models/model.pt" --paddle-model "models/model.pdparams"
python "compare_torch_paddle_forward.py" --torch-model "models/model.pt" --paddle-model "models/model.pdparams" --seed 0 --num-points 8 --input-dim 1
```

Expected:
- 第一条命令输出 `[OK] ...` 和 `[DONE] ...`
- 第二条命令输出 `logits_shape` 与四项误差指标

- [ ] **Step 5: 检查最终变更集只包含本次工作**

Run:

```bash
git status --short -- \
  "PhysicsRegressionPaddle/PhysicsRegression.py" \
  "PhysicsRegressionPaddle/symbolicregression/checkpoint_io.py" \
  "PhysicsRegressionPaddle/symbolicregression/model/__init__.py" \
  "PhysicsRegressionPaddle/symbolicregression/trainer.py" \
  "PhysicsRegressionPaddle/bash/eval_bash.py" \
  "PhysicsRegressionPaddle/bash/eval_synthetic.sh" \
  "PhysicsRegressionPaddle/bash/eval_feynman.sh" \
  "PhysicsRegressionPaddle/README.md" \
  "PhysicsRegressionPaddle/CLAUDE.md" \
  "PhysicsRegressionPaddle/PADDLE_MIGRATION.md" \
  "PhysicsRegressionPaddle/symbolicregression/CLAUDE.md" \
  "PhysicsRegressionPaddle/Oracle/CLAUDE.md" \
  "PhysicsRegressionPaddle/physical/CLAUDE.md" \
  "PhysicsRegressionPaddle/example.ipynb" \
  "PhysicsRegressionPaddle/physical/case1_SSN.ipynb" \
  "PhysicsRegressionPaddle/physical/case2_Plasma.ipynb" \
  "PhysicsRegressionPaddle/physical/case3_DifferentialRotation.ipynb" \
  "PhysicsRegressionPaddle/physical/case4_ContributionFunction.ipynb" \
  "PhysicsRegressionPaddle/physical/case5_LunarTide.ipynb" \
  "PhysicsRegressionPaddle/unitTest/test_checkpoint_io.py" \
  "PhysicsRegressionPaddle/unitTest/test_phyreg_native_policy.py" \
  "PhysicsRegressionPaddle/unitTest/test_training_entrypoints_native_only.py" \
  "PhysicsRegressionPaddle/unitTest/test_root_model_scripts.py" \
  "convert_torch_to_paddle.py" \
  "compare_torch_paddle_forward.py"
```

Expected: only the intended files appear

- [ ] **Step 6: Commit**

```bash
git add \
  "PhysicsRegressionPaddle/PhysicsRegression.py" \
  "PhysicsRegressionPaddle/symbolicregression/checkpoint_io.py" \
  "PhysicsRegressionPaddle/symbolicregression/model/__init__.py" \
  "PhysicsRegressionPaddle/symbolicregression/trainer.py" \
  "PhysicsRegressionPaddle/bash/eval_bash.py" \
  "PhysicsRegressionPaddle/bash/eval_synthetic.sh" \
  "PhysicsRegressionPaddle/bash/eval_feynman.sh" \
  "PhysicsRegressionPaddle/README.md" \
  "PhysicsRegressionPaddle/CLAUDE.md" \
  "PhysicsRegressionPaddle/PADDLE_MIGRATION.md" \
  "PhysicsRegressionPaddle/symbolicregression/CLAUDE.md" \
  "PhysicsRegressionPaddle/Oracle/CLAUDE.md" \
  "PhysicsRegressionPaddle/physical/CLAUDE.md" \
  "PhysicsRegressionPaddle/example.ipynb" \
  "PhysicsRegressionPaddle/physical/case1_SSN.ipynb" \
  "PhysicsRegressionPaddle/physical/case2_Plasma.ipynb" \
  "PhysicsRegressionPaddle/physical/case3_DifferentialRotation.ipynb" \
  "PhysicsRegressionPaddle/physical/case4_ContributionFunction.ipynb" \
  "PhysicsRegressionPaddle/physical/case5_LunarTide.ipynb" \
  "PhysicsRegressionPaddle/unitTest/test_checkpoint_io.py" \
  "PhysicsRegressionPaddle/unitTest/test_phyreg_native_policy.py" \
  "PhysicsRegressionPaddle/unitTest/test_training_entrypoints_native_only.py" \
  "PhysicsRegressionPaddle/unitTest/test_root_model_scripts.py"
git commit -m "refactor: converge paddle model io to native checkpoints"
```

---

## Self-Review

- Spec coverage:
  - 删除旧 `.pkl/.pt` 兼容路径：Task 1-3 覆盖
  - 只保留 Paddle 原生模型与 checkpoint：Task 1-3 覆盖
  - 删除旧转换脚本：Task 3 覆盖
  - 文档和 notebook 统一：Task 4 覆盖
  - 使用 `PhyE2E/models` 真模型测试：Task 1、2、5 覆盖
  - 单测保证修改无误：Task 1、2、3、5 覆盖

- Placeholder scan:
  - 无 `TODO` / `TBD` / “implement later” / “similar to”

- Type consistency:
  - 推理模型统一称为 `model.pdparams`
  - 训练恢复统一称为 `checkpoint.pth`
  - 预训练模型入口统一是 `reload_model`
  - 训练恢复入口统一是 `reload_checkpoint`
