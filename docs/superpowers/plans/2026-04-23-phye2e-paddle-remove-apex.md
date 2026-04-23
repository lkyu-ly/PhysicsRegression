# PhyE2E Paddle 侧 Apex 依赖拔除 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从 `PhysicsRegressionPaddle` 彻底删除对 Torch Apex 的代码依赖，移除 `--nvidia_apex` 和所有 Apex 训练/续训分支，只保留 Paddle 原生 AMP、原生 checkpoint 和现有默认脚本行为。

**Architecture:** 这次改动只碰真实存在 Apex 依赖的 3 个代码点: `parsers.py` 的 CLI 开关、`trainer.py` 的 AMP/优化器/续训分支、`transformer.py` 的 Apex 状态位。测试分成 3 组: 公共 CLI/源码契约、Trainer 原生 AMP/续训回归、现有 native-only 入口测试清理；不改 README、notebook 或 shell 脚本，因为这些文件当前没有对 Apex 做显式承诺。

**Tech Stack:** Python 3.10, PaddlePaddle 3.3, pytest, argparse, `paddle.amp.GradScaler`, 仓库现有 `models/model.pdparams` / `models/model.pt`

已确认的行为边界:
- 删除 `--nvidia_apex`，外部再传这个参数会直接触发 argparse 报错
- `--amp` 保留现有整数接口，但从实现上只表示“是否启用 Paddle 原生 AMP”；所有非负值都不再映射 Apex `O1/O2/O3`
- 默认训练/评估脚本不传 `--nvidia_apex`，因此默认脚本行为不变
- 历史 Apex checkpoint 不再被承诺可恢复；Paddle 侧只支持原生 optimizer/scaler 状态恢复

---

## File Structure

- Create: `PhysicsRegressionPaddle/unitTest/test_amp_public_contract.py`
  单一职责：锁定 CLI 和模型源码的公共契约，确保 Paddle 侧不再暴露 `--nvidia_apex`，也不再在 Transformer 中保留 Apex 状态位
- Create: `PhysicsRegressionPaddle/unitTest/test_trainer_native_amp_only.py`
  单一职责：验证 Trainer 不再引用 Apex，并且原生 AMP checkpoint 可以恢复 optimizer/scaler
- Modify: `PhysicsRegressionPaddle/parsers.py`
  删除 `--nvidia_apex`，把 `--amp` help 文案改成 Paddle 原生 AMP 语义
- Modify: `PhysicsRegressionPaddle/symbolicregression/trainer.py`
  删除 `import apex` / `has_apex` / 所有 Apex 分支，统一到 Paddle 原生 AMP、原生 optimizer 恢复、原生 scaler 恢复
- Modify: `PhysicsRegressionPaddle/symbolicregression/model/transformer.py`
  删除 `self.apex`，去掉 `generate_beam()` 中基于 Apex 的条件 cast
- Modify: `PhysicsRegressionPaddle/unitTest/test_training_entrypoints_native_only.py`
  去掉测试夹具中的 `nvidia_apex` 历史参数，补一个 native-only 约束测试，并在最终回归里继续覆盖现有入口行为

---

### Task 1: 删除公共 Apex CLI 与 Transformer 状态位

**Files:**
- Create: `PhysicsRegressionPaddle/unitTest/test_amp_public_contract.py`
- Modify: `PhysicsRegressionPaddle/parsers.py`
- Modify: `PhysicsRegressionPaddle/symbolicregression/model/transformer.py`
- Test: `PhysicsRegressionPaddle/unitTest/test_amp_public_contract.py`

- [ ] **Step 1: 写公共契约失败测试**

```python
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PADDLE_ROOT = PROJECT_ROOT / "PhysicsRegressionPaddle"
if str(PADDLE_ROOT) not in sys.path:
    sys.path.insert(0, str(PADDLE_ROOT))

from parsers import get_parser

PARSER_FILE = PROJECT_ROOT / "PhysicsRegressionPaddle/parsers.py"
TRANSFORMER_FILE = (
    PROJECT_ROOT / "PhysicsRegressionPaddle/symbolicregression/model/transformer.py"
)


def test_parser_no_longer_registers_nvidia_apex():
    parser = get_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert "--nvidia_apex" not in option_strings
    with pytest.raises(SystemExit):
        parser.parse_args(["--nvidia_apex", "true"])


def test_amp_help_text_describes_paddle_native_amp():
    parser = get_parser()
    amp_action = next(
        action for action in parser._actions if "--amp" in action.option_strings
    )
    assert "Paddle native AMP" in amp_action.help
    assert "Level of optimization" not in amp_action.help


def test_transformer_source_no_longer_mentions_apex():
    source = TRANSFORMER_FILE.read_text(encoding="utf-8")
    assert "self.apex" not in source
    assert "nvidia_apex" not in source


def test_parser_source_no_longer_mentions_nvidia_apex():
    source = PARSER_FILE.read_text(encoding="utf-8")
    assert "--nvidia_apex" not in source
```

- [ ] **Step 2: 运行测试，确认当前失败**

Run:

```bash
pytest "PhysicsRegressionPaddle/unitTest/test_amp_public_contract.py" -q
```

Expected: FAIL，至少包含 `assert '--nvidia_apex' not in option_strings` 或 `assert 'self.apex' not in source`

- [ ] **Step 3: 实现最小公共契约清理**

在 `PhysicsRegressionPaddle/parsers.py` 中，把 `--amp` 和 Apex 相关部分改成下面这样：

```python
    parser.add_argument(
        "--fp16", type=bool_flag, default=False, help="Run model with float16"
    )
    parser.add_argument(
        "--amp",
        type=int,
        default=-1,
        help=(
            "Use Paddle native AMP. -1 disables AMP; any non-negative value "
            "enables native autocast and GradScaler."
        ),
    )
```

并删除下面整段：

```python
    parser.add_argument(
        "--nvidia_apex", type=bool_flag, default=False, help="NVIDIA version of apex"
    )
```

在 `PhysicsRegressionPaddle/symbolicregression/model/transformer.py` 中，把 Apex 状态位和 beam 分支改成下面这样：

```python
        super().__init__()
        self.dtype = paddle.float16 if params.fp16 else paddle.float32
        self.is_encoder = is_encoder
        self.is_decoder = not is_encoder
        self.with_output = with_output
        self.id2word = id2word
```

以及：

```python
            assert tensor.size() == (1, bs * beam_size, self.dim)
            tensor = tensor.data[-1, :, :]
            scores = self.proj(tensor)
            scores = paddle.nn.functional.log_softmax(x=scores.float(), axis=-1)
```

- [ ] **Step 4: 重新运行公共契约测试**

Run:

```bash
pytest "PhysicsRegressionPaddle/unitTest/test_amp_public_contract.py" -q
```

Expected: PASS，输出 `4 passed`

- [ ] **Step 5: 提交这一批改动**

```bash
git add \
  "PhysicsRegressionPaddle/parsers.py" \
  "PhysicsRegressionPaddle/symbolicregression/model/transformer.py" \
  "PhysicsRegressionPaddle/unitTest/test_amp_public_contract.py"
git commit -m "refactor: remove apex public contract from paddle frontend"
```

### Task 2: 把 Trainer 收敛到 Paddle 原生 AMP 和原生 checkpoint

**Files:**
- Create: `PhysicsRegressionPaddle/unitTest/test_trainer_native_amp_only.py`
- Modify: `PhysicsRegressionPaddle/symbolicregression/trainer.py`
- Test: `PhysicsRegressionPaddle/unitTest/test_trainer_native_amp_only.py`

- [ ] **Step 1: 写 Trainer native-only 失败测试**

```python
import sys
from pathlib import Path
from types import SimpleNamespace

import paddle

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PADDLE_ROOT = PROJECT_ROOT / "PhysicsRegressionPaddle"
if str(PADDLE_ROOT) not in sys.path:
    sys.path.insert(0, str(PADDLE_ROOT))

from symbolicregression.trainer import Trainer

TRAINER_FILE = PROJECT_ROOT / "PhysicsRegressionPaddle/symbolicregression/trainer.py"


def make_fake_trainer(tmp_path, amp, fp16):
    trainer = Trainer.__new__(Trainer)
    checkpoint_path = tmp_path / "checkpoint.pth"
    trainer.params = SimpleNamespace(
        reload_checkpoint=str(checkpoint_path),
        dump_path=str(tmp_path),
        is_master=True,
        amp=amp,
        fp16=fp16,
        clip_grad_norm=0,
        accumulate_gradients=1,
    )
    trainer.modules = {
        "embedder": paddle.nn.Linear(2, 2),
        "encoder": paddle.nn.Linear(2, 2),
        "decoder": paddle.nn.Linear(2, 2),
    }
    parameters = []
    for module in trainer.modules.values():
        parameters.extend(module.parameters())
    trainer.parameters = {"model": parameters}
    trainer.optimizer = paddle.optimizer.Adam(
        learning_rate=0.001, parameters=parameters
    )
    trainer.scaler = paddle.amp.GradScaler() if amp >= 0 else None
    trainer.epoch = 2
    trainer.n_total_iter = 7
    trainer.best_metrics = {"valid_loss": 1.0}
    trainer.best_stopping_criterion = 1.0
    return trainer, checkpoint_path


def test_trainer_source_no_longer_mentions_apex():
    source = TRAINER_FILE.read_text(encoding="utf-8")
    for token in ("import apex", "nvidia_apex", "apex.amp", "master_params"):
        assert token not in source


def test_reload_checkpoint_restores_native_optimizer_and_scaler(tmp_path):
    saved, checkpoint_path = make_fake_trainer(tmp_path, amp=1, fp16=True)
    Trainer.save_checkpoint(saved, "checkpoint")

    loaded, _ = make_fake_trainer(tmp_path, amp=1, fp16=True)
    loaded.params.reload_checkpoint = str(checkpoint_path)
    Trainer.reload_checkpoint(loaded, path="checkpoint.pth")

    assert loaded.epoch == saved.epoch + 1
    assert loaded.n_total_iter == saved.n_total_iter
    assert loaded.scaler is not None


def test_reload_checkpoint_without_amp_keeps_scaler_none(tmp_path):
    saved, checkpoint_path = make_fake_trainer(tmp_path, amp=-1, fp16=False)
    Trainer.save_checkpoint(saved, "checkpoint")

    loaded, _ = make_fake_trainer(tmp_path, amp=-1, fp16=False)
    loaded.params.reload_checkpoint = str(checkpoint_path)
    Trainer.reload_checkpoint(loaded, path="checkpoint.pth")

    assert loaded.scaler is None
```

- [ ] **Step 2: 运行测试，确认当前失败**

Run:

```bash
pytest "PhysicsRegressionPaddle/unitTest/test_trainer_native_amp_only.py" -q
```

Expected: FAIL，至少包含 `assert 'import apex' not in source`，并且当前 `reload_checkpoint()` 的 scaler 恢复条件也可能失败

- [ ] **Step 3: 实现 Trainer native-only 清理**

在 `PhysicsRegressionPaddle/symbolicregression/trainer.py` 顶部删除 Apex 导入和探测逻辑，让导入区收敛成下面这样：

```python
from .checkpoint_io import (
    load_paddle_payload,
    require_keys,
    set_grad_scaler_state,
    set_modules_state,
    set_optimizer_state,
)
from .optim import get_optimizer
from .utils import to_cuda

logger = getLogger()
```

在 `Trainer.__init__()` 中删除这句断言：

```python
        assert not params.nvidia_apex or has_apex
```

在 `init_amp()` 中删掉 Apex 分支，保留原有组合约束，但始终初始化 Paddle 原生 scaler：

```python
    def init_amp(self):
        """
        Initialize Paddle native AMP.
        """
        params = self.params
        assert (
            params.amp == 0
            and params.fp16 is False
            or params.amp in [1, 2, 3]
            and params.fp16 is True
        )
        self.scaler = paddle.amp.GradScaler(
            incr_every_n_steps=2000, init_loss_scaling=65536.0
        )
```

在 `optimize()` 中删掉 Apex 反传分支，只保留普通路径和原生 scaler 路径：

```python
        if params.amp == -1:
            optimizer.zero_grad()
            loss.backward()
            if params.clip_grad_norm > 0:
                paddle.nn.utils.clip_grad_norm_(
                    parameters=self.parameters["model"], max_norm=params.clip_grad_norm
                )
            optimizer.step()
        else:
            if params.accumulate_gradients > 1:
                loss = loss / params.accumulate_gradients
            self.scaler.scale(loss).backward()
            if (self.n_iter + 1) % params.accumulate_gradients == 0:
                if params.clip_grad_norm > 0:
                    self.scaler.unscale_(optimizer)
                    paddle.nn.utils.clip_grad_norm_(
                        parameters=self.parameters["model"],
                        max_norm=params.clip_grad_norm,
                    )
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()
```

在 `reload_checkpoint()` 中删掉基于 `nvidia_apex` 的 optimizer/scaler 分叉，统一成下面这样：

```python
        logger.warning("Reloading checkpoint optimizer ...")
        set_optimizer_state(self.optimizer, data["optimizer"])

        if hasattr(self.optimizer, "num_updates") and "optimizer_num_updates" in data:
            self.optimizer.num_updates = data["optimizer_num_updates"]
            logger.warning(f"Restored optimizer num_updates: {self.optimizer.num_updates}")

            if hasattr(self.optimizer, "get_lr_for_step"):
                restored_lr = self.optimizer.get_lr_for_step(self.optimizer.num_updates)
                self.optimizer._learning_rate = restored_lr
                logger.warning(f"Restored learning rate: {restored_lr}")
        else:
            if "optimizer_num_updates" not in data:
                logger.warning(
                    "Old checkpoint format detected. Learning rate scheduling will restart."
                )
                logger.warning(
                    "Consider retraining from a newer checkpoint for optimal performance."
                )
            else:
                logger.warning(
                    "No num_updates found in optimizer or optimizer doesn't support it"
                )

        if self.params.amp >= 0:
            logger.warning("Reloading gradient scaler ...")
            set_grad_scaler_state(self.scaler, data["scaler"])
        else:
            assert self.scaler is None and "scaler" not in data
```

在 `enc_dec_step()` 中删掉 `params.nvidia_apex` 判定：

```python
        if params.amp == -1:
            encoded = encoder("fwd", x=x1, lengths=len1, causal=False)
            decoded = decoder(
                "fwd",
                x=x2,
                lengths=len2,
                causal=True,
                src_enc=encoded.transpose(0, 1),
                src_len=len1,
                units=units,
            )
            _, loss = decoder(
                "predict",
                tensor=decoded,
                pred_mask=pred_mask,
                y=y,
                get_scores=False,
                y_units=y_units,
            )
        else:
            with paddle.cuda.amp.autocast():
                encoded = encoder("fwd", x=x1, lengths=len1, causal=False)
                decoded = decoder(
                    "fwd",
                    x=x2,
                    lengths=len2,
                    causal=True,
                    src_enc=encoded.transpose(0, 1),
                    src_len=len1,
                    units=units,
                )
                _, loss = decoder(
                    "predict",
                    tensor=decoded,
                    pred_mask=pred_mask,
                    y=y,
                    get_scores=False,
                    y_units=y_units,
                )
```

- [ ] **Step 4: 重新运行 Trainer native-only 测试**

Run:

```bash
pytest "PhysicsRegressionPaddle/unitTest/test_trainer_native_amp_only.py" -q
```

Expected: PASS，输出 `3 passed`

- [ ] **Step 5: 提交这一批改动**

```bash
git add \
  "PhysicsRegressionPaddle/symbolicregression/trainer.py" \
  "PhysicsRegressionPaddle/unitTest/test_trainer_native_amp_only.py"
git commit -m "refactor: remove apex branches from paddle trainer"
```

### Task 3: 清理现有 native-only 夹具并做回归验证

**Files:**
- Modify: `PhysicsRegressionPaddle/unitTest/test_training_entrypoints_native_only.py`
- Test: `PhysicsRegressionPaddle/unitTest/test_training_entrypoints_native_only.py`
- Test: `PhysicsRegressionPaddle/unitTest/test_amp_public_contract.py`
- Test: `PhysicsRegressionPaddle/unitTest/test_trainer_native_amp_only.py`
- Test: `PhysicsRegressionPaddle/unitTest/test_checkpoint_io.py`
- Test: `PhysicsRegressionPaddle/unitTest/test_phyreg_native_policy.py`
- Test: `PhysicsRegressionPaddle/unitTest/test_root_model_scripts.py`

- [ ] **Step 1: 给现有入口测试补一个 native-only 失败断言**

把 `PhysicsRegressionPaddle/unitTest/test_training_entrypoints_native_only.py` 改成下面这样：

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


def test_fake_trainer_fixture_is_native_only():
    trainer = make_fake_trainer(MODEL_PDPARAMS)
    assert not hasattr(trainer.params, "nvidia_apex")


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

- [ ] **Step 2: 运行现有入口测试，确认当前失败**

Run:

```bash
pytest "PhysicsRegressionPaddle/unitTest/test_training_entrypoints_native_only.py" -q
```

Expected: FAIL，包含 `assert not hasattr(trainer.params, "nvidia_apex")`

- [ ] **Step 3: 清理旧夹具中的 Apex 历史参数**

确保 `make_fake_trainer()` 最终只保留下面这组 native-only 参数：

```python
    trainer.params = SimpleNamespace(
        reload_checkpoint=str(reload_checkpoint),
        dump_path=str(PROJECT_ROOT / "tmp"),
        amp=-1,
        fp16=False,
    )
```

- [ ] **Step 4: 跑完整回归集**

Run:

```bash
pytest \
  "PhysicsRegressionPaddle/unitTest/test_amp_public_contract.py" \
  "PhysicsRegressionPaddle/unitTest/test_trainer_native_amp_only.py" \
  "PhysicsRegressionPaddle/unitTest/test_training_entrypoints_native_only.py" \
  "PhysicsRegressionPaddle/unitTest/test_checkpoint_io.py" \
  "PhysicsRegressionPaddle/unitTest/test_phyreg_native_policy.py" \
  "PhysicsRegressionPaddle/unitTest/test_root_model_scripts.py" \
  -q
```

Expected: PASS，输出 `20 passed`

- [ ] **Step 5: 提交回归与夹具清理**

```bash
git add \
  "PhysicsRegressionPaddle/unitTest/test_training_entrypoints_native_only.py"
git commit -m "test: align paddle native-only fixtures after apex removal"
```

---

## Self-Review

- 覆盖检查：计划只覆盖真实存在 Apex 依赖的 `parsers.py`、`trainer.py`、`transformer.py` 和对应测试，没有扩展到 README / notebook / shell，因为本次范围是“代码依赖拔除”，这些文件当前也没有 Apex 内容
- 占位符检查：所有任务都给出了明确文件、测试代码、实施代码和命令，没有 `TODO/TBD`
- 一致性检查：最终口径始终一致
  - `--nvidia_apex` 被删除
  - `--amp` 保留整数接口但只表示 Paddle 原生 AMP
  - Trainer 只保留普通路径和原生 AMP 路径
  - Apex checkpoint 不再属于支持范围
