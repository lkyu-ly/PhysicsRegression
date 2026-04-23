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
