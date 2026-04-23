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
