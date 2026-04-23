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
        assert bool(paddle.equal_all(target.state_dict()[key], value))

    optimizer = paddle.optimizer.Adam(
        learning_rate=0.001, parameters=target.parameters()
    )
    optimizer_state = optimizer.state_dict()
    set_optimizer_state(optimizer, optimizer_state)

    scaler = paddle.amp.GradScaler()
    scaler_state = scaler.state_dict()
    set_grad_scaler_state(scaler, scaler_state)
