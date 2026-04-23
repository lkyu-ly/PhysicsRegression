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
    import os

    old_cwd = Path.cwd()
    os.chdir(PADDLE_ROOT)
    try:
        model = PhyReg(MODEL_PDPARAMS, device="cpu")
    finally:
        os.chdir(old_cwd)
    assert set(model.modules) == {"embedder", "encoder", "decoder"}


def test_phyreg_rejects_torch_pt():
    with pytest.raises(ValueError, match="convert_torch_to_paddle.py"):
        PhyReg(MODEL_PT, device="cpu")


def test_phyreg_rejects_legacy_pickle(tmp_path):
    legacy_path = tmp_path / "legacy.pkl"
    make_legacy_pickle(legacy_path)
    with pytest.raises(ValueError, match="仅支持 Paddle 原生"):
        PhyReg(legacy_path, device="cpu")
