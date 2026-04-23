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
