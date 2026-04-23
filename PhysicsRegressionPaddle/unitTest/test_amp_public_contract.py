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
