import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir) + "/../")

from simsiam.model import SimSiam


def test_model():
    model = SimSiam()
    assert model is not None
