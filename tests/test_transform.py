import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir) + "/../")

from simsiam.transforms import get_transforms


def test_transform():
    transforms = get_transforms()
    assert transforms is not None
