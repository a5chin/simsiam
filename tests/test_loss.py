import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir) + "/../")

from simsiam.loss import NegativeCosineSimilarity


def test_loss():
    criterion = NegativeCosineSimilarity()
    assert criterion is not None
