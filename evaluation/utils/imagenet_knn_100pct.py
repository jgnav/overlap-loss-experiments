import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.utils.imagenet import imagenet_entrypoint

if __name__ == "__main__":
    imagenet_entrypoint("evaluation.utils.imagenet_knn_100pct", "knn", "imagenet_knn_100pct")
