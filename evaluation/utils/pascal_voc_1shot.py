import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.utils.classification import classification_entrypoint

if __name__ == "__main__":
    classification_entrypoint("evaluation.utils.pascal_voc_1shot", "pascal_voc", "pascal_voc_1shot")
