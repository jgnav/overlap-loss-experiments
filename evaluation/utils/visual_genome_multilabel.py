"""CRISP VG500 full-data frozen linear probe."""

from evaluation.utils.classification import classification_entrypoint


if __name__ == "__main__":
    classification_entrypoint(
        "evaluation.utils.visual_genome_multilabel", "visual_genome", "visual_genome_multilabel"
    )
