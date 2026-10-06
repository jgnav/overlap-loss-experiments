"""Dispatch the explicitly selected video propagation protocol."""
from evaluation.utils.common import base_parser


def main(dataset):
    args, _ = base_parser("Video propagation protocol").parse_known_args()
    if args.video_protocol in ("dino_480p_last4", "dino_square_last4"):
        from evaluation.utils.video_dino import main as evaluate
    else:
        from evaluation.utils.video_dinov3 import main as evaluate
    evaluate(dataset)
