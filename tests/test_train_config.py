import unittest
from pathlib import Path
from unittest import mock

from train import load_config
from utils import training as utils


class ContinuationConfigTest(unittest.TestCase):
    def test_production_config_is_bf16_200_epochs_without_lr_warmup(self):
        config = load_config(Path(__file__).parents[1] / "train.yaml")

        self.assertEqual(config.additional_epochs, 200)
        self.assertEqual(config.epochs, 200)
        self.assertEqual(config.precision, "bf16")
        self.assertFalse(config.use_fp16)
        self.assertEqual(config.warmup_epochs, 0)
        self.assertEqual(config.batch_size_per_gpu * config.gpu_count, 256)
        self.assertEqual(config.saveckp_freq, 50)

    def test_continuation_cosine_schedule_starts_at_configured_lr(self):
        schedule = utils.cosine_scheduler(
            1e-4,
            1e-6,
            epochs=200,
            iterations_per_epoch=2,
            warmup_epochs=0,
        )

        self.assertAlmostEqual(schedule[0], 1e-4)

    def test_debug_environment_can_override_precision_and_batch(self):
        with mock.patch.dict(
            "os.environ",
            {
                "IBOT_PRECISION_OVERRIDE": "fp32",
                "IBOT_BATCH_SIZE_PER_GPU_OVERRIDE": "4",
                "IBOT_GPU_COUNT_OVERRIDE": "1",
            },
        ):
            config = load_config(Path(__file__).parents[1] / "train.yaml")

        self.assertEqual(config.precision, "fp32")
        self.assertEqual(config.batch_size_per_gpu, 4)
        self.assertEqual(config.gpu_count, 1)


if __name__ == "__main__":
    unittest.main()
