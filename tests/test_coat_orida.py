import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image
import torch

import coat_orida as coat


class CoatMetricTest(unittest.TestCase):
    def test_perfect_parallelogram_has_zero_l2(self):
        rng = np.random.default_rng(1)
        a, b, c = rng.normal(size=(3, 12))
        d = b - a + c
        self.assertAlmostEqual(coat.l2_loss(a, b, c, d), 0, places=12)

    def test_perfect_parallel_transform_has_zero_angle(self):
        a = np.array([1., 2., 3.])
        delta = np.array([.4, -.2, .8])
        c = np.array([-1., 3., 2.])
        self.assertAlmostEqual(coat.angular_loss(a, a + delta, c, c + delta), 0, places=7)

    def test_opposite_transform_has_pi_angle(self):
        a = np.zeros(3)
        delta = np.array([1., -2., .5])
        c = np.ones(3)
        self.assertAlmostEqual(coat.angular_loss(a, delta, c, c - delta), np.pi, places=7)

    def test_zero_positive_loss_normalizes_to_one(self):
        self.assertAlmostEqual(coat.coat_score(0, .42), 1)

    def test_pixel_algebra_is_exact(self):
        rng = np.random.default_rng(2)
        a, b, c = rng.normal(size=(3, 3, 16, 16)).astype(np.float32)
        d = b - a + c
        self.assertLess(coat.l2_loss(a.ravel(), b.ravel(), c.ravel(), d.ravel()), 1e-8)


class CheckpointGuardTest(unittest.TestCase):
    def test_identical_checkpoints_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.pth"
            second = Path(directory) / "second.pth"
            torch.save({"args": {"lambda3": 0}}, first)
            shutil.copyfile(first, second)
            with self.assertRaisesRegex(ValueError, "byte-identical"):
                coat.validate_checkpoints({"Control +200": first, "Region +200": second})


class ManifestTest(unittest.TestCase):
    def _sets(self, root):
        sets = []
        for scene_index, scene in enumerate(("s1", "s2", "s3")):
            background = root / f"{scene}_background.png"
            Image.new("RGB", (100, 100), "black").save(background)
            factuals = []
            for position, (x, y) in enumerate(((10, 10), (60, 10), (10, 60), (60, 60))):
                path = root / f"{scene}_{position}.png"
                Image.new("RGB", (100, 100), "white").save(path)
                factuals.append(coat.FactualImage(str(path), (x, y, 20, 20), str(position)))
            sets.append(coat.FCFSet("physical-object", scene, str(background), tuple(factuals)))
        return sets

    def test_tuple_and_random_manifest_reproducibility(self):
        with tempfile.TemporaryDirectory() as directory:
            sets = self._sets(Path(directory))
            first = coat.build_tuples(sets, seed=0)
            second = coat.build_tuples(sets, seed=0)
            self.assertEqual(first, second)
            # Lower the production pool size for this synthetic fixture.
            with mock.patch.object(coat, "NUM_RANDOM_D", 2):
                random_first = coat.build_random_baselines(first, seed=0)
                random_second = coat.build_random_baselines(second, seed=0)
            self.assertEqual(random_first, random_second)


class RepresentationTest(unittest.TestCase):
    class Backbone(torch.nn.Module):
        num_register_tokens = 2

        def forward(self, tensor, return_all_tokens=True):
            batch = len(tensor)
            # CLS + two registers + four spatial tokens.
            special = torch.full((batch, 3, 3), 99., device=tensor.device)
            spatial = torch.zeros((batch, 4, 3), device=tensor.device)
            return torch.cat((special, spatial), dim=1)

    class Head(torch.nn.Module):
        def forward(self, tokens):
            # A failing special-token slice would change the expected patch count.
            logits = torch.tensor(
                [[[2., 0.], [0., 2.], [2., 0.], [0., 2.]]],
                device=tokens.device,
            ).repeat(len(tokens), 1, 1)
            return tokens[:, 0], logits

    def test_representation_is_probability_and_excludes_special_tokens(self):
        raw = torch.rand(2, 3, 4, 4)
        with (
            mock.patch.object(coat, "DEVICE", "cpu"),
            mock.patch.object(coat, "INPUT_SIZE", 4),
        ):
            z = coat.encode_tensor_batch(
                raw, self.Backbone(), self.Head(), patch_size=2,
                prototypes=2, temperature=1,
            )
        self.assertEqual(z.shape, (2, 2))
        self.assertTrue(np.all(z >= 0))
        np.testing.assert_allclose(z.sum(axis=1), 1, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
