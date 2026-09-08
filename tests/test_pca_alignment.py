"""Small CPU-only checks; no PyTorch, checkpoints, or image dataset required."""

import itertools
import unittest

import numpy as np

from utils.pca_alignment import align_pca_components


class PCAAlignmentTest(unittest.TestCase):
    @staticmethod
    def _orthogonal_scores(columns=3):
        samples = np.random.default_rng(7).normal(size=(64, columns))
        samples -= samples.mean(axis=0)
        return np.linalg.qr(samples)[0]

    def test_recovers_all_component_permutations_and_sign_flips(self):
        reference = self._orthogonal_scores().astype(np.float32)
        for order in itertools.permutations(range(3)):
            for flips in itertools.product((-1, 1), repeat=3):
                with self.subTest(order=order, flips=flips):
                    target = reference[:, order] * np.array(flips, dtype=np.float32)
                    before = target.copy()

                    alignment = align_pca_components(reference, target)

                    np.testing.assert_array_equal(alignment.projected, reference)
                    np.testing.assert_array_equal(target, before)
                    np.testing.assert_array_equal(alignment.permutation, np.argsort(order))
                    matched = alignment.correlations[np.arange(3), alignment.permutation]
                    np.testing.assert_allclose(matched * alignment.signs, 1.0, atol=1e-12)
                    self.assertEqual(alignment.projected.dtype, np.float32)

    def test_assignment_is_global_optimum_not_greedy_row_matching(self):
        basis = self._orthogonal_scores(columns=6)
        reference = basis[:, :3]
        correlations = np.array([
            [0.60, 0.58, 0.0],
            [0.59, 0.02, 0.0],
            [0.0, 0.0, 0.4],
        ])
        # Build orthonormal target PCs with this exact cross-correlation.
        values, vectors = np.linalg.eigh(np.eye(3) - correlations.T @ correlations)
        residual = (vectors * np.sqrt(values)) @ vectors.T
        target = reference @ correlations + basis[:, 3:] @ residual

        alignment = align_pca_components(reference, target)

        np.testing.assert_allclose(alignment.correlations, correlations, atol=1e-12)
        np.testing.assert_array_equal(alignment.permutation, [1, 0, 2])
        np.testing.assert_array_equal(alignment.signs, [1, 1, 1])
        np.testing.assert_array_equal(alignment.projected, target[:, [1, 0, 2]])
        # Alignment must preserve differences, rather than copy the reference.
        self.assertFalse(np.allclose(alignment.projected, reference))

    def test_pearson_matching_ignores_offsets_and_scales_without_removing_them(self):
        reference = self._orthogonal_scores()
        target = reference[:, [2, 0, 1]] * [-2.0, 0.5, -4.0] + [10.0, -3.0, 9.0]

        alignment = align_pca_components(reference, target)

        np.testing.assert_array_equal(alignment.permutation, [1, 2, 0])
        np.testing.assert_array_equal(alignment.signs, [1, -1, -1])
        np.testing.assert_allclose(
            alignment.projected, reference * [0.5, 4.0, 2.0] + [-3.0, -9.0, -10.0]
        )
        matched = alignment.correlations[np.arange(3), alignment.permutation]
        np.testing.assert_allclose(np.abs(matched), 1.0, atol=1e-12)

    def test_constant_components_have_zero_correlation_without_nan(self):
        reference = np.ones((16, 3))
        target = np.full((16, 3), 2.0)

        alignment = align_pca_components(reference, target)

        np.testing.assert_array_equal(alignment.correlations, np.zeros((3, 3)))
        np.testing.assert_array_equal(np.sort(alignment.permutation), [0, 1, 2])
        np.testing.assert_array_equal(alignment.signs, [1, 1, 1])
        np.testing.assert_array_equal(alignment.projected, target)

    def test_matching_shapes_and_corresponding_patch_count_are_required(self):
        reference = self._orthogonal_scores()
        for target in (np.zeros((63, 3)), np.zeros((64, 4)), np.zeros(3)):
            with self.subTest(shape=target.shape):
                with self.assertRaisesRegex(ValueError, "corresponding patches"):
                    align_pca_components(reference, target)
        for shape in ((0, 3), (1, 3), (4, 2), (3,)):
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    align_pca_components(np.zeros(shape), np.zeros(shape))

    def test_non_finite_scores_are_rejected(self):
        reference = self._orthogonal_scores()
        for invalid in (np.nan, np.inf, -np.inf):
            target = reference.copy()
            target[0, 0] = invalid
            for left, right in ((reference, target), (target, reference)):
                with self.subTest(invalid=invalid):
                    with self.assertRaisesRegex(ValueError, "finite"):
                        align_pca_components(left, right)


if __name__ == "__main__":
    unittest.main()
