import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

import evaluate_nopair_niqe_brisque as evaluation


class Brighten(torch.nn.Module):
    def forward(self, image):
        return image + 0.3


class HighlightProtectionTests(unittest.TestCase):
    def test_color_preservation_keeps_rgb_ratios_with_clipped_model_colors(self):
        original = np.full((16, 16, 3), [0.7, 0.25, 0.1], dtype=np.float32)
        enhanced = np.ones_like(original)
        output = evaluation.preserve_input_color(original, enhanced)
        np.testing.assert_allclose(output / output.sum(axis=2, keepdims=True),
                                   original / original.sum(axis=2, keepdims=True),
                                   rtol=1e-6, atol=1e-7)
        self.assertGreater(output.mean(), original.mean())
        self.assertLess(output.max(), 1)

    def test_color_preservation_black_zero_strength_and_invalid_values(self):
        original = np.zeros((2, 2, 3), dtype=np.float32)
        enhanced = np.ones_like(original)
        output = evaluation.preserve_input_color(original, enhanced)
        self.assertTrue(np.isfinite(output).all())
        np.testing.assert_array_equal(output, original)
        original[:] = [0.4, 0.2, 0.1]
        np.testing.assert_array_equal(
            evaluation.preserve_input_color(original, enhanced, strength=0), original)
        for strength in (-1, float('nan'), 1.1):
            with self.assertRaises(ValueError):
                evaluation.preserve_input_color(original, enhanced, strength)

    def test_color_preservation_runs_after_padding_and_ensemble(self):
        image = np.full((25, 31, 3), [180, 70, 30], dtype=np.uint8)
        tensor = evaluation.rgb_to_tensor(image, 'cpu')
        baseline = evaluation.infer_image(Brighten(), tensor, 8, False,
                                         color_options={'strength': 0.45})
        ensemble = evaluation.infer_image(Brighten(), tensor, 8, True,
                                         color_options={'strength': 0.45})
        np.testing.assert_array_equal(baseline, ensemble)
        self.assertEqual(baseline.shape, image.shape)
        self.assertLess(int(baseline[..., 0].max()), 255)

    def test_shadows_and_model_darkening_are_unchanged(self):
        rng = np.random.default_rng(42)
        original = rng.uniform(0, 0.45, (25, 31, 3)).astype(np.float32)
        enhanced = np.clip(original + 0.3, 0, 1)
        output, mask = evaluation.protect_highlights(original, enhanced)
        np.testing.assert_array_equal(output, enhanced)
        np.testing.assert_array_equal(mask, 0)
        original = rng.uniform(0.7, 1, (25, 31, 3)).astype(np.float32)
        enhanced = original * 0.8
        output, mask = evaluation.protect_highlights(original, enhanced)
        np.testing.assert_array_equal(output, enhanced)
        np.testing.assert_array_equal(mask, 0)

    def test_restores_source_highlight_variation(self):
        ramp = np.linspace(0.85, 0.99, 64, dtype=np.float32)
        original = np.broadcast_to(ramp[None, :, None], (32, 64, 3)).copy()
        enhanced = np.ones_like(original)
        output, mask = evaluation.protect_highlights(original, enhanced)
        self.assertTrue(np.isfinite(output).all())
        self.assertLess(np.mean(output >= 0.999), np.mean(enhanced >= 0.999))
        self.assertGreater(np.std(output), 0)
        self.assertTrue(np.all(output >= original))
        self.assertTrue(np.all(output <= enhanced))
        self.assertGreater(mask.max(), 0)
        self.assertLessEqual(mask.max(), 0.8 + 1e-6)

    def test_edge_does_not_darken_adjacent_shadows(self):
        original = np.full((48, 64, 3), 0.1, dtype=np.float32)
        original[:, 32:] = 0.85
        enhanced = np.clip(original + 0.3, 0, 1)
        output, mask = evaluation.protect_highlights(original, enhanced)
        np.testing.assert_array_equal(output[:, :32], enhanced[:, :32])
        np.testing.assert_array_equal(mask[:, :32], 0)
        self.assertTrue(np.all(output[:, 32:] < enhanced[:, 32:]))
        self.assertTrue(np.all(output >= original))

    def test_zero_strength_and_tiny_images(self):
        original = np.full((1, 1, 3), 0.8, dtype=np.float32)
        enhanced = np.ones_like(original)
        output, mask = evaluation.protect_highlights(original, enhanced, strength=0)
        np.testing.assert_array_equal(output, enhanced)
        np.testing.assert_array_equal(mask, 0)
        output, mask = evaluation.protect_highlights(original, enhanced)
        self.assertTrue(np.isfinite(output).all())
        self.assertGreater(mask.item(), 0)

    def test_no_color_overshoot(self):
        original = np.full((20, 20, 3), [0.95, 0.75, 0.65], dtype=np.float32)
        enhanced = np.full_like(original, [1, 0.98, 0.9])
        output, _ = evaluation.protect_highlights(original, enhanced, radius=0)
        self.assertTrue(np.all(output >= original))
        self.assertTrue(np.all(output <= enhanced))

    def test_invalid_parameters(self):
        image = np.ones((8, 8, 3), dtype=np.float32)
        for options in ({'strength': -0.1}, {'strength': float('nan')},
                        {'low': 0.9, 'high': 0.8}, {'high': float('inf')},
                        {'radius': -1}, {'radius': 1.5}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                evaluation.protect_highlights(image, image, **options)

    def test_inference_padding_ensemble_and_disabled_compatibility(self):
        image = np.full((25, 31, 3), 26, dtype=np.uint8)
        image[:, 16:] = 217
        tensor = evaluation.rgb_to_tensor(image, 'cpu')
        for ensemble in (False, True):
            baseline = evaluation.infer_image(Brighten(), tensor, 8, ensemble)
            disabled = evaluation.infer_image(
                Brighten(), tensor, 8, ensemble, {'strength': 0})
            protected = evaluation.infer_image(
                Brighten(), tensor, 8, ensemble, {'strength': 0.8})
            np.testing.assert_array_equal(disabled, baseline)
            np.testing.assert_array_equal(protected[:, :16], baseline[:, :16])
            self.assertEqual(protected.shape, image.shape)
            self.assertEqual(protected.dtype, np.uint8)
            self.assertTrue(np.all(protected[:, 16:] < baseline[:, 16:]))

    def test_cli_saves_and_scores_same_protected_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'input' / 'scene'
            source.mkdir(parents=True)
            image = np.full((25, 31, 3), 217, dtype=np.uint8)
            cv2.imwrite(str(source / 'test.png'), image)
            result = root / 'output'
            scored = []

            def metrics(rgb, path, metric, args):
                saved = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
                np.testing.assert_array_equal(rgb, saved)
                scored.append(rgb.copy())
                return 1.0, 2.0

            argv = ['evaluate_nopair_niqe_brisque.py', '--gpu_id', '-1',
                    '--input_root', str(root / 'input'), '--result_root', str(result)]
            with patch.object(sys, 'argv', argv), \
                    patch.object(evaluation, 'build_model', return_value=Brighten()), \
                    patch.object(evaluation, 'build_brisque_metric', return_value=object()), \
                    patch.object(evaluation, 'calculate_metrics', side_effect=metrics), \
                    patch.object(evaluation, 'tqdm', None), \
                    contextlib.redirect_stdout(io.StringIO()):
                evaluation.main()
            self.assertEqual(len(scored), 1)
            self.assertLess(scored[0].max(), 255)
            self.assertTrue((result / 'metrics.csv').is_file())
            self.assertTrue((result / 'summary.csv').is_file())

            saved_path = result / 'scene' / 'test.png'
            before = saved_path.read_bytes()
            with patch.object(sys, 'argv', argv + ['--metrics_only']), \
                    patch.object(evaluation, 'build_model') as build, \
                    patch.object(evaluation, 'protect_highlights') as protect, \
                    patch.object(evaluation, 'build_brisque_metric', return_value=object()), \
                    patch.object(evaluation, 'calculate_metrics', side_effect=metrics), \
                    patch.object(evaluation, 'tqdm', None), \
                    contextlib.redirect_stdout(io.StringIO()):
                evaluation.main()
                build.assert_not_called()
                protect.assert_not_called()
            self.assertEqual(saved_path.read_bytes(), before)
            np.testing.assert_array_equal(scored[0], scored[1])


if __name__ == '__main__':
    unittest.main()
