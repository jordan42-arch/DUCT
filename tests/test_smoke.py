"""Data-free execution checks; not an accuracy or benchmark reproduction."""
import importlib.util
import unittest

import torch

from models import DUCT, SUPPORTED_MODELS, build_baseline


class DUCTSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(2024)

    def model(self):
        return DUCT(
            n_vars=3, lookback=32, pred_len=12, patch_len=8, stride=4,
            d_model=16, n_heads=4, n_layers=1, d_ff=32, dropout=0.0,
            proj_dim=8,
        )

    def test_forecast_and_fusion(self):
        model = self.model().eval()
        with torch.no_grad():
            out = model(torch.randn(2, 3, 32), return_components=True)
        for key in ("y", "y_patch", "y_var"):
            self.assertEqual(tuple(out[key].shape), (2, 3, 12))
            self.assertTrue(torch.isfinite(out[key]).all())
        torch.testing.assert_close(out["patch_weight"] + out["var_weight"], torch.tensor(1.0))
        expected = out["patch_weight"] * out["y_patch"] + out["var_weight"] * out["y_var"]
        torch.testing.assert_close(out["y"], expected)

    def test_pretraining_backward(self):
        model = self.model().train()
        out = model.pretrain_forward(torch.randn(4, 3, 32))
        for key in ("loss_mae", "loss_contrast", "loss_total"):
            self.assertTrue(torch.isfinite(out[key]).all())
        out["loss_total"].backward()
        for module in (model.patch_branch, model.variate_branch, model.rep_fusion):
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))

    def test_registered_baselines(self):
        x = torch.randn(2, 3, 96)
        for name in SUPPORTED_MODELS:
            if name in ("DUCT", "WPMixer"):
                continue
            with self.subTest(model=name), torch.no_grad():
                model = build_baseline(name, 3, 96, 12).eval()
                y = model(x)
                self.assertEqual(tuple(y.shape), (2, 3, 12))
                self.assertTrue(torch.isfinite(y).all())

    @unittest.skipUnless(importlib.util.find_spec("pywt"), "PyWavelets is not installed")
    def test_wpmixer(self):
        with torch.no_grad():
            model = build_baseline("WPMixer", 3, 96, 12).eval()
            y = model(torch.randn(2, 3, 96))
        self.assertEqual(tuple(y.shape), (2, 3, 12))
        self.assertTrue(torch.isfinite(y).all())


if __name__ == "__main__":
    unittest.main()
