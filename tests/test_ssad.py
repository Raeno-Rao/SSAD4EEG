import unittest

import torch

from models.ssad import SSAD, band_log_power


class SpectralTests(unittest.TestCase):
    def test_projection_commutes_with_fft(self):
        model = SSAD(22, 4, 7, 250)
        x = torch.randn(2, 22, 250)
        actual = torch.fft.rfft(model.encoder(x), dim=-1)
        expected = torch.einsum("dc,bcf->bdf", model.encoder.projection.weight[:, :, 0].to(actual.dtype), torch.fft.rfft(x, dim=-1))
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))

    def test_physical_alpha_frequency(self):
        rate, length = 250, 1000
        time = torch.arange(length) / rate
        wave = torch.sin(2 * torch.pi * 10 * time)[None, None]
        powers = band_log_power(wave, rate)
        self.assertEqual(powers.argmax(dim=-1).item(), 2)

    def test_geometry_matches_fft_grid_and_backpropagates(self):
        model = SSAD(22, 4, 7, 250, steps=1)
        x = torch.randn(2, 22, 250)
        output = model(x, training_losses=True, deterministic=True)
        loss = output["logits"].square().mean() + output["score_loss"] + output["align_loss"]
        loss.backward()
        self.assertEqual(output["gate"].numel(), 126)
        self.assertIsNotNone(model.geometry.centers.grad)
        self.assertTrue(torch.isfinite(model.geometry.centers.grad).all())

    def test_fixed_inference_noise(self):
        model = SSAD(22, 4, 7, 250, steps=1).eval()
        x = torch.randn(2, 22, 250)
        with torch.no_grad():
            first = model(x)["logits"]
            second = model(x)["logits"]
        self.assertTrue(torch.equal(first, second))

    def test_covariance_controls_match_trace(self):
        reference = SSAD(22, 4, 7, 250)
        gate, learned = reference.geometry.responses(250, torch.device("cpu"))
        target = reference.trace_mean(learned, 250)
        for mode in ("isotropic", "random", "fixed"):
            model = SSAD(22, 4, 7, 250, covariance_mode=mode)
            response = model.covariance_response(gate, learned, 250)
            self.assertTrue(torch.all(response > 0))
            self.assertTrue(torch.allclose(model.trace_mean(response, 250), target, atol=1e-6))

    def test_step_sweep_and_no_refinement(self):
        x = torch.randn(2, 22, 250)
        model = SSAD(22, 4, 7, 250)
        with torch.no_grad():
            for steps in (1, 2, 4, 10, 20):
                model.steps = steps
                output = model(x, return_details=True)
                self.assertEqual(len(output["score_fields"]), steps)
                self.assertEqual(output["logits"].shape, (2, 4))
            baseline = SSAD(22, 4, 7, 250, filtering=False, refinement="none", noise_enabled=False)
            output = baseline(x, training_losses=True, return_details=True)
            self.assertEqual(len(output["score_fields"]), 0)
            self.assertTrue(torch.equal(output["clean"], output["refined"]))


if __name__ == "__main__":
    unittest.main()
