"""Unit tests for TwoHandAugmentationEngine and _renormalize_two_hand."""

import numpy as np
import pytest

from src.lab.augment import TwoHandAugmentationEngine, _renormalize_two_hand


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_template(seed: int = 0, secondary_absent: bool = False) -> np.ndarray:
    """Build a plausible 84-float two-hand template programmatically."""
    rng = np.random.default_rng(seed)
    primary = rng.uniform(-1.0, 1.0, (21, 2)).astype(np.float32)
    primary[0] = 0.0                                    # wrist at origin
    primary /= np.max(np.abs(primary[1:]))              # scale to [-1, 1]

    if secondary_absent:
        secondary = np.zeros((21, 2), dtype=np.float32)
    else:
        # Secondary can exceed [-1, 1] — it is in primary's frame
        secondary = rng.uniform(-2.0, 2.0, (21, 2)).astype(np.float32)

    return np.concatenate([primary.flatten(), secondary.flatten()]).astype(np.float32)


# ---------------------------------------------------------------------------
# TwoHandAugmentationEngine tests
# ---------------------------------------------------------------------------

class TestTwoHandAugmentationEngine:

    def test_output_shape(self):
        engine   = TwoHandAugmentationEngine(seed=0)
        template = _make_template()
        result   = engine.generate(template, 50)
        assert result.shape == (50, 84)

    def test_output_dtype(self):
        engine   = TwoHandAugmentationEngine(seed=0)
        template = _make_template()
        result   = engine.generate(template, 10)
        assert result.dtype == np.float32

    def test_primary_wrist_at_origin(self):
        engine   = TwoHandAugmentationEngine(seed=0)
        template = _make_template()
        result   = engine.generate(template, 50)
        # Primary wrist = first two floats (x0_p, y0_p)
        np.testing.assert_allclose(result[:, 0], 0.0, atol=1e-5,
                                   err_msg="Primary wrist x must be 0")
        np.testing.assert_allclose(result[:, 1], 0.0, atol=1e-5,
                                   err_msg="Primary wrist y must be 0")

    def test_primary_values_bounded(self):
        engine   = TwoHandAugmentationEngine(seed=0)
        template = _make_template()
        result   = engine.generate(template, 100)
        primary_maxabs = np.max(np.abs(result[:, :42]), axis=1)
        assert np.all(primary_maxabs <= 1.0 + 1e-5), (
            f"Primary half exceeds [-1, 1]: max={primary_maxabs.max():.6f}"
        )

    def test_secondary_stays_zero_when_absent(self):
        engine   = TwoHandAugmentationEngine(seed=0)
        template = _make_template(secondary_absent=True)
        result   = engine.generate(template, 50)
        assert np.all(result[:, 42:] == 0.0), (
            "Secondary half must remain all zeros when template secondary is absent"
        )

    def test_secondary_non_zero_when_present(self):
        engine   = TwoHandAugmentationEngine(seed=0)
        template = _make_template(secondary_absent=False)
        result   = engine.generate(template, 50)
        # At least some rows should have non-zero secondary halves
        assert not np.all(result[:, 42:] == 0.0), (
            "Secondary half must not be all zeros when template secondary is present"
        )

    def test_reproducible_with_seed(self):
        template = _make_template()
        result_a = TwoHandAugmentationEngine(seed=42).generate(template, 30)
        result_b = TwoHandAugmentationEngine(seed=42).generate(template, 30)
        np.testing.assert_array_equal(result_a, result_b,
                                      err_msg="Same seed must produce identical output")

    def test_different_seeds_differ(self):
        template = _make_template()
        result_a = TwoHandAugmentationEngine(seed=1).generate(template, 30)
        result_b = TwoHandAugmentationEngine(seed=2).generate(template, 30)
        assert not np.array_equal(result_a, result_b), (
            "Different seeds must produce different output"
        )

    def test_n_samples_one(self):
        engine   = TwoHandAugmentationEngine(seed=0)
        template = _make_template()
        result   = engine.generate(template, 1)
        assert result.shape == (1, 84)

    def test_wrong_shape_raises(self):
        engine   = TwoHandAugmentationEngine(seed=0)
        template = _make_template()[:42]  # (42,) — wrong for two-hand engine
        with pytest.raises(ValueError, match="84"):
            engine.generate(template, 10)

    def test_no_nan_or_inf(self):
        engine   = TwoHandAugmentationEngine(seed=0)
        template = _make_template()
        result   = engine.generate(template, 100)
        assert not np.any(np.isnan(result)), "Output must not contain NaN"
        assert not np.any(np.isinf(result)), "Output must not contain Inf"


# ---------------------------------------------------------------------------
# _renormalize_two_hand tests
# ---------------------------------------------------------------------------

class TestRenormalizeTwoHand:

    def test_primary_wrist_at_origin(self):
        template = _make_template()
        # Shift the primary wrist off-origin to simulate post-noise state
        template[0] += 0.1   # x0_p
        template[1] += 0.05  # y0_p
        result = _renormalize_two_hand(template)
        np.testing.assert_allclose(result[0], 0.0, atol=1e-6)
        np.testing.assert_allclose(result[1], 0.0, atol=1e-6)

    def test_output_shape(self):
        result = _renormalize_two_hand(_make_template())
        assert result.shape == (84,)

    def test_output_dtype(self):
        result = _renormalize_two_hand(_make_template())
        assert result.dtype == np.float32

    def test_secondary_absent_stays_zero(self):
        template = _make_template(secondary_absent=True)
        result   = _renormalize_two_hand(template)
        assert np.all(result[42:] == 0.0)

    def test_does_not_mutate_input(self):
        template = _make_template()
        original = template.copy()
        _renormalize_two_hand(template)
        np.testing.assert_array_equal(template, original,
                                      err_msg="_renormalize_two_hand must not mutate input")
