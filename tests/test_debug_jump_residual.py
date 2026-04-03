import numpy as np

from tools.debug_jump_residual import build_threshold_masks, parse_thresholds, summarize_threshold


def test_parse_thresholds_sorts_values():
    assert parse_thresholds("0.6,0.4,0.8") == [0.4, 0.6, 0.8]


def test_build_threshold_masks_reconstructs_refined_and_rejection_buckets():
    shock_props = {
        "mask": np.array([True, True, True, True]),
        "sr_mach_normal": np.array([1.5, 1.1, 1.6, 1.7]),
        "jump_residual_light": np.array([0.3, 0.2, 0.7, 0.3]),
        "theta_Bn": np.array([0.1, 0.2, 0.3, 0.4]),
        "entropy_jump": np.array([1.0, 1.0, 1.0, -1.0]),
    }

    masks = build_threshold_masks(shock_props, sr_mach_min=1.2, jump_residual_max=0.4)

    assert np.array_equal(masks["refined_mask"], np.array([True, False, False, False]))
    assert np.array_equal(masks["rejected_low_mach"], np.array([False, True, False, False]))
    assert np.array_equal(masks["rejected_jump"], np.array([False, False, True, False]))
    assert np.array_equal(masks["rejected_entropy"], np.array([False, False, False, True]))


def test_summarize_threshold_reports_expected_counts_and_medians():
    shock_props = {
        "mask": np.array([True, True, True, True]),
        "sr_mach_normal": np.array([1.5, 1.1, 1.6, 1.7]),
        "jump_residual_light": np.array([0.3, 0.2, 0.7, 0.3]),
        "theta_Bn": np.array([0.1, 0.2, 0.3, 0.4]),
        "entropy_jump": np.array([1.0, 1.0, 1.0, -1.0]),
    }
    masks = build_threshold_masks(shock_props, sr_mach_min=1.2, jump_residual_max=0.4)
    summary = summarize_threshold(shock_props, masks, jump_residual_max=0.4)

    assert summary.verified_count == 4
    assert summary.refined_count == 1
    assert np.isclose(summary.refined_fraction, 0.25)
    assert summary.rejected_low_mach_count == 1
    assert summary.rejected_jump_count == 1
    assert summary.rejected_entropy_count == 1
    assert np.isclose(summary.refined_sr_mach_median, 1.5)
    assert np.isclose(summary.refined_sr_mach_p90, 1.5)
