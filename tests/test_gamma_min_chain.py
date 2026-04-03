import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy.special", reason="scipy is required for nonthermal electron tests")

try:
    import h5py
except ModuleNotFoundError:
    h5py = None



from src.core.nt_electron_v1 import calculate_nonthermal_electrons
from src.core.shock_v1 import find_shocks_in_roi_mhd
from src.workflows.base_workflow import save_h5_file


def _build_minimal_roi():
    shape = (1, 3, 7)
    rho = np.ones(shape)
    press = np.ones(shape)
    press[0, 1, 2] = 1.0
    press[0, 1, 3] = 2.0
    press[0, 1, 4] = 10.0
    press[0, 1, 5] = 20.0
    press[0, 1, 6] = 20.0

    vel1 = np.zeros(shape)
    vel1[0, 1, :] = [0.0, 1.0, 2.0, 3.0, 8.0, 9.0, 9.0]
    vel2 = np.zeros(shape)
    vel3 = np.zeros(shape)

    zeros = np.zeros(shape)
    return {
        "rho": rho,
        "press": press,
        "vel1": vel1,
        "vel2": vel2,
        "vel3": vel3,
        "Bcc1": zeros.copy(),
        "Bcc2": zeros.copy(),
        "Bcc3": zeros.copy(),
        "x1f": np.arange(8, dtype=float) + 1.0,
        "x2f": np.array([1.0, 1.4, 1.8, 2.2]),
        "x3f": np.array([0.0, 2 * np.pi]),
        "x1v": np.arange(7, dtype=float) + 1.5,
        "x2v": np.array([1.2, 1.6, 2.0]),
        "x3v": np.array([np.pi]),
        "Time": 0.0,
    }


def test_find_shocks_returns_downstream_primitives_and_sampling_diagnostics():
    roi_data = _build_minimal_roi()

    shock_props = find_shocks_in_roi_mhd(
        roi_data,
        grad_p_filter_quantile=0.0,
        march_cells=1,
    )

    assert "rho2_code_grid" in shock_props
    assert "press2_code_grid" in shock_props
    assert "press2_over_rho2_grid" in shock_props
    assert "sample_i2_grid" in shock_props
    assert "sample_j2_grid" in shock_props
    assert "sample_k2_grid" in shock_props
    assert "sample_boundary_clipped_grid" in shock_props
    assert "sampling_stats" in shock_props
    assert "verified_mask" in shock_props
    assert "sr_mach_normal" in shock_props
    assert "theta_Bn" in shock_props
    assert "jump_residual_light" in shock_props
    assert shock_props["mask"].any()
    assert shock_props["verified_mask"].any()
    assert shock_props["sampling_stats"]["geom_candidate_count"] >= shock_props["sampling_stats"]["verified_count"] >= shock_props["sampling_stats"]["sr_refined_count"]
    assert "utilde_sq_upstream" in shock_props
    assert "gamma_lorentz_upstream" in shock_props
    assert "utilde_n_upstream" in shock_props


def test_find_shocks_rejects_removed_classical_candidate_controls():
    roi_data = _build_minimal_roi()

    with pytest.raises(ValueError, match="Removed classical shock controls"):
        find_shocks_in_roi_mhd(
            roi_data,
            mach_threshold_loose=0.5,
        )

    with pytest.raises(ValueError, match="Removed classical shock controls"):
        find_shocks_in_roi_mhd(
            roi_data,
            min_physical_mach=1.1,
        )


def test_find_shocks_mainline_mach_replaces_legacy_upstream_mach_field():
    roi_data = _build_minimal_roi()

    shock_props = find_shocks_in_roi_mhd(
        roi_data,
        grad_p_filter_quantile=0.0,
        march_cells=1,
    )

    assert "mainline_mach" in shock_props
    assert "upstream_mach" not in shock_props
    active = shock_props["mask"]
    assert np.allclose(shock_props["mainline_mach"][active], shock_props["sr_mach_normal"][active])


def test_save_h5_rejects_missing_mainline_mach_field():
    roi_data = _build_minimal_roi()
    mask = np.zeros_like(roi_data["rho"], dtype=bool)
    mask[0, 1, 3] = True
    shock_props = {
        "mask": mask,
        "verified_mask": mask.copy(),
        "sr_mach_normal": np.ones_like(roi_data["rho"]) * 1.5,
        "theta_Bn": np.zeros_like(roi_data["rho"]),
        "h_rel_upstream": np.ones_like(roi_data["rho"]),
        "cfast_n_upstream": np.ones_like(roi_data["rho"]) * 0.5,
        "utilde_sq_upstream": np.ones_like(roi_data["rho"]) * 3.0,
        "gamma_lorentz_upstream": np.ones_like(roi_data["rho"]) * 2.0,
        "utilde_n_upstream": np.ones_like(roi_data["rho"]) * 1.5,
        "jump_residual_light": np.zeros_like(roi_data["rho"]),
        "ptot_jump": np.ones_like(roi_data["rho"]),
        "entropy_jump": np.ones_like(roi_data["rho"]),
        "v_n_upstream": np.ones_like(roi_data["rho"]) * 1.5,
        "u_n_upstream": np.ones_like(roi_data["rho"]) * 1.5,
    }
    nonthermal_props = {
        "C_grid": np.zeros_like(roi_data["rho"]),
        "q_grid": np.full_like(roi_data["rho"], 2.5),
        "gamma_min_grid": np.ones_like(roi_data["rho"]),
        "gamma_min_grid_physical": np.ones_like(roi_data["rho"]),
        "gamma_min_failure_code_grid": np.zeros_like(roi_data["rho"], dtype=int),
    }
    config = {
        "shock_params": {"gamma": 4.0 / 3.0},
        "physics": {"spin": 0.98, "hslope": 0.3, "R0": 0.0},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        output_h5 = Path(tmpdir) / "test.h5"
        with pytest.raises(KeyError, match="mainline_mach"):
            save_h5_file(str(output_h5), roi_data, shock_props, nonthermal_props, config)


def test_run_pipeline_default_config_omits_removed_classical_candidate_controls():
    from run_dsa_pipeline import create_default_config

    config = create_default_config()
    assert "mach_threshold_loose" not in config["shock_params"]
    assert "min_physical_mach" not in config["shock_params"]
    assert set(config["shock_params"].keys()) == {"gamma", "grad_p_filter_quantile", "march_cells"}


def test_process_snapshot_rejects_removed_classical_candidate_controls_in_config():
    from src.workflows.workflowFull_v2 import process_snapshot

    config = {
        "output_directory": ".",
        "data_output_directory": ".",
        "roi_params": {},
        "shock_params": {
            "gamma": 4.0 / 3.0,
            "mach_threshold_loose": 1.05,
            "grad_p_filter_quantile": 0.2,
            "march_cells": 6,
        },
        "shock_sr": {"sr_mach_min": 1.2, "jump_residual_max": 0.8},
        "nt_params": {},
        "physics": {"M_unit": 1e25, "MBH_solar": 6.2e9, "enable_advection": False},
    }

    assert process_snapshot("dummy.athdf", config, logger=None) is False


def test_gamma_min_uses_physical_downstream_branch_not_legacy_downstream_temp():
    mask = np.zeros((1, 1, 1), dtype=bool)
    mask[0, 0, 0] = True
    shock_properties = {
        "mask": mask,
        "upstream_mach": np.full((1, 1, 1), 3.0),
        "downstream_temp": np.full((1, 1, 1), 1.0e-12),
        "downstream_n_e": np.full((1, 1, 1), 1.0),
        "rho2_code_grid": np.full((1, 1, 1), 2.0),
        "press2_code_grid": np.full((1, 1, 1), 4.0),
        "press2_over_rho2_grid": np.full((1, 1, 1), 2.0),
        "sample_boundary_clipped_grid": np.zeros((1, 1, 1), dtype=bool),
    }

    result = calculate_nonthermal_electrons(
        shock_properties,
        rho_unit=2.0,
        u_unit=18.0,
    )

    assert result["gamma_min_grid"][0, 0, 0] > 1.0
    assert result["gamma_min_failure_code_grid"][0, 0, 0] == 0


def test_nonthermal_rejects_removed_mask_switch_control():
    mask = np.ones((1, 1, 1), dtype=bool)
    shock_properties = {
        "mask": mask,
        "upstream_mach": np.full((1, 1, 1), 3.0),
        "downstream_temp": np.full((1, 1, 1), 1.0e8),
        "downstream_n_e": np.full((1, 1, 1), 10.0),
        "rho2_code_grid": np.full((1, 1, 1), 2.0),
        "press2_code_grid": np.full((1, 1, 1), 4.0),
        "press2_over_rho2_grid": np.full((1, 1, 1), 2.0),
        "beta2_grid": np.full((1, 1, 1), 1.0),
        "sample_boundary_clipped_grid": np.zeros((1, 1, 1), dtype=bool),
    }

    with pytest.raises(ValueError, match="use_sr_refined_mask has been removed"):
        calculate_nonthermal_electrons(
            shock_properties,
            use_sr_refined_mask=True,
        )


@pytest.mark.skipif(h5py is None, reason="h5py not installed")
def test_save_h5_uses_gamma_min_grid_without_overwriting_valid_values():
    roi_data = _build_minimal_roi()
    mask = np.zeros_like(roi_data["rho"], dtype=bool)
    mask[0, 1, 3] = True
    shock_props = {
        "mask": mask,
        "upstream_mach": np.ones_like(roi_data["rho"]),
        "verified_mask": mask.copy(),
        "sr_mach_normal": np.ones_like(roi_data["rho"]) * 1.5,
        "theta_Bn": np.zeros_like(roi_data["rho"]),
        "h_rel_upstream": np.ones_like(roi_data["rho"]),
        "cfast_n_upstream": np.ones_like(roi_data["rho"]) * 0.5,
        "utilde_sq_upstream": np.ones_like(roi_data["rho"]) * 3.0,
        "gamma_lorentz_upstream": np.ones_like(roi_data["rho"]) * 2.0,
        "utilde_n_upstream": np.ones_like(roi_data["rho"]) * 1.5,
        "jump_residual_light": np.zeros_like(roi_data["rho"]),
        "ptot_jump": np.ones_like(roi_data["rho"]),
        "entropy_jump": np.ones_like(roi_data["rho"]),
        "v_n_upstream": np.ones_like(roi_data["rho"]) * 1.5,
        "u_n_upstream": np.ones_like(roi_data["rho"]) * 1.5,
    }
    nonthermal_props = {
        "C_grid": np.zeros_like(roi_data["rho"]),
        "q_grid": np.full_like(roi_data["rho"], 2.5),
        "gamma_min_grid": np.ones_like(roi_data["rho"]),
        "gamma_min_grid_physical": np.ones_like(roi_data["rho"]),
        "gamma_min_failure_code_grid": np.zeros_like(roi_data["rho"], dtype=int),
    }
    nonthermal_props["gamma_min_grid"][0, 1, 3] = 4.2
    nonthermal_props["gamma_min_grid_physical"][0, 1, 3] = 4.2

    config = {
        "shock_params": {"gamma": 4.0 / 3.0},
        "physics": {"spin": 0.98, "hslope": 0.3, "R0": 0.0},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        output_h5 = Path(tmpdir) / "test.h5"
        save_h5_file(str(output_h5), roi_data, shock_props, nonthermal_props, config)
        with h5py.File(output_h5, "r") as handle:
            gamma_min = handle["GAMMA_MIN"][:]
            assert "SR_MACH_NORMAL" in handle
            assert "THETA_BN" in handle
            assert "H_REL_UPSTREAM" in handle
            assert "CFAST_N_UPSTREAM" in handle
            assert "JUMP_RESIDUAL_LIGHT" in handle
            assert "PTOT_JUMP" in handle
            assert "ENTROPY_JUMP" in handle
            assert "UTILDE_SQ_UPSTREAM" in handle
            assert "GAMMA_LORENTZ_UPSTREAM" in handle
            assert "UTILDE_N_UPSTREAM" in handle
            assert np.isclose(handle["UTILDE_SQ_UPSTREAM"][3, 1, 0], 3.0)
            assert np.isclose(handle["GAMMA_LORENTZ_UPSTREAM"][3, 1, 0], 2.0)
            assert np.isclose(handle["UTILDE_N_UPSTREAM"][3, 1, 0], 1.5)

    assert np.isclose(gamma_min[3, 1, 0], 4.2)
