import tempfile
from pathlib import Path

import numpy as np
import pytest

try:
    import h5py
except ModuleNotFoundError:
    h5py = None



from src.core.nt_electron_v1 import (
    POWERLAW_GAMMA_MAX_DEFAULT,
    _compute_effective_powerlaw_index,
    _compute_relativistic_k_inj,
    _compute_relativistic_p_min,
    calculate_nonthermal_electrons,
)
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
    assert "rho1_code_grid" in shock_props
    assert "press1_code_grid" in shock_props
    assert "beta1_grid" in shock_props
    assert "press2_over_rho2_grid" in shock_props
    assert "sample_i2_grid" in shock_props
    assert "sample_j2_grid" in shock_props
    assert "sample_k2_grid" in shock_props
    assert "sample_boundary_clipped_grid" in shock_props
    assert "sampling_stats" in shock_props
    assert "verified_mask" in shock_props
    assert "sr_mach_normal" in shock_props
    assert "sr_sonic_mach" in shock_props
    assert "theta_Bn" in shock_props
    assert "jump_residual_light" in shock_props
    assert shock_props["mask"].any()
    assert shock_props["verified_mask"].any()
    assert shock_props["sampling_stats"]["geom_candidate_count"] >= shock_props["sampling_stats"]["verified_count"] >= shock_props["sampling_stats"]["sr_refined_count"]
    assert "utilde_sq_upstream" in shock_props
    assert "gamma_lorentz_upstream" in shock_props
    assert "utilde_n_upstream" in shock_props
    active = shock_props["mask"]
    assert np.all(shock_props["sr_sonic_mach"][active] > 0)


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
        "q_budget_grid": np.full_like(roi_data["rho"], 3.2),
        "p_eff_grid": np.full_like(roi_data["rho"], 2.2),
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
    assert config["nt_params"]["inj_model"] == "pic_dual_cap"
    assert config["nt_params"]["energy_budget_model"] == "total_internal_energy_excess"
    assert config["nt_params"]["p_eff_model"] == "hybrid_classical_relativistic"
    assert config["nt_params"]["r_high"] == pytest.approx(10.0)
    assert config["nt_params"]["eta_inj_e0"] == pytest.approx(2.0e-1)
    assert config["nt_params"]["eps_nth_e0"] == pytest.approx(2.0e-1)


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


def test_two_temp_chain_uses_upstream_beta_and_sonic_mach():
    mask = np.zeros((1, 1, 1), dtype=bool)
    mask[0, 0, 0] = True
    shock_properties = {
        "mask": mask,
        "mainline_mach": np.full((1, 1, 1), 3.0),
        "rho1_code_grid": np.full((1, 1, 1), 2.0),
        "press1_code_grid": np.full((1, 1, 1), 1.0),
        "beta1_grid": np.full((1, 1, 1), 2.0),
        "sr_sonic_mach": np.full((1, 1, 1), 4.0),
        "rho2_code_grid": np.full((1, 1, 1), 2.0),
        "press2_code_grid": np.full((1, 1, 1), 4.0),
        "sample_boundary_clipped_grid": np.zeros((1, 1, 1), dtype=bool),
    }

    result = calculate_nonthermal_electrons(
        shock_properties,
        rho_unit=2.0,
        u_unit=18.0,
    )

    beta1 = 2.0
    r1 = 80.0 * beta1**2 / (1.0 + beta1**2) + 1.0 / (1.0 + beta1**2)
    theta_e1 = (1.0 / 2.0) * (1.6726e-24 / 9.1094e-28) / (1.0 + r1)
    theta_e2_ad = theta_e1
    theta_e2 = theta_e2_ad * (1.0 + 0.0016 * 4.0**3.6)
    assert np.isclose(result["R1_grid"][0, 0, 0], r1)
    assert np.isclose(result["theta_e1_grid"][0, 0, 0], theta_e1)
    assert np.isclose(result["theta_e2_ad_grid"][0, 0, 0], theta_e2_ad)
    assert np.isclose(result["theta_e_grid"][0, 0, 0], theta_e2)
    assert np.isclose(result["gamma_min_grid"][0, 0, 0], 1.0 + 3.0 * theta_e2)
    assert result["gamma_min_failure_code_grid"][0, 0, 0] == 0
    assert result["C_grid"][0, 0, 0] >= 0.0


def test_relativistic_p_min_matches_low_theta_limit():
    theta_e2 = np.array([1.0e-6, 1.0e-4, 1.0e-3])
    x_inj = 3.5
    p_min = _compute_relativistic_p_min(theta_e2, x_inj)
    low_theta_limit = x_inj * np.sqrt(6.0 * theta_e2)
    assert np.allclose(p_min, low_theta_limit, rtol=5.0e-4, atol=0.0)


def test_relativistic_k_inj_is_positive_and_monotone_in_p_min():
    q = np.array([2.3, 2.3, 2.3])
    p_min = np.array([0.5, 1.0, 2.0])
    k_inj = _compute_relativistic_k_inj(q, p_min)
    assert np.all(np.isfinite(k_inj))
    assert np.all(k_inj > 0.0)
    assert np.all(np.diff(k_inj) > 0.0)


def test_relativistic_k_inj_handles_q_outside_old_beta_domain():
    q = np.array([1.8, 2.2, 3.4])
    p_min = np.array([1.0, 1.0, 1.0])
    k_inj = _compute_relativistic_k_inj(q, p_min)
    assert np.all(np.isfinite(k_inj))
    assert np.all(k_inj > 0.0)

    harder = _compute_relativistic_k_inj(np.array([1.8]), np.array([1.0]))[0]
    softer = _compute_relativistic_k_inj(np.array([3.4]), np.array([1.0]))[0]
    assert harder > softer


def test_pic_dual_cap_injection_quenches_without_irreversible_heating():
    mask = np.ones((1, 1, 1), dtype=bool)
    shock_properties = {
        "mask": mask,
        "mainline_mach": np.full((1, 1, 1), 3.0),
        "rho1_code_grid": np.full((1, 1, 1), 2.0),
        "press1_code_grid": np.full((1, 1, 1), 1.0),
        "beta1_grid": np.full((1, 1, 1), 1.0),
        "sr_sonic_mach": np.full((1, 1, 1), 4.0),
        "theta_Bn": np.zeros((1, 1, 1)),
        "rho2_code_grid": np.full((1, 1, 1), 2.0),
        "press2_code_grid": np.full((1, 1, 1), 1.0),
        "sample_boundary_clipped_grid": np.zeros((1, 1, 1), dtype=bool),
        "sigma_grid": np.zeros((1, 1, 1)),
    }

    result = calculate_nonthermal_electrons(shock_properties)

    assert result["theta_e_grid"][0, 0, 0] == pytest.approx(result["theta_e2_ad_grid"][0, 0, 0])
    assert result["e_diss_e_grid"][0, 0, 0] == pytest.approx(0.0)
    assert result["inj_gate_grid"][0, 0, 0] == pytest.approx(0.0)
    assert result["C_grid"][0, 0, 0] == pytest.approx(0.0)
    assert result["inj_limit_mode_grid"][0, 0, 0] == result["inj_limit_modes"]["quenched_by_gate"]


def test_pic_dual_cap_injection_is_capped_by_dual_constraints():
    mask = np.ones((1, 1, 1), dtype=bool)
    shock_properties = {
        "mask": mask,
        "mainline_mach": np.full((1, 1, 1), 3.0),
        "rho1_code_grid": np.full((1, 1, 1), 2.0),
        "press1_code_grid": np.full((1, 1, 1), 1.0),
        "beta1_grid": np.full((1, 1, 1), 0.2),
        "sr_sonic_mach": np.full((1, 1, 1), 5.0),
        "theta_Bn": np.zeros((1, 1, 1)),
        "rho2_code_grid": np.full((1, 1, 1), 4.0),
        "press2_code_grid": np.full((1, 1, 1), 10.0),
        "sample_boundary_clipped_grid": np.zeros((1, 1, 1), dtype=bool),
        "sigma_grid": np.zeros((1, 1, 1)),
    }

    result = calculate_nonthermal_electrons(shock_properties, eta_inj_e0=1.0e-3, eps_nth_e0=3.0e-3)

    unth_code = result["C_grid"][0, 0, 0]
    n_nth = result["n_nth_phys_grid"][0, 0, 0]
    n_eta = result["n_nth_eta_phys_grid"][0, 0, 0]
    n_eps = result["n_nth_eps_phys_grid"][0, 0, 0]
    assert result["inj_gate_grid"][0, 0, 0] > 0.0
    assert unth_code > 0.0
    assert n_nth > 0.0
    assert n_nth <= n_eta + 1e-30
    assert n_nth <= n_eps + 1e-30
    assert result["inj_limit_mode_grid"][0, 0, 0] in (
        result["inj_limit_modes"]["eta_cap"],
        result["inj_limit_modes"]["eps_cap"],
    )
    assert np.isfinite(result["p_min_physical_grid"][0, 0, 0])
    assert result["p_min_physical_grid"][0, 0, 0] > 0.0
    assert np.isfinite(result["c_eps_grid"][0, 0, 0])


def test_total_internal_energy_budget_is_exported_in_default_chain():
    mask = np.ones((1, 1, 1), dtype=bool)
    shock_properties = {
        "mask": mask,
        "mainline_mach": np.full((1, 1, 1), 3.0),
        "rho1_code_grid": np.full((1, 1, 1), 2.0),
        "press1_code_grid": np.full((1, 1, 1), 1.0),
        "beta1_grid": np.full((1, 1, 1), 0.2),
        "sr_sonic_mach": np.full((1, 1, 1), 5.0),
        "theta_Bn": np.zeros((1, 1, 1)),
        "rho2_code_grid": np.full((1, 1, 1), 4.0),
        "press2_code_grid": np.full((1, 1, 1), 10.0),
        "sample_boundary_clipped_grid": np.zeros((1, 1, 1), dtype=bool),
        "sigma_grid": np.zeros((1, 1, 1)),
    }

    result = calculate_nonthermal_electrons(shock_properties, rho_unit=2.0, u_unit=18.0)
    assert result["e_diss_tot_grid"][0, 0, 0] > 0.0
    assert result["u_nth_budget_grid"][0, 0, 0] > 0.0
    assert result["energy_budget_model"] == "total_internal_energy_excess"


def test_energy_budget_model_changes_energy_limited_branch():
    mask = np.ones((1, 1, 1), dtype=bool)
    shock_properties = {
        "mask": mask,
        "mainline_mach": np.full((1, 1, 1), 3.0),
        "rho1_code_grid": np.full((1, 1, 1), 2.0),
        "press1_code_grid": np.full((1, 1, 1), 1.0),
        "beta1_grid": np.full((1, 1, 1), 0.2),
        "sr_sonic_mach": np.full((1, 1, 1), 5.0),
        "theta_Bn": np.zeros((1, 1, 1)),
        "rho2_code_grid": np.full((1, 1, 1), 4.0),
        "press2_code_grid": np.full((1, 1, 1), 10.0),
        "sample_boundary_clipped_grid": np.zeros((1, 1, 1), dtype=bool),
        "sigma_grid": np.zeros((1, 1, 1)),
    }

    result_total = calculate_nonthermal_electrons(
        shock_properties,
        rho_unit=2.0,
        u_unit=18.0,
        energy_budget_model="total_internal_energy_excess",
        eta_inj_e0=1.0,
        eps_nth_e0=3.0e-3,
    )
    result_electron = calculate_nonthermal_electrons(
        shock_properties,
        rho_unit=2.0,
        u_unit=18.0,
        energy_budget_model="electron_thermal_excess",
        eta_inj_e0=1.0,
        eps_nth_e0=3.0e-3,
    )

    assert result_total["u_nth_budget_grid"][0, 0, 0] != pytest.approx(result_electron["u_nth_budget_grid"][0, 0, 0])
    assert result_total["n_nth_eps_phys_grid"][0, 0, 0] != pytest.approx(result_electron["n_nth_eps_phys_grid"][0, 0, 0])
    assert result_total["n_nth_phys_grid"][0, 0, 0] != pytest.approx(result_electron["n_nth_phys_grid"][0, 0, 0])


def test_nonthermal_rejects_unknown_energy_budget_model():
    mask = np.ones((1, 1, 1), dtype=bool)
    shock_properties = {
        "mask": mask,
        "mainline_mach": np.full((1, 1, 1), 3.0),
        "rho1_code_grid": np.full((1, 1, 1), 2.0),
        "press1_code_grid": np.full((1, 1, 1), 1.0),
        "beta1_grid": np.full((1, 1, 1), 1.0),
        "sr_sonic_mach": np.full((1, 1, 1), 3.0),
        "rho2_code_grid": np.full((1, 1, 1), 2.0),
        "press2_code_grid": np.full((1, 1, 1), 4.0),
        "sample_boundary_clipped_grid": np.zeros((1, 1, 1), dtype=bool),
    }

    with pytest.raises(ValueError, match="Unsupported energy_budget_model"):
        calculate_nonthermal_electrons(
            shock_properties,
            energy_budget_model="invalid_budget",
        )


def test_relativistic_k_inj_matches_python_closure_cutoff_assumption():
    q = np.array([2.2])
    p_min = np.array([1.0])
    k_inj = _compute_relativistic_k_inj(q, p_min, gamma_max=POWERLAW_GAMMA_MAX_DEFAULT)[0]
    k_inj_lower_cutoff = _compute_relativistic_k_inj(q, p_min, gamma_max=1.0e3)[0]
    assert k_inj > 0.0
    assert k_inj >= k_inj_lower_cutoff


def test_effective_powerlaw_index_uses_classical_branch_for_weak_shocks():
    p_eff = _compute_effective_powerlaw_index(
        mainline_mach=np.array([1.5]),
        p_classical=np.array([2.4]),
        theta_bn=np.array([0.1]),
        sigma=np.array([1.0e-4]),
        classical_fast_mach_max=1.8,
        relativistic_fast_mach_min=3.0,
        theta_bn_parallel_max_deg=35.0,
        theta_bn_oblique_max_deg=60.0,
        sigma_rel_parallel_max=1.0e-3,
        sigma_rel_oblique_max=1.0e-2,
        p_eff_parallel=2.35,
        p_eff_oblique=2.8,
        p_eff_steep=3.5,
        p_eff_floor=1.5,
        p_eff_ceiling=4.5,
    )
    assert p_eff[0] == pytest.approx(2.4)


def test_effective_powerlaw_index_uses_relativistic_classification_for_strong_shocks():
    p_eff = _compute_effective_powerlaw_index(
        mainline_mach=np.array([4.0, 4.0, 4.0]),
        p_classical=np.array([0.8, 0.8, 0.8]),
        theta_bn=np.array([np.deg2rad(20.0), np.deg2rad(50.0), np.deg2rad(80.0)]),
        sigma=np.array([1.0e-4, 5.0e-3, 5.0e-2]),
        classical_fast_mach_max=1.8,
        relativistic_fast_mach_min=3.0,
        theta_bn_parallel_max_deg=35.0,
        theta_bn_oblique_max_deg=60.0,
        sigma_rel_parallel_max=1.0e-3,
        sigma_rel_oblique_max=1.0e-2,
        p_eff_parallel=2.35,
        p_eff_oblique=2.8,
        p_eff_steep=3.5,
        p_eff_floor=1.5,
        p_eff_ceiling=4.5,
    )
    assert p_eff[0] == pytest.approx(2.35)
    assert p_eff[1] == pytest.approx(2.8)
    assert p_eff[2] == pytest.approx(3.5)


def test_nonthermal_rejects_removed_mask_switch_control():
    mask = np.ones((1, 1, 1), dtype=bool)
    shock_properties = {
        "mask": mask,
        "mainline_mach": np.full((1, 1, 1), 3.0),
        "rho1_code_grid": np.full((1, 1, 1), 2.0),
        "press1_code_grid": np.full((1, 1, 1), 1.0),
        "beta1_grid": np.full((1, 1, 1), 1.0),
        "sr_sonic_mach": np.full((1, 1, 1), 3.0),
        "rho2_code_grid": np.full((1, 1, 1), 2.0),
        "press2_code_grid": np.full((1, 1, 1), 4.0),
        "sample_boundary_clipped_grid": np.zeros((1, 1, 1), dtype=bool),
    }

    with pytest.raises(ValueError, match="use_sr_refined_mask has been removed"):
        calculate_nonthermal_electrons(
            shock_properties,
            use_sr_refined_mask=True,
        )


def test_nonthermal_returns_effective_powerlaw_grid():
    mask = np.ones((1, 1, 1), dtype=bool)
    shock_properties = {
        "mask": mask,
        "mainline_mach": np.full((1, 1, 1), 4.0),
        "rho1_code_grid": np.full((1, 1, 1), 1.0),
        "press1_code_grid": np.full((1, 1, 1), 1.0e-4),
        "beta1_grid": np.full((1, 1, 1), 0.2),
        "sr_sonic_mach": np.full((1, 1, 1), 5.0),
        "theta_Bn": np.full((1, 1, 1), np.deg2rad(20.0)),
        "rho2_code_grid": np.full((1, 1, 1), 4.0),
        "press2_code_grid": np.full((1, 1, 1), 0.1),
        "sample_boundary_clipped_grid": np.zeros((1, 1, 1), dtype=bool),
        "sigma_grid": np.full((1, 1, 1), 1.0e-4),
    }

    result = calculate_nonthermal_electrons(shock_properties, rho_unit=2.0, u_unit=18.0)
    assert result["p_eff_grid"][0, 0, 0] == pytest.approx(2.35)
    assert result["q_budget_grid"][0, 0, 0] == pytest.approx(3.35)


@pytest.mark.skipif(h5py is None, reason="h5py not installed")
def test_save_h5_uses_gamma_min_grid_without_overwriting_valid_values():
    roi_data = _build_minimal_roi()
    mask = np.zeros_like(roi_data["rho"], dtype=bool)
    mask[0, 1, 3] = True
    shock_props = {
        "mask": mask,
        "mainline_mach": np.ones_like(roi_data["rho"]),
        "verified_mask": mask.copy(),
        "sr_mach_normal": np.ones_like(roi_data["rho"]) * 1.5,
        "sr_sonic_mach": np.ones_like(roi_data["rho"]) * 2.5,
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
        "theta_e_grid": np.ones_like(roi_data["rho"]) * 0.1,
        "theta_e1_grid": np.ones_like(roi_data["rho"]) * 0.05,
        "theta_e2_ad_grid": np.ones_like(roi_data["rho"]) * 0.08,
        "sironi_boost_grid": np.ones_like(roi_data["rho"]) * 1.2,
        "R1_grid": np.ones_like(roi_data["rho"]) * 10.0,
        "Te2_grid": np.ones_like(roi_data["rho"]) * 1.0e8,
        "eta_inj_e_grid": np.ones_like(roi_data["rho"]) * 1.0e-3,
        "eps_nth_e_grid": np.ones_like(roi_data["rho"]) * 3.0e-3,
        "e_diss_e_grid": np.ones_like(roi_data["rho"]) * 1.0e-6,
        "inj_gate_grid": np.ones_like(roi_data["rho"]) * 0.5,
        "n_nth_eta_phys_grid": np.ones_like(roi_data["rho"]) * 2.0e-4,
        "n_nth_eps_phys_grid": np.ones_like(roi_data["rho"]) * 1.0e-4,
        "n_nth_phys_grid": np.ones_like(roi_data["rho"]) * 1.0e-4,
        "spectral_norm_eta_grid": np.ones_like(roi_data["rho"]) * 1.0e-7,
        "spectral_norm_eps_grid": np.ones_like(roi_data["rho"]) * 5.0e-8,
        "c_eta_grid": np.ones_like(roi_data["rho"]) * 1.0e-7,
        "c_eps_grid": np.ones_like(roi_data["rho"]) * 5.0e-8,
        "inj_limit_mode_grid": np.ones_like(roi_data["rho"], dtype=np.int16),
        "energy_budget_model": "total_internal_energy_excess",
        "p_eff_model": "hybrid_classical_relativistic",
        "inj_limit_modes": {"none": 0, "eta_cap": 1, "eps_cap": 2, "quenched_by_gate": 3, "invalid_or_boundary": 4},
    }
    nonthermal_props["gamma_min_grid"][0, 1, 3] = 4.2
    nonthermal_props["gamma_min_grid_physical"][0, 1, 3] = 4.2

    config = {
        "shock_params": {"gamma": 4.0 / 3.0},
        "physics": {"spin": 0.98, "hslope": 0.3, "R0": 0.0},
        "hdf5_options": {"include_3d_diagnostics": True},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        output_h5 = Path(tmpdir) / "test.h5"
        save_h5_file(str(output_h5), roi_data, shock_props, nonthermal_props, config)
        with h5py.File(output_h5, "r") as handle:
            gamma_min = handle["GAMMA_MIN"][:]
            assert "SR_MACH_NORMAL" in handle
            assert "SONIC_MACH" in handle
            assert "THETA_BN" in handle
            assert "H_REL_UPSTREAM" in handle
            assert "CFAST_N_UPSTREAM" in handle
            assert "JUMP_RESIDUAL_LIGHT" in handle
            assert "PTOT_JUMP" in handle
            assert "ENTROPY_JUMP" in handle
            assert "UTILDE_SQ_UPSTREAM" in handle
            assert "GAMMA_LORENTZ_UPSTREAM" in handle
            assert "UTILDE_N_UPSTREAM" in handle
            assert "THETA_E1" in handle
            assert "THETA_E2_AD" in handle
            assert "THETA_E" in handle
            assert "SIRONI_BOOST" in handle
            assert "R1" in handle
            assert "TE2" in handle
            assert "ETA_INJ_E" in handle
            assert "EPS_NTH_E" in handle
            assert "E_DISS_E" in handle
            assert "E_DISS_TOT" in handle
            assert "U_NTH_BUDGET" in handle
            assert "INJ_GATE" in handle
            assert "P_EFF" in handle
            assert "Q_BUDGET" in handle
            assert "N_NTH_ETA" in handle
            assert "N_NTH_EPS" in handle
            assert "N_NTH" in handle
            assert "PL_NORM_ETA" in handle
            assert "PL_NORM_EPS" in handle
            assert "C_ETA" in handle
            assert "C_EPS" in handle
            assert "INJ_LIMIT_MODE" in handle
            assert np.isclose(handle["UTILDE_SQ_UPSTREAM"][3, 1, 0], 3.0)
            assert np.isclose(handle["GAMMA_LORENTZ_UPSTREAM"][3, 1, 0], 2.0)
            assert np.isclose(handle["UTILDE_N_UPSTREAM"][3, 1, 0], 1.5)
            assert np.isclose(handle["SONIC_MACH"][3, 1, 0], 2.5)

    assert np.isclose(gamma_min[3, 1, 0], 4.2)


def test_nonthermal_rejects_removed_beta_closure_controls():
    mask = np.ones((1, 1, 1), dtype=bool)
    shock_properties = {
        "mask": mask,
        "mainline_mach": np.full((1, 1, 1), 3.0),
        "rho1_code_grid": np.full((1, 1, 1), 2.0),
        "press1_code_grid": np.full((1, 1, 1), 1.0),
        "beta1_grid": np.full((1, 1, 1), 1.0),
        "sr_sonic_mach": np.full((1, 1, 1), 3.0),
        "rho2_code_grid": np.full((1, 1, 1), 2.0),
        "press2_code_grid": np.full((1, 1, 1), 4.0),
        "sample_boundary_clipped_grid": np.zeros((1, 1, 1), dtype=bool),
    }

    with pytest.raises(ValueError, match="Removed beta-closure controls"):
        calculate_nonthermal_electrons(
            shock_properties,
            r_low=1.0,
        )
