import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from run_dsa_pipeline import create_default_config
from src.core.advection_v0 import (
    _build_shock_local_active_region,
    _compute_spherical_cell_volumes,
    solve_steady_advection,
)


def _build_roi(shape=(3, 3, 6), radial_velocity=0.0, magnetic_field=0.0):
    nk, nj, ni = shape
    zeros = np.zeros(shape, dtype=float)
    return {
        "rho": np.ones(shape, dtype=float),
        "press": np.ones(shape, dtype=float),
        "vel1": np.full(shape, radial_velocity, dtype=float),
        "vel2": zeros.copy(),
        "vel3": zeros.copy(),
        "Bcc1": np.full(shape, magnetic_field, dtype=float),
        "Bcc2": zeros.copy(),
        "Bcc3": zeros.copy(),
        "x1f": np.linspace(10.0, 16.0, ni + 1),
        "x2f": np.linspace(0.3, 1.5, nj + 1),
        "x3f": np.linspace(0.0, 2.0 * np.pi, nk + 1),
        "x1v": np.linspace(10.5, 15.5, ni),
        "x2v": np.linspace(0.5, 1.3, nj),
        "x3v": np.linspace(0.5, 5.5, nk),
        "Time": 0.0,
    }


def _build_nonthermal(shape, density, q=2.5):
    density = np.asarray(density, dtype=float).reshape(shape)
    return {
        "unth_code_grid": density.copy(),
        "C_grid": density.copy(),
        "q_grid": np.full(shape, q, dtype=float),
        "p_eff_grid": np.full(shape, q - 1.0, dtype=float),
        "gamma_min_grid": np.ones(shape, dtype=float) * 2.0,
        "gamma_min_failure_code_grid": np.zeros(shape, dtype=np.int16),
        "n_nth_phys_grid": density.copy() * 10.0,
        "mask": density > 0.0,
    }


def _build_shock_props(shape, shock_cells):
    mask = np.zeros(shape, dtype=bool)
    sample_k2 = np.full(shape, -1, dtype=int)
    sample_j2 = np.full(shape, -1, dtype=int)
    sample_i2 = np.full(shape, -1, dtype=int)
    for k, j, i, kd, jd, id_ in shock_cells:
        mask[k, j, i] = True
        sample_k2[k, j, i] = kd
        sample_j2[k, j, i] = jd
        sample_i2[k, j, i] = id_
    return {
        "mask": mask,
        "sample_k2_grid": sample_k2,
        "sample_j2_grid": sample_j2,
        "sample_i2_grid": sample_i2,
    }


def test_default_config_uses_shock_local_relaxation_defaults():
    config = create_default_config()

    assert config["physics"]["advection_model"] == "sr_radial"
    assert config["physics"]["advection_domain"] == "shock_local"
    assert config["physics"]["advection_seed_mode"] == "downstream_sample"
    assert config["physics"]["advection_injection_layer"] == "downstream_sample"
    assert config["physics"]["advection_tau_inj_cell_crossing_fraction"] == pytest.approx(1.0e-3)
    assert config["physics"]["advection_shock_pad_r"] >= 0
    assert "advection_seed_kernel_r" not in config["physics"]
    assert "advection_seed_merge_fraction" not in config["physics"]


def test_source_mapping_preserves_volume_weighted_unth_budget():
    shape = (3, 3, 6)
    roi_data = _build_roi(shape=shape, radial_velocity=0.5, magnetic_field=0.0)
    density = np.zeros(shape, dtype=float)
    density[0, 1, 1] = 5.0
    density[0, 1, 2] = 4.0
    shock_props = _build_shock_props(
        shape,
        [
            (0, 1, 1, 1, 1, 3),
            (0, 1, 2, 1, 1, 3),
        ],
    )
    config = create_default_config()

    active_mask, source_mask, source_density, bbox, stats = _build_shock_local_active_region(
        shock_props,
        density,
        config["physics"],
        roi_data=roi_data,
    )
    volumes = _compute_spherical_cell_volumes(roi_data, shape)
    expected_budget = density[0, 1, 1] * volumes[0, 1, 1] + density[0, 1, 2] * volumes[0, 1, 2]

    assert stats["seed_samples_valid"] == 2
    assert stats["source_cells_unique"] == 1
    assert source_mask[1, 1, 3]
    assert np.sum(source_density * volumes) == pytest.approx(expected_budget)
    assert stats["source_budget_relative_error"] == pytest.approx(0.0)
    assert np.any(active_mask)
    assert bbox["i_max"] >= bbox["i_min"]


def test_shock_local_advection_without_shock_sources_stays_zero():
    shape = (3, 3, 6)
    roi_data = _build_roi(shape=shape, radial_velocity=0.4, magnetic_field=0.0)
    nonthermal_props = _build_nonthermal(shape, np.zeros(shape, dtype=float))
    shock_props = _build_shock_props(shape, [])
    config = create_default_config()
    config["physics"]["cooling_factor"] = 1.0e30

    result = solve_steady_advection(roi_data, shock_props, nonthermal_props, config)

    assert np.allclose(result["unth_code_grid"], 0.0)
    assert np.allclose(result["n_nth_phys_grid"], 0.0)


def test_stiff_relaxation_source_approaches_equilibrium_without_dirichlet_floor():
    shape = (3, 3, 6)
    roi_data = _build_roi(shape=shape, radial_velocity=0.5, magnetic_field=0.0)
    density = np.zeros(shape, dtype=float)
    density[0, 0, 1] = 4.0
    nonthermal_props = _build_nonthermal(shape, density, q=2.7)
    shock_props = _build_shock_props(shape, [(0, 0, 1, 1, 1, 2)])
    config = create_default_config()
    config["physics"]["cooling_factor"] = 1.0e30
    config["physics"]["advection_tau_inj_cell_crossing_fraction"] = 1.0e-6
    config["physics"]["advection_line_sweeps"] = 8
    config["physics"]["advection_shock_pad_r"] = 2
    config["physics"]["advection_shock_pad_theta"] = 1
    config["physics"]["advection_shock_pad_phi"] = 1

    active_mask, source_mask, source_density, _, _ = _build_shock_local_active_region(
        shock_props,
        density,
        config["physics"],
        roi_data=roi_data,
    )
    result = solve_steady_advection(roi_data, shock_props, nonthermal_props, config)
    evolved = result["unth_code_grid"]

    assert active_mask[1, 1, 2]
    assert source_mask[1, 1, 2]
    assert evolved[1, 1, 2] == pytest.approx(source_density[1, 1, 2], rel=1.0e-4)
    assert np.sum(evolved > 0.0) > 1
    assert np.allclose(evolved[:, :, 0], 0.0)
    assert np.allclose(result["q_grid"], nonthermal_props["q_grid"])
    assert np.allclose(result["p_eff_grid"], nonthermal_props["p_eff_grid"])
    assert np.allclose(result["gamma_min_grid"], nonthermal_props["gamma_min_grid"])


def test_physical_density_is_mapped_with_the_same_source_budget():
    shape = (3, 3, 6)
    roi_data = _build_roi(shape=shape, radial_velocity=0.5, magnetic_field=0.0)
    density = np.zeros(shape, dtype=float)
    density[0, 0, 1] = 2.0
    nonthermal_props = _build_nonthermal(shape, density)
    shock_props = _build_shock_props(shape, [(0, 0, 1, 1, 1, 2)])
    config = create_default_config()
    config["physics"]["cooling_factor"] = 1.0e30
    config["physics"]["advection_tau_inj_cell_crossing_fraction"] = 1.0e-6
    config["physics"]["advection_shock_pad_r"] = 1
    config["physics"]["advection_shock_pad_theta"] = 0
    config["physics"]["advection_shock_pad_phi"] = 0

    result = solve_steady_advection(roi_data, shock_props, nonthermal_props, config)

    assert result["unth_code_grid"][1, 1, 2] > 0.0
    assert result["n_nth_phys_grid"][1, 1, 2] == pytest.approx(
        10.0 * result["unth_code_grid"][1, 1, 2]
    )


def test_shock_local_cooling_suppresses_solution_relative_to_no_cooling():
    shape = (3, 3, 6)
    roi_data = _build_roi(shape=shape, radial_velocity=0.5, magnetic_field=1.0)
    density = np.zeros(shape, dtype=float)
    density[0, 0, 1] = 3.0
    nonthermal_props = _build_nonthermal(shape, density)
    shock_props = _build_shock_props(shape, [(0, 0, 1, 1, 1, 2)])

    config_hot = create_default_config()
    config_hot["physics"]["cooling_factor"] = 1.0e30
    config_hot["physics"]["advection_shock_pad_r"] = 2
    config_hot["physics"]["advection_shock_pad_theta"] = 1
    config_hot["physics"]["advection_shock_pad_phi"] = 1

    config_cool = create_default_config()
    config_cool["physics"]["cooling_factor"] = 0.1
    config_cool["physics"]["advection_shock_pad_r"] = 2
    config_cool["physics"]["advection_shock_pad_theta"] = 1
    config_cool["physics"]["advection_shock_pad_phi"] = 1

    no_cooling = solve_steady_advection(roi_data, shock_props, nonthermal_props, config_hot)
    cooling = solve_steady_advection(roi_data, shock_props, nonthermal_props, config_cool)

    assert np.sum(cooling["unth_code_grid"]) < np.sum(no_cooling["unth_code_grid"])


def test_invalid_tau_inj_fraction_fails_loudly():
    shape = (3, 3, 6)
    roi_data = _build_roi(shape=shape, radial_velocity=0.5, magnetic_field=0.0)
    density = np.zeros(shape, dtype=float)
    density[0, 0, 1] = 1.0
    nonthermal_props = _build_nonthermal(shape, density)
    shock_props = _build_shock_props(shape, [(0, 0, 1, 1, 1, 2)])
    config = create_default_config()
    config["physics"]["advection_tau_inj_cell_crossing_fraction"] = 0.0

    with pytest.raises(ValueError, match="advection_tau_inj_cell_crossing_fraction"):
        solve_steady_advection(roi_data, shock_props, nonthermal_props, config)


def test_invalid_downstream_sample_is_rejected_without_creating_source():
    shape = (3, 3, 6)
    roi_data = _build_roi(shape=shape, radial_velocity=0.5, magnetic_field=0.0)
    density = np.zeros(shape, dtype=float)
    density[0, 0, 1] = 1.0
    shock_props = _build_shock_props(shape, [(0, 0, 1, 9, 1, 2)])
    config = create_default_config()

    _, source_mask, source_density, _, stats = _build_shock_local_active_region(
        shock_props,
        density,
        config["physics"],
        roi_data=roi_data,
    )

    assert stats["seed_samples_rejected_invalid"] == 1
    assert not np.any(source_mask)
    assert np.allclose(source_density, 0.0)
