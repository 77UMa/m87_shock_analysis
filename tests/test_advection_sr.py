import numpy as np

from run_dsa_pipeline import create_default_config
from src.core.advection_v0 import solve_steady_advection


def _build_roi(shape=(1, 2, 4), radial_velocity=0.0, magnetic_field=0.0):
    nk, nj, ni = shape
    zeros = np.zeros(shape, dtype=float)
    vel1 = np.full(shape, radial_velocity, dtype=float)
    bcc1 = np.full(shape, magnetic_field, dtype=float)
    return {
        "rho": np.ones(shape, dtype=float),
        "press": np.ones(shape, dtype=float),
        "vel1": vel1,
        "vel2": zeros.copy(),
        "vel3": zeros.copy(),
        "Bcc1": bcc1,
        "Bcc2": zeros.copy(),
        "Bcc3": zeros.copy(),
        "x1f": np.linspace(10.0, 14.0, ni + 1),
        "x2f": np.linspace(0.3, 1.3, nj + 1),
        "x3f": np.linspace(0.0, 2.0 * np.pi, nk + 1),
        "x1v": np.linspace(10.5, 13.5, ni),
        "x2v": np.linspace(0.55, 1.05, nj),
        "x3v": np.linspace(np.pi, np.pi, nk),
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


def test_default_config_enables_relativistic_advection_model_selection():
    config = create_default_config()
    assert config["physics"]["advection_model"] == "sr_radial"
    assert config["physics"]["advection_cooling_model"] == "synchrotron_local_sink"
    assert config["physics"]["advection_line_sweeps"] >= 1


def test_sr_advection_reduces_to_local_source_without_transport_or_cooling():
    roi_data = _build_roi(radial_velocity=0.0, magnetic_field=0.0)
    source = np.array([[[1.0, 0.5, 2.0, 0.0], [0.2, 0.1, 0.3, 0.4]]], dtype=float)
    nonthermal_props = _build_nonthermal(source.shape, source)
    config = create_default_config()
    config["physics"]["enable_advection"] = True
    config["physics"]["cooling_factor"] = 1.0e30

    result = solve_steady_advection(roi_data, nonthermal_props, config)

    assert np.allclose(result["unth_code_grid"], source)
    assert np.allclose(result["C_grid"], source)
    assert np.allclose(result["q_grid"], nonthermal_props["q_grid"])
    assert np.allclose(result["gamma_min_grid"], nonthermal_props["gamma_min_grid"])


def test_sr_advection_cooling_is_monotone_in_gamma_min():
    shape = (1, 1, 4)
    roi_data = _build_roi(shape=shape, radial_velocity=0.0, magnetic_field=1.0)
    density = np.ones(shape, dtype=float)
    nonthermal_props = _build_nonthermal(shape, density)
    nonthermal_props["gamma_min_grid"][0, 0, :] = np.array([2.0, 4.0, 8.0, 16.0])
    config = create_default_config()
    config["physics"]["enable_advection"] = True
    config["physics"]["cooling_factor"] = 0.1

    result = solve_steady_advection(roi_data, nonthermal_props, config)
    evolved = result["unth_code_grid"][0, 0, :]

    assert np.all(evolved > 0.0)
    assert np.all(np.diff(evolved) < 0.0)
    assert np.all(evolved < density[0, 0, :])


def test_sr_advection_updates_density_but_preserves_spectral_fields():
    shape = (1, 1, 4)
    roi_data = _build_roi(shape=shape, radial_velocity=0.5, magnetic_field=0.0)
    density = np.array([[[1.0, 0.0, 0.0, 0.0]]], dtype=float)
    nonthermal_props = _build_nonthermal(shape, density, q=2.7)
    nonthermal_props["gamma_min_grid"][0, 0, :] = np.array([2.0, 3.0, 4.0, 5.0])
    config = create_default_config()
    config["physics"]["enable_advection"] = True
    config["physics"]["cooling_factor"] = 1.0e30
    config["physics"]["advection_line_sweeps"] = 8

    result = solve_steady_advection(roi_data, nonthermal_props, config)

    assert not np.allclose(result["unth_code_grid"], density)
    assert np.allclose(result["unth_code_grid"], result["C_grid"])
    assert np.allclose(result["q_grid"], nonthermal_props["q_grid"])
    assert np.allclose(result["p_eff_grid"], nonthermal_props["p_eff_grid"])
    assert np.allclose(result["gamma_min_grid"], nonthermal_props["gamma_min_grid"])
    assert np.all(result["n_nth_phys_grid"] >= 0.0)


def test_sr_advection_handles_inward_radial_flow_without_negative_densities():
    shape = (1, 1, 5)
    roi_data = _build_roi(shape=shape, radial_velocity=-0.4, magnetic_field=0.2)
    density = np.array([[[0.0, 0.0, 0.2, 0.5, 1.0]]], dtype=float)
    nonthermal_props = _build_nonthermal(shape, density)
    config = create_default_config()
    config["physics"]["enable_advection"] = True
    config["physics"]["cooling_factor"] = 10.0
    config["physics"]["advection_line_sweeps"] = 8

    result = solve_steady_advection(roi_data, nonthermal_props, config)

    assert np.all(np.isfinite(result["unth_code_grid"]))
    assert np.all(result["unth_code_grid"] >= 0.0)
    assert np.all(result["n_nth_phys_grid"] >= 0.0)
