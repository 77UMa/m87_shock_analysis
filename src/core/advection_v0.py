import copy
import traceback

import numpy as np

try:
    from numba import njit, prange
except ModuleNotFoundError:
    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    prange = range

from src.core.shock_v1 import compute_comoving_magnetic_geometry


_TINY = 1.0e-30
_M_E = 9.1094e-28
_M_P = 1.6726e-24
_M_SUN = 1.989e33
_G_CGS = 6.674e-8
_C_LIGHT = 2.9979e10


def _format_stat_triplet(values):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "min=nan, median=nan, max=nan"
    return (
        f"min={np.min(finite):.3e}, "
        f"median={np.median(finite):.3e}, "
        f"max={np.max(finite):.3e}"
    )


def _get_physics_cfg(config):
    return config.get("physics", {})


def _log_advection_debug(logger, message):
    print(message)
    if logger:
        logger.ai.debug(message)


def _log_advection_codepath(logger, branch, details=""):
    if logger:
        logger.ai.codepath(branch, details)


def _log_advection_data(logger, name, value, unit=""):
    if logger:
        logger.ai.data(name, value, unit)


def _extract_nonthermal_density(nonthermal_props):
    if "unth_code_grid" in nonthermal_props:
        return np.ascontiguousarray(nonthermal_props["unth_code_grid"], dtype=np.float64)
    return np.ascontiguousarray(nonthermal_props["C_grid"], dtype=np.float64)


@njit(fastmath=True)
def estimate_cooling_time_jit(b_sq_comoving, gamma_char, cooling_factor):
    b_sq_safe = np.maximum(b_sq_comoving, 1.0e-12)
    gamma_safe = np.maximum(gamma_char, 1.0 + 1.0e-6)
    return cooling_factor / (b_sq_safe * gamma_safe)


def _rescale_physical_density(evolved_props, initial_code_density, updated_code_density):
    if "n_nth_phys_grid" not in evolved_props:
        return

    n_nth_phys = np.ascontiguousarray(evolved_props["n_nth_phys_grid"], dtype=np.float64)
    positive_mask = initial_code_density > 0.0
    factor_grid = np.zeros_like(updated_code_density)
    np.divide(n_nth_phys, initial_code_density, out=factor_grid, where=positive_mask)
    updated_phys = np.zeros_like(updated_code_density)
    updated_phys[positive_mask] = updated_code_density[positive_mask] * factor_grid[positive_mask]
    evolved_props["n_nth_phys_grid"] = updated_phys


def _compute_code_density_summary(values):
    finite = np.asarray(values[np.isfinite(values)], dtype=np.float64)
    if finite.size == 0:
        return {
            "sum": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }
    return {
        "sum": float(np.sum(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "p95": float(np.percentile(finite, 95.0)),
        "p99": float(np.percentile(finite, 99.0)),
    }


def _compute_nnth_ratio_summary(roi_data, evolved_props, config):
    if "n_nth_phys_grid" not in evolved_props:
        return None

    physics_cfg = _get_physics_cfg(config)
    m_unit = float(physics_cfg.get("M_unit", 1.0e25))
    mbh_solar = float(physics_cfg.get("MBH_solar", 6.2e9))
    l_unit = _G_CGS * (mbh_solar * _M_SUN) / (_C_LIGHT**2)
    rho_unit = m_unit / (l_unit**3)
    particle_mass = _M_P + _M_E
    n_e_phys = np.divide(
        np.ascontiguousarray(roi_data["rho"], dtype=np.float64) * rho_unit,
        particle_mass,
        out=np.zeros_like(roi_data["rho"], dtype=np.float64),
        where=roi_data["rho"] > 0.0,
    )
    ratio = np.divide(
        np.ascontiguousarray(evolved_props["n_nth_phys_grid"], dtype=np.float64),
        n_e_phys,
        out=np.zeros_like(n_e_phys, dtype=np.float64),
        where=n_e_phys > 0.0,
    )
    finite = ratio[np.isfinite(ratio)]
    if finite.size == 0:
        return None
    return {
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95.0)),
        "p99": float(np.percentile(finite, 99.0)),
        "max": float(np.max(finite)),
    }


def _compute_advection_coverage_diagnostics(
    updated_density,
    active_mask,
    seed_mask,
    shock_props,
    n_init,
    b_sq_comoving,
    p_eff_grid,
):
    shock_mask = np.ascontiguousarray(shock_props.get("mask", np.zeros_like(updated_density, dtype=bool)), dtype=bool)
    total_shocks = int(np.count_nonzero(shock_mask))
    total_active = int(np.count_nonzero(active_mask))
    total_seed = int(np.count_nonzero(seed_mask))
    if updated_density.size == 0:
        return {
            "shock_unth_nonzero_ratio": 0.0,
            "seed_unth_nonzero_ratio": 0.0,
            "active_unth_nonzero_ratio": 0.0,
            "emissivity_weighted_coverage": 0.0,
        }

    threshold = max(float(np.max(updated_density)) * 1.0e-12, 1.0e-30)
    nonzero_mask = updated_density > threshold

    shock_unth_nonzero_ratio = (
        float(np.count_nonzero(nonzero_mask & shock_mask) / total_shocks) if total_shocks else 0.0
    )
    seed_unth_nonzero_ratio = (
        float(np.count_nonzero(nonzero_mask & seed_mask) / total_seed) if total_seed else 0.0
    )
    active_unth_nonzero_ratio = (
        float(np.count_nonzero(nonzero_mask & active_mask) / total_active) if total_active else 0.0
    )

    p_eff = np.ascontiguousarray(p_eff_grid, dtype=np.float64)
    emissivity_proxy = np.zeros_like(updated_density, dtype=np.float64)
    power = 0.25 * (np.maximum(p_eff, 1.0) + 1.0)
    valid_weight_mask = shock_mask & np.isfinite(n_init) & (n_init > 0.0)
    emissivity_proxy[valid_weight_mask] = (
        n_init[valid_weight_mask]
        * np.power(np.maximum(b_sq_comoving[valid_weight_mask], 1.0e-30), power[valid_weight_mask])
    )
    total_weight = float(np.sum(emissivity_proxy))
    covered_weight = float(np.sum(emissivity_proxy[nonzero_mask]))
    emissivity_weighted_coverage = covered_weight / total_weight if total_weight > 0.0 else 0.0

    return {
        "shock_unth_nonzero_ratio": shock_unth_nonzero_ratio,
        "seed_unth_nonzero_ratio": seed_unth_nonzero_ratio,
        "active_unth_nonzero_ratio": active_unth_nonzero_ratio,
        "emissivity_weighted_coverage": emissivity_weighted_coverage,
    }


def _compute_spherical_cell_volumes(roi_data, shape):
    r_face = np.ascontiguousarray(roi_data["x1f"], dtype=np.float64)
    theta_face = np.ascontiguousarray(roi_data["x2f"], dtype=np.float64)
    phi_face = np.ascontiguousarray(roi_data["x3f"], dtype=np.float64)
    radial_volume = (r_face[1:] ** 3 - r_face[:-1] ** 3) / 3.0
    polar_volume = np.cos(theta_face[:-1]) - np.cos(theta_face[1:])
    azimuthal_volume = phi_face[1:] - phi_face[:-1]
    volumes = (
        azimuthal_volume[:, np.newaxis, np.newaxis]
        * polar_volume[np.newaxis, :, np.newaxis]
        * radial_volume[np.newaxis, np.newaxis, :]
    )
    if volumes.shape != shape:
        raise ValueError(f"Cell-volume shape {volumes.shape} does not match density shape {shape}")
    if not np.all(np.isfinite(volumes)) or np.any(volumes <= 0.0):
        raise ValueError("Invalid spherical cell volumes: all volumes must be finite and positive")
    return np.ascontiguousarray(volumes, dtype=np.float64)


def _build_shock_local_active_region(shock_props, n_inj, physics_cfg, roi_data=None):
    if roi_data is None:
        raise ValueError("roi_data is required for budget-preserving shock-local source construction")
    injection_layer = physics_cfg.get("advection_injection_layer", "downstream_sample")
    if injection_layer != "downstream_sample":
        raise ValueError(
            f"Unsupported advection_injection_layer={injection_layer!r}. "
            "'downstream_sample' is the only implemented injection layer."
        )
    if not np.all(np.isfinite(n_inj)):
        raise ValueError("Non-finite UNTH input encountered before advection")

    nk, nj, ni = n_inj.shape
    active_mask = np.zeros((nk, nj, ni), dtype=np.bool_)
    source_budget = np.zeros((nk, nj, ni), dtype=np.float64)
    cell_volume = _compute_spherical_cell_volumes(roi_data, n_inj.shape)

    mask = np.ascontiguousarray(shock_props.get("mask", np.zeros_like(n_inj, dtype=bool)), dtype=bool)
    sample_k2 = np.ascontiguousarray(shock_props.get("sample_k2_grid", np.full_like(n_inj, -1, dtype=int)), dtype=int)
    sample_j2 = np.ascontiguousarray(shock_props.get("sample_j2_grid", np.full_like(n_inj, -1, dtype=int)), dtype=int)
    sample_i2 = np.ascontiguousarray(shock_props.get("sample_i2_grid", np.full_like(n_inj, -1, dtype=int)), dtype=int)

    pad_r = max(int(physics_cfg.get("advection_shock_pad_r", 8)), 0)
    pad_theta = max(int(physics_cfg.get("advection_shock_pad_theta", 2)), 0)
    pad_phi = max(int(physics_cfg.get("advection_shock_pad_phi", 2)), 0)

    shock_count = int(np.count_nonzero(mask))
    valid_seed_samples = 0
    rejected_invalid_sample = 0
    rejected_same_cell = 0
    rejected_nonpositive_injection = 0
    input_budget = 0.0

    for k, j, i in np.argwhere(mask):
        kd = int(sample_k2[k, j, i])
        jd = int(sample_j2[k, j, i])
        id_ = int(sample_i2[k, j, i])
        if not (0 <= kd < nk and 0 <= jd < nj and 0 <= id_ < ni):
            rejected_invalid_sample += 1
            continue
        if kd == k and jd == j and id_ == i:
            rejected_same_cell += 1
            continue

        injection_value = float(n_inj[k, j, i])
        if injection_value <= 0.0:
            rejected_nonpositive_injection += 1
            continue

        valid_seed_samples += 1
        injected_number = injection_value * cell_volume[k, j, i]
        input_budget += injected_number
        source_budget[kd, jd, id_] += injected_number

        for ok in range(-pad_phi, pad_phi + 1):
            kk = (kd + ok) % nk
            for oj in range(-pad_theta, pad_theta + 1):
                jj = jd + oj
                if not (0 <= jj < nj):
                    continue
                for oi in range(-pad_r, pad_r + 1):
                    ii = id_ + oi
                    if not (0 <= ii < ni):
                        continue
                    active_mask[kk, jj, ii] = True

    source_density = np.divide(
        source_budget,
        cell_volume,
        out=np.zeros_like(source_budget),
        where=cell_volume > 0.0,
    )
    source_mask = source_density > 0.0
    active_mask |= source_mask
    active_count = int(np.count_nonzero(active_mask))
    unique_source_cells = int(np.count_nonzero(source_mask))
    remapped_budget = float(np.sum(source_density * cell_volume))

    if active_count > 0:
        kk, jj, ii = np.where(active_mask)
        bbox = {
            "k_min": int(np.min(kk)),
            "k_max": int(np.max(kk)),
            "j_min": int(np.min(jj)),
            "j_max": int(np.max(jj)),
            "i_min": int(np.min(ii)),
            "i_max": int(np.max(ii)),
        }
        bbox_shape = [
            bbox["k_max"] - bbox["k_min"] + 1,
            bbox["j_max"] - bbox["j_min"] + 1,
            bbox["i_max"] - bbox["i_min"] + 1,
        ]
    else:
        bbox = {"k_min": 0, "k_max": -1, "j_min": 0, "j_max": -1, "i_min": 0, "i_max": -1}
        bbox_shape = [0, 0, 0]

    stats = {
        "shock_count": shock_count,
        "seed_samples_valid": int(valid_seed_samples),
        "seed_cells_unique": unique_source_cells,
        "source_cells_unique": unique_source_cells,
        "seed_samples_rejected_invalid": int(rejected_invalid_sample),
        "seed_samples_rejected_same_cell": int(rejected_same_cell),
        "seed_samples_rejected_nonpositive": int(rejected_nonpositive_injection),
        "active_cell_count": active_count,
        "active_fraction": float(active_count / n_inj.size) if n_inj.size else 0.0,
        "bbox_shape": bbox_shape,
        "pad_r": pad_r,
        "pad_theta": pad_theta,
        "pad_phi": pad_phi,
        "injection_layer": injection_layer,
        "source_input_budget": float(input_budget),
        "source_remapped_budget": remapped_budget,
        "source_budget_relative_error": (
            float(abs(remapped_budget - input_budget) / input_budget) if input_budget > 0.0 else 0.0
        ),
    }
    return active_mask, source_mask, source_density, bbox, stats


@njit(fastmath=True)
def _neighbor_active_value(values, active_mask, k, j, i):
    nk, nj, ni = values.shape
    if 0 <= k < nk and 0 <= j < nj and 0 <= i < ni and active_mask[k, j, i]:
        return values[k, j, i]
    return 0.0


@njit(fastmath=True)
def _compute_cell_crossing_time(
    u1_hat,
    u2_hat,
    u3_hat,
    dr,
    dtheta,
    dphi,
    r_coords,
    sin_theta,
    fallback_time,
    k,
    j,
    i,
):
    advective_rate = (
        abs(u1_hat[k, j, i]) / max(dr[i], _TINY)
        + abs(u2_hat[k, j, i]) / max(r_coords[i] * dtheta[j], _TINY)
        + abs(u3_hat[k, j, i]) / max(r_coords[i] * sin_theta[j] * dphi[k], _TINY)
    )
    if advective_rate <= _TINY:
        return fallback_time
    return 1.0 / advective_rate


@njit(fastmath=True)
def _update_relaxation_cell(
    values,
    active_mask,
    source_density,
    source_mask,
    cool_rate,
    u1_hat,
    u2_hat,
    u3_hat,
    dr,
    dtheta,
    dphi,
    r_coords,
    sin_theta,
    tau_inj_fraction,
    fallback_crossing_time,
    k,
    j,
    i,
):
    inv_dr = 1.0 / max(dr[i], _TINY)
    inv_dtheta = 1.0 / max(r_coords[i] * dtheta[j], _TINY)
    inv_dphi = 1.0 / max(r_coords[i] * sin_theta[j] * dphi[k], _TINY)

    weight_r = 0.0
    rhs_r = 0.0
    ur = u1_hat[k, j, i]
    if ur >= 0.0:
        weight_r = ur * inv_dr
        rhs_r = weight_r * _neighbor_active_value(values, active_mask, k, j, i - 1)
    else:
        weight_r = (-ur) * inv_dr
        rhs_r = weight_r * _neighbor_active_value(values, active_mask, k, j, i + 1)

    weight_theta = 0.0
    rhs_theta = 0.0
    uth = u2_hat[k, j, i]
    if uth >= 0.0:
        weight_theta = uth * inv_dtheta
        rhs_theta = weight_theta * _neighbor_active_value(values, active_mask, k, j - 1, i)
    else:
        weight_theta = (-uth) * inv_dtheta
        rhs_theta = weight_theta * _neighbor_active_value(values, active_mask, k, j + 1, i)

    weight_phi = 0.0
    rhs_phi = 0.0
    uph = u3_hat[k, j, i]
    k_prev = k - 1 if k > 0 else values.shape[0] - 1
    k_next = k + 1 if k + 1 < values.shape[0] else 0
    if uph >= 0.0:
        weight_phi = uph * inv_dphi
        rhs_phi = weight_phi * _neighbor_active_value(values, active_mask, k_prev, j, i)
    else:
        weight_phi = (-uph) * inv_dphi
        rhs_phi = weight_phi * _neighbor_active_value(values, active_mask, k_next, j, i)

    total_weight = weight_r + weight_theta + weight_phi
    rhs = rhs_r + rhs_theta + rhs_phi
    coeff = cool_rate[k, j, i] + total_weight

    if source_mask[k, j, i]:
        tau_cross = _compute_cell_crossing_time(
            u1_hat,
            u2_hat,
            u3_hat,
            dr,
            dtheta,
            dphi,
            r_coords,
            sin_theta,
            fallback_crossing_time,
            k,
            j,
            i,
        )
        tau_inj = tau_inj_fraction * tau_cross
        inj_rate = 1.0 / tau_inj
        coeff += inj_rate
        rhs += inj_rate * source_density[k, j, i]

    updated = 0.0 if coeff <= _TINY else rhs / coeff

    if updated < 0.0:
        return 0.0, 1
    return updated, 0


@njit(fastmath=True)
def solve_shock_local_relaxation_jit(
    active_mask,
    source_density,
    source_mask,
    cool_rate,
    u1_hat,
    u2_hat,
    u3_hat,
    dr,
    dtheta,
    dphi,
    r_coords,
    sin_theta,
    line_sweeps,
    tau_inj_fraction,
    fallback_crossing_time,
    k_min,
    k_max,
    j_min,
    j_max,
    i_min,
    i_max,
):
    values = np.zeros_like(source_density)
    clip_count = 0

    if k_max < k_min or j_max < j_min or i_max < i_min:
        return values, clip_count

    for sweep in range(line_sweeps):
        if sweep % 2 == 0:
            k_range = range(k_min, k_max + 1)
            j_range = range(j_min, j_max + 1)
            i_range = range(i_min, i_max + 1)
        else:
            k_range = range(k_max, k_min - 1, -1)
            j_range = range(j_max, j_min - 1, -1)
            i_range = range(i_max, i_min - 1, -1)

        for k in k_range:
            for j in j_range:
                for i in i_range:
                    if not active_mask[k, j, i]:
                        continue
                    updated, clipped = _update_relaxation_cell(
                        values,
                        active_mask,
                        source_density,
                        source_mask,
                        cool_rate,
                        u1_hat,
                        u2_hat,
                        u3_hat,
                        dr,
                        dtheta,
                        dphi,
                        r_coords,
                        sin_theta,
                        tau_inj_fraction,
                        fallback_crossing_time,
                        k,
                        j,
                        i,
                    )
                    values[k, j, i] = updated
                    clip_count += clipped

    return values, clip_count


def _compute_shock_local_residual_stats(
    solution,
    active_mask,
    source_density,
    source_mask,
    cool_rate,
    u1_hat,
    u2_hat,
    u3_hat,
    dr,
    dtheta,
    dphi,
    r_coords,
    sin_theta,
    tau_inj_fraction,
    fallback_crossing_time,
    bbox,
):
    if bbox["k_max"] < bbox["k_min"]:
        return {"max_abs": 0.0, "mean_abs": 0.0}

    max_abs = 0.0
    mean_abs = 0.0
    count = 0

    for k in range(bbox["k_min"], bbox["k_max"] + 1):
        for j in range(bbox["j_min"], bbox["j_max"] + 1):
            for i in range(bbox["i_min"], bbox["i_max"] + 1):
                if not active_mask[k, j, i]:
                    continue
                updated, _ = _update_relaxation_cell(
                    solution,
                    active_mask,
                    source_density,
                    source_mask,
                    cool_rate,
                    u1_hat,
                    u2_hat,
                    u3_hat,
                    dr,
                    dtheta,
                    dphi,
                    r_coords,
                    sin_theta,
                    tau_inj_fraction,
                    fallback_crossing_time,
                    k,
                    j,
                    i,
                )
                abs_residual = abs(updated - solution[k, j, i])
                if abs_residual > max_abs:
                    max_abs = abs_residual
                mean_abs += abs_residual
                count += 1

    if count == 0:
        return {"max_abs": 0.0, "mean_abs": 0.0}
    return {"max_abs": max_abs, "mean_abs": mean_abs / count}


def _validate_advection_model(config):
    physics_cfg = _get_physics_cfg(config)
    advection_model = physics_cfg.get("advection_model", "sr_radial")
    if advection_model != "sr_radial":
        raise ValueError(
            f"Unsupported advection_model={advection_model!r}. "
            "'sr_radial' is the only implemented relativistic advection model."
        )

    cooling_model = physics_cfg.get("advection_cooling_model", "synchrotron_local_sink")
    if cooling_model != "synchrotron_local_sink":
        raise ValueError(
            f"Unsupported advection_cooling_model={cooling_model!r}. "
            "'synchrotron_local_sink' is the only implemented cooling closure."
        )

    injection_layer = physics_cfg.get("advection_injection_layer", "downstream_sample")
    if injection_layer != "downstream_sample":
        raise ValueError(
            f"Unsupported advection_injection_layer={injection_layer!r}. "
            "'downstream_sample' is the only implemented injection layer."
        )

    tau_fraction = float(physics_cfg.get("advection_tau_inj_cell_crossing_fraction", 1.0e-3))
    if not np.isfinite(tau_fraction) or tau_fraction <= 0.0:
        raise ValueError(
            "physics.advection_tau_inj_cell_crossing_fraction must be finite and positive"
        )

    return advection_model, cooling_model


def solve_steady_advection(roi_data, shock_props_or_nonthermal, nonthermal_props_or_config, config=None, logger=None):
    if config is None:
        shock_props = {}
        nonthermal_props = shock_props_or_nonthermal
        config = nonthermal_props_or_config
    else:
        shock_props = shock_props_or_nonthermal
        nonthermal_props = nonthermal_props_or_config

    advection_model, cooling_model = _validate_advection_model(config)
    physics_cfg = _get_physics_cfg(config)
    _log_advection_codepath(
        logger,
        "Advection entry",
        f"model={advection_model}, cooling_model={cooling_model}",
    )

    _log_advection_codepath(logger, "Advection checkpoint", "extract nonthermal density")
    n_init = _extract_nonthermal_density(nonthermal_props)
    gamma_min_grid = np.ascontiguousarray(
        nonthermal_props.get("gamma_min_grid", np.ones_like(n_init)),
        dtype=np.float64,
    )
    p_eff_grid = np.ascontiguousarray(
        nonthermal_props.get("p_eff_grid", np.full_like(n_init, 3.0)),
        dtype=np.float64,
    )
    v1 = np.ascontiguousarray(roi_data["vel1"], dtype=np.float64)
    _log_advection_data(logger, "advection.n_init", n_init)
    _log_advection_data(logger, "advection.gamma_min_grid", gamma_min_grid)
    _log_advection_data(logger, "advection.p_eff_grid", p_eff_grid)
    _log_advection_data(logger, "advection.vel1", v1)
    initial_summary = _compute_code_density_summary(n_init)
    _log_advection_data(logger, "advection.unth_initial_summary", initial_summary)

    _log_advection_codepath(logger, "Advection checkpoint", "build shock-local relaxation source")
    active_mask, source_mask, source_density, bbox, active_stats = _build_shock_local_active_region(
        shock_props,
        n_init,
        physics_cfg,
        roi_data=roi_data,
    )
    _log_advection_data(logger, "advection.active_region_stats", active_stats)
    _log_advection_data(logger, "advection.source_density", source_density[source_mask])

    _log_advection_codepath(logger, "Advection checkpoint", "compute comoving magnetic geometry")
    magnetic_geom = compute_comoving_magnetic_geometry(roi_data)
    b_sq_comoving = np.ascontiguousarray(magnetic_geom["b_sq_comoving"], dtype=np.float64)
    u1_hat = np.ascontiguousarray(magnetic_geom["u1_hat"], dtype=np.float64)
    u2_hat = np.ascontiguousarray(magnetic_geom["u2_hat"], dtype=np.float64)
    u3_hat = np.ascontiguousarray(magnetic_geom["u3_hat"], dtype=np.float64)
    _log_advection_data(logger, "advection.b_sq_comoving", b_sq_comoving)
    gamma_floor = float(physics_cfg.get("gamma_cool_floor", 1.0e-3))
    gamma_char = np.maximum(gamma_min_grid, 1.0 + gamma_floor)
    cooling_factor = float(physics_cfg.get("cooling_factor", 50.0))
    tau_inj_fraction = float(physics_cfg.get("advection_tau_inj_cell_crossing_fraction", 1.0e-3))
    _log_advection_debug(
        logger,
        (
            "    Advection config summary: "
            f"gamma_floor={gamma_floor:.3e}, cooling_factor={cooling_factor:.3e}, "
            f"line_sweeps={int(physics_cfg.get('advection_line_sweeps', 6))}, "
            f"tau_inj_cell_crossing_fraction={tau_inj_fraction:.3e}, "
            f"injection_layer={physics_cfg.get('advection_injection_layer', 'downstream_sample')}"
        ),
    )

    _log_advection_codepath(logger, "Advection checkpoint", "estimate effective cooling time")
    tau_cool_eff = estimate_cooling_time_jit(b_sq_comoving, gamma_char, cooling_factor)
    cool_rate = np.divide(
        1.0,
        np.maximum(tau_cool_eff, 1.0e-12),
        out=np.zeros_like(tau_cool_eff),
        where=tau_cool_eff > 0.0,
    )
    if not np.all(np.isfinite(cool_rate)) or np.any(cool_rate < 0.0):
        raise ValueError("Invalid cooling rate: all entries must be finite and non-negative")
    _log_advection_data(logger, "advection.gamma_char", gamma_char)
    _log_advection_data(logger, "advection.tau_cool_eff", tau_cool_eff)
    _log_advection_data(logger, "advection.cool_rate", cool_rate)
    line_sweeps = int(physics_cfg.get("advection_line_sweeps", 6))
    if line_sweeps < 1:
        line_sweeps = 1

    dr = np.ascontiguousarray(np.diff(roi_data["x1f"]), dtype=np.float64)
    dtheta = np.ascontiguousarray(np.diff(roi_data["x2f"]), dtype=np.float64)
    dphi = np.ascontiguousarray(np.diff(roi_data["x3f"]), dtype=np.float64)
    r_coords = np.ascontiguousarray(0.5 * (roi_data["x1f"][:-1] + roi_data["x1f"][1:]), dtype=np.float64)
    theta_coords = np.ascontiguousarray(0.5 * (roi_data["x2f"][:-1] + roi_data["x2f"][1:]), dtype=np.float64)
    sin_theta = np.ascontiguousarray(np.maximum(np.sin(theta_coords), 1.0e-12), dtype=np.float64)
    if np.any(dr <= 0.0) or np.any(dtheta <= 0.0) or np.any(dphi <= 0.0):
        raise ValueError("Invalid grid spacing: x1f/x2f/x3f must be strictly increasing")
    min_physical_width = min(
        float(np.min(dr)),
        float(np.min(r_coords[:, np.newaxis] * dtheta[np.newaxis, :])),
        float(np.min(r_coords[:, np.newaxis] * sin_theta[np.newaxis, :] * np.min(dphi))),
    )
    fallback_crossing_time = max(min_physical_width, _TINY)

    nk, nj, ni = n_init.shape
    _log_advection_debug(logger, ">>> [Advection-SR] Solving shock-local finite-volume relaxation transport...")
    _log_advection_debug(
        logger,
        f"    Model={advection_model}, Cooling={cooling_model}, Domain=shock_local, "
        f"Grid={ni}x{nj}x{nk}, Line Sweeps={line_sweeps}",
    )
    _log_advection_debug(logger, f"    UNTH(code) init stats: {_format_stat_triplet(n_init)}")
    _log_advection_debug(
        logger,
        "    Active region: "
        f"cells={active_stats['active_cell_count']}, "
        f"fraction={active_stats['active_fraction']:.3e}, "
        f"seed_samples_valid={active_stats['seed_samples_valid']}, "
        f"source_cells_unique={active_stats['source_cells_unique']}, "
        f"bbox_shape={tuple(active_stats['bbox_shape'])}",
    )
    _log_advection_debug(
        logger,
        "    Source budget: "
        f"input={active_stats['source_input_budget']:.3e}, "
        f"remapped={active_stats['source_remapped_budget']:.3e}, "
        f"relative_error={active_stats['source_budget_relative_error']:.3e}",
    )
    _log_advection_debug(logger, f"    gamma_char stats: {_format_stat_triplet(gamma_char)}")
    _log_advection_debug(logger, f"    tau_cool_eff stats: {_format_stat_triplet(tau_cool_eff)}")

    try:
        _log_advection_codepath(logger, "Advection checkpoint", "enter solve_shock_local_relaxation_jit")
        updated_density, clip_count = solve_shock_local_relaxation_jit(
            active_mask,
            source_density,
            source_mask,
            cool_rate,
            u1_hat,
            u2_hat,
            u3_hat,
            dr,
            dtheta,
            dphi,
            r_coords,
            sin_theta,
            line_sweeps,
            tau_inj_fraction,
            fallback_crossing_time,
            bbox["k_min"],
            bbox["k_max"],
            bbox["j_min"],
            bbox["j_max"],
            bbox["i_min"],
            bbox["i_max"],
        )
        _log_advection_codepath(logger, "Advection checkpoint", "solve_shock_local_relaxation_jit completed")
    except Exception as exc:
        _log_advection_debug(
            logger,
            f"    Advection solver raised {type(exc).__name__}: {exc}",
        )
        _log_advection_debug(logger, traceback.format_exc())
        raise

    _log_advection_codepath(logger, "Advection checkpoint", "compute residual stats")
    residual_stats = _compute_shock_local_residual_stats(
        updated_density,
        active_mask,
        source_density,
        source_mask,
        cool_rate,
        u1_hat,
        u2_hat,
        u3_hat,
        dr,
        dtheta,
        dphi,
        r_coords,
        sin_theta,
        tau_inj_fraction,
        fallback_crossing_time,
        bbox,
    )

    _log_advection_debug(
        logger,
        "    Active region stats: "
        f"seed_rejected_invalid={active_stats['seed_samples_rejected_invalid']}, "
        f"seed_rejected_nonpositive={active_stats['seed_samples_rejected_nonpositive']}",
    )
    _log_advection_debug(logger, f"    UNTH(code) final stats: {_format_stat_triplet(updated_density)}")
    _log_advection_data(logger, "advection.updated_density", updated_density)
    updated_summary = _compute_code_density_summary(updated_density)
    _log_advection_data(logger, "advection.unth_final_summary", updated_summary)
    coverage_diagnostics = _compute_advection_coverage_diagnostics(
        updated_density,
        active_mask,
        source_mask,
        shock_props,
        n_init,
        b_sq_comoving,
        p_eff_grid,
    )
    _log_advection_data(logger, "advection.coverage_diagnostics", coverage_diagnostics)
    _log_advection_debug(
        logger,
        "    Residual stats: "
        f"max_abs={residual_stats['max_abs']:.3e}, mean_abs={residual_stats['mean_abs']:.3e}, "
        f"negative_clips={clip_count}",
    )
    _log_advection_debug(
        logger,
        "    Coverage diagnostics: "
        f"shock_unth_nonzero_ratio={coverage_diagnostics['shock_unth_nonzero_ratio']:.3e}, "
        f"seed_unth_nonzero_ratio={coverage_diagnostics['seed_unth_nonzero_ratio']:.3e}, "
        f"active_unth_nonzero_ratio={coverage_diagnostics['active_unth_nonzero_ratio']:.3e}, "
        f"emissivity_weighted_coverage={coverage_diagnostics['emissivity_weighted_coverage']:.3e}",
    )

    evolved_props = copy.deepcopy(nonthermal_props)
    evolved_props["unth_code_grid"] = updated_density
    evolved_props["C_grid"] = updated_density
    evolved_props["advection_active_region_stats"] = active_stats
    evolved_props["advection_coverage_diagnostics"] = coverage_diagnostics
    if "n_nth_phys_grid" in nonthermal_props:
        phys_init = np.ascontiguousarray(nonthermal_props["n_nth_phys_grid"], dtype=np.float64)
        _, phys_source_mask, phys_source_density, _, phys_active_stats = _build_shock_local_active_region(
            shock_props,
            phys_init,
            physics_cfg,
            roi_data=roi_data,
        )
        if not np.array_equal(phys_source_mask, source_mask):
            raise ValueError("Physical and code-unit advection source masks diverged")
        updated_phys, phys_clip_count = solve_shock_local_relaxation_jit(
            active_mask,
            phys_source_density,
            phys_source_mask,
            cool_rate,
            u1_hat,
            u2_hat,
            u3_hat,
            dr,
            dtheta,
            dphi,
            r_coords,
            sin_theta,
            line_sweeps,
            tau_inj_fraction,
            fallback_crossing_time,
            bbox["k_min"],
            bbox["k_max"],
            bbox["j_min"],
            bbox["j_max"],
            bbox["i_min"],
            bbox["i_max"],
        )
        evolved_props["n_nth_phys_grid"] = updated_phys
        _log_advection_data(logger, "advection.physical_source_stats", phys_active_stats)
        if phys_clip_count:
            _log_advection_debug(logger, f"    Physical density negative clips={phys_clip_count}")
    else:
        _rescale_physical_density(evolved_props, n_init, updated_density)
    ratio_summary = _compute_nnth_ratio_summary(roi_data, evolved_props, config)
    if ratio_summary is not None:
        _log_advection_data(logger, "advection.n_nth_over_n_e_summary", ratio_summary)
    _log_advection_codepath(logger, "Advection exit", "state update completed")
    return evolved_props
