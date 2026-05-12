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


def _build_radial_transport_coefficients(x1f):
    r_face = np.ascontiguousarray(x1f, dtype=np.float64)
    radial_shell = np.maximum(r_face[1:] ** 3 - r_face[:-1] ** 3, _TINY)
    coeff_in = 3.0 * r_face[:-1] ** 2 / radial_shell
    coeff_out = 3.0 * r_face[1:] ** 2 / radial_shell
    return coeff_in, coeff_out


def _extract_nonthermal_density(nonthermal_props):
    if "unth_code_grid" in nonthermal_props:
        return np.ascontiguousarray(nonthermal_props["unth_code_grid"], dtype=np.float64)
    return np.ascontiguousarray(nonthermal_props["C_grid"], dtype=np.float64)


def _compute_face_speed_line_py(v_line):
    ni = v_line.shape[0]
    face_speed = np.empty(ni + 1, dtype=np.float64)
    face_speed[0] = v_line[0]
    face_speed[ni] = v_line[-1]
    if ni > 1:
        face_speed[1:ni] = 0.5 * (v_line[:-1] + v_line[1:])
    return face_speed


@njit(fastmath=True)
def _compute_face_speed_line(v_line):
    ni = v_line.shape[0]
    face_speed = np.empty(ni + 1, dtype=np.float64)
    face_speed[0] = v_line[0]
    face_speed[ni] = v_line[-1]
    if ni > 1:
        face_speed[1:ni] = 0.5 * (v_line[:-1] + v_line[1:])
    return face_speed


@njit(fastmath=True)
def estimate_cooling_time_jit(b_sq_comoving, gamma_char, cooling_factor):
    b_sq_safe = np.maximum(b_sq_comoving, 1.0e-12)
    gamma_safe = np.maximum(gamma_char, 1.0 + 1.0e-6)
    return cooling_factor / (b_sq_safe * gamma_safe)


@njit(fastmath=True)
def _classify_line(face_speed):
    has_positive = False
    has_negative = False
    for idx in range(face_speed.shape[0]):
        if face_speed[idx] > 0.0:
            has_positive = True
        elif face_speed[idx] < 0.0:
            has_negative = True
    if has_positive and has_negative:
        return 2
    if has_positive:
        return 1
    if has_negative:
        return -1
    return 0


@njit(fastmath=True)
def _update_radial_cell(
    line_values,
    cool_rate_line,
    face_speed,
    coeff_in,
    coeff_out,
    dirichlet_face_mask,
    dirichlet_face_value,
    cell_idx,
):
    ni = line_values.shape[0]
    rhs = 0.0
    coeff = cool_rate_line[cell_idx]

    u_in = face_speed[cell_idx]
    if u_in >= 0.0:
        if dirichlet_face_mask[cell_idx]:
            neighbor_left = dirichlet_face_value[cell_idx]
        elif cell_idx == 0:
            neighbor_left = 0.0
        else:
            neighbor_left = line_values[cell_idx - 1]
        rhs += coeff_in[cell_idx] * u_in * neighbor_left
    else:
        coeff += -coeff_in[cell_idx] * u_in

    u_out = face_speed[cell_idx + 1]
    if u_out < 0.0:
        if dirichlet_face_mask[cell_idx + 1]:
            neighbor_right = dirichlet_face_value[cell_idx + 1]
        elif cell_idx == ni - 1:
            neighbor_right = 0.0
        else:
            neighbor_right = line_values[cell_idx + 1]
        rhs += -coeff_out[cell_idx] * u_out * neighbor_right
    else:
        coeff += coeff_out[cell_idx] * u_out

    if coeff <= _TINY:
        updated = 0.0 if rhs <= 0.0 else rhs / _TINY
    else:
        updated = rhs / coeff

    if updated < 0.0:
        updated = 0.0
        return updated, 1
    return updated, 0


@njit(fastmath=True)
def solve_radial_transport_jit(
    n_init,
    cool_rate,
    v1,
    coeff_in,
    coeff_out,
    dirichlet_face_mask,
    dirichlet_face_value,
    line_sweeps,
):
    nk, nj, ni = n_init.shape
    n_out = n_init.copy()
    clip_count = 0
    outward_lines = 0
    inward_lines = 0
    mixed_lines = 0
    stagnant_lines = 0

    for k in range(nk):
        local_clips = 0
        local_outward = 0
        local_inward = 0
        local_mixed = 0
        local_stagnant = 0
        for j in range(nj):
            face_speed = _compute_face_speed_line(v1[k, j, :])
            line_class = _classify_line(face_speed)
            if line_class > 0:
                local_outward += 1
            elif line_class < 0:
                local_inward += 1
            elif line_class == 2:
                local_mixed += 1
            elif line_class == 0:
                local_stagnant += 1

            for sweep in range(line_sweeps):
                if sweep % 2 == 0:
                    for i in range(ni):
                        updated, clipped = _update_radial_cell(
                            n_out[k, j, :],
                            cool_rate[k, j, :],
                            face_speed,
                            coeff_in,
                            coeff_out,
                            dirichlet_face_mask[k, j, :],
                            dirichlet_face_value[k, j, :],
                            i,
                        )
                        n_out[k, j, i] = updated
                        local_clips += clipped
                else:
                    for i in range(ni - 1, -1, -1):
                        updated, clipped = _update_radial_cell(
                            n_out[k, j, :],
                            cool_rate[k, j, :],
                            face_speed,
                            coeff_in,
                            coeff_out,
                            dirichlet_face_mask[k, j, :],
                            dirichlet_face_value[k, j, :],
                            i,
                        )
                        n_out[k, j, i] = updated
                        local_clips += clipped

        clip_count += local_clips
        outward_lines += local_outward
        inward_lines += local_inward
        mixed_lines += local_mixed
        stagnant_lines += local_stagnant

    return n_out, clip_count, outward_lines, inward_lines, mixed_lines, stagnant_lines


def _compute_residual_stats(
    solution,
    cool_rate,
    v1,
    coeff_in,
    coeff_out,
    dirichlet_face_mask,
    dirichlet_face_value,
):
    nk, nj, ni = solution.shape
    max_abs_residual = 0.0
    mean_abs_residual = 0.0
    sample_count = 0

    for k in range(nk):
        for j in range(nj):
            line = solution[k, j, :]
            face_speed = _compute_face_speed_line(v1[k, j, :])
            for i in range(ni):
                u_in = face_speed[i]
                if u_in >= 0.0:
                    if dirichlet_face_mask[k, j, i]:
                        n_left = dirichlet_face_value[k, j, i]
                    elif i == 0:
                        n_left = 0.0
                    else:
                        n_left = line[i - 1]
                    flux_in = coeff_in[i] * u_in * n_left
                else:
                    flux_in = coeff_in[i] * u_in * line[i]

                u_out = face_speed[i + 1]
                if u_out >= 0.0:
                    flux_out = coeff_out[i] * u_out * line[i]
                else:
                    if dirichlet_face_mask[k, j, i + 1]:
                        n_right = dirichlet_face_value[k, j, i + 1]
                    elif i == ni - 1:
                        n_right = 0.0
                    else:
                        n_right = line[i + 1]
                    flux_out = coeff_out[i] * u_out * n_right

                residual = (flux_out - flux_in) + cool_rate[k, j, i] * line[i]
                abs_residual = abs(residual)
                if abs_residual > max_abs_residual:
                    max_abs_residual = abs_residual
                mean_abs_residual += abs_residual
                sample_count += 1

    if sample_count == 0:
        return {"max_abs": 0.0, "mean_abs": 0.0}
    return {"max_abs": max_abs_residual, "mean_abs": mean_abs_residual / sample_count}


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


def _build_dirichlet_face_map(shock_props, n_inj, v1):
    nk, nj, ni = n_inj.shape
    dirichlet_face_mask = np.zeros((nk, nj, ni + 1), dtype=np.bool_)
    dirichlet_face_value = np.zeros((nk, nj, ni + 1), dtype=np.float64)
    mask = np.ascontiguousarray(shock_props.get("mask", np.zeros_like(n_inj, dtype=bool)), dtype=bool)
    sample_k2 = np.ascontiguousarray(shock_props.get("sample_k2_grid", np.full_like(n_inj, -1, dtype=int)), dtype=int)
    sample_j2 = np.ascontiguousarray(shock_props.get("sample_j2_grid", np.full_like(n_inj, -1, dtype=int)), dtype=int)
    sample_i2 = np.ascontiguousarray(shock_props.get("sample_i2_grid", np.full_like(n_inj, -1, dtype=int)), dtype=int)

    outward_lines = 0
    inward_lines = 0
    mixed_lines = 0
    stagnant_lines = 0
    outer_boundary_inflow_count = 0
    same_line_count = 0
    rejected_nonradial = 0
    rejected_same_cell = 0
    rejected_flow_inconsistent = 0
    rejected_nonpositive_injection = 0

    for k in range(nk):
        for j in range(nj):
            face_speed = _compute_face_speed_line_py(v1[k, j, :])
            line_class = _classify_line(face_speed)
            if line_class == 2:
                mixed_lines += 1
            elif line_class > 0:
                outward_lines += 1
            elif line_class < 0:
                inward_lines += 1
            else:
                stagnant_lines += 1

            if face_speed[0] > 0.0:
                outer_boundary_inflow_count += 1
            if face_speed[ni] < 0.0:
                outer_boundary_inflow_count += 1

            shock_indices = np.flatnonzero(mask[k, j, :])
            for i in shock_indices:
                kd = int(sample_k2[k, j, i])
                jd = int(sample_j2[k, j, i])
                id_ = int(sample_i2[k, j, i])
                if kd != k or jd != j:
                    rejected_nonradial += 1
                    continue

                same_line_count += 1
                if id_ == i or id_ < 0 or id_ >= ni:
                    rejected_same_cell += 1
                    continue

                injection_value = float(n_inj[k, j, i])
                if injection_value <= 0.0:
                    rejected_nonpositive_injection += 1
                    continue

                if id_ > i:
                    face_idx = i + 1
                    if face_speed[face_idx] <= 0.0:
                        rejected_flow_inconsistent += 1
                        continue
                else:
                    face_idx = i
                    if face_speed[face_idx] >= 0.0:
                        rejected_flow_inconsistent += 1
                        continue

                dirichlet_face_mask[k, j, face_idx] = True
                if injection_value > dirichlet_face_value[k, j, face_idx]:
                    dirichlet_face_value[k, j, face_idx] = injection_value

    stats = {
        "dirichlet_faces_total": int(np.count_nonzero(dirichlet_face_mask)),
        "dirichlet_faces_same_line": int(same_line_count),
        "dirichlet_faces_rejected_nonradial": int(rejected_nonradial),
        "dirichlet_faces_rejected_same_cell": int(rejected_same_cell),
        "dirichlet_faces_rejected_flow_inconsistent": int(rejected_flow_inconsistent),
        "dirichlet_faces_rejected_nonpositive_injection": int(rejected_nonpositive_injection),
        "dirichlet_faces_outer_boundary_inflow_count": int(outer_boundary_inflow_count),
        "line_classification": {
            "outward": int(outward_lines),
            "inward": int(inward_lines),
            "mixed": int(mixed_lines),
            "stagnant": int(stagnant_lines),
        },
    }
    return dirichlet_face_mask, dirichlet_face_value, stats


def _build_shock_local_active_region(shock_props, n_inj, physics_cfg):
    nk, nj, ni = n_inj.shape
    active_mask = np.zeros((nk, nj, ni), dtype=np.bool_)
    seed_mask = np.zeros((nk, nj, ni), dtype=np.bool_)
    seed_value = np.zeros((nk, nj, ni), dtype=np.float64)

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
    rejected_nonpositive_injection = 0

    for k, j, i in np.argwhere(mask):
        kd = int(sample_k2[k, j, i])
        jd = int(sample_j2[k, j, i])
        id_ = int(sample_i2[k, j, i])
        if not (0 <= kd < nk and 0 <= jd < nj and 0 <= id_ < ni):
            rejected_invalid_sample += 1
            continue

        injection_value = float(n_inj[k, j, i])
        if injection_value <= 0.0:
            rejected_nonpositive_injection += 1
            continue

        valid_seed_samples += 1
        seed_mask[kd, jd, id_] = True
        if injection_value > seed_value[kd, jd, id_]:
            seed_value[kd, jd, id_] = injection_value

        k0 = max(kd - pad_phi, 0)
        k1 = min(kd + pad_phi + 1, nk)
        j0 = max(jd - pad_theta, 0)
        j1 = min(jd + pad_theta + 1, nj)
        i0 = max(id_ - pad_r, 0)
        i1 = min(id_ + pad_r + 1, ni)
        active_mask[k0:k1, j0:j1, i0:i1] = True

    active_mask |= seed_mask
    active_count = int(np.count_nonzero(active_mask))
    unique_seed_cells = int(np.count_nonzero(seed_mask))

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
        "seed_cells_unique": unique_seed_cells,
        "seed_samples_rejected_invalid": int(rejected_invalid_sample),
        "seed_samples_rejected_nonpositive": int(rejected_nonpositive_injection),
        "active_cell_count": active_count,
        "active_fraction": float(active_count / n_inj.size) if n_inj.size else 0.0,
        "bbox_shape": bbox_shape,
        "pad_r": pad_r,
        "pad_theta": pad_theta,
        "pad_phi": pad_phi,
    }
    return active_mask, seed_mask, seed_value, bbox, stats


@njit(fastmath=True)
def _seed_density(seed_mask, seed_value):
    density = np.zeros_like(seed_value)
    nk, nj, ni = seed_mask.shape
    for k in range(nk):
        for j in range(nj):
            for i in range(ni):
                if seed_mask[k, j, i]:
                    density[k, j, i] = seed_value[k, j, i]
    return density


@njit(fastmath=True)
def _neighbor_active_value(values, active_mask, k, j, i):
    nk, nj, ni = values.shape
    if 0 <= k < nk and 0 <= j < nj and 0 <= i < ni and active_mask[k, j, i]:
        return values[k, j, i]
    return 0.0


@njit(fastmath=True)
def _update_local_cell(
    values,
    active_mask,
    seed_mask,
    seed_value,
    cool_rate,
    u1_hat,
    u2_hat,
    u3_hat,
    dr,
    dtheta,
    dphi,
    r_coords,
    sin_theta,
    k,
    j,
    i,
):
    coeff = cool_rate[k, j, i]
    rhs = 0.0

    inv_dr = 1.0 / max(dr[i], _TINY)
    inv_dtheta = 1.0 / max(r_coords[i] * dtheta[j], _TINY)
    inv_dphi = 1.0 / max(r_coords[i] * sin_theta[j] * dphi[k], _TINY)

    ur = u1_hat[k, j, i]
    if ur >= 0.0:
        rhs += ur * inv_dr * _neighbor_active_value(values, active_mask, k, j, i - 1)
        coeff += ur * inv_dr
    else:
        rhs += (-ur) * inv_dr * _neighbor_active_value(values, active_mask, k, j, i + 1)
        coeff += (-ur) * inv_dr

    uth = u2_hat[k, j, i]
    if uth >= 0.0:
        rhs += uth * inv_dtheta * _neighbor_active_value(values, active_mask, k, j - 1, i)
        coeff += uth * inv_dtheta
    else:
        rhs += (-uth) * inv_dtheta * _neighbor_active_value(values, active_mask, k, j + 1, i)
        coeff += (-uth) * inv_dtheta

    uph = u3_hat[k, j, i]
    k_prev = k - 1 if k > 0 else values.shape[0] - 1
    k_next = k + 1 if k + 1 < values.shape[0] else 0
    if uph >= 0.0:
        rhs += uph * inv_dphi * _neighbor_active_value(values, active_mask, k_prev, j, i)
        coeff += uph * inv_dphi
    else:
        rhs += (-uph) * inv_dphi * _neighbor_active_value(values, active_mask, k_next, j, i)
        coeff += (-uph) * inv_dphi

    if coeff <= _TINY:
        updated = 0.0
    else:
        updated = rhs / coeff

    if seed_mask[k, j, i] and updated < seed_value[k, j, i]:
        updated = seed_value[k, j, i]

    if updated < 0.0:
        return 0.0, 1
    return updated, 0


@njit(fastmath=True)
def solve_shock_local_transport_jit(
    active_mask,
    seed_mask,
    seed_value,
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
    k_min,
    k_max,
    j_min,
    j_max,
    i_min,
    i_max,
):
    values = _seed_density(seed_mask, seed_value)
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
                    updated, clipped = _update_local_cell(
                        values,
                        active_mask,
                        seed_mask,
                        seed_value,
                        cool_rate,
                        u1_hat,
                        u2_hat,
                        u3_hat,
                        dr,
                        dtheta,
                        dphi,
                        r_coords,
                        sin_theta,
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
    seed_mask,
    seed_value,
    cool_rate,
    u1_hat,
    u2_hat,
    u3_hat,
    dr,
    dtheta,
    dphi,
    r_coords,
    sin_theta,
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
                if not active_mask[k, j, i] or seed_mask[k, j, i]:
                    continue

                inv_dr = 1.0 / max(dr[i], _TINY)
                inv_dtheta = 1.0 / max(r_coords[i] * dtheta[j], _TINY)
                inv_dphi = 1.0 / max(r_coords[i] * sin_theta[j] * dphi[k], _TINY)

                ur = u1_hat[k, j, i]
                if ur >= 0.0:
                    radial = ur * inv_dr * (solution[k, j, i] - _neighbor_active_value(solution, active_mask, k, j, i - 1))
                else:
                    radial = (-ur) * inv_dr * (solution[k, j, i] - _neighbor_active_value(solution, active_mask, k, j, i + 1))

                uth = u2_hat[k, j, i]
                if uth >= 0.0:
                    polar = uth * inv_dtheta * (solution[k, j, i] - _neighbor_active_value(solution, active_mask, k, j - 1, i))
                else:
                    polar = (-uth) * inv_dtheta * (solution[k, j, i] - _neighbor_active_value(solution, active_mask, k, j + 1, i))

                uph = u3_hat[k, j, i]
                k_prev = k - 1 if k > 0 else solution.shape[0] - 1
                k_next = k + 1 if k + 1 < solution.shape[0] else 0
                if uph >= 0.0:
                    azimuthal = uph * inv_dphi * (solution[k, j, i] - _neighbor_active_value(solution, active_mask, k_prev, j, i))
                else:
                    azimuthal = (-uph) * inv_dphi * (solution[k, j, i] - _neighbor_active_value(solution, active_mask, k_next, j, i))

                residual = radial + polar + azimuthal + cool_rate[k, j, i] * solution[k, j, i]
                abs_residual = abs(residual)
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
    v1 = np.ascontiguousarray(roi_data["vel1"], dtype=np.float64)
    _log_advection_data(logger, "advection.n_init", n_init)
    _log_advection_data(logger, "advection.gamma_min_grid", gamma_min_grid)
    _log_advection_data(logger, "advection.vel1", v1)
    initial_summary = _compute_code_density_summary(n_init)
    _log_advection_data(logger, "advection.unth_initial_summary", initial_summary)

    _log_advection_codepath(logger, "Advection checkpoint", "build shock-local active region")
    active_mask, seed_mask, seed_value, bbox, active_stats = _build_shock_local_active_region(
        shock_props,
        n_init,
        physics_cfg,
    )
    _log_advection_data(logger, "advection.active_region_stats", active_stats)
    _log_advection_data(logger, "advection.seed_values", seed_value[seed_mask])

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
    _log_advection_debug(
        logger,
        (
            "    Advection config summary: "
            f"gamma_floor={gamma_floor:.3e}, cooling_factor={cooling_factor:.3e}, "
            f"line_sweeps={int(physics_cfg.get('advection_line_sweeps', 6))}"
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

    nk, nj, ni = n_init.shape
    _log_advection_debug(logger, ">>> [Advection-SR] Solving shock-local sparse relativistic transport...")
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
        f"seed_cells_unique={active_stats['seed_cells_unique']}, "
        f"bbox_shape={tuple(active_stats['bbox_shape'])}",
    )
    _log_advection_debug(logger, f"    gamma_char stats: {_format_stat_triplet(gamma_char)}")
    _log_advection_debug(logger, f"    tau_cool_eff stats: {_format_stat_triplet(tau_cool_eff)}")

    try:
        _log_advection_codepath(logger, "Advection checkpoint", "enter solve_shock_local_transport_jit")
        updated_density, clip_count = solve_shock_local_transport_jit(
            active_mask,
            seed_mask,
            seed_value,
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
            bbox["k_min"],
            bbox["k_max"],
            bbox["j_min"],
            bbox["j_max"],
            bbox["i_min"],
            bbox["i_max"],
        )
        _log_advection_codepath(logger, "Advection checkpoint", "solve_shock_local_transport_jit completed")
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
        seed_mask,
        seed_value,
        cool_rate,
        u1_hat,
        u2_hat,
        u3_hat,
        dr,
        dtheta,
        dphi,
        r_coords,
        sin_theta,
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
    _log_advection_debug(
        logger,
        "    Residual stats: "
        f"max_abs={residual_stats['max_abs']:.3e}, mean_abs={residual_stats['mean_abs']:.3e}, "
        f"negative_clips={clip_count}",
    )

    evolved_props = copy.deepcopy(nonthermal_props)
    evolved_props["unth_code_grid"] = updated_density
    evolved_props["C_grid"] = updated_density
    _rescale_physical_density(evolved_props, n_init, updated_density)
    ratio_summary = _compute_nnth_ratio_summary(roi_data, evolved_props, config)
    if ratio_summary is not None:
        _log_advection_data(logger, "advection.n_nth_over_n_e_summary", ratio_summary)
    _log_advection_codepath(logger, "Advection exit", "state update completed")
    return evolved_props
