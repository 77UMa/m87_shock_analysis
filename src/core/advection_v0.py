import copy

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
    source_line,
    cool_rate_line,
    face_speed,
    coeff_in,
    coeff_out,
    left_bc,
    right_bc,
    cell_idx,
):
    ni = line_values.shape[0]
    rhs = source_line[cell_idx]
    coeff = cool_rate_line[cell_idx]

    u_in = face_speed[cell_idx]
    if u_in >= 0.0:
        neighbor_left = left_bc if cell_idx == 0 else line_values[cell_idx - 1]
        rhs += coeff_in[cell_idx] * u_in * neighbor_left
    else:
        coeff += -coeff_in[cell_idx] * u_in

    u_out = face_speed[cell_idx + 1]
    if u_out >= 0.0:
        coeff += coeff_out[cell_idx] * u_out
    else:
        neighbor_right = right_bc if cell_idx == ni - 1 else line_values[cell_idx + 1]
        rhs += -coeff_out[cell_idx] * u_out * neighbor_right

    if coeff <= _TINY:
        updated = source_line[cell_idx]
    else:
        updated = rhs / coeff

    if updated < 0.0:
        updated = 0.0
        return updated, 1
    return updated, 0


@njit(parallel=True, fastmath=True)
def solve_radial_transport_jit(
    n_init,
    source_density,
    cool_rate,
    v1,
    coeff_in,
    coeff_out,
    line_sweeps,
):
    nk, nj, ni = n_init.shape
    n_out = n_init.copy()
    clip_count = 0
    outward_lines = 0
    inward_lines = 0
    mixed_lines = 0
    stagnant_lines = 0

    for k in prange(nk):
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
            elif line_class == 0:
                local_stagnant += 1
            else:
                local_mixed += 1

            left_bc = n_init[k, j, 0]
            right_bc = n_init[k, j, ni - 1]
            for sweep in range(line_sweeps):
                if sweep % 2 == 0:
                    for i in range(ni):
                        updated, clipped = _update_radial_cell(
                            n_out[k, j, :],
                            source_density[k, j, :],
                            cool_rate[k, j, :],
                            face_speed,
                            coeff_in,
                            coeff_out,
                            left_bc,
                            right_bc,
                            i,
                        )
                        n_out[k, j, i] = updated
                        local_clips += clipped
                else:
                    for i in range(ni - 1, -1, -1):
                        updated, clipped = _update_radial_cell(
                            n_out[k, j, :],
                            source_density[k, j, :],
                            cool_rate[k, j, :],
                            face_speed,
                            coeff_in,
                            coeff_out,
                            left_bc,
                            right_bc,
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


def _compute_residual_stats(solution, source_density, cool_rate, v1, coeff_in, coeff_out):
    nk, nj, ni = solution.shape
    max_abs_residual = 0.0
    mean_abs_residual = 0.0
    sample_count = 0

    for k in range(nk):
        for j in range(nj):
            line = solution[k, j, :]
            face_speed = _compute_face_speed_line(v1[k, j, :])
            left_bc = line[0]
            right_bc = line[-1]
            for i in range(ni):
                u_in = face_speed[i]
                if u_in >= 0.0:
                    n_left = left_bc if i == 0 else line[i - 1]
                    flux_in = coeff_in[i] * u_in * n_left
                else:
                    flux_in = coeff_in[i] * u_in * line[i]

                u_out = face_speed[i + 1]
                if u_out >= 0.0:
                    flux_out = coeff_out[i] * u_out * line[i]
                else:
                    n_right = right_bc if i == ni - 1 else line[i + 1]
                    flux_out = coeff_out[i] * u_out * n_right

                residual = (flux_out - flux_in) - source_density[k, j, i] + cool_rate[k, j, i] * line[i]
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
    factor_positive = factor_grid[positive_mask]
    fallback_factor = float(np.median(factor_positive)) if factor_positive.size else 0.0
    factor_grid[~positive_mask] = fallback_factor
    evolved_props["n_nth_phys_grid"] = updated_code_density * factor_grid


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


def solve_steady_advection(roi_data, nonthermal_props, config):
    advection_model, cooling_model = _validate_advection_model(config)
    physics_cfg = _get_physics_cfg(config)

    n_init = _extract_nonthermal_density(nonthermal_props)
    source_density = np.ascontiguousarray(nonthermal_props.get("C_grid", n_init), dtype=np.float64)
    gamma_min_grid = np.ascontiguousarray(
        nonthermal_props.get("gamma_min_grid", np.ones_like(n_init)),
        dtype=np.float64,
    )
    v1 = np.ascontiguousarray(roi_data["vel1"], dtype=np.float64)
    coeff_in, coeff_out = _build_radial_transport_coefficients(roi_data["x1f"])

    magnetic_geom = compute_comoving_magnetic_geometry(roi_data)
    b_sq_comoving = np.ascontiguousarray(magnetic_geom["b_sq_comoving"], dtype=np.float64)
    gamma_floor = float(physics_cfg.get("gamma_cool_floor", 1.0e-3))
    gamma_char = np.maximum(gamma_min_grid, 1.0 + gamma_floor)
    cooling_factor = float(physics_cfg.get("cooling_factor", 50.0))
    tau_cool_eff = estimate_cooling_time_jit(b_sq_comoving, gamma_char, cooling_factor)
    cool_rate = np.divide(
        1.0,
        np.maximum(tau_cool_eff, 1.0e-12),
        out=np.zeros_like(tau_cool_eff),
        where=tau_cool_eff > 0.0,
    )
    line_sweeps = int(physics_cfg.get("advection_line_sweeps", 6))
    if line_sweeps < 1:
        line_sweeps = 1

    nk, nj, ni = n_init.shape
    print(">>> [Advection-SR] Solving steady-state relativistic radial transport...")
    print(
        f"    Model={advection_model}, Cooling={cooling_model}, "
        f"Grid={ni}x{nj}x{nk}, Line Sweeps={line_sweeps}"
    )
    print(f"    UNTH(code) init stats: {_format_stat_triplet(n_init)}")
    print(f"    Source stats: {_format_stat_triplet(source_density)}")
    print(f"    gamma_char stats: {_format_stat_triplet(gamma_char)}")
    print(f"    tau_cool_eff stats: {_format_stat_triplet(tau_cool_eff)}")

    updated_density, clip_count, outward_lines, inward_lines, mixed_lines, stagnant_lines = solve_radial_transport_jit(
        n_init,
        source_density,
        cool_rate,
        v1,
        coeff_in,
        coeff_out,
        line_sweeps,
    )
    residual_stats = _compute_residual_stats(updated_density, source_density, cool_rate, v1, coeff_in, coeff_out)

    print(
        "    Line classification: "
        f"outward={outward_lines}, inward={inward_lines}, mixed={mixed_lines}, stagnant={stagnant_lines}"
    )
    print(f"    UNTH(code) final stats: {_format_stat_triplet(updated_density)}")
    print(
        "    Residual stats: "
        f"max_abs={residual_stats['max_abs']:.3e}, mean_abs={residual_stats['mean_abs']:.3e}, "
        f"negative_clips={clip_count}"
    )

    evolved_props = copy.deepcopy(nonthermal_props)
    evolved_props["unth_code_grid"] = updated_density
    evolved_props["C_grid"] = updated_density
    _rescale_physical_density(evolved_props, n_init, updated_density)
    return evolved_props
