"""Non-thermal electron calculations and diagnostics."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


POWERLAW_GAMMA_MAX_DEFAULT = 1.0e5
K_INJ_QUADRATURE_ORDER = 48


def _compute_relativistic_p_min(theta_e2, x_inj):
    """Return relativistic injection threshold momentum p/(m_e c)."""
    gamma_mean = 1.0 + 3.0 * np.maximum(theta_e2, 0.0)
    p_th_rel_sq = np.maximum(gamma_mean * gamma_mean - 1.0, 0.0)
    return x_inj * np.sqrt(p_th_rel_sq)


def _compute_relativistic_k_inj(q, p_min, gamma_max=POWERLAW_GAMMA_MAX_DEFAULT):
    """Return average kinetic energy per injected electron in units of m_e c^2.

    The power-law tail is integrated from p_min to the finite cutoff implied by gamma_max.
    A finite cutoff is required because the mean energy diverges for q <= 2 if the tail
    is extended to infinity. The default gamma_max matches the ipole/Symphony default.
    """
    q = np.asarray(q, dtype=float)
    p_min = np.asarray(p_min, dtype=float)
    k_inj = np.zeros_like(q, dtype=float)

    valid = np.isfinite(q) & np.isfinite(p_min) & (q > 1.0) & (p_min > 0.0)
    if not np.any(valid):
        return k_inj

    gamma_max = max(float(gamma_max), 1.0 + 1.0e-12)
    p_max = np.sqrt(max(gamma_max * gamma_max - 1.0, 0.0))
    valid &= p_min < p_max
    if not np.any(valid):
        return k_inj

    qv = q[valid]
    pminv = p_min[valid]
    log_span = np.log(p_max / pminv)
    finite_span = np.isfinite(log_span) & (log_span > 0.0)
    if not np.any(finite_span):
        return k_inj

    qv = qv[finite_span]
    pminv = pminv[finite_span]
    log_span = log_span[finite_span]

    nodes, weights = np.polynomial.legendre.leggauss(K_INJ_QUADRATURE_ORDER)
    x = 0.5 * (nodes + 1.0)
    w = 0.5 * weights

    log_p = np.log(pminv)[:, None] + log_span[:, None] * x[None, :]
    p = np.exp(log_p)
    gamma_p = np.sqrt(1.0 + p * p)
    numerator_integrand = (gamma_p - 1.0) * np.exp((1.0 - qv)[:, None] * log_p)
    numerator = log_span * np.sum(numerator_integrand * w[None, :], axis=1)

    denom = np.empty_like(qv)
    near_one = np.isclose(qv, 1.0, rtol=0.0, atol=1.0e-10)
    denom[near_one] = np.log(p_max / pminv[near_one])
    not_one = ~near_one
    denom[not_one] = (
        np.power(pminv[not_one], 1.0 - qv[not_one]) - np.power(p_max, 1.0 - qv[not_one])
    ) / (qv[not_one] - 1.0)

    kv = np.divide(
        numerator,
        denom,
        out=np.zeros_like(numerator),
        where=np.isfinite(denom) & (denom > 0.0),
    )
    kv = np.where(np.isfinite(kv) & (kv > 0.0), kv, 0.0)

    valid_indices = np.flatnonzero(valid)
    k_inj[valid_indices[finite_span]] = kv
    return k_inj


def calculate_nonthermal_electrons(
    shock_properties,
    gamma=4.0 / 3.0,
    x_inj=3.5,
    xi_max=0.05,
    eta_inj_e0=1.0e-3,
    eps_nth_e0=3.0e-3,
    theta_bn_quench=50.0,
    theta_bn_width=10.0,
    sonic_mach_inj_min=1.5,
    inj_model="pic_dual_cap",
    sigma_crit=0.1,
    alpha_sigma=2,
    rho_unit=1.0,
    u_unit=None,
    sironi_tran_coeff=0.0016,
    sironi_tran_exp=3.6,
    sironi_tran_delta_max=3.0,
    use_sr_refined_mask=None,
    logger=None,
    **deprecated_controls,
):
    """Calculate non-thermal electron properties for shock cells."""
    if u_unit is None:
        u_unit = rho_unit
    print(
        "Starting non-thermal electron calculation "
        f"(inj_model={inj_model}, sigma_crit={sigma_crit}, alpha_sigma={alpha_sigma}, "
        f"rho_unit={rho_unit:.3e}, u_unit={u_unit:.3e})..."
    )
    if logger:
        logger.ai.func_enter(
            "calculate_nonthermal_electrons",
            {"gamma": gamma, "x_inj": x_inj, "xi_max": xi_max, "eta_inj_e0": eta_inj_e0,
             "eps_nth_e0": eps_nth_e0, "theta_bn_quench": theta_bn_quench,
             "theta_bn_width": theta_bn_width, "sonic_mach_inj_min": sonic_mach_inj_min,
             "inj_model": inj_model, "sigma_crit": sigma_crit,
             "alpha_sigma": alpha_sigma, "rho_unit": rho_unit, "u_unit": u_unit,
             "sironi_tran_coeff": sironi_tran_coeff, "sironi_tran_exp": sironi_tran_exp,
             "sironi_tran_delta_max": sironi_tran_delta_max,
             "shock_cells": int(np.sum(shock_properties["mask"]))},
        )
    removed_controls = [key for key in ("r_low", "r_high", "beta_crit") if key in deprecated_controls]
    if removed_controls:
        raise ValueError(
            "Removed beta-closure controls detected: "
            + ", ".join(removed_controls)
            + ". Two-temperature Sironi-Tran heating is now the only active electron-heating chain."
        )
    if use_sr_refined_mask is not None:
        raise ValueError('use_sr_refined_mask has been removed; shock_properties["mask"] is already the SRMHD mainline mask.')
    mask = shock_properties["mask"]
    q_grid = np.zeros_like(mask, dtype=float)
    unth_code_grid = np.zeros_like(mask, dtype=float)
    gamma_min_grid = np.ones_like(mask, dtype=float)
    gamma_min_grid_physical = np.ones_like(mask, dtype=float)
    sigma_suppression_grid = np.ones_like(mask, dtype=float)
    gamma_min_failure_code_grid = np.zeros_like(mask, dtype=np.int16)
    theta_e_grid = np.zeros_like(mask, dtype=float)
    theta_e1_grid = np.zeros_like(mask, dtype=float)
    theta_e2_ad_grid = np.zeros_like(mask, dtype=float)
    sironi_boost_grid = np.ones_like(mask, dtype=float)
    r1_grid = np.zeros_like(mask, dtype=float)
    te2_grid = np.zeros_like(mask, dtype=float)
    p_min_physical_grid = np.zeros_like(mask, dtype=float)
    eta_inj_e_grid = np.zeros_like(mask, dtype=float)
    eps_nth_e_grid = np.zeros_like(mask, dtype=float)
    e_diss_e_grid = np.zeros_like(mask, dtype=float)
    inj_gate_grid = np.zeros_like(mask, dtype=float)
    n_nth_eta_phys_grid = np.zeros_like(mask, dtype=float)
    n_nth_eps_phys_grid = np.zeros_like(mask, dtype=float)
    n_nth_phys_grid = np.zeros_like(mask, dtype=float)
    spectral_norm_eta_grid = np.zeros_like(mask, dtype=float)
    spectral_norm_eps_grid = np.zeros_like(mask, dtype=float)
    inj_limit_mode_grid = np.zeros_like(mask, dtype=np.int16)
    inj_limit_modes = {
        "none": 0,
        "eta_cap": 1,
        "eps_cap": 2,
        "quenched_by_gate": 3,
        "invalid_or_boundary": 4,
    }
    gamma_failure_codes = {
        "ok": 0,
        "press1_nonpositive": 1,
        "rho1_nonpositive": 2,
        "press2_nonpositive": 3,
        "rho2_nonpositive": 4,
        "beta1_invalid": 5,
        "sonic_mach_invalid": 6,
        "boundary_clipped": 7,
        "theta_e1_invalid": 8,
        "theta_e2_invalid": 9,
        "gamma_min_le_one": 10,
    }
    if np.any(mask):
        m_e = 9.1094e-28
        m_p = 1.6726e-24
        c_light = 2.9979e10
        k_b = 1.3806e-16
        if inj_model != "pic_dual_cap":
            raise ValueError(
                f"Unsupported inj_model={inj_model!r}. 'pic_dual_cap' is the only active mainline injection model."
            )
        mp_over_me = m_p / m_e
        mach_field = shock_properties["mainline_mach"] if "mainline_mach" in shock_properties else shock_properties["upstream_mach"]
        m1 = mach_field[mask]
        rho1 = shock_properties.get("rho1_code_grid", np.zeros_like(mask, dtype=float))[mask]
        press1 = shock_properties.get("press1_code_grid", np.zeros_like(mask, dtype=float))[mask]
        beta1 = shock_properties.get("beta1_grid", np.zeros_like(mask, dtype=float))[mask]
        sonic_mach = shock_properties.get("sr_sonic_mach", np.zeros_like(mask, dtype=float))[mask]
        theta_bn = shock_properties.get("theta_Bn", np.zeros_like(mask, dtype=float))[mask]
        rho2 = shock_properties.get("rho2_code_grid", np.zeros_like(mask, dtype=float))[mask]
        press2 = shock_properties.get("press2_code_grid", np.zeros_like(mask, dtype=float))[mask]
        clipped = shock_properties.get("sample_boundary_clipped_grid", np.zeros_like(mask, dtype=bool))[mask]
        with np.errstate(divide="ignore", invalid="ignore"):
            press1_over_rho1 = np.divide(press1, rho1, out=np.zeros_like(press1), where=rho1 > 0)
            compression = np.divide(rho2, rho1, out=np.zeros_like(rho2), where=rho1 > 0)
            r1 = 80.0 * np.square(beta1) / (1.0 + np.square(beta1)) + 1.0 / (1.0 + np.square(beta1))
            theta_e1 = press1_over_rho1 * mp_over_me / (1.0 + r1)
            theta_e2_ad = theta_e1 * np.cbrt(np.maximum(compression, 0.0))
            sironi_delta_raw = sironi_tran_coeff * np.power(np.maximum(sonic_mach, 0.0), sironi_tran_exp)
            sironi_delta_sat = sironi_tran_delta_max * np.tanh(sironi_delta_raw / sironi_tran_delta_max)
            sironi_boost = 1.0 + sironi_delta_sat
            theta_e2 = theta_e2_ad * sironi_boost
            t2 = theta_e2 * m_e * c_light**2 / k_b
        particle_mass = m_p + m_e
        n_e2_phys = np.divide(rho2 * rho_unit, particle_mass, out=np.zeros_like(rho2), where=rho2 > 0)
        print(
            "  Electron heating branch: "
            f"Theta_e1 median={np.median(theta_e1[np.isfinite(theta_e1)]) if np.any(np.isfinite(theta_e1)) else 0.0:.3e}, "
            f"Theta_e2 median={np.median(theta_e2[np.isfinite(theta_e2)]) if np.any(np.isfinite(theta_e2)) else 0.0:.3e}"
        )
        inv_tau = (gamma - 1.0) / (gamma + 1.0) + (2.0 / (gamma + 1.0)) / m1**2
        tau = 1.0 / inv_tau
        q = (tau + 2.0) / (tau - 1.0)
        q_grid[mask] = q
        print("  Power-law index 'q' calculated.")
        if logger:
            logger.ai.data("nt.q", q)
        p_min_phys = _compute_relativistic_p_min(theta_e2, x_inj)
        p_min_phys_sq = p_min_phys * p_min_phys
        gamma_min_raw = 1.0 + 3.0 * theta_e2
        failure = np.full(gamma_min_raw.shape, gamma_failure_codes["ok"], dtype=np.int16)
        failure[press1 <= 0] = gamma_failure_codes["press1_nonpositive"]
        failure[(failure == 0) & (rho1 <= 0)] = gamma_failure_codes["rho1_nonpositive"]
        failure[(failure == 0) & (press2 <= 0)] = gamma_failure_codes["press2_nonpositive"]
        failure[(failure == 0) & (rho2 <= 0)] = gamma_failure_codes["rho2_nonpositive"]
        invalid_beta1 = (~np.isfinite(beta1)) | (beta1 < 0)
        failure[(failure == 0) & invalid_beta1] = gamma_failure_codes["beta1_invalid"]
        invalid_sonic_mach = (~np.isfinite(sonic_mach)) | (sonic_mach <= 0)
        failure[(failure == 0) & invalid_sonic_mach] = gamma_failure_codes["sonic_mach_invalid"]
        failure[(failure == 0) & clipped] = gamma_failure_codes["boundary_clipped"]
        invalid_theta_e1 = (~np.isfinite(theta_e1)) | (theta_e1 <= 0)
        failure[(failure == 0) & invalid_theta_e1] = gamma_failure_codes["theta_e1_invalid"]
        invalid_theta_e2 = (~np.isfinite(theta_e2)) | (~np.isfinite(t2)) | (theta_e2 <= 0) | (t2 <= 0)
        failure[(failure == 0) & invalid_theta_e2] = gamma_failure_codes["theta_e2_invalid"]
        failure[(failure == 0) & (gamma_min_raw <= 1.0)] = gamma_failure_codes["gamma_min_le_one"]
        gamma_min = np.where(failure == 0, gamma_min_raw, 1.0)
        gamma_min_grid[mask] = gamma_min
        gamma_min_grid_physical[mask] = gamma_min
        gamma_min_failure_code_grid[mask] = failure
        theta_e_grid[mask] = theta_e2
        theta_e1_grid[mask] = theta_e1
        theta_e2_ad_grid[mask] = theta_e2_ad
        sironi_boost_grid[mask] = sironi_boost
        r1_grid[mask] = r1
        te2_grid[mask] = t2
        p_min_physical_grid[mask] = p_min_phys
        unique_codes, unique_counts = np.unique(failure, return_counts=True)
        failure_summary = {int(code): int(count) for code, count in zip(unique_codes, unique_counts)}
        valid_gamma = failure == 0
        print(f"  gamma_min failure summary: {failure_summary}")
        print(f"  gamma_min two-temp branch: valid={np.sum(valid_gamma)}/{failure.size}, median_valid={np.median(gamma_min[valid_gamma]) if np.any(valid_gamma) else 1.0:.2f}")
        if logger:
            logger.ai.codepath("Electron heating branch", "upstream R-beta + adiabatic compression + saturated Sironi-Tran boost")
            logger.ai.codepath("Gamma-min branch", f"failure_counts={failure_summary}")
            logger.ai.data(
                "summary.thermal_chain",
                {
                    "beta1_median": float(np.median(beta1[np.isfinite(beta1)])) if np.any(np.isfinite(beta1)) else 0.0,
                    "R1_median": float(np.median(r1[np.isfinite(r1)])) if np.any(np.isfinite(r1)) else 0.0,
                    "theta_e1_median": float(np.median(theta_e1[np.isfinite(theta_e1)])) if np.any(np.isfinite(theta_e1)) else 0.0,
                    "theta_e2_ad_median": float(np.median(theta_e2_ad[np.isfinite(theta_e2_ad)])) if np.any(np.isfinite(theta_e2_ad)) else 0.0,
                    "sironi_boost_median": float(np.median(sironi_boost[np.isfinite(sironi_boost)])) if np.any(np.isfinite(sironi_boost)) else 0.0,
                    "theta_e2_median": float(np.median(theta_e2[np.isfinite(theta_e2)])) if np.any(np.isfinite(theta_e2)) else 0.0,
                    "Te2_median_K": float(np.median(t2[np.isfinite(t2)])) if np.any(np.isfinite(t2)) else 0.0,
                    "gamma_min_valid_fraction": float(np.mean(valid_gamma)),
                    "gamma_failure_counts": failure_summary,
                },
            )
        k_inj = _compute_relativistic_k_inj(q, p_min_phys, gamma_max=POWERLAW_GAMMA_MAX_DEFAULT)
        print(
            "  Mean kinetic energy 'K_inj' calculated "
            f"(relativistic finite-cutoff closure, gamma_max={POWERLAW_GAMMA_MAX_DEFAULT:.1e})."
        )
        finite_theta_bn = np.isfinite(theta_bn)
        theta_bn_rad = np.deg2rad(theta_bn_quench)
        theta_bn_width_rad = np.deg2rad(max(theta_bn_width, 1.0e-6))
        obliquity_gate = np.ones_like(theta_e2)
        obliquity_gate[finite_theta_bn] = 1.0 / (
            1.0 + np.exp((theta_bn[finite_theta_bn] - theta_bn_rad) / theta_bn_width_rad)
        )
        obliquity_gate[~finite_theta_bn] = 0.0

        if "sigma_grid" in shock_properties:
            sigma = shock_properties["sigma_grid"][mask]
            sigma_gate = 1.0 / (1.0 + np.power(np.maximum(sigma, 0.0) / sigma_crit, alpha_sigma))
            sigma_suppression_grid[mask] = sigma_gate
            print(
                "  Sigma/obliquity gating: "
                f"sigma median={np.median(sigma):.3e}, sigma_gate median={np.median(sigma_gate):.3f}, "
                f"obliquity_gate median={np.median(obliquity_gate):.3f}"
            )
            if logger:
                logger.ai.codepath("Injection gate branch", "sigma_grid found and obliquity gate applied")
        else:
            sigma = np.zeros_like(theta_e2)
            sigma_gate = np.ones_like(theta_e2)
            print("  No sigma_grid found in shock_properties; using only obliquity/sonic gating.")
            if logger:
                logger.ai.codepath("Injection gate branch", "sigma_grid missing; sigma gate defaults to unity")

        finite_q = np.isfinite(q) & (q > 1.0)
        finite_p = np.isfinite(p_min_phys) & (p_min_phys > 0)
        valid_tail = finite_q & finite_p
        valid_energy = valid_tail & np.isfinite(k_inj) & (k_inj > 0)
        positive_heating = np.isfinite(theta_e2) & np.isfinite(theta_e2_ad) & (theta_e2 > theta_e2_ad)
        valid_gate = (
            (failure == 0)
            & (~clipped)
            & np.isfinite(sonic_mach)
            & (sonic_mach >= sonic_mach_inj_min)
            & positive_heating
            & valid_tail
        )
        inj_gate = obliquity_gate * sigma_gate
        inj_gate = np.where(valid_gate, inj_gate, 0.0)
        inj_gate = np.clip(inj_gate, 0.0, 1.0)

        eta_inj_e = eta_inj_e0 * inj_gate
        e_diss_e = 3.0 * n_e2_phys * m_e * c_light**2 * np.maximum(theta_e2 - theta_e2_ad, 0.0)
        eps_nth_e = np.where(positive_heating, eps_nth_e0 * inj_gate, 0.0)
        n_nth_eta_phys = eta_inj_e * n_e2_phys
        u_eps = eps_nth_e * e_diss_e
        n_nth_eps_phys = np.divide(
            u_eps,
            k_inj * m_e * c_light**2,
            out=np.zeros_like(u_eps),
            where=valid_energy,
        )
        n_nth_phys = np.minimum(n_nth_eta_phys, n_nth_eps_phys)
        n_nth_phys = np.where(inj_gate > 0, n_nth_phys, 0.0)
        n_nth_phys = np.where(np.isfinite(n_nth_phys) & (n_nth_phys > 0), n_nth_phys, 0.0)
        unth_code = np.divide(
            n_nth_phys * particle_mass,
            rho_unit,
            out=np.zeros_like(n_nth_phys),
            where=rho_unit > 0,
        )

        number_norm_factor = (q - 1.0) * np.power(p_min_phys, q - 1.0)
        spectral_norm_eta = np.where(valid_tail, n_nth_eta_phys * number_norm_factor, 0.0)
        spectral_norm_eps = np.where(valid_energy, n_nth_eps_phys * number_norm_factor, 0.0)
        spectral_norm_eta = np.where(np.isfinite(spectral_norm_eta) & (spectral_norm_eta > 0), spectral_norm_eta, 0.0)
        spectral_norm_eps = np.where(np.isfinite(spectral_norm_eps) & (spectral_norm_eps > 0), spectral_norm_eps, 0.0)

        limit_mode = np.full_like(failure, inj_limit_modes["none"], dtype=np.int16)
        limit_mode[(failure != 0) | clipped | (~valid_tail)] = inj_limit_modes["invalid_or_boundary"]
        limit_mode[(limit_mode == inj_limit_modes["none"]) & (inj_gate <= 0)] = inj_limit_modes["quenched_by_gate"]
        eta_selected = (limit_mode == inj_limit_modes["none"]) & (n_nth_eta_phys <= n_nth_eps_phys)
        eps_selected = (limit_mode == inj_limit_modes["none"]) & (n_nth_eps_phys < n_nth_eta_phys)
        limit_mode[eta_selected] = inj_limit_modes["eta_cap"]
        limit_mode[eps_selected] = inj_limit_modes["eps_cap"]

        unth_code_grid[mask] = unth_code
        eta_inj_e_grid[mask] = eta_inj_e
        eps_nth_e_grid[mask] = eps_nth_e
        e_diss_e_grid[mask] = e_diss_e
        inj_gate_grid[mask] = inj_gate
        n_nth_eta_phys_grid[mask] = n_nth_eta_phys
        n_nth_eps_phys_grid[mask] = n_nth_eps_phys
        n_nth_phys_grid[mask] = n_nth_phys
        spectral_norm_eta_grid[mask] = spectral_norm_eta
        spectral_norm_eps_grid[mask] = spectral_norm_eps
        inj_limit_mode_grid[mask] = limit_mode
        print("  Final UNTH density calculated using PIC dual-cap injection.")
        if logger:
            logger.ai.codepath(
                "Injection branch",
                "PIC dual-cap injection: UNTH is code-unit nonthermal number density after sonic/obliquity/sigma gating",
            )
            logger.ai.codepath("Gamma-min branch", "two-temperature/Sironi-Tran branch used for gamma_min_grid")
            logger.ai.codepath(
                "Relativistic closure",
                f"p_min uses relativistic thermal momentum and K_inj uses finite-cutoff numerical integration with gamma_max={POWERLAW_GAMMA_MAX_DEFAULT:.1e}",
            )
            logger.ai.data(
                "summary.injection_gate",
                {
                    "theta_bn_median_rad": float(np.median(theta_bn[finite_theta_bn])) if np.any(finite_theta_bn) else 0.0,
                    "sigma_median": float(np.median(sigma[np.isfinite(sigma)])) if np.any(np.isfinite(sigma)) else 0.0,
                    "sigma_gate_median": float(np.median(sigma_gate[np.isfinite(sigma_gate)])) if np.any(np.isfinite(sigma_gate)) else 0.0,
                    "obliquity_gate_median": float(np.median(obliquity_gate[np.isfinite(obliquity_gate)])) if np.any(np.isfinite(obliquity_gate)) else 0.0,
                    "inj_gate_median": float(np.median(inj_gate[np.isfinite(inj_gate)])) if np.any(np.isfinite(inj_gate)) else 0.0,
                    "inj_gate_open_fraction": float(np.mean(inj_gate > 0)),
                    "sr_sonic_mach_median": float(np.median(sonic_mach[np.isfinite(sonic_mach)])) if np.any(np.isfinite(sonic_mach)) else 0.0,
                },
            )
            logger.ai.data(
                "summary.nonthermal_closure",
                {
                    "p_min_median": float(np.median(p_min_phys[np.isfinite(p_min_phys)])) if np.any(np.isfinite(p_min_phys)) else 0.0,
                    "k_inj_median": float(np.median(k_inj[np.isfinite(k_inj) & (k_inj > 0)])) if np.any(np.isfinite(k_inj) & (k_inj > 0)) else 0.0,
                    "k_inj_finite_cutoff_gamma_max": POWERLAW_GAMMA_MAX_DEFAULT,
                    "eta_inj_e_median": float(np.median(eta_inj_e[np.isfinite(eta_inj_e)])) if np.any(np.isfinite(eta_inj_e)) else 0.0,
                    "eps_nth_e_median": float(np.median(eps_nth_e[np.isfinite(eps_nth_e)])) if np.any(np.isfinite(eps_nth_e)) else 0.0,
                    "e_diss_e_median": float(np.median(e_diss_e[np.isfinite(e_diss_e)])) if np.any(np.isfinite(e_diss_e)) else 0.0,
                    "n_nth_eta_median_cm^-3": float(np.median(n_nth_eta_phys[np.isfinite(n_nth_eta_phys)])) if np.any(np.isfinite(n_nth_eta_phys)) else 0.0,
                    "n_nth_eps_median_cm^-3": float(np.median(n_nth_eps_phys[np.isfinite(n_nth_eps_phys)])) if np.any(np.isfinite(n_nth_eps_phys)) else 0.0,
                    "n_nth_median_cm^-3": float(np.median(n_nth_phys[np.isfinite(n_nth_phys)])) if np.any(np.isfinite(n_nth_phys)) else 0.0,
                    "unth_code_median": float(np.median(unth_code[np.isfinite(unth_code)])) if np.any(np.isfinite(unth_code)) else 0.0,
                    "gamma_min_median": float(np.median(gamma_min[np.isfinite(gamma_min)])) if np.any(np.isfinite(gamma_min)) else 1.0,
                    "inj_limit_counts": {int(code): int(count) for code, count in zip(*np.unique(limit_mode, return_counts=True))},
                },
            )
        print(
            "  SRMHD mainline injection: "
            f"UNTH_code median={np.median(unth_code):.3e}, n_nth_phys median={np.median(n_nth_phys):.3e} cm^-3, "
            f"gamma_min median={np.median(gamma_min):.2f}, "
            f"gate_open_fraction={np.mean(inj_gate > 0):.1%}"
        )
    elif logger:
        logger.ai.codepath("Nonthermal branch", "no shock cells, returned zero grids")
    nonthermal_properties = {
        "q_grid": q_grid,
        "unth_code_grid": unth_code_grid,
        "C_grid": unth_code_grid,
        "mask": mask,
        "sigma_suppression_grid": sigma_suppression_grid,
        "gamma_min_grid": gamma_min_grid,
        "gamma_min_grid_physical": gamma_min_grid_physical,
        "gamma_min_failure_code_grid": gamma_min_failure_code_grid,
        "theta_e_grid": theta_e_grid,
        "theta_e1_grid": theta_e1_grid,
        "theta_e2_ad_grid": theta_e2_ad_grid,
        "sironi_boost_grid": sironi_boost_grid,
        "R1_grid": r1_grid,
        "Te2_grid": te2_grid,
        "p_min_physical_grid": p_min_physical_grid,
        "eta_inj_e_grid": eta_inj_e_grid,
        "eps_nth_e_grid": eps_nth_e_grid,
        "e_diss_e_grid": e_diss_e_grid,
        "inj_gate_grid": inj_gate_grid,
        "n_nth_eta_phys_grid": n_nth_eta_phys_grid,
        "n_nth_eps_phys_grid": n_nth_eps_phys_grid,
        "n_nth_phys_grid": n_nth_phys_grid,
        "spectral_norm_eta_grid": spectral_norm_eta_grid,
        "spectral_norm_eps_grid": spectral_norm_eps_grid,
        "c_eta_grid": spectral_norm_eta_grid,
        "c_eps_grid": spectral_norm_eps_grid,
        "inj_limit_mode_grid": inj_limit_mode_grid,
        "inj_limit_modes": inj_limit_modes,
        "gamma_failure_codes": gamma_failure_codes,
    }
    if logger:
        logger.ai.func_exit("calculate_nonthermal_electrons", {"result_keys": sorted(nonthermal_properties.keys()), "shock_cells": int(np.sum(mask))})
    return nonthermal_properties


def plot_diagnostic_histograms(shock_properties, nonthermal_props, snapshot_name, output_filename):
    print(f"Generating non-thermal 1D diagnostic histograms for {snapshot_name}...")
    mask = shock_properties["mask"]
    if np.sum(mask) == 0:
        print("  No shock cells found. Skipping 1D histograms.")
        return

    mach_field = shock_properties["mainline_mach"] if "mainline_mach" in shock_properties else shock_properties["upstream_mach"]
    m1 = mach_field[mask]
    q = nonthermal_props["q_grid"][mask]
    unth_vals = nonthermal_props["C_grid"][mask]
    log_unth = np.log10(unth_vals[unth_vals > 0])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle(f"Non-Thermal Electron Diagnostics (1D Histograms) for {snapshot_name}", fontsize=16)

    ax1.hist(m1, bins=50, color="blue", alpha=0.7, log=True)
    ax1.set_title("Upstream Mach Number ($M_1$) Distribution")
    ax1.set_xlabel("Mach Number ($M_1$)")
    ax1.set_ylabel("Count (log scale)")
    ax1.axvline(m1.mean(), color="red", linestyle="dashed", linewidth=2, label=f"Mean: {m1.mean():.2f}")
    ax1.legend()

    ax2.hist(q, bins=50, color="green", alpha=0.7, log=True)
    ax2.set_title("Power-law Index (q) Distribution")
    ax2.set_xlabel("Index (q)")
    ax2.set_ylabel("Count (log scale)")
    ax2.axvline(q.mean(), color="red", linestyle="dashed", linewidth=2, label=f"Mean: {q.mean():.2f}")
    ax2.axvline(1.5, color="black", linestyle="dotted", linewidth=2, label="q_min (M->inf) = 1.5")
    ax2.legend()

    if len(log_unth) > 0:
        ax3.hist(log_unth, bins=50, color="purple", alpha=0.7, log=True)
        ax3.set_title("UNTH Code Density Distribution")
        ax3.set_xlabel("log10(UNTH_code)")
        ax3.set_ylabel("Count (log scale)")
        ax3.axvline(log_unth.mean(), color="red", linestyle="dashed", linewidth=2, label=f"Mean: {log_unth.mean():.2f}")
        ax3.legend()
    else:
        ax3.set_title("UNTH Code Density Distribution")
        ax3.text(0.5, 0.5, "No UNTH > 0 values found", ha="center", va="center", transform=ax3.transAxes)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  1D diagnostic histograms saved to {output_filename}")


def plot_diagnostic_correlations(shock_properties, nonthermal_props, snapshot_name, output_filename):
    print(f"Generating non-thermal 2D diagnostic correlations for {snapshot_name}...")
    mask = shock_properties["mask"]
    if np.sum(mask) == 0:
        print("  No shock cells found. Skipping 2D correlations.")
        return

    mach_field = shock_properties["mainline_mach"] if "mainline_mach" in shock_properties else shock_properties["upstream_mach"]
    m1 = mach_field[mask]
    q = nonthermal_props["q_grid"][mask]
    unth_vals = nonthermal_props["C_grid"][mask]
    log_unth = np.log10(unth_vals[unth_vals > 0])
    m1_for_unth = m1[unth_vals > 0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"Nonthermal Electron Diagnostics (2D Correlations) for {snapshot_name}", fontsize=16)

    hb1 = ax1.hexbin(m1, q, gridsize=50, cmap="viridis", norm=LogNorm())
    fig.colorbar(hb1, ax=ax1, label="Count (log scale)")
    ax1.set_title("Physical Check: $M_1$ vs. $q$")
    ax1.set_xlabel("Upstream Mach Number ($M_1$)")
    ax1.set_ylabel("Power-law Index ($q$)")
    ax1.set_ylim(bottom=1.4)
    ax1.axhline(1.5, color="red", linestyle="dashed", linewidth=2, label="q_min (M->inf) = 1.5")
    ax1.legend()

    if len(log_unth) > 0:
        hb2 = ax2.hexbin(m1_for_unth, log_unth, gridsize=50, cmap="inferno", norm=LogNorm())
        fig.colorbar(hb2, ax=ax2, label="Count (log scale)")
        ax2.set_title("Injection Check: $M_1$ vs. $log_{10}(UNTH_{code})$")
        ax2.set_xlabel("Upstream Mach Number ($M_1$)")
        ax2.set_ylabel("Log10(UNTH_code)")
    else:
        ax2.set_title("Injection Check: $M_1$ vs. $log_{10}(UNTH_{code})$")
        ax2.text(0.5, 0.5, "No UNTH > 0 values found", ha="center", va="center", transform=ax2.transAxes)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  2D diagnostic correlations saved to {output_filename}")
