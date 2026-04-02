"""Non-thermal electron calculations and diagnostics."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.special import betainc


def calculate_nonthermal_electrons(
    shock_properties,
    gamma=4.0 / 3.0,
    x_inj=3.5,
    xi_max=0.05,
    sigma_crit=0.1,
    alpha_sigma=2,
    rho_unit=1.0,
    u_unit=None,
    r_low=1.0,
    r_high=80.0,
    beta_crit=1.0,
    logger=None,
):
    """Calculate non-thermal electron properties for shock cells."""
    if u_unit is None:
        u_unit = rho_unit
    print(
        "Starting non-thermal electron calculation "
        f"(sigma_crit={sigma_crit}, alpha_sigma={alpha_sigma}, rho_unit={rho_unit:.3e}, u_unit={u_unit:.3e})..."
    )
    if logger:
        logger.ai.func_enter(
            "calculate_nonthermal_electrons",
            {"gamma": gamma, "x_inj": x_inj, "xi_max": xi_max, "sigma_crit": sigma_crit,
             "alpha_sigma": alpha_sigma, "rho_unit": rho_unit, "u_unit": u_unit,
             "r_low": r_low, "r_high": r_high, "beta_crit": beta_crit,
             "shock_cells": int(np.sum(shock_properties["mask"]))},
        )
    mask = shock_properties["mask"]
    q_grid = np.zeros_like(mask, dtype=float)
    c_grid = np.zeros_like(mask, dtype=float)
    gamma_min_grid = np.ones_like(mask, dtype=float)
    gamma_min_grid_physical = np.ones_like(mask, dtype=float)
    sigma_suppression_grid = np.ones_like(mask, dtype=float)
    gamma_min_failure_code_grid = np.zeros_like(mask, dtype=np.int16)
    theta_e_grid = np.zeros_like(mask, dtype=float)
    p_min_physical_grid = np.zeros_like(mask, dtype=float)
    gamma_failure_codes = {"ok": 0, "press2_nonpositive": 1, "rho2_nonpositive": 2, "boundary_clipped": 3, "temperature_invalid": 4, "gamma_min_le_one": 5}
    if np.any(mask):
        m_e = 9.1094e-28
        c_light = 2.9979e10
        k_b = 1.3806e-16
        mu_i, mu_e, mu_tot, game = 1.0, 1.0, 0.5, 4.0 / 3.0
        m1 = shock_properties["upstream_mach"][mask]
        t2 = shock_properties["downstream_temp"][mask]
        n_e2 = shock_properties["downstream_n_e"][mask]
        rho2 = shock_properties.get("rho2_code_grid", np.zeros_like(mask, dtype=float))[mask]
        press2 = shock_properties.get("press2_code_grid", np.zeros_like(mask, dtype=float))[mask]
        press2_over_rho2 = shock_properties.get("press2_over_rho2_grid", np.zeros_like(mask, dtype=float))[mask]
        bsq2 = shock_properties.get("bsq2_code_grid", np.zeros_like(mask, dtype=float))[mask]
        beta2 = shock_properties.get("beta2_grid", np.zeros_like(mask, dtype=float))[mask]
        clipped = shock_properties.get("sample_boundary_clipped_grid", np.zeros_like(mask, dtype=bool))[mask]
        print(f"  Downstream state: T2 median={np.median(t2):.3e} K, ne median={np.median(n_e2):.3e} cm^-3")
        if logger:
            logger.ai.data("nt.T2_shocks", t2, "K")
            logger.ai.data("nt.n_e2_shocks", n_e2, "cm^-3")
        inv_tau = (gamma - 1.0) / (gamma + 1.0) + (2.0 / (gamma + 1.0)) / m1**2
        tau = 1.0 / inv_tau
        q = (tau + 2.0) / (tau - 1.0)
        q_grid[mask] = q
        print("  Power-law index 'q' calculated.")
        if logger:
            logger.ai.data("nt.q", q)
        p_min_sq = np.maximum(x_inj**2 * 2.0 * k_b * t2 / (m_e * c_light**2), 0.0)
        p_min = np.sqrt(p_min_sq)
        with np.errstate(divide="ignore", invalid="ignore"):
            # gamma_min physical branch stays in the same code-unit thermodynamic ratio
            # used by shock sampling. Multiplying by u_unit/rho_unit would inject an
            # extra ~c^2 factor and catastrophically inflate theta_e.
            uu_over_rho = np.divide(press2, rho2, out=np.zeros_like(press2), where=rho2 > 0)
            uu_over_rho = np.where(np.isfinite(press2_over_rho2), press2_over_rho2, uu_over_rho)
            beta_ratio_sq = np.square(beta2 / beta_crit)
            trat = (r_high * beta_ratio_sq + r_low) / (1.0 + beta_ratio_sq)
            dfactor = mu_tot / mu_e + mu_tot / mu_i * trat
            theta_e = np.divide(uu_over_rho, dfactor, out=np.zeros_like(uu_over_rho), where=dfactor > 0) * (game - 1.0)
        t_e = np.divide(theta_e * m_e * c_light**2, k_b, out=np.zeros_like(theta_e), where=theta_e > 0)
        p_min_phys_sq = np.maximum(2.0 * x_inj**2 * theta_e, 0.0)
        p_min_phys = np.sqrt(p_min_phys_sq)
        gamma_min_raw = np.sqrt(1.0 + p_min_phys_sq)
        failure = np.full(gamma_min_raw.shape, gamma_failure_codes["ok"], dtype=np.int16)
        failure[press2 <= 0] = gamma_failure_codes["press2_nonpositive"]
        failure[(failure == 0) & (rho2 <= 0)] = gamma_failure_codes["rho2_nonpositive"]
        failure[(failure == 0) & clipped] = gamma_failure_codes["boundary_clipped"]
        invalid_temp = ((~np.isfinite(beta2)) | (~np.isfinite(trat)) | (~np.isfinite(uu_over_rho)) | (~np.isfinite(theta_e)) | (~np.isfinite(t_e)) | (trat <= 0) | (theta_e <= 0) | (t_e <= 0))
        failure[(failure == 0) & invalid_temp] = gamma_failure_codes["temperature_invalid"]
        failure[(failure == 0) & (gamma_min_raw <= 1.0)] = gamma_failure_codes["gamma_min_le_one"]
        gamma_min = np.where(failure == 0, gamma_min_raw, 1.0)
        gamma_min_grid[mask] = gamma_min
        gamma_min_grid_physical[mask] = gamma_min
        gamma_min_failure_code_grid[mask] = failure
        theta_e_grid[mask] = theta_e
        p_min_physical_grid[mask] = p_min_phys
        beta_med = np.median(beta2[np.isfinite(beta2)]) if np.any(np.isfinite(beta2)) else 0.0
        trat_med = np.median(trat[np.isfinite(trat)]) if np.any(np.isfinite(trat)) else 0.0
        theta_med = np.median(theta_e[np.isfinite(theta_e) & (theta_e > 0)]) if np.any(np.isfinite(theta_e) & (theta_e > 0)) else 0.0
        print(f"  gamma_min beta closure: beta median={beta_med:.3e}, trat median={trat_med:.3e}, theta_e median={theta_med:.3e}")
        unique_codes, unique_counts = np.unique(failure, return_counts=True)
        failure_summary = {int(code): int(count) for code, count in zip(unique_codes, unique_counts)}
        valid_gamma = failure == 0
        print(f"  gamma_min physical branch failure summary: {failure_summary}")
        print(f"  gamma_min physical branch: valid={np.sum(valid_gamma)}/{failure.size}, median_valid={np.median(gamma_min[valid_gamma]) if np.any(valid_gamma) else 1.0:.2f}")
        if logger:
            logger.ai.data("nt.rho2_code_shocks", rho2)
            logger.ai.data("nt.press2_code_shocks", press2)
            logger.ai.data("nt.press2_over_rho2_shocks", uu_over_rho)
            logger.ai.data("nt.bsq2_code_shocks", bsq2)
            logger.ai.data("nt.beta2_shocks", beta2)
            logger.ai.data("nt.trat_shocks", trat)
            logger.ai.data("nt.theta_e2_shocks", theta_e)
            logger.ai.data("nt.gamma_min_failure", failure)
            logger.ai.codepath("Gamma-min branch", f"failure_counts={failure_summary}")
            logger.ai.codepath("Gamma-min closure", f"ipole beta closure r_low={r_low}, r_high={r_high}, beta_crit={beta_crit}")
            logger.ai.debug(f"Gamma-min beta sampling stats={{'finite_beta2_count': {int(np.sum(np.isfinite(beta2)))}, 'finite_thetae_count': {int(np.sum(np.isfinite(theta_e) & (theta_e > 0)))}}}")
            logger.ai.debug(f"Gamma-min closure metadata={{'r_low': {r_low}, 'r_high': {r_high}, 'beta_crit': {beta_crit}, 'mu_i': 1.0, 'mu_e': 1.0, 'mu_tot': 0.5, 'game': {game}}}")
        eta_lin = (4.0 / np.sqrt(np.pi)) * (x_inj**3 / (q - 1.0)) * np.exp(-x_inj**2)
        print("  Linear injection fraction 'eta_lin' calculated.")
        if logger:
            logger.ai.data("nt.eta_lin", eta_lin)
        k_inj = np.zeros_like(q)
        valid_q = (q > 2.0) & (q < 3.0)
        if np.any(valid_q):
            qv = q[valid_q]
            p2v = p_min_sq[valid_q]
            x_beta = 1.0 / (1.0 + p2v)
            incomplete_beta = betainc((qv - 2.0) / 2.0, (3.0 - qv) / 2.0, x_beta)
            k_inj[valid_q] = (p_min[valid_q] ** (qv - 1.0)) / 2.0 * incomplete_beta + np.sqrt(1.0 + p2v) - 1.0
        print("  Mean kinetic energy 'K_inj' calculated.")
        if logger:
            logger.ai.data("nt.K_inj", k_inj)
        e_nonthermal = eta_lin * k_inj * n_e2 * m_e * c_light**2
        e_thermal = 3.0 * n_e2 * k_b * t2
        xi_lin = np.divide(e_nonthermal, e_thermal, out=np.zeros_like(e_nonthermal), where=e_thermal != 0)
        delta = xi_lin / xi_max
        with np.errstate(divide="ignore", invalid="ignore"):
            prefactor = np.power(m_e * c_light**2 / (2.0 * np.pi * k_b * t2), 1.5)
        f_e = 4.0 * np.pi * n_e2 * p_min**2 * prefactor * np.exp(-m_e * c_light**2 * p_min_sq / (2.0 * k_b * t2))
        norm_factor = np.divide(1.0 - np.exp(-delta), delta, out=np.zeros_like(delta), where=delta != 0)
        norm_factor[delta == 0] = 1.0
        c_norm = norm_factor * f_e * p_min**q
        n_inj = np.divide(c_norm * p_min ** (1.0 - q), q - 1.0, out=np.zeros_like(c_norm), where=(q - 1.0) != 0)
        n_inj = np.where(np.isfinite(n_inj) & (n_inj > 0), n_inj, 0.0)
        if "sigma_grid" in shock_properties:
            sigma = shock_properties["sigma_grid"][mask]
            suppression = 1.0 / (1.0 + (sigma / sigma_crit) ** alpha_sigma)
            n_inj *= suppression
            sigma_suppression_grid[mask] = suppression
            print(f"  Sigma suppression applied: median factor={np.median(suppression):.3f}, cells with factor<0.5: {np.mean(suppression < 0.5):.1%}")
            if logger:
                logger.ai.codepath("Sigma suppression branch", "sigma_grid found and applied")
                logger.ai.data("nt.sigma_at_shocks", sigma)
                logger.ai.data("nt.suppression", suppression)
        else:
            print("  No sigma_grid found in shock_properties; skipping sigma suppression.")
            if logger:
                logger.ai.codepath("Sigma suppression branch", "sigma_grid missing")
        c_grid[mask] = n_inj
        print("  Final normalization 'C' calculated using full physics model.")
        if logger:
            logger.ai.data("nt.N_inj", n_inj)
            logger.ai.data("nt.gamma_min", gamma_min)
            logger.ai.codepath("Injection branch", "legacy empirical interface retained for C_grid/UNTH")
            logger.ai.codepath("Gamma-min branch", "physical downstream thermodynamic interface used for gamma_min_grid")
        print(f"  legacy injection branch preserved: C median={np.median(n_inj):.3e}, gamma_min median={np.median(gamma_min):.2f}")
    elif logger:
        logger.ai.codepath("Nonthermal branch", "no shock cells, returned zero grids")
    nonthermal_properties = {"q_grid": q_grid, "C_grid": c_grid, "mask": mask, "sigma_suppression_grid": sigma_suppression_grid, "gamma_min_grid": gamma_min_grid, "gamma_min_grid_physical": gamma_min_grid_physical, "gamma_min_failure_code_grid": gamma_min_failure_code_grid, "theta_e_grid": theta_e_grid, "p_min_physical_grid": p_min_physical_grid, "gamma_failure_codes": gamma_failure_codes}
    if logger:
        logger.ai.func_exit("calculate_nonthermal_electrons", {"result_keys": sorted(nonthermal_properties.keys()), "shock_cells": int(np.sum(mask))})
    return nonthermal_properties


def plot_diagnostic_histograms(shock_properties, nonthermal_props, snapshot_name, output_filename):
    print(f"Generating non-thermal 1D diagnostic histograms for {snapshot_name}...")
    mask = shock_properties["mask"]
    if np.sum(mask) == 0:
        print("  No shock cells found. Skipping 1D histograms.")
        return

    m1 = shock_properties["upstream_mach"][mask]
    q = nonthermal_props["q_grid"][mask]
    c_vals = nonthermal_props["C_grid"][mask]
    log_c = np.log10(c_vals[c_vals > 0])

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

    if len(log_c) > 0:
        ax3.hist(log_c, bins=50, color="purple", alpha=0.7, log=True)
        ax3.set_title("Nonthermal Electron Density (log10 N) Distribution")
        ax3.set_xlabel("log10(N)")
        ax3.set_ylabel("Count (log scale)")
        ax3.axvline(log_c.mean(), color="red", linestyle="dashed", linewidth=2, label=f"Mean: {log_c.mean():.2f}")
        ax3.legend()
    else:
        ax3.set_title("Normalization (log10 N) Distribution")
        ax3.text(0.5, 0.5, "No C > 0 values found", ha="center", va="center", transform=ax3.transAxes)

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

    m1 = shock_properties["upstream_mach"][mask]
    q = nonthermal_props["q_grid"][mask]
    c_vals = nonthermal_props["C_grid"][mask]
    log_c = np.log10(c_vals[c_vals > 0])
    m1_for_c = m1[c_vals > 0]

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

    if len(log_c) > 0:
        hb2 = ax2.hexbin(m1_for_c, log_c, gridsize=50, cmap="inferno", norm=LogNorm())
        fig.colorbar(hb2, ax=ax2, label="Count (log scale)")
        ax2.set_title("Injection Check: $M_1$ vs. $log_{10}(N)$")
        ax2.set_xlabel("Upstream Mach Number ($M_1$)")
        ax2.set_ylabel("Log10(Normalization N)")
    else:
        ax2.set_title("Injection Check: $M_1$ vs. $log_{10}(N)$")
        ax2.text(0.5, 0.5, "No C > 0 values found", ha="center", va="center", transform=ax2.transAxes)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  2D diagnostic correlations saved to {output_filename}")
