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
    logger=None,
):
    """Calculate non-thermal electron properties for shock cells."""
    print(
        "Starting non-thermal electron calculation "
        f"(sigma_crit={sigma_crit}, alpha_sigma={alpha_sigma}, rho_unit={rho_unit:.3e})..."
    )
    if logger:
        logger.ai.func_enter(
            "calculate_nonthermal_electrons",
            {
                "gamma": gamma,
                "x_inj": x_inj,
                "xi_max": xi_max,
                "sigma_crit": sigma_crit,
                "alpha_sigma": alpha_sigma,
                "rho_unit": rho_unit,
                "shock_cells": int(np.sum(shock_properties["mask"])),
            },
        )

    mask = shock_properties["mask"]
    m1 = shock_properties["upstream_mach"]
    t2 = shock_properties["downstream_temp"]
    n_e2 = shock_properties["downstream_n_e"]

    m_e = 9.1094e-28
    c_light = 2.9979e10
    k_b = 1.3806e-16

    q_grid = np.zeros_like(mask, dtype=float)
    c_grid = np.zeros_like(mask, dtype=float)
    gamma_min_grid = np.ones_like(mask, dtype=float)
    sigma_suppression_grid = np.ones_like(mask, dtype=float)

    if np.any(mask):
        m1_shocks = m1[mask]
        t2_shocks = t2[mask] * c_light**2
        n_e2_shocks = n_e2[mask] * rho_unit

        print(
            f"  Unit conversion applied: T2 median={np.median(t2_shocks):.3e} K, "
            f"ne median={np.median(n_e2_shocks):.3e} cm^-3"
        )
        if logger:
            logger.ai.data("nt.T2_shocks", t2_shocks, "K")
            logger.ai.data("nt.n_e2_shocks", n_e2_shocks, "cm^-3")

        inv_tau = (gamma - 1.0) / (gamma + 1.0) + (2.0 / (gamma + 1.0)) / m1_shocks**2
        tau = 1.0 / inv_tau
        q = (tau + 2.0) / (tau - 1.0)
        q_grid[mask] = q
        print("  Power-law index 'q' calculated.")
        if logger:
            logger.ai.data("nt.q", q)

        thermal_term = 2.0 * k_b * t2_shocks / (m_e * c_light**2)
        p_min_sq = x_inj**2 * thermal_term
        p_min = np.sqrt(p_min_sq)

        eta_lin = (4.0 / np.sqrt(np.pi)) * (x_inj**3 / (q - 1.0)) * np.exp(-x_inj**2)
        print("  Linear injection fraction 'eta_lin' calculated.")
        if logger:
            logger.ai.data("nt.eta_lin", eta_lin)

        valid_q_mask = (q > 2.0) & (q < 3.0)
        k_inj = np.zeros_like(q)
        if np.any(valid_q_mask):
            q_valid = q[valid_q_mask]
            p_min_sq_valid = p_min_sq[valid_q_mask]
            x_beta = 1.0 / (1.0 + p_min_sq_valid)
            a_beta = (q_valid - 2.0) / 2.0
            b_beta = (3.0 - q_valid) / 2.0
            incomplete_beta_func = betainc(a_beta, b_beta, x_beta)
            term1 = (p_min[valid_q_mask] ** (q_valid - 1.0)) / 2.0
            term2 = np.sqrt(1.0 + p_min_sq_valid) - 1.0
            k_inj[valid_q_mask] = term1 * incomplete_beta_func + term2
        print("  Mean kinetic energy 'K_inj' calculated.")
        if logger:
            logger.ai.data("nt.K_inj", k_inj)

        e_nonthermal = eta_lin * k_inj * n_e2_shocks * m_e * c_light**2
        e_thermal_increase = 3.0 * n_e2_shocks * k_b * t2_shocks
        xi_lin = np.divide(
            e_nonthermal,
            e_thermal_increase,
            out=np.zeros_like(e_nonthermal),
            where=e_thermal_increase != 0,
        )

        delta = xi_lin / xi_max
        f_e_p_min = (
            4.0
            * np.pi
            * n_e2_shocks
            * p_min**2
            * (m_e * c_light**2 / (2.0 * np.pi * k_b * t2_shocks)) ** 1.5
            * np.exp(-m_e * c_light**2 * p_min_sq / (2.0 * k_b * t2_shocks))
        )

        norm_factor = np.divide(1.0 - np.exp(-delta), delta, out=np.zeros_like(delta), where=delta != 0)
        norm_factor[delta == 0] = 1.0

        c_norm = norm_factor * f_e_p_min * p_min**q
        n_inj = (c_norm * p_min ** (1.0 - q)) / (q - 1.0)

        if "sigma_grid" in shock_properties:
            sigma_at_shocks = shock_properties["sigma_grid"][mask]
            suppression = 1.0 / (1.0 + (sigma_at_shocks / sigma_crit) ** alpha_sigma)
            n_inj *= suppression
            sigma_suppression_grid[mask] = suppression
            print(
                f"  Sigma suppression applied: median factor={np.median(suppression):.3f}, "
                f"cells with factor<0.5: {np.mean(suppression < 0.5):.1%}"
            )
            if logger:
                logger.ai.codepath("Sigma suppression branch", "sigma_grid found and applied")
                logger.ai.data("nt.sigma_at_shocks", sigma_at_shocks)
                logger.ai.data("nt.suppression", suppression)
        else:
            print("  No sigma_grid found in shock_properties; skipping sigma suppression.")
            if logger:
                logger.ai.codepath("Sigma suppression branch", "sigma_grid missing")

        c_grid[mask] = n_inj
        gamma_min_at_shocks = np.sqrt(1.0 + p_min**2)
        gamma_min_grid[mask] = gamma_min_at_shocks
        print(
            f"  gamma_min computed from p_min: median={np.median(gamma_min_at_shocks):.2f}, "
            f"range=[{np.min(gamma_min_at_shocks):.2f}, {np.max(gamma_min_at_shocks):.2f}]"
        )
        print("  Final normalization 'C' calculated using full physics model.")

        if logger:
            logger.ai.data("nt.N_inj", n_inj)
            logger.ai.data("nt.gamma_min", gamma_min_at_shocks)
    elif logger:
        logger.ai.codepath("Nonthermal branch", "no shock cells, returned zero grids")

    nonthermal_properties = {
        "q_grid": q_grid,
        "C_grid": c_grid,
        "mask": mask,
        "sigma_suppression_grid": sigma_suppression_grid,
        "gamma_min_grid": gamma_min_grid,
    }
    if logger:
        logger.ai.func_exit(
            "calculate_nonthermal_electrons",
            {"result_keys": sorted(nonthermal_properties.keys()), "shock_cells": int(np.sum(mask))},
        )
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
