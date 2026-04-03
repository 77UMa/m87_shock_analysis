#!/usr/bin/env python3
"""Shared data loading, physics calculation, and HDF5 export helpers."""

import gc
import os
import sys

import h5py
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

try:
    from pyathena import athena_read
    from src.core.nt_electron_v1 import calculate_nonthermal_electrons
    from src.core.shock_v1 import find_shocks_in_roi_mhd
except ImportError as exc:
    print(f"Fatal error: failed to import scientific modules. {exc}")
    sys.exit(1)


def _human_info(logger, message: str) -> None:
    if logger:
        logger.human.info(message)
    else:
        print(message)


def _ai_debug(logger, message: str) -> None:
    if logger:
        logger.ai.debug(message)


def get_coords(data, axis_idx):
    key_v = f"x{axis_idx}v"
    key_f = f"x{axis_idx}f"
    if key_v in data:
        return data[key_v]
    if key_f in data:
        face = data[key_f]
        return 0.5 * (face[:-1] + face[1:])
    raise KeyError(f"Critical error: neither '{key_v}' nor '{key_f}' found.")


def load_and_slice_data(filename, roi_params, logger=None):
    """Load Athena++ data and extract the requested ROI."""
    basename = os.path.basename(filename)
    _human_info(logger, f"Loading data from {basename}")
    if logger:
        logger.ai.func_enter("load_and_slice_data", {"filename": basename, "roi_params": roi_params})

    full_data = athena_read.athdf(filename, level=4)
    full_data["x1v"] = get_coords(full_data, 1)
    full_data["x2v"] = get_coords(full_data, 2)
    full_data["x3v"] = get_coords(full_data, 3)

    r_coords = full_data["x1f"]
    th_coords = full_data["x2f"]
    ph_coords = full_data["x3f"]

    i_start = np.searchsorted(r_coords, roi_params["r_min"], side="left")
    i_end = np.searchsorted(r_coords, roi_params["r_max"], side="right")
    j_start = np.searchsorted(th_coords, roi_params["theta_min"], side="left")
    j_end = np.searchsorted(th_coords, roi_params["theta_max"], side="right")
    k_start = np.searchsorted(ph_coords, roi_params["phi_min"], side="left")
    k_end = np.searchsorted(ph_coords, roi_params["phi_max"], side="right")

    if logger:
        logger.ai.codepath(
            "ROI slice indices",
            f"i=[{i_start}, {i_end}), j=[{j_start}, {j_end}), k=[{k_start}, {k_end})",
        )

    if i_start >= i_end or j_start >= j_end or k_start >= k_end:
        if logger:
            logger.ai.codepath("Invalid ROI slice", f"filename={basename}")
            logger.ai.func_exit("load_and_slice_data", {"result": None})
        _human_info(logger, f"Invalid ROI slice for {basename}")
        return None

    roi_data = {"Time": full_data.get("Time", 0.0)}
    target_keys = ["rho", "press", "vel1", "vel2", "vel3", "Bcc1", "Bcc2", "Bcc3"]
    if "Bcc1" not in full_data:
        target_keys = ["rho", "press", "vel1", "vel2", "vel3", "B1", "B2", "B3"]
        if logger:
            logger.ai.codepath("Fallback magnetic field keys", "Using B1/B2/B3 instead of Bcc1/Bcc2/Bcc3")

    for key in target_keys:
        roi_data[key] = full_data[key][k_start:k_end, j_start:j_end, i_start:i_end]
        if key.startswith("B") and not key.startswith("Bcc"):
            roi_data[f"Bcc{key[-1]}"] = roi_data[key]

    roi_data["x1f"] = r_coords[i_start : i_end + 1]
    roi_data["x2f"] = th_coords[j_start : j_end + 1]
    roi_data["x3f"] = ph_coords[k_start : k_end + 1]
    roi_data["x1v"] = full_data["x1v"][i_start:i_end]
    roi_data["x2v"] = full_data["x2v"][j_start:j_end]
    roi_data["x3v"] = full_data["x3v"][k_start:k_end]

    if logger:
        for key in ("rho", "press", "vel1", "Bcc1"):
            logger.ai.data(f"roi_data.{key}", roi_data[key])
        logger.ai.data("roi_data.x1v", roi_data["x1v"], "r_g")
        logger.ai.data("roi_data.x2v", roi_data["x2v"], "rad")
        logger.ai.data("roi_data.x3v", roi_data["x3v"], "rad")

    del full_data
    gc.collect()

    if logger:
        logger.ai.func_exit(
            "load_and_slice_data",
            {"keys": sorted(roi_data.keys()), "rho_shape": roi_data["rho"].shape, "time": roi_data["Time"]},
        )
    return roi_data


def calculate_dsa_physics(roi_data, shock_params, nt_params, logger=None):
    """Run shock detection and non-thermal electron calculations."""
    _human_info(logger, "Calculating shock properties")
    nt_params = dict(nt_params)
    deprecated_temp_fraction = nt_params.pop("electron_temp_fraction", None)
    nt_params.setdefault("r_low", 1.0)
    nt_params.setdefault("r_high", 80.0)
    nt_params.setdefault("beta_crit", 1.0)
    if logger:
        logger.ai.func_enter(
            "calculate_dsa_physics",
            {"shock_params": shock_params, "nt_params": nt_params, "rho_shape": roi_data["rho"].shape},
        )
        if deprecated_temp_fraction is not None:
            logger.ai.codepath(
                "NT parameter migration",
                "deprecated electron_temp_fraction ignored; using r_low/r_high/beta_crit closure",
            )

    shock_props = find_shocks_in_roi_mhd(roi_data, logger=logger, **shock_params)

    gamma = shock_params.get("gamma", 4.0 / 3.0)
    b_sq = roi_data["Bcc1"] ** 2 + roi_data["Bcc2"] ** 2 + roi_data["Bcc3"] ** 2
    uu = roi_data["press"] / (gamma - 1.0)
    denom = roi_data["rho"] + uu + roi_data["press"]
    sigma_grid = np.where(denom > 0, b_sq / (2.0 * denom), 0.0)
    shock_props["sigma_grid"] = sigma_grid
    shock_props["sigma2_grid"] = np.where(shock_props["mask"], sigma_grid, 0.0)

    if logger:
        logger.ai.data("shock_props.sigma_grid", sigma_grid)

    if np.any(shock_props["mask"]):
        sigma_vals = sigma_grid[shock_props["mask"]]
        _human_info(
            logger,
            "Shock sigma summary: "
            f"median={np.median(sigma_vals):.3e}, max={np.max(sigma_vals):.3e}, fraction sigma>0.1={np.mean(sigma_vals > 0.1):.1%}",
        )
        _human_info(logger, "Calculating non-thermal electrons")
        nonthermal_props = calculate_nonthermal_electrons(shock_props, logger=logger, **nt_params)
    else:
        _human_info(logger, "No shocks found, initializing non-thermal properties to zero")
        if logger:
            logger.ai.codepath("No shock branch", "calculate_nonthermal_electrons skipped")
        nonthermal_props = {
            "q_grid": np.zeros_like(roi_data["rho"]),
            "C_grid": np.zeros_like(roi_data["rho"]),
            "mask": shock_props["mask"],
            "sigma_suppression_grid": np.ones_like(roi_data["rho"]),
            "gamma_min_grid": np.ones_like(roi_data["rho"]),
            "gamma_min_grid_physical": np.ones_like(roi_data["rho"]),
            "gamma_min_failure_code_grid": np.zeros_like(roi_data["rho"], dtype=np.int16),
            "theta_e_grid": np.zeros_like(roi_data["rho"]),
            "p_min_physical_grid": np.zeros_like(roi_data["rho"]),
            "gamma_failure_codes": {
                "ok": 0,
                "press2_nonpositive": 1,
                "rho2_nonpositive": 2,
                "boundary_clipped": 3,
                "temperature_invalid": 4,
                "gamma_min_le_one": 5,
            },
        }

        shock_defaults = {
            "rho2_code_grid": np.zeros_like(roi_data["rho"]),
            "press2_code_grid": np.zeros_like(roi_data["rho"]),
            "press2_over_rho2_grid": np.zeros_like(roi_data["rho"]),
            "bsq2_code_grid": np.zeros_like(roi_data["rho"]),
            "beta2_grid": np.zeros_like(roi_data["rho"]),
            "mask_sr_refined": np.zeros_like(roi_data["rho"], dtype=bool),
            "h_rel_upstream": np.zeros_like(roi_data["rho"]),
            "w_rel_upstream": np.zeros_like(roi_data["rho"]),
            "v_n_upstream": np.zeros_like(roi_data["rho"]),
            "u_n_upstream": np.zeros_like(roi_data["rho"]),
            "theta_Bn": np.zeros_like(roi_data["rho"]),
            "cfast_n_upstream": np.zeros_like(roi_data["rho"]),
            "sr_mach_normal": np.zeros_like(roi_data["rho"]),
            "ptot_jump": np.zeros_like(roi_data["rho"]),
            "entropy_jump": np.zeros_like(roi_data["rho"]),
            "jump_residual_light": np.zeros_like(roi_data["rho"]),
            "sample_boundary_clipped_grid": np.zeros_like(roi_data["rho"], dtype=bool),
            "sample_k2_grid": np.full_like(roi_data["rho"], -1, dtype=int),
            "sample_j2_grid": np.full_like(roi_data["rho"], -1, dtype=int),
            "sample_i2_grid": np.full_like(roi_data["rho"], -1, dtype=int),
            "sampling_stats": {
                "candidate_count": 0,
                "verified_count": 0,
                "boundary_clipped_candidate_count": 0,
                "boundary_clipped_verified_count": 0,
                "sr_refined_count": 0,
                "sr_rejected_low_mach_count": 0,
                "sr_rejected_jump_count": 0,
                "sr_rejected_entropy_count": 0,
            },
        }
        for key, value in shock_defaults.items():
            shock_props.setdefault(key, value)

        if logger:
            logger.ai.codepath("No shock branch", "initialized split gamma_min diagnostics to defaults")

    if logger:
        logger.ai.func_exit(
            "calculate_dsa_physics",
            {
                "shock_cells": int(np.sum(shock_props["mask"])),
                "nonthermal_keys": sorted(nonthermal_props.keys()),
            },
        )
    return shock_props, nonthermal_props


def save_h5_file(output_h5, roi_data, shock_props, nonthermal_props, config, logger=None):
    """Write the ipole input HDF5 file."""
    if logger:
        logger.ai.func_enter("save_h5_file", {"output_h5": output_h5})
    _human_info(logger, f"Saving HDF5 to {os.path.basename(output_h5)}")

    gamma = config["shock_params"]["gamma"]
    spin = config["physics"]["spin"]
    hslope = config["physics"]["hslope"]
    r0 = config["physics"]["R0"]

    rho = roi_data["rho"]
    nk, nj, ni = rho.shape
    r_vals = roi_data["x1v"]
    r_3d = r_vals[np.newaxis, np.newaxis, :]

    scale_1 = 1.0 / r_3d
    scale_2 = 1.0 / np.pi
    scale_3 = 1.0

    if logger:
        logger.ai.data("save.scale_1", scale_1)
        logger.ai.debug(f"Transformation scales: scale_2={scale_2}, scale_3={scale_3}")

    with h5py.File(output_h5, "w") as handle:
        handle.create_dataset("t", data=roi_data.get("Time", 0.0))
        handle.create_dataset("dump_cadence", data=1.0)

        hdr = handle.create_group("header")
        hdr.create_dataset("n1", data=ni, dtype="i4")
        hdr.create_dataset("n2", data=nj, dtype="i4")
        hdr.create_dataset("n3", data=nk, dtype="i4")
        hdr.create_dataset("n_prim", data=8, dtype="i4")
        hdr.create_dataset("gam", data=gamma)
        hdr.create_dataset("has_electrons", data=0, dtype="i4")
        hdr.create_dataset("metric", data=np.bytes_("MKS"))

        geom = hdr.create_group("geom")
        r_f = roi_data["x1f"]
        th_f = roi_data["x2f"]
        ph_f = roi_data["x3f"]

        geom.create_dataset("startx1", data=np.log(r_f[0]))
        geom.create_dataset("startx2", data=th_f[0] / np.pi)
        geom.create_dataset("startx3", data=ph_f[0])
        geom.create_dataset("dx1", data=np.log(r_f[-1] / r_f[0]) / ni)
        geom.create_dataset("dx2", data=(th_f[-1] - th_f[0]) / (np.pi * nj))
        geom.create_dataset("dx3", data=(ph_f[-1] - ph_f[0]) / nk if nk > 1 else 2 * np.pi)

        mks = geom.create_group("mks")
        mks.create_dataset("a", data=spin)
        mks.create_dataset("hslope", data=hslope)
        mks.create_dataset("R0", data=r0)
        mks.create_dataset("r_in", data=r_f[0])
        mks.create_dataset("r_out", data=r_f[-1])
        mks.create_dataset("r_eh", data=1.0 + np.sqrt(1.0 - spin**2))

        uu = roi_data["press"] / (gamma - 1.0)
        b1_mks = roi_data["Bcc1"] * scale_1
        b2_mks = roi_data["Bcc2"] * scale_2
        b3_mks = roi_data["Bcc3"] * scale_3
        v1_mks = roi_data["vel1"] * scale_1
        v2_mks = roi_data["vel2"] * scale_2
        v3_mks = roi_data["vel3"] * scale_3

        prims = np.stack([rho, uu, v1_mks, v2_mks, v3_mks, b1_mks, b2_mks, b3_mks], axis=-1)
        handle.create_dataset("prims", data=prims.transpose(2, 1, 0, 3).astype("f4"))

        mask = shock_props["mask"]
        c_grid = nonthermal_props.get("C_grid", np.zeros_like(rho))
        q_grid = nonthermal_props.get("q_grid", np.zeros_like(rho))
        gamma_min_grid = nonthermal_props.get("gamma_min_grid", np.ones_like(rho))
        gamma_failure_grid = nonthermal_props.get("gamma_min_failure_code_grid", np.zeros_like(rho, dtype=np.int16))
        p_grid = np.where(mask, q_grid - 1.0, 3.0)

        shock_gamma_vals = gamma_min_grid[mask] if np.any(mask) else np.array([])
        fallback_risk_count = int(np.sum(mask & (gamma_min_grid <= 1.0)))
        if shock_gamma_vals.size > 0:
            gamma_stats = {
                "min": float(np.min(shock_gamma_vals)),
                "median": float(np.median(shock_gamma_vals)),
                "max": float(np.max(shock_gamma_vals)),
                "shock_count": int(shock_gamma_vals.size),
                "fallback_risk_count": fallback_risk_count,
                "fallback_risk_fraction": float(fallback_risk_count / shock_gamma_vals.size),
            }
        else:
            gamma_stats = {
                "min": 1.0,
                "median": 1.0,
                "max": 1.0,
                "shock_count": 0,
                "fallback_risk_count": 0,
                "fallback_risk_fraction": 0.0,
            }

        _human_info(
            logger,
            "Pre-HDF5 gamma_min summary: "
            f"min={gamma_stats['min']:.3e}, median={gamma_stats['median']:.3e}, max={gamma_stats['max']:.3e}, "
            f"fallback_risk={gamma_stats['fallback_risk_count']}/{gamma_stats['shock_count']}",
        )

        handle.create_dataset("KEL", data=mask.transpose(2, 1, 0).astype("f8"))
        if "mask_sr_refined" in shock_props:
            handle.create_dataset("KEL_SR", data=shock_props["mask_sr_refined"].transpose(2, 1, 0).astype("f8"))
        handle.create_dataset("UNTH", data=c_grid.transpose(2, 1, 0).astype("f8"))
        handle.create_dataset("p", data=p_grid.transpose(2, 1, 0).astype("f8"))
        handle.create_dataset("GAMMA_MIN", data=gamma_min_grid.transpose(2, 1, 0).astype("f8"))
        handle.create_dataset("GAMMA_MIN_FAILURE_CODE", data=gamma_failure_grid.transpose(2, 1, 0).astype("i2"))

        if "sigma_grid" in shock_props:
            handle.create_dataset("sigma", data=shock_props["sigma_grid"].transpose(2, 1, 0).astype("f4"))
        if "sigma2_grid" in shock_props:
            handle.create_dataset("sigma2", data=shock_props["sigma2_grid"].transpose(2, 1, 0).astype("f4"))
        if "sigma_suppression_grid" in nonthermal_props:
            handle.create_dataset(
                "sigma_suppression",
                data=nonthermal_props["sigma_suppression_grid"].transpose(2, 1, 0).astype("f4"),
            )
        if "rho2_code_grid" in shock_props:
            handle.create_dataset("RHO2_CODE", data=shock_props["rho2_code_grid"].transpose(2, 1, 0).astype("f8"))
        if "press2_code_grid" in shock_props:
            handle.create_dataset("PRESS2_CODE", data=shock_props["press2_code_grid"].transpose(2, 1, 0).astype("f8"))
        if "press2_over_rho2_grid" in shock_props:
            handle.create_dataset(
                "PRESS2_OVER_RHO2",
                data=shock_props["press2_over_rho2_grid"].transpose(2, 1, 0).astype("f8"),
            )
        if "sr_mach_normal" in shock_props:
            handle.create_dataset("SR_MACH_NORMAL", data=shock_props["sr_mach_normal"].transpose(2, 1, 0).astype("f8"))
        if "theta_Bn" in shock_props:
            handle.create_dataset("THETA_BN", data=shock_props["theta_Bn"].transpose(2, 1, 0).astype("f8"))
        if "h_rel_upstream" in shock_props:
            handle.create_dataset("H_REL_UPSTREAM", data=shock_props["h_rel_upstream"].transpose(2, 1, 0).astype("f8"))
        if "utilde_sq_upstream" in shock_props:
            handle.create_dataset(
                "UTILDE_SQ_UPSTREAM",
                data=shock_props["utilde_sq_upstream"].transpose(2, 1, 0).astype("f8"),
            )
        if "gamma_lorentz_upstream" in shock_props:
            handle.create_dataset(
                "GAMMA_LORENTZ_UPSTREAM",
                data=shock_props["gamma_lorentz_upstream"].transpose(2, 1, 0).astype("f8"),
            )
        if "utilde_n_upstream" in shock_props:
            handle.create_dataset(
                "UTILDE_N_UPSTREAM",
                data=shock_props["utilde_n_upstream"].transpose(2, 1, 0).astype("f8"),
            )
        if "cfast_n_upstream" in shock_props:
            handle.create_dataset(
                "CFAST_N_UPSTREAM",
                data=shock_props["cfast_n_upstream"].transpose(2, 1, 0).astype("f8"),
            )
        if "jump_residual_light" in shock_props:
            handle.create_dataset(
                "JUMP_RESIDUAL_LIGHT",
                data=shock_props["jump_residual_light"].transpose(2, 1, 0).astype("f8"),
            )
        if "ptot_jump" in shock_props:
            handle.create_dataset("PTOT_JUMP", data=shock_props["ptot_jump"].transpose(2, 1, 0).astype("f8"))
        if "entropy_jump" in shock_props:
            handle.create_dataset(
                "ENTROPY_JUMP",
                data=shock_props["entropy_jump"].transpose(2, 1, 0).astype("f8"),
            )
        if "sample_boundary_clipped_grid" in shock_props:
            handle.create_dataset(
                "SAMPLE_BOUNDARY_CLIPPED",
                data=shock_props["sample_boundary_clipped_grid"].transpose(2, 1, 0).astype("i1"),
            )
        if "theta_e_grid" in nonthermal_props:
            handle.create_dataset("THETA_E", data=nonthermal_props["theta_e_grid"].transpose(2, 1, 0).astype("f8"))
        if "p_min_physical_grid" in nonthermal_props:
            handle.create_dataset(
                "P_MIN_PHYSICAL",
                data=nonthermal_props["p_min_physical_grid"].transpose(2, 1, 0).astype("f8"),
            )

        handle.attrs["gamma_min_shock_min"] = gamma_stats["min"]
        handle.attrs["gamma_min_shock_median"] = gamma_stats["median"]
        handle.attrs["gamma_min_shock_max"] = gamma_stats["max"]
        handle.attrs["gamma_min_fallback_risk_count"] = gamma_stats["fallback_risk_count"]
        handle.attrs["gamma_min_fallback_risk_fraction"] = gamma_stats["fallback_risk_fraction"]

        sampling_stats = shock_props.get("sampling_stats", {})
        for key, value in sampling_stats.items():
            handle.attrs[f"shock_sampling_{key}"] = value

        failure_codes = nonthermal_props.get("gamma_failure_codes", {})
        for name, code in failure_codes.items():
            handle.attrs[f"gamma_failure_code_{name}"] = code

        if np.any(mask):
            unique_codes, unique_counts = np.unique(gamma_failure_grid[mask], return_counts=True)
            for code, count in zip(unique_codes, unique_counts):
                handle.attrs[f"gamma_failure_count_{int(code)}"] = int(count)

        if logger:
            logger.ai.data("save.gamma_min.active", shock_gamma_vals if shock_gamma_vals.size else np.array([1.0]))
            logger.ai.data(
                "save.gamma_failure_code.active",
                gamma_failure_grid[mask] if np.any(mask) else np.array([0], dtype=np.int16),
            )
            logger.ai.codepath("Gamma-min export", f"fallback risk count={fallback_risk_count}")

    if logger:
        logger.ai.codepath(
            "HDF5 datasets written",
            "t,dump_cadence,header,prims,KEL,KEL_SR,UNTH,p,GAMMA_MIN,GAMMA_MIN_FAILURE_CODE,sigma,sigma2,sigma_suppression,RHO2_CODE,PRESS2_CODE,PRESS2_OVER_RHO2,SR_MACH_NORMAL,THETA_BN,H_REL_UPSTREAM,UTILDE_SQ_UPSTREAM,GAMMA_LORENTZ_UPSTREAM,UTILDE_N_UPSTREAM,CFAST_N_UPSTREAM,JUMP_RESIDUAL_LIGHT,PTOT_JUMP,ENTROPY_JUMP,SAMPLE_BOUNDARY_CLIPPED,THETA_E,P_MIN_PHYSICAL",
        )
        logger.ai.func_exit("save_h5_file", {"output_h5": output_h5, "grid_shape": [ni, nj, nk]})

    return gamma_stats
