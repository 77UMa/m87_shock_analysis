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
    if logger:
        logger.ai.func_enter(
            "calculate_dsa_physics",
            {"shock_params": shock_params, "nt_params": nt_params, "rho_shape": roi_data["rho"].shape},
        )

    shock_props = find_shocks_in_roi_mhd(roi_data, logger=logger, **shock_params)

    gamma = shock_params.get("gamma", 4.0 / 3.0)
    b_sq = roi_data["Bcc1"] ** 2 + roi_data["Bcc2"] ** 2 + roi_data["Bcc3"] ** 2
    uu = roi_data["press"] / (gamma - 1.0)
    denom = roi_data["rho"] + uu + roi_data["press"]
    sigma_grid = np.where(denom > 0, b_sq / (2.0 * denom), 0.0)
    shock_props["sigma_grid"] = sigma_grid

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
        }

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
        p_grid = np.where(mask, q_grid - 1.0, 3.0)

        handle.create_dataset("KEL", data=mask.transpose(2, 1, 0).astype("f8"))
        handle.create_dataset("UNTH", data=c_grid.transpose(2, 1, 0).astype("f8"))
        handle.create_dataset("p", data=p_grid.transpose(2, 1, 0).astype("f8"))

        if "sigma_grid" in shock_props:
            handle.create_dataset("sigma", data=shock_props["sigma_grid"].transpose(2, 1, 0).astype("f4"))
        if "sigma_suppression_grid" in nonthermal_props:
            handle.create_dataset(
                "sigma_suppression",
                data=nonthermal_props["sigma_suppression_grid"].transpose(2, 1, 0).astype("f4"),
            )

    if logger:
        logger.ai.codepath(
            "HDF5 datasets written",
            "t,dump_cadence,header,prims,KEL,UNTH,p,sigma,sigma_suppression",
        )
        logger.ai.func_exit("save_h5_file", {"output_h5": output_h5, "grid_shape": [ni, nj, nk]})
