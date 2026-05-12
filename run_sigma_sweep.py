#!/usr/bin/env python3
"""Run a sigma_crit sweep and keep large HDF5 artifacts outside the project tree."""

import argparse
import copy
import logging
import os
import shutil
import sys
import time

import h5py
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
script_dir = os.path.dirname(os.path.abspath(__file__))
pyathena_path = os.path.join(script_dir, "..", "pyathena")
sys.path.insert(0, pyathena_path)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.utils.logging_config import setup_logging
from src.utils.paths import IPOLE_DSA, PATHS
from src.workflows.workflowFull_v2 import process_snapshot

try:
    import MAD98_DSA_Postprocessing.ipole as ipole_api
except ImportError as exc:
    print(f"Fatal error: could not import ipole module. {exc}")
    sys.exit(1)


IPOLE_PARAMS = {
    "thetacam": 163,
    "freqcgs": 86e9,
    "M_unit": 1e25,
    "trat_j": 1.0,
    "trat_d": 80.0,
    "sigma_cut": 5.0,
    "fov": 300,
    "nx": 300,
    "ny": 300,
}


def _log(message, logger=None, level=logging.INFO):
    print(message, flush=True)
    if logger:
        logger.log(level, message)


def _base_pipeline_config():
    return {
        "output_directory": PATHS["output"],
        "data_output_directory": PATHS["data_output"],
        "ipole_executable_path": IPOLE_DSA,
        "roi_params": {
            "r_min": 10,
            "r_max": 1200,
            "theta_min": 0.0,
            "theta_max": 3.14159,
            "phi_min": 0.0,
            "phi_max": 6.28319,
        },
        "shock_params": {
            "gamma": 4.0 / 3.0,
            "grad_p_filter_quantile": 0.20,
            "march_cells": 6,
        },
        "shock_sr": {
            "sr_mach_min": 1.7,
            "jump_residual_max": 0.8,
        },
        "nt_params": {
            "gamma": 4.0 / 3.0,
            "x_inj": 3.5,
            "eta_inj_e0": 1.0e-3,
            "eps_nth_e0": 3.0e-3,
            "theta_bn_quench": 50.0,
            "theta_bn_width": 10.0,
            "sonic_mach_inj_min": 1.5,
            "inj_model": "pic_dual_cap",
            "sigma_crit": 0.1,
            "alpha_sigma": 2,
        },
        "physics": {
            "spin": 0.98,
            "hslope": 1.0,
            "R0": 0.0,
            "enable_advection": False,
            "advection_model": "sr_radial",
            "advection_domain": "shock_local",
            "advection_cooling_model": "synchrotron_local_sink",
            "advection_line_sweeps": 6,
            "advection_shock_pad_r": 8,
            "advection_shock_pad_theta": 2,
            "advection_shock_pad_phi": 2,
            "advection_seed_mode": "downstream_sample",
            "cooling_factor": 50.0,
            "advection_steps": 2000,
            "M_unit": 1e25,
            "MBH_solar": 6.2e9,
        },
        "max_concurrent_tasks": 1,
    }


def run_ipole(input_h5, output_h5, ipole_bin, emission_type=None, logger=None):
    params = IPOLE_PARAMS.copy()
    params["dump"] = input_h5
    params["outfile"] = output_h5
    if emission_type is not None:
        params["emission_type"] = emission_type

    label = os.path.basename(output_h5)
    _log(f"  [ipole] START: {label} (emission_type={emission_type})", logger)
    start = time.time()
    ipole_api.run(params, exe=ipole_bin, verbose=2)
    elapsed = time.time() - start
    _log(f"  [ipole] DONE : {label} ({elapsed:.1f}s)", logger)
    return elapsed


def prepare_input(source_h5, target_h5, zero_kel=False, logger=None):
    os.makedirs(os.path.dirname(target_h5), exist_ok=True)
    shutil.copy2(source_h5, target_h5)
    if zero_kel:
        with h5py.File(target_h5, "r+") as handle:
            if "KEL" in handle:
                handle["KEL"][...] = np.zeros_like(handle["KEL"][:])
        _log(f"  KEL zeroed -> {os.path.basename(target_h5)}", logger)


def load_intensity(h5_file, logger=None):
    ny, nx = IPOLE_PARAMS["ny"], IPOLE_PARAMS["nx"]
    if not h5_file or not os.path.exists(h5_file):
        _log(f"  Warning: output not found: {h5_file}", logger, logging.WARNING)
        return np.zeros((ny, nx))

    with h5py.File(h5_file, "r") as handle:
        if "pol" in handle:
            data = handle["pol"][:, :, 0]
        elif "unpol" in handle:
            data = handle["unpol"][:]
        else:
            _log(f"  Warning: no intensity data in {os.path.basename(h5_file)}", logger, logging.WARNING)
            return np.zeros((ny, nx))

        scale = 1.0
        for key in ("scale", "header/scale"):
            if key in handle:
                scale = float(handle[key][()])
                break
    return data * scale


def phase1_generate_h5(snapshot_file, sigma_values, alpha_sigma, output_dir, data_output_dir, logger):
    snap_base = os.path.basename(snapshot_file).replace(".athdf", "")
    sigma_h5_map = {}

    for sigma_crit in sigma_values:
        sigma_tag = f"sigma_{sigma_crit:.3f}"
        sigma_dir = os.path.join(output_dir, sigma_tag)
        sigma_data_dir = os.path.join(data_output_dir, sigma_tag)
        os.makedirs(sigma_dir, exist_ok=True)
        os.makedirs(sigma_data_dir, exist_ok=True)

        sub_logger, _ = setup_logging(
            log_dir=os.path.join(sigma_dir, "logs"),
            log_level=logging.INFO,
            log_name=f"generate_{snap_base}.log",
            logger_name=f"Gen_{sigma_tag}",
        )
        _log(f"\n[Phase 1] Generating h5 for {sigma_tag} (alpha={alpha_sigma})...", logger)

        cfg = copy.deepcopy(_base_pipeline_config())
        cfg["output_directory"] = sigma_dir
        cfg["data_output_directory"] = sigma_data_dir
        cfg["nt_params"]["sigma_crit"] = sigma_crit
        cfg["nt_params"]["alpha_sigma"] = alpha_sigma

        try:
            success = process_snapshot(snapshot_file, cfg, sub_logger)
            if not success:
                raise RuntimeError("process_snapshot returned False")

            expected_h5 = os.path.join(sigma_data_dir, "ipole_inputs", f"{snap_base}_dsa_input.h5")
            if not os.path.exists(expected_h5):
                raise FileNotFoundError(f"Expected h5 not found: {expected_h5}")

            sigma_h5_map[sigma_crit] = expected_h5
            _log(f"  [Phase 1] OK: {sigma_tag} -> {expected_h5}", logger)
        except Exception as exc:
            _log(f"  [Phase 1] FAILED {sigma_tag}: {exc}", logger, logging.ERROR)
            if logger:
                logger.exception(exc)

    return sigma_h5_map


def phase2_shared_models(any_h5, snap_base, output_dir, data_output_dir, logger):
    shared_dir = os.path.join(output_dir, "shared")
    shared_data_dir = os.path.join(data_output_dir, "shared")
    os.makedirs(shared_dir, exist_ok=True)
    os.makedirs(shared_data_dir, exist_ok=True)

    h5_ab = os.path.join(shared_data_dir, "input_AB_kel0.h5")
    prepare_input(any_h5, h5_ab, zero_kel=True, logger=logger)

    img_a = os.path.join(shared_data_dir, f"img_A_thermal_{snap_base}.h5")
    img_b = os.path.join(shared_data_dir, f"img_B_reconnection_{snap_base}.h5")
    elapsed = {}

    _log("\n[Phase 2] Model A: Thermal Only | ipole-DSA + emission_type=1 + KEL zeroed", logger)
    try:
        elapsed["A"] = run_ipole(h5_ab, img_a, IPOLE_DSA, emission_type=1, logger=logger)
    except Exception as exc:
        _log(f"  [Phase 2] Model A FAILED: {exc}", logger, logging.ERROR)
        if logger:
            logger.exception(exc)
        elapsed["A"] = None

    _log("\n[Phase 2] Model B: Reconnection B^2 | ipole-DSA + emission_type=3 + KEL zeroed", logger)
    try:
        elapsed["B"] = run_ipole(h5_ab, img_b, IPOLE_DSA, emission_type=3, logger=logger)
    except Exception as exc:
        _log(f"  [Phase 2] Model B FAILED: {exc}", logger, logging.ERROR)
        if logger:
            logger.exception(exc)
        elapsed["B"] = None

    return img_a, img_b, elapsed


def phase3_dsa_models(sigma_h5_map, snap_base, output_dir, data_output_dir, logger):
    sigma_img_map = {}
    elapsed = {}

    _log("\n[Phase 3] Model C: DSA Shock | ipole-DSA + default emission + KEL preserved", logger)
    for sigma_crit, src_h5 in sigma_h5_map.items():
        sigma_tag = f"sigma_{sigma_crit:.3f}"
        sigma_dir = os.path.join(output_dir, sigma_tag)
        sigma_data_dir = os.path.join(data_output_dir, sigma_tag)
        os.makedirs(sigma_dir, exist_ok=True)
        os.makedirs(sigma_data_dir, exist_ok=True)

        h5_c = os.path.join(sigma_data_dir, f"input_C_shock_{sigma_tag}.h5")
        img_c = os.path.join(sigma_data_dir, f"img_C_shock_{snap_base}_{sigma_tag}.h5")

        _log(f"\n  [Phase 3] Model C [{sigma_tag}]", logger)
        try:
            prepare_input(src_h5, h5_c, zero_kel=False, logger=logger)
            elapsed[f"C_{sigma_tag}"] = run_ipole(h5_c, img_c, IPOLE_DSA, emission_type=None, logger=logger)
            sigma_img_map[sigma_crit] = img_c
        except Exception as exc:
            _log(f"  [Phase 3] FAILED [{sigma_tag}]: {exc}", logger, logging.ERROR)
            if logger:
                logger.exception(exc)
            sigma_img_map[sigma_crit] = None
            elapsed[f"C_{sigma_tag}"] = None

    return sigma_img_map, elapsed


def phase4_comparison_figure(img_a_path, img_b_path, sigma_img_map, sigma_values, alpha_sigma, snap_base, output_dir, timestamp, logger):
    _log("\n[Phase 4] Generating comparison figure...", logger)

    n_sigma = len(sigma_values)
    n_cols = 2 + n_sigma
    fig, axes = plt.subplots(2, n_cols, figsize=(4.5 * n_cols, 9))

    img_a = load_intensity(img_a_path, logger)
    img_b = load_intensity(img_b_path, logger)
    imgs_c = [load_intensity(sigma_img_map.get(sigma_crit), logger) for sigma_crit in sigma_values]

    all_imgs = [img_a, img_b] + imgs_c
    valid_max = [np.max(image) for image in all_imgs if np.max(image) > 0]
    vmax = max(valid_max) if valid_max else 1.0
    vmin = vmax * 1e-4
    norm_log = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    row1_titles = ["A: Thermal Only", "B: Reconnection (B^2)"] + [f"C: DSA sigma_crit={value}" for value in sigma_values]
    for ax, image, title in zip(axes[0], all_imgs, row1_titles):
        im = ax.imshow(image, cmap="afmhot", origin="lower", norm=norm_log)
        ax.set_title(title, fontsize=9, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)

    diff_titles = [f"(C sigma={value} - A) / A" for value in sigma_values]
    diff_imgs = [np.divide(imgs_c[idx] - img_a, img_a + 1e-30 * vmax) for idx in range(n_sigma)]
    if n_sigma > 0:
        diff_imgs.append(np.divide(imgs_c[0] - img_b, img_b + 1e-30 * vmax))
        diff_titles.append(f"(C sigma={sigma_values[0]} - B) / B")
    while len(diff_imgs) < n_cols:
        diff_imgs.append(None)
        diff_titles.append("")

    diff_abs_max = max((np.nanpercentile(np.abs(diff), 99) for diff in diff_imgs if diff is not None), default=1.0)
    norm_div = mcolors.TwoSlopeNorm(vmin=-diff_abs_max, vcenter=0, vmax=diff_abs_max)
    for ax, diff, title in zip(axes[1], diff_imgs, diff_titles):
        if diff is None:
            ax.axis("off")
            continue
        im = ax.imshow(diff, cmap="RdBu_r", origin="lower", norm=norm_div)
        ax.set_title(title, fontsize=9, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)

    fig.suptitle(f"Sigma Sweep: {snap_base} | alpha={alpha_sigma} | sigma_crit={sigma_values}", fontsize=11, y=1.01)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"sweep_{snap_base}_{timestamp}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log(f"  Figure saved: {plot_path}", logger)
    return plot_path


def main():
    parser = argparse.ArgumentParser(description="Run a sigma_crit sweep for one snapshot.")
    parser.add_argument("--snapshot", required=True, help="Input .athdf file")
    parser.add_argument("--sigma-values", nargs="+", type=float, default=[0.01, 0.03, 0.1], help="sigma_crit values")
    parser.add_argument("--alpha-sigma", type=float, default=2, help="Sigma suppression steepness")
    parser.add_argument("--output-dir", required=True, help="Metadata output directory")
    parser.add_argument(
        "--data-output-dir",
        help="Large HDF5 output root; defaults to PATHS['data_output']/<basename(output-dir)>",
    )
    args = parser.parse_args()

    snapshot = args.snapshot
    sigma_values = sorted(args.sigma_values)
    alpha_sigma = args.alpha_sigma
    output_dir = args.output_dir
    data_output_dir = args.data_output_dir or os.path.join(
        PATHS["data_output"], os.path.basename(os.path.normpath(output_dir))
    )
    snap_base = os.path.basename(snapshot).replace(".athdf", "")
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_output_dir, exist_ok=True)

    logger, log_file = setup_logging(
        log_dir=output_dir,
        log_level=logging.INFO,
        log_name=f"sweep_{snap_base}_{timestamp}.log",
        logger_name="SigmaSweep",
    )

    _log("=" * 70, logger)
    _log("SIGMA SWEEP START", logger)
    _log(f"  Snapshot        : {snapshot}", logger)
    _log(f"  Sigma values    : {sigma_values}", logger)
    _log(f"  Alpha           : {alpha_sigma}", logger)
    _log(f"  Metadata output : {output_dir}", logger)
    _log(f"  Large-data output: {data_output_dir}", logger)
    _log(f"  IPOLE DSA bin   : {IPOLE_DSA}", logger)
    _log(f"  IPOLE DSA exists: {os.path.exists(IPOLE_DSA)}", logger)
    _log(
        f"  IPOLE DSA exec  : {os.access(IPOLE_DSA, os.X_OK) if os.path.exists(IPOLE_DSA) else False}",
        logger,
    )
    _log(f"  Structured log  : {log_file}", logger)
    _log("=" * 70, logger)

    if not os.path.exists(IPOLE_DSA):
        _log(f"ABORT: ipole-DSA executable not found: {IPOLE_DSA}", logger, logging.ERROR)
        _log("Set M87_IPOLE_DSA to a valid ipole-DSA binary before running sigma_sweep.", logger, logging.ERROR)
        sys.exit(1)

    start = time.time()
    elapsed_all = {}

    _log("\n" + "-" * 60, logger)
    _log("Phase 1 / 4 - Generate h5 files for each sigma", logger)
    _log("-" * 60, logger)
    sigma_h5_map = phase1_generate_h5(snapshot, sigma_values, alpha_sigma, output_dir, data_output_dir, logger)
    if not sigma_h5_map:
        _log("ABORT: No h5 files generated successfully.", logger, logging.ERROR)
        sys.exit(1)

    _log("\n" + "-" * 60, logger)
    _log("Phase 2 / 4 - Shared models A and B", logger)
    _log("-" * 60, logger)
    any_h5 = next(iter(sigma_h5_map.values()))
    img_a, img_b, elapsed_ab = phase2_shared_models(any_h5, snap_base, output_dir, data_output_dir, logger)
    elapsed_all.update(elapsed_ab)

    _log("\n" + "-" * 60, logger)
    _log("Phase 3 / 4 - Model C for each sigma", logger)
    _log("-" * 60, logger)
    sigma_img_map, elapsed_c = phase3_dsa_models(sigma_h5_map, snap_base, output_dir, data_output_dir, logger)
    elapsed_all.update(elapsed_c)

    _log("\n" + "-" * 60, logger)
    _log("Phase 4 / 4 - Comparison figure", logger)
    _log("-" * 60, logger)
    plot_path = phase4_comparison_figure(
        img_a,
        img_b,
        sigma_img_map,
        sigma_values,
        alpha_sigma,
        snap_base,
        output_dir,
        timestamp,
        logger,
    )

    total_time = time.time() - start
    _log("\n" + "=" * 70, logger)
    _log("SIGMA SWEEP COMPLETE", logger)
    _log(f"  Total wall time   : {total_time:.1f}s ({total_time / 60:.1f} min)", logger)
    for key, value in elapsed_all.items():
        status = f"{value:.1f}s" if value is not None else "FAILED"
        _log(f"  {key:<20} {status}", logger)
    _log(f"  Comparison figure : {plot_path}", logger)
    _log(f"  Metadata output   : {output_dir}", logger)
    _log(f"  Large-data output : {data_output_dir}", logger)
    _log("=" * 70, logger)


if __name__ == "__main__":
    main()
