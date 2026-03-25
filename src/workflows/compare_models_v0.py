"""Compare thermal, reconnection, and shock-DSA image models."""

import glob
import logging
import os
import sys
import time

import h5py
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

_script_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_script_dir)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from src.utils.logging_config import setup_logging
from src.utils.paths import PATHS

try:
    import ipole as ipole_api
except ImportError as exc:
    print(f"Fatal error: could not import ipole. {exc}")
    sys.exit(1)


IPOLE_DSA_BIN = PATHS["ipole_dsa"]
FOV = 300
PARAMS = {
    "thetacam": 163,
    "freqcgs": 86e9,
    "M_unit": 1e25,
    "trat_j": 1.0,
    "trat_d": 80.0,
    "sigma_cut": 5.0,
    "fov": FOV,
    "nx": FOV,
    "ny": FOV,
}


def _default_metadata_output() -> str:
    return os.path.join(PATHS["output"], "sigmaTest_run01")


def _default_data_output() -> str:
    return os.path.join(PATHS["data_output"], "sigmaTest_run01")


def discover_input_h5(input_h5=None):
    if input_h5:
        return input_h5

    patterns = [
        os.path.join(PATHS["data_output"], "**", "*_dsa_input.h5"),
        os.path.join(PATHS["output"], "**", "*_dsa_input.h5"),
    ]
    candidates = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern, recursive=True))

    if not candidates:
        return None

    candidates = sorted(set(candidates), key=lambda path: os.path.getmtime(path), reverse=True)
    return candidates[0]


def run_ipole(input_file, output_file, ipole_bin, emission_type=None, logger=None):
    args = PARAMS.copy()
    args["dump"] = input_file
    args["outfile"] = output_file
    if emission_type is not None:
        args["emission_type"] = emission_type

    label = os.path.basename(output_file)
    if logger:
        logger.info(f"Starting IPOLE: {label} (emission_type={emission_type})")
    else:
        print(f"\n>>> Running IPOLE: {label}")

    start = time.time()
    ipole_api.run(args, exe=ipole_bin, verbose=2)
    elapsed = time.time() - start
    if logger:
        logger.info(f"Finished IPOLE: {label} ({elapsed:.1f}s)")
    return elapsed


def load_intensity(h5_file, logger=None):
    if not h5_file or not os.path.exists(h5_file):
        msg = f"Output file not found: {h5_file}"
        if logger:
            logger.warning(msg)
        else:
            print(f"Warning: {msg}")
        return np.zeros((PARAMS["ny"], PARAMS["nx"])), 1.0

    with h5py.File(h5_file, "r") as handle:
        if "pol" in handle:
            data = handle["pol"][:, :, 0]
        elif "unpol" in handle:
            data = handle["unpol"][:]
        else:
            msg = f"No intensity data in {h5_file}. Keys: {list(handle.keys())}"
            if logger:
                logger.error(msg)
            else:
                print(f"Error: {msg}")
            return np.zeros((PARAMS["ny"], PARAMS["nx"])), 1.0

        scale = 1.0
        if "scale" in handle:
            scale = handle["scale"][()]
        elif "header/scale" in handle:
            scale = handle["header/scale"][()]
        elif "header" in handle and "scale" in handle["header"].attrs:
            scale = handle["header"].attrs["scale"]
        else:
            msg = f"'scale' not found in {os.path.basename(h5_file)}, flux in code units."
            if logger:
                logger.warning(msg)
            else:
                print(f"Warning: {msg}")

    return data * scale, scale


def prepare_input(source_h5, target_h5, mode="shock", logger=None):
    import shutil

    os.makedirs(os.path.dirname(target_h5), exist_ok=True)
    shutil.copy2(source_h5, target_h5)
    if mode == "none":
        with h5py.File(target_h5, "r+") as handle:
            if "KEL" in handle:
                handle["KEL"][...] = np.zeros_like(handle["KEL"][:])
                msg = f"KEL zeroed in {os.path.basename(target_h5)}"
                if logger:
                    logger.info(msg)
                else:
                    print(f"  {msg}")


def main(args=None):
    input_h5 = discover_input_h5(getattr(args, "input_h5", None))
    output_dir = getattr(args, "output_dir", None) or _default_metadata_output()
    data_output_dir = getattr(args, "data_output_dir", None) or _default_data_output()

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "logs")
    logger, log_file = setup_logging(
        log_dir=log_dir,
        log_level=logging.INFO,
        log_name=f"compare_models_{time.strftime('%Y%m%d_%H%M%S')}.log",
        logger_name="CompareModels",
    )

    logger.info("=" * 60)
    logger.info("Compare Models Run Started")
    logger.info(f"  Input H5            : {input_h5}")
    logger.info(f"  Metadata Output Dir : {output_dir}")
    logger.info(f"  Large-data Output   : {data_output_dir}")
    logger.info(f"  IPOLE DSA bin       : {IPOLE_DSA_BIN}")
    logger.info(f"  IPOLE DSA exists    : {os.path.exists(IPOLE_DSA_BIN)}")
    logger.info(f"  IPOLE DSA executable: {os.access(IPOLE_DSA_BIN, os.X_OK) if os.path.exists(IPOLE_DSA_BIN) else False}")
    logger.info(f"  Params              : {PARAMS}")
    logger.info(f"  Log file            : {log_file}")
    logger.info("=" * 60)

    if not input_h5 or not os.path.exists(input_h5):
        logger.error("No valid input H5 found for compare_models.")
        logger.error("Pass --input-h5 explicitly or ensure *_dsa_input.h5 exists under the data output tree.")
        return
    if not os.path.exists(IPOLE_DSA_BIN):
        logger.error(f"ipole-DSA executable not found: {IPOLE_DSA_BIN}")
        logger.error("Set M87_IPOLE_DSA to a valid ipole-DSA binary before running compare_models.")
        return

    start_total = time.time()
    h5_thermal_in = os.path.join(data_output_dir, "input_thermal.h5")
    h5_reconn_in = os.path.join(data_output_dir, "input_reconnection.h5")
    h5_shock_in = os.path.join(data_output_dir, "input_shock.h5")
    out_thermal = os.path.join(data_output_dir, "img_thermal.h5")
    out_reconn = os.path.join(data_output_dir, "img_reconnection.h5")
    out_shock = os.path.join(data_output_dir, "img_shock.h5")
    elapsed = {}

    logger.info("--- Model A: Thermal Only | ipole-DSA + emission_type=1 + KEL zeroed ---")
    try:
        prepare_input(input_h5, h5_thermal_in, mode="none", logger=logger)
        elapsed["A"] = run_ipole(h5_thermal_in, out_thermal, IPOLE_DSA_BIN, emission_type=1, logger=logger)
    except Exception as exc:
        logger.error(f"Model A failed: {exc}", exc_info=True)
        elapsed["A"] = None

    logger.info("--- Model B: Reconnection | ipole-DSA + emission_type=3 + KEL zeroed ---")
    try:
        prepare_input(input_h5, h5_reconn_in, mode="none", logger=logger)
        elapsed["B"] = run_ipole(h5_reconn_in, out_reconn, IPOLE_DSA_BIN, emission_type=3, logger=logger)
    except Exception as exc:
        logger.error(f"Model B failed: {exc}", exc_info=True)
        elapsed["B"] = None

    logger.info("--- Model C: Shock-DSA | ipole-DSA + default emission + KEL preserved ---")
    try:
        prepare_input(input_h5, h5_shock_in, mode="shock", logger=logger)
        elapsed["C"] = run_ipole(h5_shock_in, out_shock, IPOLE_DSA_BIN, emission_type=None, logger=logger)
    except Exception as exc:
        logger.error(f"Model C failed: {exc}", exc_info=True)
        elapsed["C"] = None

    logger.info("=" * 60)
    logger.info("IPOLE Runtime Summary:")
    for model, runtime in elapsed.items():
        logger.info(f"  Model {model}: {f'{runtime:.1f}s' if runtime is not None else 'FAILED'}")
    logger.info(f"  Total elapsed: {time.time() - start_total:.1f}s")
    logger.info("=" * 60)

    if all(runtime is None for runtime in elapsed.values()):
        logger.error("All three compare_models runs failed. Skip plot generation.")
        return

    img_a, _ = load_intensity(out_thermal, logger=logger)
    img_b, _ = load_intensity(out_reconn, logger=logger)
    img_c, _ = load_intensity(out_shock, logger=logger)

    logger.info("FLUX STATISTICS (Jy):")
    logger.info(f"  Model A (Thermal Only)     : {np.sum(img_a):.4f}")
    logger.info(f"  Model B (Reconnection/B^2) : {np.sum(img_b):.4f}")
    logger.info(f"  Model C (Shock-DSA/Ours)   : {np.sum(img_c):.4f}")

    valid_max = max(np.max(img_a), np.max(img_b), np.max(img_c))
    if valid_max <= 0:
        logger.error("No positive image intensity available after compare_models runs. Skip plot generation.")
        return

    logger.info("Generating comparison plot...")
    vmin_log = valid_max * 1e-4
    norm_log = mcolors.LogNorm(vmin=vmin_log, vmax=valid_max)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    im0 = axes[0, 0].imshow(img_a, cmap="afmhot", origin="lower", norm=norm_log)
    axes[0, 0].set_title("A: Thermal Only")
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(img_b, cmap="afmhot", origin="lower", norm=norm_log)
    axes[0, 1].set_title("B: Reconnection Model")
    plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[0, 2].imshow(img_c, cmap="afmhot", origin="lower", norm=norm_log)
    axes[0, 2].set_title("C: Shock-DSA (Xia+25)")
    plt.colorbar(im2, ax=axes[0, 2])

    diff_shock = img_c - img_a
    im3 = axes[1, 0].imshow(diff_shock, cmap="viridis", origin="lower")
    axes[1, 0].set_title("C - A: Net Shock Contribution")
    plt.colorbar(im3, ax=axes[1, 0])

    diff_model = img_c - img_b
    im4 = axes[1, 1].imshow(diff_model, cmap="RdBu_r", origin="lower")
    axes[1, 1].set_title("C - B: Shock vs Reconnection")
    plt.colorbar(im4, ax=axes[1, 1])

    axes[1, 2].axis("off")
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"comparison_results_mhd_native_fov{FOV}.png")
    plt.savefig(plot_path)
    plt.close(fig)

    logger.info(f"Plot saved to: {plot_path}")
    logger.info(f"Metadata results saved to: {output_dir}")
    logger.info(f"Large HDF5 results saved to: {data_output_dir}")
    logger.info(f"Log saved to: {log_file}")
