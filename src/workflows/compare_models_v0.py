"""Compare thermal, reconnection, and shock-DSA image models."""

import glob
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


FOV = 300
PARAMS = {
    "thetacam": 163,
    "freqcgs": 230e9,
    "M_unit": 1e25,
    "trat_j": 1.0,
    "trat_d": 80.0,
    "sigma_cut": 5.0,
    "fov": FOV,
    "nx": FOV,
    "ny": FOV,
}


def _default_metadata_output() -> str:
    return os.path.join(PATHS["output"], "logs_compareModels")


def _default_data_output() -> str:
    return os.path.join(PATHS["data_output"], "logs_compareModels")


def _resolve_scratch_work_dir(scratch_dir=None):
    if not scratch_dir:
        return None
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    work_dir = os.path.join(scratch_dir, f"compare_models_{run_tag}")
    os.makedirs(work_dir, exist_ok=True)
    return work_dir


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

    def _log_ipole_line(line):
        if logger:
            logger.info(f"[IPOLE] {line}")

    start = time.time()
    ipole_api.run(args, exe=ipole_bin, verbose=2, line_callback=_log_ipole_line if logger else None)
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


def _copy_artifact_to_output(work_file, output_file, logger=None):
    if not work_file or not output_file or os.path.abspath(work_file) == os.path.abspath(output_file):
        return
    if not os.path.exists(work_file):
        if logger:
            logger.warning(f"Scratch artifact missing, cannot copy: {work_file}")
        return
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    shutil.copy2(work_file, output_file)
    if logger:
        logger.info(f"Copied artifact from scratch to data output: {output_file}")


def main(args=None):
    run_timestamp = time.strftime('%Y%m%d_%H%M%S')
    ipole_dsa_bin = getattr(args, "ipole_dsa_bin", None) or PATHS["ipole_dsa"]
    input_h5 = discover_input_h5(getattr(args, "input_h5", None))
    output_dir = getattr(args, "output_dir", None) or _default_metadata_output()
    data_output_dir = getattr(args, "data_output_dir", None) or _default_data_output()
    scratch_dir = getattr(args, "scratch_dir", None)
    work_data_dir = _resolve_scratch_work_dir(scratch_dir) or data_output_dir
    selected_models = list(dict.fromkeys(getattr(args, "models", None) or ["A", "B", "C"]))

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "logs")
    logger, log_file = setup_logging(
        log_dir=log_dir,
        log_level=logging.INFO,
        log_name=f"compare_models_{run_timestamp}.log",
        logger_name="CompareModels",
    )

    logger.info("=" * 60)
    logger.info("Compare Models Run Started")
    logger.info(f"  Input H5            : {input_h5}")
    logger.info(f"  Metadata Output Dir : {output_dir}")
    logger.info(f"  Large-data Output   : {data_output_dir}")
    logger.info(f"  Scratch Work Dir    : {work_data_dir if scratch_dir else 'disabled'}")
    logger.info(f"  Models              : {selected_models}")
    logger.info(f"  IPOLE DSA bin       : {ipole_dsa_bin}")
    logger.info(f"  IPOLE DSA exists    : {os.path.exists(ipole_dsa_bin)}")
    logger.info(f"  IPOLE DSA executable: {os.access(ipole_dsa_bin, os.X_OK) if os.path.exists(ipole_dsa_bin) else False}")
    logger.info(f"  Params              : {PARAMS}")
    logger.info(f"  Log file            : {log_file}")
    logger.info("=" * 60)

    if not input_h5 or not os.path.exists(input_h5):
        logger.error("No valid input H5 found for compare_models.")
        logger.error("Pass --input-h5 explicitly or ensure *_dsa_input.h5 exists under the data output tree.")
        return
    if not os.path.exists(ipole_dsa_bin):
        logger.error(f"ipole-DSA executable not found: {ipole_dsa_bin}")
        logger.error("Set M87_IPOLE_DSA to a valid ipole-DSA binary before running compare_models.")
        return

    start_total = time.time()
    h5_thermal_in = os.path.join(work_data_dir, "input_thermal.h5")
    h5_reconn_in = os.path.join(work_data_dir, "input_reconnection.h5")
    h5_shock_in = os.path.join(work_data_dir, "input_shock.h5")
    out_thermal = os.path.join(work_data_dir, "img_thermal.h5")
    out_reconn = os.path.join(work_data_dir, "img_reconnection.h5")
    out_shock = os.path.join(work_data_dir, "img_shock.h5")
    final_out_thermal = os.path.join(data_output_dir, "img_thermal.h5")
    final_out_reconn = os.path.join(data_output_dir, "img_reconnection.h5")
    final_out_shock = os.path.join(data_output_dir, "img_shock.h5")
    elapsed = {}
    try:
        if "A" in selected_models:
            logger.info("--- Model A: Thermal Only | ipole-DSA + emission_type=1 + KEL zeroed ---")
            try:
                prepare_input(input_h5, h5_thermal_in, mode="none", logger=logger)
                if scratch_dir:
                    logger.info("Model A copied source H5 to scratch and will run IPOLE from local scratch.")
                elapsed["A"] = run_ipole(h5_thermal_in, out_thermal, ipole_dsa_bin, emission_type=1, logger=logger)
                _copy_artifact_to_output(out_thermal, final_out_thermal, logger=logger)
            except Exception as exc:
                logger.error(f"Model A failed: {exc}", exc_info=True)
                elapsed["A"] = None
        else:
            elapsed["A"] = None
            logger.info("--- Model A skipped ---")

        if "B" in selected_models:
            logger.info("--- Model B: Reconnection | ipole-DSA + emission_type=3 + KEL zeroed ---")
            try:
                prepare_input(input_h5, h5_reconn_in, mode="none", logger=logger)
                if scratch_dir:
                    logger.info("Model B copied source H5 to scratch and will run IPOLE from local scratch.")
                elapsed["B"] = run_ipole(h5_reconn_in, out_reconn, ipole_dsa_bin, emission_type=3, logger=logger)
                _copy_artifact_to_output(out_reconn, final_out_reconn, logger=logger)
            except Exception as exc:
                logger.error(f"Model B failed: {exc}", exc_info=True)
                elapsed["B"] = None
        else:
            elapsed["B"] = None
            logger.info("--- Model B skipped ---")

        if "C" in selected_models:
            logger.info("--- Model C: Shock-DSA | ipole-DSA + default emission + KEL preserved ---")
            try:
                if scratch_dir:
                    prepare_input(input_h5, h5_shock_in, mode="shock", logger=logger)
                    shock_input = h5_shock_in
                    logger.info("Model C copied source H5 to scratch and will run IPOLE from local scratch.")
                else:
                    shock_input = input_h5
                    logger.info("Model C uses the source H5 directly; no shock-side input copy is needed.")
                elapsed["C"] = run_ipole(shock_input, out_shock, ipole_dsa_bin, emission_type=None, logger=logger)
                _copy_artifact_to_output(out_shock, final_out_shock, logger=logger)
            except Exception as exc:
                logger.error(f"Model C failed: {exc}", exc_info=True)
                elapsed["C"] = None
        else:
            elapsed["C"] = None
            logger.info("--- Model C skipped ---")

        logger.info("=" * 60)
        logger.info("IPOLE Runtime Summary:")
        for model, runtime in elapsed.items():
            if model not in selected_models:
                status = "SKIPPED"
            else:
                status = f"{runtime:.1f}s" if runtime is not None else "FAILED"
            logger.info(f"  Model {model}: {status}")
        logger.info(f"  Total elapsed: {time.time() - start_total:.1f}s")
        logger.info("=" * 60)

        if all(model not in selected_models or runtime is None for model, runtime in elapsed.items()):
            logger.error(f"All requested compare_models runs failed for models={selected_models}. Skip plot generation.")
            return

        model_specs = {
            "A": {
                "label": "Model A (Thermal Only)",
                "title": "A: Thermal Only",
                "load_path": final_out_thermal if os.path.exists(final_out_thermal) else out_thermal,
            },
            "B": {
                "label": "Model B (Reconnection/B^2)",
                "title": "B: Reconnection Model",
                "load_path": final_out_reconn if os.path.exists(final_out_reconn) else out_reconn,
            },
            "C": {
                "label": "Model C (Shock-DSA/Ours)",
                "title": "C: Shock-DSA",
                "load_path": final_out_shock if os.path.exists(final_out_shock) else out_shock,
            },
        }
        successful_models = [model for model in selected_models if elapsed.get(model) is not None]
        if not successful_models:
            logger.error("No successful model run available for flux report/plot generation.")
            return

        images = {}
        for model in successful_models:
            img, _ = load_intensity(model_specs[model]["load_path"], logger=logger)
            images[model] = img

        logger.info("FLUX STATISTICS (Jy):")
        for model in successful_models:
            logger.info(f"  {model_specs[model]['label']:<28}: {np.sum(images[model]):.4f}")

        valid_max = max(np.max(images[model]) for model in successful_models)
        if valid_max <= 0:
            logger.error("No positive image intensity available after compare_models runs. Skip plot generation.")
            return

        logger.info("Generating comparison plot...")
        vmin_log = valid_max * 1e-4
        norm_log = mcolors.LogNorm(vmin=vmin_log, vmax=valid_max)

        if successful_models == ["A", "B", "C"]:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            img_a = images["A"]
            img_b = images["B"]
            img_c = images["C"]

            im0 = axes[0, 0].imshow(img_a, cmap="afmhot", origin="lower", norm=norm_log)
            axes[0, 0].set_title("A: Thermal Only")
            plt.colorbar(im0, ax=axes[0, 0])

            im1 = axes[0, 1].imshow(img_b, cmap="afmhot", origin="lower", norm=norm_log)
            axes[0, 1].set_title("B: Reconnection Model")
            plt.colorbar(im1, ax=axes[0, 1])

            im2 = axes[0, 2].imshow(img_c, cmap="afmhot", origin="lower", norm=norm_log)
            axes[0, 2].set_title("C: Shock-DSA")
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
        else:
            ncols = len(successful_models)
            fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
            if ncols == 1:
                axes = [axes]
            for idx, model in enumerate(successful_models):
                im = axes[idx].imshow(images[model], cmap="afmhot", origin="lower", norm=norm_log)
                axes[idx].set_title(model_specs[model]["title"])
                plt.colorbar(im, ax=axes[idx])

        plt.tight_layout()
        plot_dir = os.path.join(output_dir, "plots", f"compare_models_{run_timestamp}")
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"comparison_results_mhd_native_fov{FOV}.png")
        plt.savefig(plot_path)
        plt.close(fig)

        logger.info(f"Plot saved to: {plot_path}")
        logger.info(f"Metadata results saved to: {output_dir}")
        logger.info(f"Large HDF5 results saved to: {data_output_dir}")
        logger.info(f"Log saved to: {log_file}")
    finally:
        if scratch_dir and work_data_dir and os.path.isdir(work_data_dir):
            try:
                shutil.rmtree(work_data_dir)
                logger.info(f"Removed scratch work dir: {work_data_dir}")
            except Exception as exc:
                logger.warning(f"Failed to remove scratch work dir {work_data_dir}: {exc}")
