#!/usr/bin/env python3
"""Entry point for the M87 DSA processing pipeline."""

import argparse
import glob
import multiprocessing as mp
import os
import shutil
import sys
import time
import traceback
import uuid

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
script_dir = os.path.dirname(os.path.abspath(__file__))
pyathena_path = os.path.join(script_dir, "..", "pyathena")
sys.path.insert(0, pyathena_path)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.utils.logging_config import DUALogger, log_summary, setup_logging
from src.workflows.compare_models_v0 import main as compare_models_main
from src.workflows.workflowFull_v2 import process_snapshot
from src.utils.paths import PATHS, ensure_dir, validate_paths, print_path_info


def _sanitize_run_label(label):
    if not label:
        return None
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(label))
    return safe.strip("_") or None


def _safe_ensure_dir(key, fallback_name):
    try:
        return ensure_dir(key)
    except OSError:
        fallback_path = os.path.join(script_dir, fallback_name)
        os.makedirs(fallback_path, exist_ok=True)
        return fallback_path


def _resolve_generate_scratch_root(cli_scratch_dir=None):
    if cli_scratch_dir:
        return os.path.abspath(cli_scratch_dir)
    if env_scratch_dir := os.environ.get("M87_SCRATCH_DIR"):
        return os.path.abspath(env_scratch_dir)
    return os.path.join(os.path.expanduser("~"), ".cache", "m87_dsa")


def process_snapshot_wrapper(task_args):
    """Wrapper used by multiprocessing workers."""
    filename, config, log_dir, scratch_run_dir = task_args
    basename = os.path.basename(filename)
    proc_logger = DUALogger(
        log_dir=log_dir,
        run_id=f"run_{time.strftime('%Y%m%d_%H%M%S')}_pid{os.getpid()}",
    )

    local_input = None
    local_task_dir = None
    start_time = time.time()
    try:
        local_task_dir = os.path.join(
            scratch_run_dir,
            f"{os.getpid()}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}",
        )
        local_input = os.path.join(local_task_dir, basename)

        copy_start = time.time()
        os.makedirs(local_task_dir, exist_ok=True)
        shutil.copy2(filename, local_input)
        copy_elapsed = time.time() - copy_start
        proc_logger.human.info(f"Scratch copy: {basename} -> {local_input}")
        proc_logger.human.time("Copy input to local scratch", copy_elapsed)

        success = process_snapshot(local_input, config, proc_logger)
        elapsed = time.time() - start_time
        if success:
            return True, filename, elapsed
        return False, filename, "Process returned False without exception"
    except Exception as exc:
        error_msg = f"Error in {basename}: {exc}\n{traceback.format_exc()}"
        return False, filename, error_msg
    finally:
        if local_input and os.path.exists(local_input):
            cleanup_start = time.time()
            try:
                os.remove(local_input)
                cleanup_elapsed = time.time() - cleanup_start
                proc_logger.human.info(f"Scratch cleanup file removed: {local_input}")
                proc_logger.human.time("Cleanup local scratch file", cleanup_elapsed)
            except Exception as exc:
                proc_logger.human.warning(f"Failed to remove scratch input {local_input}: {exc}")
        if local_task_dir and os.path.isdir(local_task_dir):
            try:
                os.rmdir(local_task_dir)
            except Exception:
                pass


def create_default_config():
    """
    创建默认配置，使用新的统一路径系统

    路径优先级：
    1. 环境变量（M87_DSA_BASE, M87_DATA_DIR 等）
    2. 默认 CPFS 路径
    """
    return {
        "output_directory": _safe_ensure_dir("output", "OUTPUT"),
        "data_output_directory": _safe_ensure_dir("data_output", os.path.join("OUTPUT", "data_output")),
        "data_directory": PATHS["data"],
        "ipole_executable_path": PATHS["ipole_std"],
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
            "x_inj": 1.2,
            "r_high": 10.0,
            "eta_inj_e0": 2.0e-1,
            "eps_nth_e0": 2.0e-1,
            "theta_bn_quench": 75.0,
            "theta_bn_width": 25.0,
            "sonic_mach_inj_min": 1.2,
            "inj_model": "pic_dual_cap",
            "energy_budget_model": "total_internal_energy_excess",
            "p_eff_model": "hybrid_classical_relativistic",
            "classical_fast_mach_max": 1.8,
            "relativistic_fast_mach_min": 3.0,
            "theta_bn_parallel_max": 35.0,
            "theta_bn_oblique_max": 60.0,
            "sigma_rel_parallel_max": 1.0e-3,
            "sigma_rel_oblique_max": 1.0e-2,
            "p_eff_parallel": 2.35,
            "p_eff_oblique": 2.8,
            "p_eff_steep": 3.5,
            "p_eff_floor": 1.5,
            "p_eff_ceiling": 4.5,
            "sigma_crit": 0.1,
            "alpha_sigma": 2,
            "sironi_tran_coeff": 0.0016,
            "sironi_tran_exp": 3.6,
            "sironi_tran_delta_max": 3.0,
        },
        "physics": {
            "spin": 0.98,
            "hslope": 1.0,
            "R0": 0.0,
            "enable_advection": True,
            "advection_model": "sr_radial",
            "advection_domain": "shock_local",
            "advection_cooling_model": "synchrotron_local_sink",
            "advection_line_sweeps": 6,
            "advection_shock_pad_r": 8,
            "advection_shock_pad_theta": 2,
            "advection_shock_pad_phi": 2,
            "advection_seed_mode": "downstream_sample",
            "advection_injection_layer": "downstream_sample",
            "advection_tau_inj_cell_crossing_fraction": 1.0e-3,
            "cooling_factor": 1e30,
            "advection_steps": 2000,
            "M_unit": 1e25,
            "MBH_solar": 6.2e9,
        },
        "hdf5_options": {
            "include_3d_diagnostics": True,
        },
        "max_concurrent_tasks": 5,
    }


def cmd_generate_h5(args):
    """生成HDF5文件的命令处理"""
    # 如果需要查看路径配置
    if args.show_paths:
        print_path_info()
        return

    config = create_default_config()

    if args.data_dir:
        config["data_directory"] = args.data_dir
    if args.output_dir:
        config["output_directory"] = args.output_dir
    if args.data_output_dir:
        config["data_output_directory"] = args.data_output_dir
    if args.n_workers:
        config["max_concurrent_tasks"] = args.n_workers
    if args.sigma_crit is not None:
        config["nt_params"]["sigma_crit"] = args.sigma_crit
    if args.alpha_sigma is not None:
        config["nt_params"]["alpha_sigma"] = args.alpha_sigma
    if args.cooling_factor is not None:
        config["physics"]["cooling_factor"] = args.cooling_factor
    if args.enable_advection is not None:
        config["physics"]["enable_advection"] = args.enable_advection == "true"
    if args.include_3d_diagnostics is not None:
        config["hdf5_options"]["include_3d_diagnostics"] = args.include_3d_diagnostics == "true"

    run_label = _sanitize_run_label(getattr(args, "run_label", None))

    os.makedirs(config["output_directory"], exist_ok=True)
    os.makedirs(config["data_output_directory"], exist_ok=True)
    log_dir = os.path.join(config["output_directory"], "logs")
    os.makedirs(log_dir, exist_ok=True)

    run_timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_stem = f"main_dsa_{run_label}_{run_timestamp}" if run_label else f"main_dsa_{run_timestamp}"
    logger, log_file = setup_logging(
        log_dir=log_dir,
        log_name=f"{log_stem}.log",
    )

    # 记录路径配置信息
    path_status = validate_paths()
    logger.info(f"Base path: {PATHS['base']}")
    logger.info(f"Data path: {PATHS['data']} (exists: {path_status['data']['exists']})")
    logger.info(f"Metadata output path: {config['output_directory']}")
    logger.info(f"Large-data output path: {config['data_output_directory']}")
    logger.info(f"ipole executable: {config['ipole_executable_path']} (exists: {path_status['ipole_std']['exists']})")
    logger.info(f"Run label: {run_label or 'none'}")

    file_pattern = os.path.join(config["data_directory"], "mad98.prim.*.athdf")
    file_list = sorted(glob.glob(file_pattern))
    if args.snapshot:
        snapshot_name = args.snapshot[:-6] if args.snapshot.endswith(".athdf") else args.snapshot
        expected_name = f"{snapshot_name}.athdf"
        file_list = [path for path in file_list if os.path.basename(path) == expected_name]

    if not file_list:
        logger.error(f"No files found: {file_pattern}")
        if args.snapshot:
            logger.error(f"Snapshot filter did not match: {args.snapshot}")
        sys.exit(1)

    scratch_root = _resolve_generate_scratch_root(getattr(args, "scratch_dir", None))
    run_tag = f"{run_label}_{run_timestamp}" if run_label else run_timestamp
    scratch_run_dir = os.path.join(scratch_root, f"generate_h5_{run_tag}")
    os.makedirs(scratch_run_dir, exist_ok=True)

    logger.info(f"Starting pipeline: {len(file_list)} files, {config['max_concurrent_tasks']} workers")
    logger.info(f"Scratch root: {scratch_root}")
    logger.info(f"Scratch work dir (run): {scratch_run_dir}")
    tasks = [(filepath, config, log_dir, scratch_run_dir) for filepath in file_list]

    start_time = time.time()
    successful = 0
    failed = 0

    try:
        with mp.Pool(processes=config["max_concurrent_tasks"], maxtasksperchild=1) as pool:
            for success, filename, info in pool.imap_unordered(process_snapshot_wrapper, tasks):
                fname = os.path.basename(filename)
                if success:
                    successful += 1
                    print(f"[SUCCESS] {fname} ({info:.2f}s)")
                else:
                    failed += 1
                    print(f"[FAILED] {fname}")
                    logger.error(f"File {fname} failed details:\n{info}")
    finally:
        if os.path.isdir(scratch_run_dir):
            cleanup_start = time.time()
            try:
                shutil.rmtree(scratch_run_dir)
                logger.info(f"Scratch cleanup: removed run directory {scratch_run_dir}")
                logger.info(f"Scratch cleanup elapsed: {time.time() - cleanup_start:.2f}s")
            except Exception as exc:
                logger.warning(f"Scratch cleanup failed for {scratch_run_dir}: {exc}")

    total_time = time.time() - start_time
    log_summary(logger, len(file_list), successful, failed, total_time)

    print("\n" + "=" * 30)
    print(f"DONE. Total Time: {total_time:.2f}s")
    print(f"Successful: {successful}, Failed: {failed}")
    print(f"Main Log: {log_file}")
    print("=" * 30)


def cmd_compare_models(args):
    compare_models_main(args)


def cmd_analyze_electrons(args):
    print("Non-thermal electron analysis mode")


def main():
    parser = argparse.ArgumentParser(description="M87 DSA Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # generate_h5 子命令
    parser_generate = subparsers.add_parser("generate_h5", help="Generate ipole input HDF5 files")
    parser_generate.add_argument("--data-dir", help="Override data directory")
    parser_generate.add_argument("--output-dir", help="Override output directory")
    parser_generate.add_argument("--data-output-dir", help="Override large-data output directory")
    parser_generate.add_argument("--snapshot", help="Run only one snapshot basename, e.g. mad98.prim.00405")
    parser_generate.add_argument("--n-workers", type=int, help="Number of parallel workers")
    parser_generate.add_argument("--sigma-crit", type=float, help="Sigma suppression critical value")
    parser_generate.add_argument("--alpha-sigma", type=float, help="Sigma suppression steepness")
    parser_generate.add_argument("--cooling-factor", type=float, help="Override synchrotron cooling factor")
    parser_generate.add_argument("--enable-advection", choices=["true", "false"], help="Enable or disable advection")
    parser_generate.add_argument(
        "--include-3d-diagnostics",
        choices=["true", "false"],
        help="Include full 3D diagnostic datasets in the HDF5 output",
    )
    parser_generate.add_argument("--run-label", help="Short label embedded in logs and scratch directories")
    parser_generate.add_argument("--scratch-dir", help="Optional local scratch root for temporary generate_h5 input copies")
    parser_generate.add_argument("--show-paths", action="store_true", help="Show path configuration and exit")
    parser_generate.set_defaults(func=cmd_generate_h5)

    # compare_models 子命令
    parser_compare = subparsers.add_parser("compare_models", help="Compare radiation images")
    parser_compare.add_argument("--input-h5", help="Explicit input DSA HDF5 file for comparison")
    parser_compare.add_argument("--output-dir", help="Metadata output directory for compare_models")
    parser_compare.add_argument("--data-output-dir", help="Large-data output directory for compare_models")
    parser_compare.add_argument("--scratch-dir", help="Optional fast local working directory for temporary compare_models HDF5 files")
    parser_compare.add_argument("--models", nargs="+", choices=["A", "B", "C"], help="Subset of models to run, e.g. --models A")
    parser_compare.add_argument("--ipole-dsa-bin", help="Explicit ipole-DSA binary to use for compare_models")
    parser_compare.add_argument("--run-label", help="Short label embedded in compare_models logs and plot directories")
    parser_compare.set_defaults(func=cmd_compare_models)

    # analyze_electrons 子命令
    parser_analyze = subparsers.add_parser("analyze_electrons", help="Analyze non-thermal electrons")
    parser_analyze.set_defaults(func=cmd_analyze_electrons)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
