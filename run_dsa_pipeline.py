#!/usr/bin/env python3
"""Entry point for the M87 DSA processing pipeline."""

import argparse
import glob
import multiprocessing as mp
import os
import sys
import time
import traceback

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


def process_snapshot_wrapper(task_args):
    """Wrapper used by multiprocessing workers."""
    filename, config, log_dir = task_args
    basename = os.path.basename(filename)
    proc_logger = DUALogger(
        log_dir=log_dir,
        run_id=f"run_{time.strftime('%Y%m%d_%H%M%S')}_pid{os.getpid()}",
    )

    start_time = time.time()
    try:
        success = process_snapshot(filename, config, proc_logger)
        elapsed = time.time() - start_time
        if success:
            return True, filename, elapsed
        return False, filename, "Process returned False without exception"
    except Exception as exc:
        error_msg = f"Error in {basename}: {exc}\n{traceback.format_exc()}"
        return False, filename, error_msg


def create_default_config():
    """
    创建默认配置，使用新的统一路径系统

    路径优先级：
    1. 环境变量（M87_DSA_BASE, M87_DATA_DIR 等）
    2. 默认 CPFS 路径
    """
    return {
        "output_directory": ensure_dir("output"),
        "data_output_directory": ensure_dir("data_output"),
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
            "mach_threshold_loose": 1.05,
            "min_physical_mach": 1.7,
            "grad_p_filter_quantile": 0.20,
            "march_cells": 6,
        },
        "nt_params": {
            "gamma": 4.0 / 3.0,
            "x_inj": 3.5,
            "xi_max": 0.05,
            "sigma_crit": 0.1,
            "alpha_sigma": 2,
        },
        "physics": {
            "spin": 0.98,
            "hslope": 1.0,
            "R0": 0.0,
            "enable_advection": False,
            "cooling_factor": 50.0,
            "advection_steps": 2000,
            "M_unit": 1e25,
            "MBH_solar": 6.2e9,
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

    os.makedirs(config["output_directory"], exist_ok=True)
    os.makedirs(config["data_output_directory"], exist_ok=True)
    log_dir = os.path.join(config["output_directory"], "logs")
    os.makedirs(log_dir, exist_ok=True)

    logger, log_file = setup_logging(
        log_dir=log_dir,
        log_name=f"main_dsa_{time.strftime('%Y%m%d_%H%M%S')}.log",
    )

    # 记录路径配置信息
    path_status = validate_paths()
    logger.info(f"Base path: {PATHS['base']}")
    logger.info(f"Data path: {PATHS['data']} (exists: {path_status['data']['exists']})")
    logger.info(f"Metadata output path: {config['output_directory']}")
    logger.info(f"Large-data output path: {config['data_output_directory']}")
    logger.info(f"ipole executable: {config['ipole_executable_path']} (exists: {path_status['ipole_std']['exists']})")

    file_pattern = os.path.join(config["data_directory"], "mad98.prim.*.athdf")
    file_list = sorted(glob.glob(file_pattern))

    if not file_list:
        logger.error(f"No files found: {file_pattern}")
        sys.exit(1)

    logger.info(f"Starting pipeline: {len(file_list)} files, {config['max_concurrent_tasks']} workers")
    tasks = [(filepath, config, log_dir) for filepath in file_list]

    start_time = time.time()
    successful = 0
    failed = 0

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
    parser_generate.add_argument("--n-workers", type=int, help="Number of parallel workers")
    parser_generate.add_argument("--sigma-crit", type=float, help="Sigma suppression critical value")
    parser_generate.add_argument("--alpha-sigma", type=float, help="Sigma suppression steepness")
    parser_generate.add_argument("--show-paths", action="store_true", help="Show path configuration and exit")
    parser_generate.set_defaults(func=cmd_generate_h5)

    # compare_models 子命令
    parser_compare = subparsers.add_parser("compare_models", help="Compare radiation images")
    parser_compare.add_argument("--input-h5", help="Explicit input DSA HDF5 file for comparison")
    parser_compare.add_argument("--output-dir", help="Metadata output directory for compare_models")
    parser_compare.add_argument("--data-output-dir", help="Large-data output directory for compare_models")
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
