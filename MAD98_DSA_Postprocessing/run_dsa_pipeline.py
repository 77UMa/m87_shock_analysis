#!/usr/bin/env python3
"""
M87喷流激波加速模型验证项目 - 主运行脚本

项目概述：
通过数值模拟验证扩散激波加速（DSA）理论是否能够解释M87星系中心黑洞喷流的观测形态。
与Yang et al. (2024)的磁重联模型进行对比实验。

运行模式：
1. generate_h5: 从原始模拟数据生成ipole输入文件
2. compare_models: 比较不同物理模型的辐射图像
3. analyze_electrons: 分析非热电子物理性质

M87喷流激波加速模型验证项目 - 主运行脚本 (修正版)
"""

import os
import sys
import argparse
import glob
import time
import logging
import multiprocessing as mp
import traceback

# 路径配置
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
script_dir = os.path.dirname(os.path.abspath(__file__))
pyathena_path = os.path.join(script_dir, '..', 'pyathena')
sys.path.insert(0, pyathena_path)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils.logging_config import setup_logging
from workflows.workflowFull_v2 import process_snapshot
from workflows.compare_models_v0 import main as compare_models_main

# --- 顶层包装函数：解决 Pickle 序列化问题 ---
def process_snapshot_wrapper(task_args):
    """
    包装函数，用于多进程调用。
    task_args: (filename, config, log_dir)
    """
    filename, config, log_dir = task_args
    basename = os.path.basename(filename)
    
    # 每个进程根据 PID 创建日志，避免冲突
    # 设置为 WARNING 级别以精简正常输出，仅保留重要信息
    proc_logger, _ = setup_logging(
        log_dir=log_dir,
        log_level=logging.WARNING, 
        log_name=f"proc_{os.getpid()}.log"
    )
    
    start_time = time.time()
    try:
        # 执行核心逻辑
        success = process_snapshot(filename, config, proc_logger)
        elapsed = time.time() - start_time
        
        if success:
            # 正常完成时不打印冗长日志，仅返回 True
            return True, filename, elapsed
        else:
            return False, filename, "Process returned False without exception"
            
    except Exception as e:
        # 捕获并记录关键报错信息
        error_msg = f"Error in {basename}: {str(e)}\n{traceback.format_exc()}"
        return False, filename, error_msg

# --- 配置函数 ---
def create_default_config():
    CPFS_PATH = "/cpfs01/projects-HDD/cfff-a7e284de52b3_HDD/cyh_22307110238/DSA"
    HOME_PATH = "/home/cyh_22307110238/project/Shockwave"

    return {
        "output_directory": os.path.join(CPFS_PATH, "workflowV2_advection02/"),
        "data_directory": os.path.join(CPFS_PATH, "data/"),
        "ipole_executable_path": os.path.join(HOME_PATH, "ipole-master/ipole"),
        "roi_params": {
            'r_min': 10, 'r_max': 1200,
            'theta_min': 0.0, 'theta_max': 3.14159,
            'phi_min': 0.0, 'phi_max': 6.28319
        },
        "shock_params": {
            "gamma": 4.0/3.0, "mach_threshold_loose": 1.05,
            "min_physical_mach": 1.7, "grad_p_filter_quantile": 0.20,
            "march_cells": 6
        },
        "nt_params": {"gamma": 4.0/3.0, "x_inj": 3.5, "xi_max": 0.05},
        "physics": {
            "spin": 0.98, "hslope": 1.0, "R0": 0.0,
            "enable_advection": True, "cooling_factor": 50.0,
            "advection_steps": 2000
        },
        "max_concurrent_tasks": 5
    }

# --- 子命令处理 ---
def cmd_generate_h5(args):
    config = create_default_config()

    if args.data_dir: config['data_directory'] = args.data_dir
    if args.output_dir: config['output_directory'] = args.output_dir
    if args.n_workers: config['max_concurrent_tasks'] = args.n_workers

    os.makedirs(config['output_directory'], exist_ok=True)
    log_dir = os.path.join(config['output_directory'], 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # 主进程日志
    logger, log_file = setup_logging(
        log_dir=log_dir,
        log_level=logging.INFO,
        log_name=f"main_dsa_{time.strftime('%Y%m%d_%H%M%S')}.log"
    )

    file_pattern = os.path.join(config['data_directory'], 'mad98.prim.*.athdf')
    file_list = sorted(glob.glob(file_pattern))

    if not file_list:
        logger.error(f"No files found: {file_pattern}")
        sys.exit(1)

    logger.info(f"Starting pipeline: {len(file_list)} files, {config['max_concurrent_tasks']} workers")
    
    # 准备并行任务
    tasks = [(f, config, log_dir) for f in file_list]
    start_time = time.time()

    successful_count = 0
    failed_count = 0

    with mp.Pool(processes=config['max_concurrent_tasks'], maxtasksperchild=1) as pool:
        # 使用 imap_unordered 可以即时获取完成情况
        for success, filename, info in pool.imap_unordered(process_snapshot_wrapper, tasks):
            fname = os.path.basename(filename)
            if success:
                successful_count += 1
                # 仅打印精简的成功信息和耗时
                print(f"[SUCCESS] {fname} ({info:.2f}s)")
            else:
                failed_count += 1
                print(f"[FAILED] {fname}")
                logger.error(f"File {fname} failed details:\n{info}")

    total_time = time.time() - start_time
    
    # 最终汇总
    logger.info(f"Summary: Total={len(file_list)}, Success={successful_count}, Failed={failed_count}")
    logger.info(f"Total time: {total_time:.2f} seconds")
    
    print("\n" + "="*30)
    print(f"DONE. Total Time: {total_time:.2f}s")
    print(f"Successful: {successful_count}, Failed: {failed_count}")
    print(f"Main Log: {log_file}")
    print("="*30)

def cmd_compare_models(args):
    compare_models_main()

def cmd_analyze_electrons(args):
    print("Non-thermal electron analysis mode")

# --- 主入口 ---
def main():
    parser = argparse.ArgumentParser(description='M87喷流激波加速模型验证项目')
    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    parser_generate = subparsers.add_parser('generate_h5', help='生成ipole输入HDF5文件')
    parser_generate.add_argument('--data-dir', help='原始数据目录路径')
    parser_generate.add_argument('--output-dir', help='输出目录路径')
    parser_generate.add_argument('--n-workers', type=int, help='并行工作进程数')
    parser_generate.set_defaults(func=cmd_generate_h5)

    parser_compare = subparsers.add_parser('compare_models', help='比较辐射图像')
    parser_compare.set_defaults(func=cmd_compare_models)

    parser_analyze = subparsers.add_parser('analyze_electrons', help='分析非热电子')
    parser_analyze.set_defaults(func=cmd_analyze_electrons)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)

if __name__ == '__main__':
    main()