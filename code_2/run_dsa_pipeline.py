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
"""

import os
import sys
import argparse
import glob

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

script_dir = os.path.dirname(os.path.abspath(__file__))
pyathena_path = os.path.join(script_dir, '..', 'pyathena')
sys.path.insert(0, pyathena_path)

# 将src目录添加到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from workflows.workflowFull_v2 import process_snapshot, run_parallel_workflow
from workflows.compare_models_v0 import main as compare_models_main


def create_default_config():
    """
    创建默认配置参数

    Returns:
        dict: 配置参数字典，包含数据路径、物理参数、并行设置等
    """
    CPFS_PATH = "/cpfs01/projects-HDD/cfff-a7e284de52b3_HDD/cyh_22307110238/DSA"
    HOME_PATH = "/home/cyh_22307110238/project/Shockwave"

    return {
        # 路径配置
        "output_directory": os.path.join(CPFS_PATH, "workflowV2_advection01/"),
        "data_directory": os.path.join(CPFS_PATH, "data/"),
        "ipole_executable_path": os.path.join(HOME_PATH, "ipole-master/ipole"),

        # ROI切片参数
        "roi_params": {
            'r_min': 10, 'r_max': 1200,
            'theta_min': 0.0, 'theta_max': 3.14159,  # 完整全域
            'phi_min': 0.0, 'phi_max': 6.28319
        },

        # 激波探测参数
        "shock_params": {
            "gamma": 4.0/3.0,
            "mach_threshold_loose": 1.05,
            "min_physical_mach": 1.7,
            "grad_p_filter_quantile": 0.20,
            "march_cells": 6
        },

        # 非热电子参数
        "nt_params": {
            "gamma": 4.0/3.0,
            "x_inj": 3.5,
            "xi_max": 0.05
        },

        # 物理参数
        "physics": {
            "spin": 0.98,   # MAD98黑洞自旋
            "hslope": 1.0,  # 无压缩theta坐标
            "R0": 0.0,      # 无径向平移
            "enable_advection": True,  # 是否启用平流-冷却扩散模型
            "cooling_factor": 50.0,     # 冷却因子
            "advection_steps": 2000     # 平流计算步数
        },

        # 并行计算配置
        "max_concurrent_tasks": 5
    }


def cmd_generate_h5(args):
    """
    命令处理函数：生成ipole输入HDF5文件

    Args:
        args: 命令行参数对象
    """
    config = create_default_config()

    # 覆盖默认配置
    if args.data_dir:
        config['data_directory'] = args.data_dir
    if args.output_dir:
        config['output_directory'] = args.output_dir
    if args.n_workers:
        config['max_concurrent_tasks'] = args.n_workers

    # 创建输出目录
    os.makedirs(config['output_directory'], exist_ok=True)

    # 查找所有模拟文件
    file_pattern = os.path.join(config['data_directory'], 'mad98.prim.*.athdf')
    file_list = sorted(glob.glob(file_pattern))

    if not file_list:
        print(f"Error: No files found matching pattern: {file_pattern}")
        sys.exit(1)

    # 运行并行处理
    start_time = args.time.time() if hasattr(args, 'time') else __import__('time').time()
    run_parallel_workflow(file_list, config, process_snapshot)
    end_time = __import__('time').time()

    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")


def cmd_compare_models(args):
    """
    命令处理函数：比较不同模型的辐射图像

    Args:
        args: 命令行参数对象
    """
    # 调用compare_models模块的main函数
    # 注意：需要设置相应的环境变量和参数
    compare_models_main()


def cmd_analyze_electrons(args):
    """
    命令处理函数：分析非热电子物理

    Args:
        args: 命令行参数对象
    """
    print("Non-thermal electron analysis mode")
    # 这里可以调用专门的非热电子分析模块
    # 目前workflowElectron已经包含了相关功能


def main():
    """
    主入口函数，解析命令行参数并执行相应操作
    """
    parser = argparse.ArgumentParser(
        description='M87喷流激波加速模型验证项目',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
使用示例:
  # 生成HDF5输入文件
  python run_dsa_pipeline.py generate_h5 --data-dir /path/to/data --output-dir /path/to/output

  # 比较不同模型
  python run_dsa_pipeline.py compare_models

  # 分析非热电子
  python run_dsa_pipeline.py analyze_electrons
        '''
    )

    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # generate_h5命令
    parser_generate = subparsers.add_parser('generate_h5', help='生成ipole输入HDF5文件')
    parser_generate.add_argument('--data-dir', help='原始数据目录路径')
    parser_generate.add_argument('--output-dir', help='输出目录路径')
    parser_generate.add_argument('--n-workers', type=int, help='并行工作进程数')
    parser_generate.set_defaults(func=cmd_generate_h5)

    # compare_models命令
    parser_compare = subparsers.add_parser('compare_models', help='比较不同物理模型的辐射图像')
    parser_compare.set_defaults(func=cmd_compare_models)

    # analyze_electrons命令
    parser_analyze = subparsers.add_parser('analyze_electrons', help='分析非热电子物理性质')
    parser_analyze.set_defaults(func=cmd_analyze_electrons)

    # 解析参数
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # 执行相应命令
    args.func(args)


if __name__ == '__main__':
    main()