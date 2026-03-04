#! /usr/bin/env python3
'''
【已恢复的稳健版本】
实现GRMHD模拟的MAD98磁囚禁盘模拟结果的激波可视化与非热电子诊断。
支持批处理，使用进程池，并将所有数据路径指向 CPFS 持久化存储。

包含：
1. 稳健的激波探测 (梯度回溯 + 双重阈值)
2. 非热电子计算 (基于 nt_electron_v1.py)
3. 激波可视化 (3D HTML, 2D 投影)
4. 新增的非热电子物理诊断图 (直方图, 相关性图)
'''

# ==============================================================================
# 导入所需模块
# ==============================================================================
import os
import sys
import glob
import time
import subprocess
import multiprocessing as mp
from functools import partial # 导入 partial
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg') # 必须在 pyplot 导入前设置，用于无GUI的服务器
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LogNorm

# 【新增】非热电子计算所需的模块
from scipy.special import betainc 
# 【新增】3D交互式可视化所需的模块
import plotly.graph_objects as go
import pandas as pd # Optional, but often convenient

# ==============================================================================
# 路径配置 (与原始脚本一致)
# ==============================================================================
# 将项目根目录添加到Python路径中，以便能找到 pyathena 包
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

script_dir = os.path.dirname(os.path.abspath(__file__))
ipole_dir = os.path.join(script_dir, '..', 'ipole-master')
ipole_scripts_path = os.path.join(ipole_dir, 'scripts')
sys.path.insert(0, ipole_scripts_path)

pyathena_path = os.path.join(script_dir, '..', 'pyathena')
sys.path.insert(0, pyathena_path)

try:
    # 尝试导入核心依赖
    from pyathena import athena_read
    from pyathena import athena_read
    # 从你的科学计算文件中导入函数
    from shock_v1 import find_shocks_in_roi_robust,  visualize_shock_projection_dual_range, visualize_shock_3d_interactive_html
    from nt_electron_v1 import calculate_nonthermal_electrons, plot_diagnostic_histograms, plot_diagnostic_correlations

except ImportError as e:
    print(f"Fatal Error: Could not import a required module. {e}")
    print("Please check the paths to 'ipole-master/scripts' and 'pyathena'.")
    sys.exit(1)

# ==============================================================================
# 工作流核心函数 (已更新)
# ==============================================================================

def analyze_snapshot_full_pipeline(filename, config):
    """
    【升级调试版】增加了非热电子计算和诊断步骤。
    """
    input_athdf_file = filename
    print(f"\n==============================================================================")
    print(f"Processing snapshot: {os.path.basename(input_athdf_file)}")
    print(f"==============================================================================")
    
    base_name = os.path.basename(input_athdf_file).replace('.athdf', '')
    
    # --- [已更新] 定义 CPFS 路径 ---
    dir_full_data = os.path.join(config['output_directory'], 'full_data_checkpoints')
    os.makedirs(dir_full_data, exist_ok=True)
    full_data_checkpoint_filename = os.path.join(dir_full_data, f"{base_name}_full_data.npz")
    
    dir_analysis_checkpoints = os.path.join(config['output_directory'], 'analysis_checkpoints')
    os.makedirs(dir_analysis_checkpoints, exist_ok=True)
    analysis_checkpoint_filename = os.path.join(dir_analysis_checkpoints, f"{base_name}_analysis.npz")
    
    dir_shock_plots = os.path.join(config['output_directory'], 'shock_visuals')
    os.makedirs(dir_shock_plots, exist_ok=True)

    # 【新增】非热电子诊断图输出目录
    dir_diag_plots = os.path.join(config['output_directory'], 'non-thermal electrons')
    os.makedirs(dir_diag_plots, exist_ok=True)

    # --- 阶段一：数据加载/重建 (与原始脚本相同) ---
    full_data = None
    if config.get('load_full_data_checkpoint', False) and os.path.exists(full_data_checkpoint_filename):
        print(f"--- Loading full reconstructed data from checkpoint: {os.path.basename(full_data_checkpoint_filename)} ---")
        with np.load(full_data_checkpoint_filename, allow_pickle=True) as data:
            full_data = data['full_data'].item()
        print("--- Full data loaded successfully. Skipping reconstruction. ---")
    
    if full_data is None:
        print("--- Running full data reconstruction from .athdf file... (This may take a while) ---")
        full_data = athena_read.athdf(input_athdf_file, level=4)

        if config.get('save_full_data_checkpoint', False):
            print(f"--- Saving full reconstructed data to checkpoint: {os.path.basename(full_data_checkpoint_filename)} ---")
            np.savez_compressed(full_data_checkpoint_filename, full_data=full_data)
            print("--- Full data checkpoint saved. ---")

    # --- 阶段二：ROI 切片与科学分析 ---
    print("  Step A: Slicing ROI from full data...")
    r_coords = full_data['x1f']
    theta_coords = full_data['x2f']
    phi_coords = full_data['x3f']
    
    # 从 config 中获取 ROI 参数
    roi_cfg = config['roi_params']
    r_min, r_max = roi_cfg['r_min'], roi_cfg['r_max']
    theta_min, theta_max = roi_cfg['theta_min'], roi_cfg['theta_max']
    phi_min, phi_max = roi_cfg['phi_min'], roi_cfg['phi_max']
    
    i_start = np.searchsorted(r_coords, r_min, side='left')
    i_end = np.searchsorted(r_coords, r_max, side='right')
    j_start = np.searchsorted(theta_coords, theta_min, side='left')
    j_end = np.searchsorted(theta_coords, theta_max, side='right')
    k_start = np.searchsorted(phi_coords, phi_min, side='left')
    k_end = np.searchsorted(phi_coords, phi_max, side='right')
    
    # 确保切片索引有效
    if i_start >= i_end or j_start >= j_end or k_start >= k_end:
        print(f"  Error: Invalid ROI slice for {base_name}. Skipping file.")
        del full_data
        import gc; gc.collect()
        return

    roi_data = {}
    for key, value in full_data.items():
        if key.startswith('x'):
            if key == 'x1f': roi_data[key] = value[i_start:i_end+1]
            if key == 'x2f': roi_data[key] = value[j_start:j_end+1]
            if key == 'x3f': roi_data[key] = value[k_start:k_end+1]
        elif key in ['prim', 'B', 'Bcc', 'vel1', 'vel2', 'vel3', 'press', 'rho', 'Bcc1', 'Bcc2', 'Bcc3']:
             # 确保我们只切片3D/4D物理数据
            if value.ndim >= 3:
                # 原始 athena_read 返回 (nz, ny, nx, nvar) 或 (nz, ny, nx)
                # 确保 k, j, i 对应 nz, ny, nx
                roi_data[key] = value[k_start:k_end, j_start:j_end, i_start:i_end, ...]
            else:
                roi_data[key] = value # 复制非空间维度的数据 (如 'Time')
    
    if 'Time' not in roi_data and 'Time' in full_data:
        roi_data['Time'] = full_data['Time']
    
    del full_data
    import gc; gc.collect()

    # --- 阶段三：激波探测 ---
    print("  Step B: Finding shocks...")
    shock_properties = find_shocks_in_roi_robust(roi_data, **config["shock_params"]) 
    
    # --- 阶段四：非热电子计算 (新增) ---
    print("  Step C: Calculating non-thermal electrons...")
    if np.any(shock_properties["mask"]):
        nonthermal_props = calculate_nonthermal_electrons(shock_properties, **config["nt_params"])
    else:
        nonthermal_props = {
            'q_grid': np.zeros_like(roi_data['rho']),
            'C_grid': np.zeros_like(roi_data['rho']),
            'mask': np.zeros_like(roi_data['rho'], dtype=bool)
        }
        print("  No shocks found, non-thermal properties initialized to zero.")

    # --- 阶段五：可视化 (激波 + 新增的诊断图) ---
    print("  Step D: Generating visualizations...")
    # ==============================================================================
    # 激波部分
    # ==============================================================================  
      
    # a. 激波：单phi切片侧视图 (R-Z平面)
    # overview_plot_filename = os.path.join(dir_shock_plots, f"{base_name}_shock_overview_slice.png")
    # visualize_shock_overview(roi_data, shock_properties, base_name, overview_plot_filename, config)

    # b. 激波：双范围俯视图
    projection_plot_filename = os.path.join(dir_shock_plots, f"{base_name}_shock_projection_dual.png")
    visualize_shock_projection_dual_range(roi_data, shock_properties, base_name, projection_plot_filename)

    # c. 激波：X-Z全景切片图
    # xz_plane_plot_filename = os.path.join(dir_shock_plots, f"{base_name}_shock_plane_xz.png")
    # visualize_shock_xz_plane(roi_data, shock_properties, base_name, xz_plane_plot_filename, config)

    # d. 激波：3D交互式HTML
    vis_3d_filename_html = os.path.join(dir_shock_plots, f"{base_name}_shock_3d_interactive.html")
    visualize_shock_3d_interactive_html(roi_data, shock_properties, base_name, vis_3d_filename_html)

    # ==============================================================================
    # 非热电子部分
    # ==============================================================================  

    # e. 【新增】非热电子：1D 统计直方图
    diag_hist_filename = os.path.join(dir_diag_plots, f"{base_name}_nt_diag_hist.png")
    plot_diagnostic_histograms(shock_properties, nonthermal_props, base_name, diag_hist_filename)
    
    # f. 【新增】非热电子：2D 物理相关性
    diag_corr_filename = os.path.join(dir_diag_plots, f"{base_name}_nt_diag_corr.png")
    plot_diagnostic_correlations(shock_properties, nonthermal_props, base_name, diag_corr_filename)

    print(f"--- Analysis complete for {base_name}. ---")
    print("==============================================================================\n")
    return # 暂停工作流，不进行ipole计算


# ==============================================================================
# 主程序入口 (已更新)
# ==============================================================================
if __name__ == '__main__':
    
    # [已更新] CPFS 路径
    CPFS_ROOT_PATH = "/cpfs01/projects-HDD/cfff-a7e284de52b3_HDD/cyh_22307110238"
    # [保持不变] Home 路径
    HOME_PATH = "/home/cyh_22307110238/project/Shockwave"


    config = {
        # --- 路径配置 ---
        "data_directory": os.path.join(CPFS_ROOT_PATH, "data_test3/"),
        "output_directory": os.path.join(CPFS_ROOT_PATH, "workflow_output_robust/"),
        "ipole_executable_path": os.path.join(HOME_PATH, "ipole-master/ipole"),
        
        # --- 工作流控制 ---
        "save_full_data_checkpoint": False,
        "load_full_data_checkpoint": False,

        # --- ROI 切片参数 ---
        "roi_params": {
            'r_min': 10, 'r_max': 1200,
            'theta_min': 0.0, 'theta_max': np.pi/2,
            'phi_min': 0.0, 'phi_max': 2*np.pi
        },

        # --- 激波探测参数 (使用新函数) ---
        "shock_params": {
            "gamma": 4.0/3.0,
            "mach_threshold_loose": 1.05, # [新] 宽松的初筛阈值
            "min_physical_mach": 1.7,  # [新] 严格的物理验证阈值
            "grad_p_filter_quantile": 0.20,
            "march_cells": 6            # [新] 回溯距离
        },

        # --- 【新增】非热电子参数 ---
        "nt_params": {
            "gamma": 4.0/3.0,
            "x_inj": 3.5,
            "xi_max": 0.05
        },

        # --- 画图压强范围配置 ---
        "v_min": -8,
        "v_max": -1,
        
        # --- 并行计算配置 ---
        "num_processes": 30,
        "max_concurrent_tasks": 12 # 限制内存密集型任务的并发数
    }
    
    # --- 准备文件和任务列表 ---
    file_pattern = os.path.join(config['data_directory'], 'mad98.prim.*.athdf')
    file_list = sorted(glob.glob(file_pattern))
    
    if not file_list:
        print(f"Error: No files found matching pattern: {file_pattern}")
        sys.exit(1)

    print(f"Found {len(file_list)} files to process.")
    print(f"Output will be saved to: {config['output_directory']}")
    
    tasks = [(filename, config) for filename in file_list]
    
    # --- 使用 partial 来固定 config 参数 ---
    # 注意：Pool.map 只接受一个可迭代的参数列表，所以我们必须打包
    # analyze_snapshot_full_pipeline 只需要一个参数 (filename)
    # 我们使用 partial 来“预填充” config 参数
    
    # [修改] 使用 partial 重新包装任务函数
    task_func = partial(analyze_snapshot_full_pipeline, config=config)
    # 准备一个只包含文件名的列表
    filenames_only = [f for f, _ in tasks] 

    SAFE_MAX_WORKERS = config['max_concurrent_tasks']
    
    print(f"Initializing a pool of {SAFE_MAX_WORKERS} worker processes.")
    
    start_total_time = time.time()
    
    # --- 设置多进程启动方式 ---
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass 

    # --- 使用 multiprocessing.Pool 并加入 maxtasksperchild=1 ---
    try:
        # maxtasksperchild=1 确保每个子进程在完成一个任务后被销毁和重建
        # 这是一种处理内存泄漏的有效（虽然粗暴）的方法
        with mp.Pool(processes=SAFE_MAX_WORKERS, maxtasksperchild=1) as pool:
            print(f"--- Submitting {len(filenames_only)} tasks with process recycling enabled ---")
            
            # pool.map 会阻塞直到所有任务完成
            results = pool.map(task_func, filenames_only)
            
            print(f"--- All {len(filenames_only)} tasks have been processed. ---")

    except Exception as e:
        print(f"\n---!!! An error occurred during parallel execution: {e} ---")
        import traceback
        traceback.print_exc()

    end_total_time = time.time()
    
    print("\n==============================================================================")
    print("--- WORKFLOW SCRIPT FINISHED ---")
    print(f"Total execution time for {len(file_list)} files: {end_total_time - start_total_time:.2f} seconds.")
    print(f"==============================================================================\n")