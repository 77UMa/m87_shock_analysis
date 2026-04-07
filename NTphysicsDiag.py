#! /usr/bin/env python3
'''
旧版本工作流，现用于诊断DSA部分物理(激波+非热电子)
从 .athdf 文件读取 -> MHD激波探测 -> 非热电子计算 -> 生成 IPOLE 输入 -> 运行 IPOLE -> 绘制最终图像。

'''

# ==============================================================================
# 导入所需模块
# ==============================================================================
import os
import sys
import glob
import time
import multiprocessing as mp
from functools import partial
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg') # 必须在 pyplot 导入前设置，用于无GUI的服务器
import matplotlib.pyplot as plt


# ==============================================================================
# 路径配置
# ==============================================================================
# 将项目根目录添加到Python路径中
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

script_dir = os.path.dirname(os.path.abspath(__file__))

# 导入统一路径配置
sys.path.insert(0, os.path.join(script_dir, 'src'))
from src.utils.paths import PATHS

# 路径配置（现在使用 paths.py 统一管理）
ipole_dir = os.path.join(PATHS['base'], 'ipole-master')
ipole_scripts_path = os.path.join(ipole_dir, 'scripts')
sys.path.insert(0, ipole_scripts_path)

pyathena_path = os.path.join(PATHS['base'], 'pyathena')
sys.path.insert(0, pyathena_path)

try:
    # 尝试导入核心依赖
    from pyathena import athena_read
    import ipole as ipole_api
    # 从你的科学计算文件中导入函数
    from src.core.shock_v1 import find_shocks_in_roi_mhd,  visualize_shock_projection_dual_range, visualize_shock_3d_interactive_html
    from src.core.nt_electron_v1 import calculate_nonthermal_electrons, plot_diagnostic_histograms, plot_diagnostic_correlations

except ImportError as e:
    print(f"Fatal Error: Could not import a required module. {e}")
    print("Please check the paths to 'ipole-master/scripts' and 'pyathena'.")
    sys.exit(1)

# ==============================================================================
# 完整工作流主函数 (已集成 IPOLE)
# ==============================================================================

def analyze_snapshot_full_pipeline(filename, config):
    """
    【完整端到端工作流 - 已修正切片逻辑】
    从 .athdf 文件加载，切片，运行稳健的激波探测，
    计算非热电子，生成诊断图，创建 IPOLE h5 输入，
    运行 IPOLE，并绘制最终的辐射图像。
    """
    input_athdf_file = filename
    print(f"\n==============================================================================")
    print(f"Processing snapshot: {os.path.basename(input_athdf_file)}")
    print(f"==============================================================================")
    
    base_name = os.path.basename(input_athdf_file).replace('.athdf', '')
    
    # --- 定义所有 CPFS 路径 ---
    dir_full_data = os.path.join(config['data_output_directory'], 'full_data_checkpoints')
    os.makedirs(dir_full_data, exist_ok=True)
    full_data_checkpoint_filename = os.path.join(dir_full_data, f"{base_name}_full_data.npz")
    
    dir_analysis_checkpoints = os.path.join(config['data_output_directory'], 'analysis_checkpoints')
    os.makedirs(dir_analysis_checkpoints, exist_ok=True)
    analysis_checkpoint_filename = os.path.join(dir_analysis_checkpoints, f"{base_name}_analysis.npz")
    
    dir_shock_plots = os.path.join(config['output_directory'], 'shock_visuals')
    os.makedirs(dir_shock_plots, exist_ok=True)
    
    dir_diag_plots = os.path.join(config['output_directory'], 'diagnostic_plots')
    os.makedirs(dir_diag_plots, exist_ok=True)
    
    dir_ipole_inputs = os.path.join(config['data_output_directory'], 'ipole_inputs') 
    dir_ipole_outputs = os.path.join(config['data_output_directory'], 'ipole_outputs')
    dir_final_images = os.path.join(config['output_directory'], 'final_images')
    os.makedirs(dir_ipole_inputs, exist_ok=True)
    os.makedirs(dir_ipole_outputs, exist_ok=True)
    os.makedirs(dir_final_images, exist_ok=True)

    # --- 阶段一：数据加载/重建 ---
    full_data = None
    if config.get('load_full_data_checkpoint', False) and os.path.exists(full_data_checkpoint_filename):
        print(f"--- Loading full reconstructed data from checkpoint: {os.path.basename(full_data_checkpoint_filename)} ---")
        try:
            with np.load(full_data_checkpoint_filename, allow_pickle=True) as data:
                full_data = data['full_data'].item()
            print("--- Full data loaded successfully. Skipping reconstruction. ---")
        except Exception as e:
            print(f"  Warning: Failed to load checkpoint {full_data_checkpoint_filename}. Error: {e}. Re-running reconstruction.")
            full_data = None
    
    if full_data is None:
        print("--- Running full data reconstruction from .athdf file... (This may take a while) ---")
        full_data = athena_read.athdf(input_athdf_file, level=4)

        if config.get('save_full_data_checkpoint', False):
            print(f"--- Saving full reconstructed data to checkpoint: {os.path.basename(full_data_checkpoint_filename)} ---")
            np.savez_compressed(full_data_checkpoint_filename, full_data=full_data)
            print("--- Full data checkpoint saved. ---")

    # --- 阶段二：ROI 切片 (*** 已修正此处的逻辑 ***) ---
    print("  Step A: Slicing ROI from full data...")
    r_coords = full_data['x1f']
    theta_coords = full_data['x2f']
    phi_coords = full_data['x3f']
    
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
    
    if i_start >= i_end or j_start >= j_end or k_start >= k_end:
        print(f"  Error: Invalid ROI slice for {base_name}. Skipping file.")
        del full_data
        import gc; gc.collect()
        return

    # --- [修正开始] ---
    # 使用您原始脚本中的通用切片逻辑
    # 这假定 athena_read 已经解包了 'prim' 和 'Bcc'
    roi_data = {}
    for key, value in full_data.items():
        if key.startswith('x'):
            if key == 'x1f': roi_data[key] = value[i_start:i_end+1]
            if key == 'x2f': roi_data[key] = value[j_start:j_end+1]
            if key == 'x3f': roi_data[key] = value[k_start:k_end+1]
        elif key in ['Time', 'dump_cadence', 'Levels', 'LogicalLocations']: # 复制元数据
             roi_data[key] = value
        elif hasattr(value, 'ndim') and value.ndim >= 3: # 假设所有其他高维数组都是需要切片的物理数据
            # (nz, ny, nx) 或 (nz, ny, nx, nvar)
            # athena_read 返回 (nz, ny, nx) [k, j, i]
            roi_data[key] = value[k_start:k_end, j_start:j_end, i_start:i_end, ...]
        else:
            # 复制其他可能的标量元数据
            roi_data[key] = value
            
    if 'Time' not in roi_data and 'Time' in full_data:
        roi_data['Time'] = full_data['Time']
    # --- [修正结束] ---
    
    del full_data
    import gc; gc.collect()
    
    # 检查 'press' 是否已成功切片
    if 'press' not in roi_data:
        print(f"  FATAL ERROR: 'press' key not found in roi_data after slicing for {base_name}.")
        print(f"  Available keys: {list(roi_data.keys())}")
        return # 立即停止此任务

    # --- 阶段三：激波探测 ---
    print("  Step B: Finding shocks...")
    shock_properties = find_shocks_in_roi_mhd(roi_data, **config["shock_params"]) 
    
    gamma = 4.0 / 3.0
    B2 = (roi_data['Bcc1']**2 + roi_data['Bcc2']**2 + roi_data['Bcc3']**2)
    uu = roi_data['press'] / (gamma - 1.0)
    denom = roi_data['rho'] + uu + roi_data['press']
    sigma_grid = np.where(denom > 0, B2 / (2.0 * denom), 0.0)
    shock_properties['sigma_grid'] = sigma_grid

    # --- 阶段四：非热电子计算 ---
    print("  Step C: Calculating non-thermal electrons...")
    if np.any(shock_properties["mask"]):
        nonthermal_props = calculate_nonthermal_electrons(shock_properties, **config["nt_params"])
    else:
        nonthermal_props = {
            'q_grid': np.zeros_like(roi_data['press']),
            'C_grid': np.zeros_like(roi_data['press']),
            'mask': np.zeros_like(roi_data['press'], dtype=bool)
        }
        print("  No shocks found, non-thermal properties initialized to zero.")

    # --- 阶段五：诊断与可视化 (激波 + 诊断图) ---
    print("  Step D: Generating diagnostic visualizations...")
    
    # projection_plot_filename = os.path.join(dir_shock_plots, f"{base_name}_shock_projection_dual.png")
    # visualize_shock_projection_dual_range(roi_data, shock_properties, base_name, projection_plot_filename)

    vis_3d_filename_html = os.path.join(dir_shock_plots, f"{base_name}_shock_3d_interactive.html")
    visualize_shock_3d_interactive_html(roi_data, shock_properties, base_name, vis_3d_filename_html)

    diag_hist_filename = os.path.join(dir_diag_plots, f"{base_name}_nt_diag_hist.png")
    plot_diagnostic_histograms(shock_properties, nonthermal_props, base_name, diag_hist_filename)
    
    diag_corr_filename = os.path.join(dir_diag_plots, f"{base_name}_nt_diag_corr.png")
    plot_diagnostic_correlations(shock_properties, nonthermal_props, base_name, diag_corr_filename)
    
    return #废止后续功能



# ==============================================================================
# 主程序入口 (使用新的统一路径系统)
# ==============================================================================
if __name__ == '__main__':
    # 导入路径配置
    from src.utils.paths import PATHS

    config = {
        # --- 路径配置 (现在使用 paths.py) ---
        "data_directory": PATHS['data'],
        "output_directory": os.path.join(PATHS['output'], "workflow_output_DSA_run04/"),
        "data_output_directory": os.path.join(PATHS['data_output'], "workflow_output_DSA_run04/"),
        "ipole_executable_path": PATHS['ipole_dsa'],
        
        # --- 工作流控制 ---
        "save_full_data_checkpoint": False, #老旧功能，数据检查点
        "load_full_data_checkpoint": False, #老旧功能，数据检查点
        "auto_cleanup": False, # [新增] 清理 ipole_input.h5

        # --- ROI 切片参数 ---
        "roi_params": {
            'r_min': 10, 'r_max': 1200,
            'theta_min': 0.0, 'theta_max': np.pi/2, # 只看北半球喷流
            'phi_min': 0.0, 'phi_max': 2*np.pi
        },

        # --- 激波探测参数 (使用新函数) ---
        "shock_params": {
            "gamma": 4.0/3.0,
            "grad_p_filter_quantile": 0.20,
            "march_cells": 5
        },
        "shock_sr": {
            "sr_mach_min": 1.7,
            "jump_residual_max": 0.8,
        },

        # --- 非热电子参数 ---
        "nt_params": {
            "gamma": 4.0/3.0,
            "x_inj": 3.5,
            "eta_inj_e0": 1.0e-3,
            "eps_nth_e0": 3.0e-3,
            "theta_bn_quench": 50.0,
            "theta_bn_width": 10.0,
            "sonic_mach_inj_min": 1.5,
            "inj_model": "pic_dual_cap",
        },

        # --- 物理参数 (IPOLE) ---
        "spin": 0.98, # [新增] [cite: 6003-6824]
        
        # --- IPOLE 观测参数 ---
        # [新增]
        "ipole_params": {
            "thetacam": 163.0,  # M87 观测倾角
            "freqcgs": 86e9,     # 86 GHz [cite: 6003-6824, 7009-7010]
            "M_unit": 1e25,      #
            "trat_j": 1.0,       #
            "trat_d": 80.0,      # [cite: 6003-6824, 7009-7010] (R_high)
            "sigma_cut": 5.0,    # [cite: 6003-6824, 7009-7010]
            "fov": 1000.0        # μas, (可调整)
        },

        # --- 画图压强范围配置 ---
        "v_min": -8,
        "v_max": -1,
        
        # --- 并行计算配置 ---
        "num_processes": 30,
        "max_concurrent_tasks": 12 
    }
    
    # --- 准备文件和任务列表 ---
    file_pattern = os.path.join(config['data_directory'], 'mad98.prim.*.athdf')
    file_list = sorted(glob.glob(file_pattern))
    
    if not file_list:
        print(f"Error: No files found matching pattern: {file_pattern}")
        sys.exit(1)

    print(f"Found {len(file_list)} files to process.")
    print(f"Metadata output will be saved to: {config['output_directory']}")
    print(f"Large-data output will be saved to: {config['data_output_directory']}")
    
    # 使用 partial 来固定 config 参数
    task_func = partial(analyze_snapshot_full_pipeline, config=config)
    filenames_only = file_list # Pool.map 会将列表中的每个元素作为 'filename' 传入

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
