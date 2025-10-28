#! /usr/bin/env python3
'''
实现GRMHD模拟的MAD98磁囚禁盘模拟结果的激波寻找、激波加速、非热电子计算和ipole辐射转移计算的工作流，支持批处理。
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
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 将项目根目录添加到Python路径中，以便能找到 pyathena 包
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# --- 确保可以找到 ipole 和 athena 的脚本 ---
# 请根据你的文件结构修改这些路径
# 假设 ipole-master 文件夹与此脚本位于同一父目录下
# 修改：需要ipole脚本和本文件在同一目录之下
script_dir = os.path.dirname(os.path.abspath(__file__))
ipole_dir = os.path.join(script_dir, '..', 'ipole-master')
ipole_scripts_path = os.path.join(ipole_dir, 'scripts')
sys.path.insert(0, ipole_scripts_path)

# 需要 athena_read.py 位于 pyathena 目录下
pyathena_path = os.path.join(script_dir, '..', 'pyathena')
sys.path.insert(0, pyathena_path)

try:
    import ipole as ipole_api
    from pyathena import athena_read
    # 从你的科学计算文件中导入函数
    from nt_electron_v2 import find_shocks_in_roi,calculate_nonthermal_electrons_full, visualize_shock_slice
except ImportError as e:
    print(f"Fatal Error: Could not import a required module. {e}")
    print("Please check the paths to 'ipole-master/scripts' and 'pyathena'.")
    sys.exit(1)

# ==============================================================================
# 辅助函数 (HDF5 生成与绘图)
# ==============================================================================
def create_ipole_input_h5(ipole_h5_filename, roi_data, shock_properties, nonthermal_props, spin):
    """为新版 ipole 生成完全兼容的 HDF5 输入文件。"""
    print(f"--- Step D: Creating IPOLE input file: {os.path.basename(ipole_h5_filename)} ---")
    gamma = 4.0/3.0 #根据Yang et. al 2024选取
    rho, press = roi_data['rho'], roi_data['press']
    uu = press / (gamma - 1.0) #内能
    vel1, vel2, vel3 = roi_data['vel1'], roi_data['vel2'], roi_data['vel3'] #从.h5文件的切片中选取
    b1, b2, b3 = roi_data['Bcc1'], roi_data['Bcc2'], roi_data['Bcc3'] #从.h5文件的切片中选取
    nk, nj, ni = rho.shape #网格数
    n1, n2, n3 = ni, nj, nk

    with h5py.File(ipole_h5_filename, 'w') as f:
        f.create_dataset('t', data=roi_data.get('Time', 0.0), dtype='float64')
        f.create_dataset('dump_cadence', data=1.0, dtype='float64')
        header = f.create_group('header')
        header.create_dataset('n1', data=n1, dtype='int32')
        header.create_dataset('n2', data=n2, dtype='int32')
        header.create_dataset('n3', data=n3, dtype='int32')
        header.create_dataset('n_prim', data=8, dtype='int32')
        header.create_dataset('gam', data=gamma, dtype='float64')
        header.create_dataset('metric', data=np.string_('MKS'))
        prim_names_list = ['RHO', 'UU', 'U1', 'U2', 'U3', 'B1', 'B2', 'B3']
        string_dt = h5py.special_dtype(vlen=str)
        header.create_dataset('prim_names', data=prim_names_list, dtype=string_dt)
        geom = header.create_group('geom')
        geom.create_dataset('startx1', data=np.log(roi_data['x1f'][0]), dtype='float64')
        geom.create_dataset('startx2', data=0.0, dtype='float64')
        geom.create_dataset('startx3', data=0.0, dtype='float64')
        dx1 = np.log(roi_data['x1f'][1] / roi_data['x1f'][0])
        dx2 = 1.0 / n2
        dx3 = (roi_data['x3f'][-1] - roi_data['x3f'][0]) / n3 if n3 > 0 else 0.0
        geom.create_dataset('dx1', data=dx1, dtype='float64')
        geom.create_dataset('dx2', data=dx2, dtype='float64')
        geom.create_dataset('dx3', data=dx3, dtype='float64')
        mks = geom.create_group('mks')
        mks.create_dataset('a', data=spin, dtype='float64')
        mks.create_dataset('r_in', data=roi_data['x1f'][0], dtype='float64')
        mks.create_dataset('r_out', data=roi_data['x1f'][-1], dtype='float64')
        mks.create_dataset('hslope', data=1.0, dtype='float64')
        r_horizon = 1.0 + np.sqrt(1.0 - spin**2)
        mks.create_dataset('r_eh', data=r_horizon, dtype='float64')
        prims_stack = np.stack([rho, uu, vel1, vel2, vel3, b1, b2, b3], axis=-1)
        prims_final = prims_stack.transpose(2, 1, 0, 3)
        f.create_dataset('prims', data=prims_final, dtype='float32')
        q_grid = nonthermal_props['q_grid']
        kel_grid = np.where(shock_properties['mask'], 1, 0)
        p_grid = q_grid - 1.0
        p_grid[~shock_properties['mask']] = 0.0
        f.create_dataset('KEL', data=kel_grid.transpose(2, 1, 0), dtype='int32')
        f.create_dataset('p', data=p_grid.transpose(2, 1, 0), dtype='float32')
    print(f"  Successfully created IPOLE input file.")

def plot_ipole_output(h5_filename, fov_muas, output_png_filename):
    """读取 ipole 输出的 HDF5 文件并可视化 Stokes I 图像。"""
    print(f"--- Step F: Plotting final image from {os.path.basename(h5_filename)} ---")
    try:
        with h5py.File(h5_filename, 'r') as f:
            if 'pol' not in f:
                print(f"  Error: Could not find 'pol' dataset in {h5_filename}.")
                return
            image_I_raw = np.copy(f['pol'])
            image_I = image_I_raw.transpose((1, 0, 2))[:, :, 0]
        extent = [-fov_muas / 2.0, fov_muas / 2.0, -fov_muas / 2.0, fov_muas / 2.0]
        fig, ax = plt.subplots(figsize=(10, 8))
        min_val = 1e-8 * np.max(image_I) if np.max(image_I) > 0 else 1e-20
        plot_data = np.log10(np.maximum(image_I, min_val))
        vmax = np.max(plot_data)
        vmin = vmax - 5
        im = ax.imshow(plot_data, extent=extent, origin='lower', cmap='afmhot', vmin=vmin, vmax=vmax)
        ax.set_xlabel("Relative RA (μas)")
        ax.set_ylabel("Relative Dec (μas)")
        ax.set_title(f"Stokes I - {os.path.basename(h5_filename)} (Log Scale, Wide FOV)")
        cbar = fig.colorbar(im)
        cbar.set_label("log10(Intensity / [Jy/pixel])")
        ax.set_aspect('equal')
        plt.savefig(output_png_filename, dpi=300, bbox_inches='tight')
        print(f"--- Successfully generated image: {os.path.basename(output_png_filename)} ---")
        plt.close(fig)
    except Exception as e:
        print(f"  An error occurred during plotting: {e}")

# ==============================================================================
# 最终版工作流核心函数
# ==============================================================================
def analyze_snapshot_full_pipeline(args):
    """
    【调试版】一个完整的工作流，但在计算完非热电子后会提前停止，用于调试。
    """
    input_athdf_file, config = args
    print(f"\n==============================================================================")
    print(f"Processing snapshot: {os.path.basename(input_athdf_file)}")
    print(f"==============================================================================")
    
    base_name = os.path.basename(input_athdf_file).replace('.athdf', '')
    dir_checkpoints = os.path.join(config['output_directory'], 'checkpoints')
    dir_shock_plots = os.path.join(config['output_directory'], 'shock_visuals')
    
    # --- [调试修改] ---
    # 仅创建调试步骤需要的目录
    for d in [dir_checkpoints, dir_shock_plots]:
        os.makedirs(d, exist_ok=True)

    checkpoint_filename = os.path.join(dir_checkpoints, f"{base_name}_checkpoint.npz")
    shock_plot_filename = os.path.join(dir_shock_plots, f"{base_name}_shock_slice.png")

    # 检查点加载逻辑保持不变
    if config['load_from_checkpoint'] and os.path.exists(checkpoint_filename):
        print(f"--- Loading data from checkpoint: {os.path.basename(checkpoint_filename)} ---")
        with np.load(checkpoint_filename, allow_pickle=True) as data:
            roi_data = data['roi_data'].item()
            shock_properties = data['shock_properties'].item()
            nonthermal_props = data['nonthermal_props'].item()
        print("  Checkpoint loaded successfully.")
    else:
        print("--- Running full pre-processing pipeline from raw data... ---")
        
        print("  Step A: Reconstructing and slicing ROI...")
        full_data = athena_read.athdf(input_athdf_file, level=4)
        
        r_coords = full_data['x1f']
        theta_coords = full_data['x2f']
        r_min, r_max = 1.1, 1200.0
        theta_min, theta_max = 0.0, 0.3 * np.pi
        
        i_start = np.searchsorted(r_coords, r_min, side='left')
        i_end = np.searchsorted(r_coords, r_max, side='right')
        j_start = np.searchsorted(theta_coords, theta_min, side='left')
        j_end = np.searchsorted(theta_coords, theta_max, side='right')
        k_start, k_end = 0, len(full_data['x3f']) - 1

        roi_data = {}
        for key, value in full_data.items():
            if key.startswith('x'):
                if key == 'x1f': roi_data[key] = value[i_start:i_end+1]
                if key == 'x2f': roi_data[key] = value[j_start:j_end+1]
                if key == 'x3f': roi_data[key] = value[k_start:k_end+1]
            else:
                roi_data[key] = value[k_start:k_end+1, j_start:j_end, i_start:i_end]
        
        if 'Time' not in roi_data and 'Time' in full_data:
            roi_data['Time'] = full_data['Time']

        del full_data
        import gc
        gc.collect()

        print("  Step B: Finding shocks and visualization...")
        # 确保此处调用您最新的激波探测函数
        # from nt_electron_v2 import find_shocks_in_roi_grmhd_corrected 
        shock_properties = find_shocks_in_roi(roi_data)
        visualize_shock_slice(roi_data, shock_properties['mask'], os.path.basename(input_athdf_file), shock_plot_filename)
        
        print("  Step C: Calculating non-thermal electrons...")
        if np.any(shock_properties["mask"]):
            nonthermal_props = calculate_nonthermal_electrons_full(shock_properties)
        else:
            nonthermal_props = {
                'q_grid': np.zeros_like(roi_data['press']),
                'C_grid': np.zeros_like(roi_data['press']),
                'mask': np.zeros_like(roi_data['press'], dtype=bool)
            }
            print("  No shocks found, non-thermal properties initialized to zero.")

        if config['save_checkpoint']:
            print(f"--- Saving checkpoint to: {os.path.basename(checkpoint_filename)} ---")
            np.savez(checkpoint_filename, 
                     roi_data=roi_data, 
                     shock_properties=shock_properties, 
                     nonthermal_props=nonthermal_props)
            print("  Checkpoint saved.")

    # --- [调试修改] ---
    # 添加诊断信息输出，帮助您快速判断计算结果
    print("\n--- DEBUGGING OUTPUT ---")
    num_shock_cells = np.sum(shock_properties['mask'])
    print(f"  Total shock cells detected: {num_shock_cells}")
    
    if num_shock_cells > 0:
        # 提取激波位置的马赫数和谱指数q
        mach_values = shock_properties['upstream_mach'][shock_properties['mask']]
        q_values = nonthermal_props['q_grid'][shock_properties['mask']]
        
        print(f"  Upstream Mach Number (M1) at shocks: Min={np.min(mach_values):.2f}, Mean={np.mean(mach_values):.2f}, Max={np.max(mach_values):.2f}")
        print(f"  Power-law index (q) at shocks:      Min={np.min(q_values):.2f}, Mean={np.mean(q_values):.2f}, Max={np.max(q_values):.2f}")
    
    print("--- Workflow paused for debugging after non-thermal electron calculation. ---")
    print("==============================================================================\n")
    
    # --- [调试修改] ---
    # 后续的所有步骤（IPOLE文件生成、运行、绘图、清理）都暂时不执行
    return

    # create_ipole_input_h5(ipole_input_h5, roi_data, shock_properties, nonthermal_props, spin=config['spin'])
    #
    # print(f"--- Step E: Running IPOLE via ipole.py API... ---")
    # args_for_ipole = {key: config[key] for key in ['thetacam', 'freqcgs', 'M_unit', 'trat_j', 'trat_d', 'sigma_cut', 'fov']}
    # args_for_ipole['dump'] = ipole_input_h5
    # args_for_ipole['outfile'] = ipole_output_h5
    # 
    # try:
    #     start_ipole_time = time.time()
    #     ipole_api.run(args_for_ipole, exe=config['ipole_executable_path'], verbose=1)
    #     end_ipole_time = time.time()
    #     
    #     print(f"  IPOLE execution time: {end_ipole_time - start_ipole_time:.2f} seconds.")
    #         
    #     if os.path.exists(ipole_output_h5):
    #         plot_ipole_output(ipole_output_h5, fov_muas=config['fov'], output_png_filename=final_png_name)
    #     else:
    #         print(f"  Error: IPOLE did not produce the expected output file.")
    #
    # except Exception as e:
    #     print(f"\n  !!! IPOLE EXECUTION FAILED: {e} !!!\n")
    # finally:
    #     if config['auto_cleanup'] and os.path.exists(ipole_input_h5):
    #         print(f"--- Cleaning up intermediate file: {os.path.basename(ipole_input_h5)} ---")
    #         os.remove(ipole_input_h5)
    #         print("  Cleanup complete.")

# ==============================================================================
# 主程序入口
# ==============================================================================
if __name__ == '__main__':
    
    config = {
        # --- 路径配置 (请务必使用绝对路径) ---
        "data_directory": "/home/cyh_22307110238/project/Shockwave/data_test2/",
        "output_directory": "/home/cyh_22307110238/project/Shockwave/workflow_output/",
        "ipole_executable_path": "/home/cyh_22307110238/project/Shockwave/ipole-master/ipole",
        
        # --- 工作流控制 ---
        "save_checkpoint": True,
        "load_from_checkpoint": False,
        "auto_cleanup": True,
        
        # --- 物理参数 ---
        "spin": 0.98,
        
        # --- IPOLE 观测参数 ---
        "thetacam": 163.0,
        "freqcgs": 86e9,
        "M_unit": 1e25,
        "trat_j": 1.0,
        "trat_d": 80.0,
        "sigma_cut": 5.0,
        "fov": 5000.0,
        
        # --- 并行计算配置 ---
        "num_processes": max(1, mp.cpu_count() - 2)
    }

    file_pattern = os.path.join(config['data_directory'], 'mad98.prim.*.athdf')
    file_list = sorted(glob.glob(file_pattern))
    
    if not file_list:
        print(f"Error: No files found matching pattern: {file_pattern}")
        sys.exit(1)

    print(f"Found {len(file_list)} files to process.")
    print(f"Output will be saved to: {config['output_directory']}")
    print(f"Initializing a pool of {config['num_processes']} worker processes.")

    tasks = [(filename, config) for filename in file_list]
    start_total_time = time.time()
    
    # --- 【核心修正】强制使用 'spawn' 启动方法 ---
    # 这会为每个子进程创建一个干净的环境，从根本上解决 h5py 的冲突问题。
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # 如果上下文已设置，可能会报错，忽略即可
        pass 

    with mp.Pool(processes=config['num_processes'], maxtasksperchild=1) as pool:
        pool.map(analyze_snapshot_full_pipeline, tasks)

    end_total_time = time.time()
    
    print("\n==============================================================================")
    print("--- ALL TASKS COMPLETED ---")
    print(f"Total execution time for {len(file_list)} files: {end_total_time - start_total_time:.2f} seconds.")
    print(f"==============================================================================\n")

    

