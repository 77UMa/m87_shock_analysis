"""
DSA工作流主模块 - 负责从原始Athena++模拟数据生成ipole输入文件

主要功能：
1. 加载原始.athdf格式的GRMHD模拟数据
2. 执行ROI（感兴趣区域）切片，减少计算量
3. 检测激波并计算激波物理性质
4. 根据激波性质计算非热电子能谱参数
5. 可选：执行平流-冷却扩散计算，扩展非热电子分布
6. 生成注入DSA物理的ipole输入HDF5文件

使用场景：批处理大量模拟快照，生成辐射转移计算的输入文件
"""

import os
import numpy as np
from src.workflows.base_workflow import load_and_slice_data, calculate_dsa_physics, save_h5_file, run_parallel_workflow

def process_snapshot(filename, config):
    """
    处理单个快照文件的完整流程

    Args:
        filename: 输入的.athdf文件路径
        config: 配置参数字典，包含ROI参数、物理参数等

    Returns:
        无，但会在输出目录生成HDF5文件
    """
    input_athdf_file = filename
    base_name = os.path.basename(input_athdf_file).replace('.athdf', '')
    print(f"\n>>> Processing: {base_name}")

    # 创建输出目录
    output_dir = os.path.join(config['output_directory'], 'ipole_inputs')
    os.makedirs(output_dir, exist_ok=True)
    output_h5 = os.path.join(output_dir, f"{base_name}_dsa_input.h5")

    # Step 1: 加载并切片数据
    roi_data = load_and_slice_data(input_athdf_file, config['roi_params'])
    if roi_data is None:
        return

    # Step 2: 计算激波和非热电子物理
    enable_advection = config.get('physics', {}).get('enable_advection', False)
    shock_props, nonthermal_props = calculate_dsa_physics(
        roi_data,
        config["shock_params"],
        config["nt_params"],
        enable_advection=enable_advection,
        config=config
    )

    # Step 3: 可选 - 执行平流-冷却扩散计算
    if config.get('physics', {}).get('enable_advection', False):
        print(">>> Enabling advection-diffusion calculation...")
        try:
            from src.core.advection_v0 import solve_steady_advection
            evolved_props = solve_steady_advection(roi_data, nonthermal_props, config)
            nonthermal_props = evolved_props
            print(">>> Advection-diffusion calculation completed.")
        except ImportError as e:
            print(f"Warning: Could not import advection module: {e}")
            print(">>> Continuing without advection-diffusion.")
        except Exception as e:
            print(f"Warning: Advection calculation failed: {e}")
            print(">>> Continuing with original non-thermal properties.")

    # Step 4: 保存HDF5文件
    save_h5_file(output_h5, roi_data, shock_props, nonthermal_props, config)
    print(f"--- Finished: {os.path.basename(output_h5)} ---")

# ==============================================================================
# 入口与配置
# ==============================================================================
if __name__ == '__main__':
    CPFS_PATH = "/cpfs01/projects-HDD/cfff-a7e284de52b3_HDD/cyh_22307110238"
    
    config = {
        "output_directory": os.path.join(CPFS_PATH, "workflowV2_native_run01/"),
        "data_directory": os.path.join(CPFS_PATH, "data_test3/"),
        
        "roi_params": {
            'r_min': 10, 'r_max': 1200,
            'theta_min': 0.0, 'theta_max': np.pi, # 完整全域
            'phi_min': 0.0, 'phi_max': 2*np.pi
        },
        
        "shock_params": {"gamma": 4.0/3.0, "mach_threshold_loose": 1.05, "min_physical_mach": 1.7},
        "nt_params": {"gamma": 4.0/3.0, "x_inj": 3.5, "xi_max": 0.05},
        
        # 【物理参数：关键补全】
        "physics": {
            "spin": 0.98,   # MAD98 
            "hslope": 1.0,  # 默认无压缩 theta 坐标
            "R0": 0.0       # 默认无径向平移
        },
        
        "max_concurrent_tasks": 8 # 内存平衡
    }

    file_list = sorted(glob.glob(os.path.join(config['data_directory'], 'mad98.prim.*.athdf')))
    task_func = partial(analyze_snapshot_full_pipeline, config=config)

    with mp.Pool(processes=config['max_concurrent_tasks'], maxtasksperchild=1) as pool:
        pool.map(task_func, file_list)