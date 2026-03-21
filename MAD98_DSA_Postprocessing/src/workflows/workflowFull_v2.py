"""
DSA工作流主模块 - 负责从原始Athena++模拟数据生成ipole输入文件

主要功能：
1. 加载原始.athdf格式的GRMHD模拟数据
2. 执行ROI（感兴趣区域）切片，减少计算量
3. 检测激波并计算激波物理性质
4. 根据激波性质计算非热电子能谱参数（可选：含平流-冷却扩散）
5. 生成注入DSA物理的ipole输入HDF5文件

使用场景：批处理大量模拟快照，生成辐射转移计算的输入文件
"""

import os
import sys
import time
import numpy as np
from src.workflows.base_workflow import load_and_slice_data, calculate_dsa_physics, save_h5_file
from src.utils.logging_config import log_start, log_finish, log_error

def process_snapshot(filename, config, logger=None):
    """
    处理单个快照文件的完整流程

    Args:
        filename: 输入的.athdf文件路径
        config: 配置参数字典，包含ROI参数、物理参数等
        logger: 日志对象（可选）

    Returns:
        bool: 处理成功返回True，失败返回False
    """
    input_athdf_file = filename
    base_name = os.path.basename(input_athdf_file).replace('.athdf', '')

    # 记录开始时间
    start_time = time.time()

    try:
        # 记录任务开始
        if logger:
            log_start(logger, base_name, config)
        else:
            print(f"\n>>> Processing: {base_name}")

        # 创建输出目录
        output_dir = os.path.join(config['output_directory'], 'ipole_inputs')
        os.makedirs(output_dir, exist_ok=True)
        output_h5 = os.path.join(output_dir, f"{base_name}_dsa_input.h5")

        # Step 1: 加载并切片数据
        roi_data = load_and_slice_data(input_athdf_file, config['roi_params'])
        if roi_data is None:
            if logger:
                logger.error(f"Failed to load and slice data for {base_name}")
            return False

        # Step 2: 计算激波和非热电子物理
        # 从 physics 配置计算 RHO_unit，用于 nt_electron_v1.py 的单位转换
        physics_cfg = config.get('physics', {})
        G_CGS    = 6.674e-8
        M_SUN    = 1.989e33
        C_LIGHT_CGS = 2.998e10
        M_unit_val  = physics_cfg.get('M_unit', 1e25)
        MBH_solar   = physics_cfg.get('MBH_solar', 6.2e9)
        L_unit_val  = G_CGS * (MBH_solar * M_SUN) / C_LIGHT_CGS**2
        rho_unit    = M_unit_val / L_unit_val**3

        # 将 rho_unit 注入 nt_params（不修改原始 config）
        nt_params = dict(config['nt_params'])
        nt_params['rho_unit'] = rho_unit

        enable_advection = physics_cfg.get('enable_advection', False)
        shock_props, nonthermal_props = calculate_dsa_physics(
            roi_data,
            config["shock_params"],
            nt_params,
        )

        # Step 3: 可选 - 执行平流-冷却扩散计算
        if enable_advection:
            if logger:
                logger.info("Enabling advection-diffusion calculation...")
            else:
                print(">>> Enabling advection-diffusion calculation...")

            try:
                from src.core.advection_v0 import solve_steady_advection
                evolved_props = solve_steady_advection(roi_data, nonthermal_props, config)
                nonthermal_props = evolved_props
                if logger:
                    logger.info("Advection-diffusion calculation completed.")
                else:
                    print(">>> Advection-diffusion calculation completed.")
            except ImportError as e:
                warning_msg = f"Could not import advection module: {e}. Continuing without advection-diffusion."
                if logger:
                    logger.warning(warning_msg)
                else:
                    print(f"Warning: {warning_msg}")
            except Exception as e:
                warning_msg = f"Advection calculation failed: {e}. Continuing with original non-thermal properties."
                if logger:
                    logger.warning(warning_msg)
                else:
                    print(f"Warning: {warning_msg}")

        # Step 4: 保存HDF5文件
        save_h5_file(output_h5, roi_data, shock_props, nonthermal_props, config)

        # 计算处理时间
        processing_time = time.time() - start_time

        # 记录完成
        if logger:
            log_finish(logger, base_name, output_h5, processing_time)
        else:
            print(f"--- Finished: {os.path.basename(output_h5)} (Time: {processing_time:.2f}s) ---")

        return True

    except Exception as e:
        # 计算处理时间
        processing_time = time.time() - start_time

        # 记录错误
        if logger:
            log_error(logger, base_name, str(e), exc_info=True)
        else:
            print(f"--- Error processing {base_name}: {e} (Time: {processing_time:.2f}s) ---")

        return False

# 本模块作为库使用，请通过 run_dsa_pipeline.py 启动完整流程：
#   python run_dsa_pipeline.py generate_h5 --data-dir /path/to/data --output-dir /path/to/output