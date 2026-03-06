'''
与`workflowThermal`平行的DSA工作流，输入原始Athena++模拟数据，计算并注入DSA非热电子物理后生成ipole输入文件。

26.2.25 update
Workflow Full V2 (Fixed Coordinate Basis)
功能：
1. 读取 Athena++ 数据 (KS 坐标)
2. 在 KS 坐标下进行激波探测和非热电子计算 (保持物理正确性)
3. 在写入 HDF5 时执行 KS -> MKS 矢量基底转换 (u^r/r 等)，适配 ipole 光线追踪
'''
#! /usr/bin/env python3
import os
import sys
import glob
import time
import gc
import multiprocessing as mp
from functools import partial
import numpy as np
import h5py

# ==============================================================================
# 路径配置
# ==============================================================================
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

script_dir = os.path.dirname(os.path.abspath(__file__))
pyathena_path = os.path.join(script_dir, '..', 'pyathena')
sys.path.insert(0, pyathena_path)

try:
    from pyathena import athena_read
    from shock_v1 import find_shocks_in_roi_mhd
    from nt_electron_v1 import calculate_nonthermal_electrons
    from advection_v0 import solve_steady_advection
except ImportError as e:
    print(f"Fatal Error: 无法加载科学计算模块. {e}")
    sys.exit(1)

# ==============================================================================
# 辅助函数：鲁棒的坐标读取
# ==============================================================================
def get_coords(data, axis_idx):
    """
    安全获取坐标：如果 x{i}v 不存在，尝试从 x{i}f 计算。
    """
    key_v = f'x{axis_idx}v'
    key_f = f'x{axis_idx}f'
    
    if key_v in data:
        return data[key_v]
    elif key_f in data:
        face = data[key_f]
        return 0.5 * (face[:-1] + face[1:])
    else:
        raise KeyError(f"Critical Error: Neither '{key_v}' nor '{key_f}' found.")

# ==============================================================================
# 核心 HDF5 写入函数 (集成 KS->MKS 变换)
# ==============================================================================
def save_native_dsa_h5(output_h5, roi_data, shock_props, nonthermal_props, config):
    """
    生成 ipole 输入文件，并执行关键的坐标基底变换。
    """
    gamma = config['shock_params']['gamma']
    spin = config['physics']['spin']
    hslope = config['physics']['hslope']
    R0 = config['physics']['R0']
    
    # 提取基本数据
    rho = roi_data['rho']
    # Athena shape: (nz, ny, nx) -> (phi, theta, r)
    nk, nj, ni = rho.shape 

    # --- 1. 准备变换因子 (Transformation Factors) ---
    # 获取径向坐标 r (用于 dx1/dr = 1/r 变换)
    # 注意：roi_data 必须包含 x1v。我们在 analyze 函数中确保了这一点。
    r_vals = roi_data['x1v'] 
    
    # 广播 r 到 3D 形状 (nk, nj, ni) 以便与数据数组相乘
    # Athena data order is (k, j, i)
    r_3d = r_vals[np.newaxis, np.newaxis, :] 

    # 计算缩放因子 (Jacobian)
    # Radial: MKS x1 = ln(r) => u^1_mks = u^r_ks / r
    scale_1 = 1.0 / r_3d
    
    # Theta: MKS x2 = theta / pi => u^2_mks = u^th_ks / pi (假设 hslope=1.0)
    scale_2 = 1.0 / np.pi
    
    # Phi: MKS x3 = phi => scale = 1.0
    scale_3 = 1.0

    with h5py.File(output_h5, 'w') as f:
        # --- 2. 基础元数据 ---
        f.create_dataset('t', data=roi_data.get('Time', 0.0))
        f.create_dataset('dump_cadence', data=1.0)
        
        # --- 3. Header ---
        hdr = f.create_group('header')
        hdr.create_dataset('n1', data=ni, dtype='i4')
        hdr.create_dataset('n2', data=nj, dtype='i4')
        hdr.create_dataset('n3', data=nk, dtype='i4')
        hdr.create_dataset('n_prim', data=8, dtype='i4')
        hdr.create_dataset('gam', data=gamma)
        hdr.create_dataset('has_electrons', data=0, dtype='i4')
        # 显式指定 MKS Metric，ipole 才会读取 geom/mks 下的参数
        hdr.create_dataset('metric', data=np.string_("MKS"))
        
        # --- 4. Geometry (MKS Grid Definition) ---
        geom = hdr.create_group('geom')
        # 使用 Face 坐标定义网格边界
        r_f = roi_data['x1f']
        th_f = roi_data['x2f']
        ph_f = roi_data['x3f']
        
        # MKS 坐标定义:
        # startx1 = ln(r_min)
        geom.create_dataset('startx1', data=np.log(r_f[0]))
        # startx2 = theta_min / pi
        geom.create_dataset('startx2', data=th_f[0] / np.pi) 
        geom.create_dataset('startx3', data=ph_f[0])
        
        # dx 计算
        geom.create_dataset('dx1', data=np.log(r_f[-1] / r_f[0]) / ni)
        geom.create_dataset('dx2', data=(th_f[-1] - th_f[0]) / (np.pi * nj))
        geom.create_dataset('dx3', data=(ph_f[-1] - ph_f[0]) / nk if nk > 1 else 2 * np.pi)
        
        # MKS 参数
        mks = geom.create_group('mks')
        mks.create_dataset('a', data=spin)
        mks.create_dataset('hslope', data=hslope)
        mks.create_dataset('R0', data=R0)
        mks.create_dataset('r_in', data=r_f[0])
        mks.create_dataset('r_out', data=r_f[-1])
        mks.create_dataset('r_eh', data=1.0 + np.sqrt(1.0 - spin**2))

        # --- 5. 原始数据写入 (应用变换) ---
        uu = roi_data['press'] / (gamma - 1.0)
        
        # 读取 KS 基底下的原始矢量
        b1_ks, b2_ks, b3_ks = roi_data['Bcc1'], roi_data['Bcc2'], roi_data['Bcc3']
        v1_ks, v2_ks, v3_ks = roi_data['vel1'], roi_data['vel2'], roi_data['vel3']
        
        # 应用变换到 MKS 基底
        b1_mks = b1_ks * scale_1
        b2_mks = b2_ks * scale_2
        b3_mks = b3_ks * scale_3
        
        v1_mks = v1_ks * scale_1
        v2_mks = v2_ks * scale_2
        v3_mks = v3_ks * scale_3

        # 堆叠数据: rho, uu, u1, u2, u3, B1, B2, B3
        prims = np.stack([rho, uu, v1_mks, v2_mks, v3_mks, b1_mks, b2_mks, b3_mks], axis=-1)
        
        # 转置为 ipole 顺序: (i, j, k, p) -> (x1, x2, x3, prim)
        # 源数据: (k, j, i, p) -> (0, 1, 2, 3)
        # 目标:   (i, j, k, p) -> (2, 1, 0, 3)
        f.create_dataset('prims', data=prims.transpose(2, 1, 0, 3).astype('f4'))

        # --- 6. DSA 物理数据 (电子分布) ---
        # 这些标量场不需要矢量基底变换，但需要转置维度
        mask = shock_props['mask']
        # 确保数据存在
        if 'C_grid' in nonthermal_props:
             c_grid = nonthermal_props['C_grid']
        else:
             c_grid = np.zeros_like(rho)

        if 'q_grid' in nonthermal_props:
             q_grid = nonthermal_props['q_grid']
        else:
             q_grid = np.zeros_like(rho) + 3.0 # default p

        p_grid = np.where(mask, q_grid - 1.0, 0.0)
        
        # 写入数据集 (转置维度 k,j,i -> i,j,k)
        f.create_dataset('KEL', data=mask.transpose(2, 1, 0).astype('f8')) # KEL 通常用作开关
        f.create_dataset('UNTH', data=c_grid.transpose(2, 1, 0).astype('f8')) # 非热电子数密度/归一化常数
        f.create_dataset('p', data=p_grid.transpose(2, 1, 0).astype('f8')) # 谱指数

# ==============================================================================
# 工作流主函数
# ==============================================================================
def analyze_snapshot_full_pipeline(filename, config):
    input_athdf_file = filename
    base_name = os.path.basename(input_athdf_file).replace('.athdf', '')
    print(f"\n>>> Processing: {base_name}")
    
    dir_ipole_inputs = os.path.join(config['output_directory'], 'ipole_inputs')
    os.makedirs(dir_ipole_inputs, exist_ok=True)
    output_h5 = os.path.join(dir_ipole_inputs, f"{base_name}_dsa_input.h5")

    try:
        # --- Step 1: 加载数据 ---
        # 读取 Level 4 以获取最高精度
        full_data = athena_read.athdf(input_athdf_file, level=4)
        
        # 确保坐标存在 (修复 KeyError)
        full_data['x1v'] = get_coords(full_data, 1)
        full_data['x2v'] = get_coords(full_data, 2)
        full_data['x3v'] = get_coords(full_data, 3)

        # --- Step 2: ROI 切片 ---
        # 我们使用 Face 坐标来确定切片索引，因为它们定义了网格边界
        r_coords = full_data['x1f']
        th_coords = full_data['x2f']
        ph_coords = full_data['x3f']
        
        roi_cfg = config['roi_params']
        
        slices = []
        for coord, c_min, c_max in zip([r_coords, th_coords, ph_coords], 
                                       [roi_cfg['r_min'], roi_cfg['theta_min'], roi_cfg['phi_min']],
                                       [roi_cfg['r_max'], roi_cfg['theta_max'], roi_cfg['phi_max']]):
            # searchsorted 查找边界
            slices.append(slice(np.searchsorted(coord, c_min, 'left'), np.searchsorted(coord, c_max, 'right')))
        
        i_s, j_s, k_s = slices # indices for (r, th, ph) - Athena stores as (ph, th, r) -> (k, j, i)
        
        roi_data = {'Time': full_data.get('Time', 0.0)}
        
        # 物理量切片 (注意 Athena 维度顺序 k, j, i)
        target_keys = ['rho', 'press', 'vel1', 'vel2', 'vel3', 'Bcc1', 'Bcc2', 'Bcc3']
        # 处理 Bcc 可能不存在的情况（回退到 B）
        if 'Bcc1' not in full_data:
             target_keys = ['rho', 'press', 'vel1', 'vel2', 'vel3', 'B1', 'B2', 'B3']

        for key in target_keys:
            roi_data[key] = full_data[key][k_s, j_s, i_s]
            # 如果 key 是 B1/B2/B3，重命名为 Bcc 以便统一处理
            if key.startswith('B') and not key.startswith('Bcc'):
                roi_data[f'Bcc{key[-1]}'] = roi_data[key]
        
        # 坐标切片
        # Face 坐标比 Cell 坐标多 1
        roi_data['x1f'] = r_coords[i_s.start : i_s.stop+1]
        roi_data['x2f'] = th_coords[j_s.start : j_s.stop+1]
        roi_data['x3f'] = ph_coords[k_s.start : k_s.stop+1]
        
        # **关键修复**: 切片并保留所有方向的胞心坐标 (x1v, x2v, x3v)
        # 之前的代码可能只保留了 x1v
        roi_data['x1v'] = full_data['x1v'][i_s]
        roi_data['x2v'] = full_data['x2v'][j_s]  # <--- [FIXED] 新增这一行
        roi_data['x3v'] = full_data['x3v'][k_s]

        # 清理内存
        del full_data; gc.collect()

        # --- Step 3 & 4: 物理计算 (在 KS 坐标下进行) ---
        # 激波探测和非热电子计算仍然使用原始的 KS 数据，这是物理上正确的
        print(f"[{base_name}] detecting shocks...")
        shock_props = find_shocks_in_roi_mhd(roi_data, **config["shock_params"])
        
        if np.any(shock_props["mask"]):
            print(f"[{base_name}] calculating NT electrons...")
            nonthermal_props = calculate_nonthermal_electrons(shock_props, **config["nt_params"])
            
            # === 新增：求解平流冷却方程 ===
            # 从 config 中读取 cooling_factor，如果没设则给个默认值
            # 建议将 cooling_factor 放入 config['physics'] 中
            print(f"[{base_name}] evolving electron distribution (Advection+Cooling)...")
            nonthermal_props = solve_steady_advection(roi_data, nonthermal_props, config)
        else:
            print(f"[{base_name}] no shocks found.")
            nonthermal_props = {
                'q_grid': np.zeros_like(roi_data['rho']) + 3.0, 
                'C_grid': np.zeros_like(roi_data['rho'])
            }

        # --- Step 5: 保存 HDF5 (执行 KS->MKS 变换) ---
        print(f"[{base_name}] saving HDF5...")
        save_native_dsa_h5(output_h5, roi_data, shock_props, nonthermal_props, config)
        print(f"--- Finished: {os.path.basename(output_h5)} ---")

    except Exception as e:
        print(f"Error processing {base_name}: {e}")
        import traceback
        traceback.print_exc()

# ==============================================================================
# 入口与配置
# ==============================================================================
if __name__ == '__main__':
    # 请根据您的实际环境修改路径
    CPFS_PATH = "/cpfs01/projects-HDD/cfff-a7e284de52b3_HDD/cyh_22307110238"
    
    config = {
        "output_directory": os.path.join(CPFS_PATH, "workflowV2_advection_run01/"), # 修改输出目录以免覆盖
        "data_directory": os.path.join(CPFS_PATH, "data_test3/"),
        
        "roi_params": {
            'r_min': 10, 'r_max': 1200,
            'theta_min': 0.0, 'theta_max': np.pi, 
            'phi_min': 0.0, 'phi_max': 2*np.pi
        },
        
        "shock_params": {
            "gamma": 4.0/3.0, 
            "mach_threshold_loose": 1.05, 
            "min_physical_mach": 1.7
        },
        "nt_params": {
            "gamma": 4.0/3.0, 
            "x_inj": 3.5, 
            "xi_max": 0.05
        },
        
        # 物理参数
        "physics": {
            "spin": 0.98,   
            "hslope": 1.0,  # 线性 Theta 映射
            "R0": 0.0,
            "cooling_factor": 50.0, # 关键参数：控制喷流长度。越大越长。
            "advection_steps": 2000 # 迭代步数
        },
        
        "max_concurrent_tasks": 5 
    }

    file_list = sorted(glob.glob(os.path.join(config['data_directory'], 'mad98.prim.*.athdf')))
    
    # 限制测试文件数量 (可选)
    # file_list = file_list[:5] 

    print(f"Found {len(file_list)} files. Starting pool with {config['max_concurrent_tasks']} workers.")
    
    task_func = partial(analyze_snapshot_full_pipeline, config=config)

    with mp.Pool(processes=config['max_concurrent_tasks'], maxtasksperchild=1) as pool:
        pool.map(task_func, file_list)