#!/usr/bin/env python3
"""
基础工作流模块，包含通用的数据处理功能
"""
import os
import sys
import glob
import gc
import multiprocessing as mp
from functools import partial
import numpy as np
import h5py

# 路径配置
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

try:
    from pyathena import athena_read
    from src.core.shock_v1 import find_shocks_in_roi_mhd
    from src.core.nt_electron_v1 import calculate_nonthermal_electrons
    # 导入平流扩散模块
    from src.core.advection_v0 import solve_steady_advection
except ImportError as e:
    print(f"Fatal Error: 无法加载科学计算模块. {e}")
    sys.exit(1)

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


def load_and_slice_data(filename, roi_params):
    """
    加载Athena++数据并进行ROI切片，修复x1v等坐标缺失问题
    """
    print(f"Loading data from {os.path.basename(filename)}...")
    
    # --- Step 1: 加载完整数据 ---
    full_data = athena_read.athdf(filename, level=4)

    # 确保胞心坐标 (Cell-centered) 存在 (修复 KeyError)
    # 使用 get_coords 函数手动生成/提取坐标
    full_data['x1v'] = get_coords(full_data, 1)
    full_data['x2v'] = get_coords(full_data, 2)
    full_data['x3v'] = get_coords(full_data, 3)

    # --- Step 2: 确定切片索引 ---
    # 使用 Face 坐标确定边界索引
    r_coords, th_coords, ph_coords = full_data['x1f'], full_data['x2f'], full_data['x3f']

    i_start = np.searchsorted(r_coords, roi_params['r_min'], side='left')
    i_end   = np.searchsorted(r_coords, roi_params['r_max'], side='right')
    j_start = np.searchsorted(th_coords, roi_params['theta_min'], side='left')
    j_end   = np.searchsorted(th_coords, roi_params['theta_max'], side='right')
    k_start = np.searchsorted(ph_coords, roi_params['phi_min'], side='left')
    k_end   = np.searchsorted(ph_coords, roi_params['phi_max'], side='right')

    if i_start >= i_end or j_start >= j_end or k_start >= k_end:
        print(f"Error: Invalid ROI slice for {filename}")
        return None

    # --- Step 3: 提取 ROI 数据 ---
    roi_data = {'Time': full_data.get('Time', 0.0)}

    # 处理物理量 (注意 Athena 的维度顺序通常是 k, j, i)
    target_keys = ['rho', 'press', 'vel1', 'vel2', 'vel3', 'Bcc1', 'Bcc2', 'Bcc3']
    # 兼容性处理：如果 Bcc 不存在，尝试读取 B1, B2, B3
    if 'Bcc1' not in full_data:
        target_keys = ['rho', 'press', 'vel1', 'vel2', 'vel3', 'B1', 'B2', 'B3']

    for key in target_keys:
        # 进行 3D 切片
        roi_data[key] = full_data[key][k_start:k_end, j_start:j_end, i_start:i_end]
        # 统一磁场键名为 Bcc
        if key.startswith('B') and not key.startswith('Bcc'):
            roi_data[f'Bcc{key[-1]}'] = roi_data[key]

    # --- Step 4: 坐标切片 (关键修复点) ---
    # Face 坐标 (长度为 N+1)
    roi_data['x1f'] = r_coords[i_start : i_end+1]
    roi_data['x2f'] = th_coords[j_start : j_end+1]
    roi_data['x3f'] = ph_coords[k_start : k_end+1]

    # Cell 坐标 (长度为 N) - 之前漏掉的赋值
    roi_data['x1v'] = full_data['x1v'][i_start : i_end]
    roi_data['x2v'] = full_data['x2v'][j_start : j_end]
    roi_data['x3v'] = full_data['x3v'][k_start : k_end]

    # --- Step 5: 清理并返回 ---
    del full_data
    gc.collect()

    return roi_data


def calculate_dsa_physics(roi_data, shock_params, nt_params, enable_advection=False, config=None):
    """
    计算激波物理和非热电子

    Args:
        roi_data: ROI切片数据
        shock_params: 激波探测参数
        nt_params: 非热电子参数
        enable_advection: 是否启用平流-冷却扩散
        config: 完整配置（用于平流计算）

    Returns:
        shock_props: 激波属性
        nonthermal_props: 非热电子属性（可能经过平流演化）
    """
    print("Calculating shock properties...")
    shock_props = find_shocks_in_roi_mhd(roi_data, **shock_params)

    if np.any(shock_props["mask"]):
        print("Calculating non-thermal electrons...")
        nonthermal_props = calculate_nonthermal_electrons(shock_props, **nt_params)

        # 可选：执行平流-冷却扩散计算
        if enable_advection and config is not None:
            print(">>> Applying advection-cooling diffusion model...")
            try:
                evolved_props = solve_steady_advection(roi_data, nonthermal_props, config)
                nonthermal_props = evolved_props
                print(">>> Advection-cooling diffusion completed.")
            except Exception as e:
                print(f"Warning: Advection calculation failed: {e}")
                print(">>> Using original non-thermal properties.")
    else:
        nonthermal_props = {
            'q_grid': np.zeros_like(roi_data['rho']),
            'C_grid': np.zeros_like(roi_data['rho'])
        }
        print("No shocks found, non-thermal properties initialized to zero.")

    return shock_props, nonthermal_props


def save_h5_file(output_h5, roi_data, shock_props, nonthermal_props, config):
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



def run_parallel_workflow(file_list, config, process_func):
    """
    运行并行工作流
    """
    SAFE_MAX_WORKERS = config.get('max_concurrent_tasks', 8)

    print(f"Found {len(file_list)} files to process.")
    print(f"Using {SAFE_MAX_WORKERS} worker processes.")

    task_func = partial(process_func, config=config)

    with mp.Pool(processes=SAFE_MAX_WORKERS, maxtasksperchild=1) as pool:
        print(f"Submitting {len(file_list)} tasks...")
        results = pool.map(task_func, file_list)
        print(f"All {len(file_list)} tasks have been processed.")

    return results