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
# 假设您的科学代码在当前目录或指定目录
# sys.path.insert(0, os.path.join(project_root, 'code'))

try:
    from pyathena import athena_read
    from shock_v1 import find_shocks_in_roi_mhd
    from nt_electron_v1 import calculate_nonthermal_electrons
except ImportError as e:
    print(f"Fatal Error: 无法加载科学计算模块. {e}")
    sys.exit(1)

# ==============================================================================
# 核心 HDF5 写入函数 (Native 模式兼容)
# ==============================================================================
def save_native_dsa_h5(output_h5, roi_data, shock_props, nonthermal_props, config):
    """
    依照 Native 模式成功经验，生成注入了 DSA 物理的 IPOLE 输入文件。
    """
    gamma = config['shock_params']['gamma']
    spin = config['physics']['spin']
    hslope = config['physics']['hslope']
    R0 = config['physics']['R0']
    
    # 提取维度信息
    rho = roi_data['rho']
    nk, nj, ni = rho.shape # athena_read 顺序: (nz, ny, nx)

    with h5py.File(output_h5, 'w') as f:
        # 1. 基础元数据
        f.create_dataset('t', data=roi_data.get('Time', 0.0))
        f.create_dataset('dump_cadence', data=1.0)
        
        # 2. Header 组
        hdr = f.create_group('header')
        hdr.create_dataset('n1', data=ni, dtype='i4')
        hdr.create_dataset('n2', data=nj, dtype='i4')
        hdr.create_dataset('n3', data=nk, dtype='i4')
        hdr.create_dataset('n_prim', data=8, dtype='i4')
        hdr.create_dataset('gam', data=gamma)
        hdr.create_dataset('has_electrons', data=0, dtype='i4')
        hdr.create_dataset('metric', data=np.string_("MKS"))
        
        # 3. 几何组 (关键：log 坐标映射)
        geom = hdr.create_group('geom')
        r_f, th_f, ph_f = roi_data['x1f'], roi_data['x2f'], roi_data['x3f']
        
        geom.create_dataset('startx1', data=np.log(r_f[0]))
        geom.create_dataset('startx2', data=th_f[0])
        geom.create_dataset('startx3', data=ph_f[0])
        
        geom.create_dataset('dx1', data=np.log(r_f[1] / r_f[0]))
        geom.create_dataset('dx2', data=th_f[1] - th_f[0])
        geom.create_dataset('dx3', data=ph_f[1] - ph_f[0] if nk > 1 else 2 * np.pi)
        
        # 4. MKS 组
        mks = geom.create_group('mks')
        mks.create_dataset('a', data=spin)
        mks.create_dataset('hslope', data=hslope)
        mks.create_dataset('R0', data=R0)
        mks.create_dataset('r_in', data=r_f[0])
        mks.create_dataset('r_out', data=r_f[-1])
        mks.create_dataset('r_eh', data=1.0 + np.sqrt(1.0 - spin**2))

        # 5. 主流体数据 (顺序调整为 i, j, k)
        # 整合为: rho, uu, v1, v2, v3, b1, b2, b3
        uu = roi_data['press'] / (gamma - 1.0)
        b1, b2, b3 = roi_data['Bcc1'], roi_data['Bcc2'], roi_data['Bcc3']
        v1, v2, v3 = roi_data['vel1'], roi_data['vel2'], roi_data['vel3']
        
        prims = np.stack([rho, uu, v1, v2, v3, b1, b2, b3], axis=-1)
        f.create_dataset('prims', data=prims.transpose(2, 1, 0, 3).astype('f4'))

        # 6. DSA 物理注入 (核心：UNTH 存储 N_inj)
        # 注意：此处假设 calculate_nonthermal_electrons 已修正为返回 N_inj 并存放在 'C_grid' 中
        mask = shock_props['mask']
        q_grid = nonthermal_props['q_grid']
        p_grid = np.where(mask, q_grid - 1.0, 0.0)
        
        # 转置为 (nx, ny, nz) 顺序
        f.create_dataset('KEL', data=mask.transpose(2, 1, 0).astype('f8'))
        f.create_dataset('UNTH', data=nonthermal_props['C_grid'].transpose(2, 1, 0).astype('f8'))
        f.create_dataset('p', data=p_grid.transpose(2, 1, 0).astype('f8'))

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

    # --- Step 1: 加载数据 (Level 4 以保证精度) ---
    full_data = athena_read.athdf(input_athdf_file, level=4)

    # --- Step 2: ROI 切片 ---
    r_coords, th_coords, ph_coords = full_data['x1f'], full_data['x2f'], full_data['x3f']
    roi_cfg = config['roi_params']
    
    slices = []
    for coord, c_min, c_max in zip([r_coords, th_coords, ph_coords], 
                                   [roi_cfg['r_min'], roi_cfg['theta_min'], roi_cfg['phi_min']],
                                   [roi_cfg['r_max'], roi_cfg['theta_max'], roi_cfg['phi_max']]):
        slices.append(slice(np.searchsorted(coord, c_min, 'left'), np.searchsorted(coord, c_max, 'right')))
    
    i_s, j_s, k_s = slices
    roi_data = {'Time': full_data.get('Time', 0.0)}
    
    # 物理量切片
    target_keys = ['rho', 'press', 'vel1', 'vel2', 'vel3', 'Bcc1', 'Bcc2', 'Bcc3']
    for key in target_keys:
        roi_data[key] = full_data[key][k_s, j_s, i_s]
    
    # 坐标边界切片 (x1f 等比胞中心多一个元素)
    roi_data['x1f'] = r_coords[i_s.start : i_s.stop+1]
    roi_data['x2f'] = th_coords[j_s.start : j_s.stop+1]
    roi_data['x3f'] = ph_coords[k_s.start : k_s.stop+1]

    del full_data; gc.collect()

    # --- Step 3 & 4: 物理计算 (MHD 激波 + 非热电子) ---
    shock_props = find_shocks_in_roi_mhd(roi_data, **config["shock_params"])
    
    if np.any(shock_props["mask"]):
        nonthermal_props = calculate_nonthermal_electrons(shock_props, **config["nt_params"])
    else:
        nonthermal_props = {'q_grid': np.zeros_like(roi_data['rho']), 'C_grid': np.zeros_like(roi_data['rho'])}

    # --- Step 5: 保存 HDF5 ---
    save_native_dsa_h5(output_h5, roi_data, shock_props, nonthermal_props, config)
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