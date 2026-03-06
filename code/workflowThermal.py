'''
修复版 workflowThermal.py
功能：读取Athena++ (.athdf) 数据，执行 KS -> MKS 坐标及矢量基底转换，生成 ipole 可读的 HDF5 文件。

增加坐标读取的鲁棒性：如果 'x1v' 缺失，自动从 'x1f' 计算。
打印可用的 keys 以便调试。
'''
import numpy as np
import h5py
import os
import sys

# 路径配置
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
script_dir = os.path.dirname(os.path.abspath(__file__))
pyathena_path = os.path.join(script_dir, '..', 'pyathena')
sys.path.insert(0, pyathena_path)

from pyathena import athena_read

def get_coords(data, axis_idx):
    """
    安全获取坐标的辅助函数。
    axis_idx: 1 for r, 2 for theta, 3 for phi
    如果 x{i}v 不存在，尝试从 x{i}f 计算。
    """
    key_v = f'x{axis_idx}v'
    key_f = f'x{axis_idx}f'
    
    if key_v in data:
        return data[key_v]
    elif key_f in data:
        # print(f"Notice: '{key_v}' not found. Calculating from '{key_f}'.")
        face = data[key_f]
        return 0.5 * (face[:-1] + face[1:])
    else:
        raise KeyError(f"Critical Error: Neither '{key_v}' nor '{key_f}' found in data. Available keys: {list(data.keys())}")

def create_thermal_native_h5(athdf_filename, output_h5, spin=0.98, hslope=1.0, R0=0.0):
    print(f"--- Processing: {os.path.basename(athdf_filename)} ---")
    
    # 1. 读取数据
    try:
        data = athena_read.athdf(athdf_filename)
    except Exception as e:
        print(f"Error reading file with athena_read: {e}")
        return

    # Debug: 打印一次 keys 确认数据结构 (仅前10个)
    # keys_list = list(data.keys())
    # print(f"DEBUG: Data keys available: {keys_list[:10]} ...")

    # 物理参数
    gamma = 4.0/3.0 
    
    # 读取变量
    try:
        rho = data['rho']
        press = data['press']
        uu = press / (gamma - 1.0)
        
        # 矢量读取
        if 'Bcc1' in data:
            b1_ks, b2_ks, b3_ks = data['Bcc1'], data['Bcc2'], data['Bcc3']
        else:
            b1_ks, b2_ks, b3_ks = data['B1'], data['B2'], data['B3']
            
        v1_ks, v2_ks, v3_ks = data['vel1'], data['vel2'], data['vel3']
        
    except KeyError as e:
        print(f"Error: Missing variable in athdf file: {e}")
        print(f"Available keys: {list(data.keys())}")
        sys.exit(1)
    
    # --- 2. 获取并修正坐标 (修复 KeyError 核心) ---
    r_coords = get_coords(data, 1) # x1v (r)
    
    # 网格维度 (Athena: phi, theta, r)
    nk, nj, ni = rho.shape 

    # 准备坐标变换所需的 scale factor
    # r_coords 是 1D，扩展为 3D (nk, nj, ni)
    r_3d = r_coords[np.newaxis, np.newaxis, :] 
    
    # --- 坐标系变换 (KS -> MKS) ---
    # 1. 径向变换: MKS x1 = ln(r), 所以 u^1_mks = u^r_ks / r
    scale_1 = 1.0 / r_3d
    
    # 2. 角向变换: theta = pi * x2, 所以 u^2_mks = u^th_ks / pi
    scale_2 = 1.0 / np.pi
    
    # 3. 方位角: phi = x3, scale = 1.0
    scale_3 = 1.0

    # 应用变换
    b1_mks = b1_ks * scale_1
    b2_mks = b2_ks * scale_2
    b3_mks = b3_ks * scale_3
    
    u1_mks = v1_ks * scale_1
    u2_mks = v2_ks * scale_2
    u3_mks = v3_ks * scale_3

    # --- 3. 准备 Header 坐标 (基于 Face) ---
    # 获取 Face 坐标 (如果只有 Center，这里也可以反推，但 athdf 通常肯定有 Face)
    r_face = data.get('x1f')
    th_face = data.get('x2f')
    ph_face = data.get('x3f')
    
    if r_face is None: # 极端情况 fallback
        print("Warning: x1f missing, estimating from x1v")
        # 简单估算，仅适用均匀对数网格
        dlogr = np.log(r_coords[1]/r_coords[0])
        r_start = r_coords[0] * np.exp(-0.5*dlogr)
        r_end = r_coords[-1] * np.exp(0.5*dlogr)
        startx1 = np.log(r_start)
        dx1 = dlogr
    else:
        startx1 = np.log(r_face[0])
        dx1 = np.log(r_face[-1] / r_face[0]) / ni

    if th_face is None:
        startx2 = 0.0
        dx2 = 1.0 / nj # assuming full 0-pi mapped to 0-1
    else:
        startx2 = th_face[0] / np.pi
        dx2 = (th_face[-1] - th_face[0]) / (np.pi * nj)
        
    if ph_face is None:
        startx3 = 0.0
        dx3 = 2*np.pi / nk if nk>0 else 0
    else:
        startx3 = ph_face[0]
        dx3 = (ph_face[-1] - ph_face[0]) / nk if nk > 1 else 2*np.pi

    # --- 4. 写入 HDF5 ---
    with h5py.File(output_h5, 'w') as f:
        f.create_dataset('t', data=data.get('Time', 0.0))
        f.create_dataset('dump_cadence', data=1.0)
        
        hdr = f.create_group('header')
        hdr.create_dataset('n1', data=ni, dtype='i4')
        hdr.create_dataset('n2', data=nj, dtype='i4')
        hdr.create_dataset('n3', data=nk, dtype='i4')
        hdr.create_dataset('n_prim', data=8, dtype='i4')
        hdr.create_dataset('gam', data=gamma)
        hdr.create_dataset('has_electrons', data=0, dtype='i4')
        hdr.create_dataset('metric', data=np.string_("MKS")) 
        
        geom = hdr.create_group('geom')
        geom.create_dataset('startx1', data=startx1)
        geom.create_dataset('startx2', data=startx2)
        geom.create_dataset('startx3', data=startx3)
        geom.create_dataset('dx1', data=dx1)
        geom.create_dataset('dx2', data=dx2)
        geom.create_dataset('dx3', data=dx3)
        
        mks = geom.create_group('mks')
        mks.create_dataset('a', data=spin)
        mks.create_dataset('r_in', data=np.exp(startx1))
        mks.create_dataset('r_out', data=np.exp(startx1 + ni*dx1))
        mks.create_dataset('hslope', data=hslope)
        mks.create_dataset('R0', data=R0)
        mks.create_dataset('r_eh', data=1.0 + np.sqrt(1.0 - spin**2))
        
        # 转置顺序：(phi, theta, r) -> (r, theta, phi) for ipole? 
        # Wait, ipole C-order is [i][j][k] (x1, x2, x3).
        # HDF5 in python is row-major.
        # Athena array is (k, j, i) -> (phi, theta, r).
        # We need (i, j, k, prim).
        prims = np.stack([rho, uu, u1_mks, u2_mks, u3_mks, b1_mks, b2_mks, b3_mks], axis=-1)
        
        # Transpose:
        # Source: (k, j, i, p) -> (0, 1, 2, 3)
        # Target: (i, j, k, p) -> (2, 1, 0, 3)
        prims_final = prims.transpose(2, 1, 0, 3).astype('f4')
        
        f.create_dataset('prims', data=prims_final)

    print(f"--- Successfully created MKS-transformed H5: {output_h5} ---")

if __name__ == "__main__":
    # 配置你的路径
    CPFS_ROOT_PATH = "/cpfs01/projects-HDD/cfff-a7e284de52b3_HDD/cyh_22307110238"
    DATA_PATH = os.path.join(CPFS_ROOT_PATH, "data_test3/") 
    
    input_file = os.path.join(DATA_PATH, "mad98.prim.00426.athdf")
    
    if os.path.exists(input_file):
        output_dir = os.path.join(CPFS_ROOT_PATH, "native_output_run02/")
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.basename(input_file).replace('.athdf', '')
        output_h5 = os.path.join(output_dir, f"{base_name}_thermal_corrected.h5")
        
        create_thermal_native_h5(input_file, output_h5)
    else:
        print(f"Warning: Input file not found at {input_file}")