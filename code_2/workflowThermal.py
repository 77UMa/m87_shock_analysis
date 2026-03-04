'''
用于验证ipole输入文件格式的HDF5生成脚本，输入原始的.athdf模拟数据，得到ipole输入文件，后续使用`compare_models_v0`来得到辐射图像
'''
import numpy as np
import h5py
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

script_dir = os.path.dirname(os.path.abspath(__file__))
pyathena_path = os.path.join(script_dir, '..', 'pyathena')
sys.path.insert(0, pyathena_path)
# 确保能找到 pyathena
# sys.path.insert(0, '/path/to/your/pyathena')
from pyathena import athena_read

def create_thermal_native_h5(athdf_filename, output_h5, spin=0.98, hslope=1.0, R0=0.0):
    """
    专门用于验证 Model A (Thermal) 的 HDF5 生成脚本。
    移除了所有非热物理量，聚焦于坐标映射和基础流体数据的准确性。
    """
    print(f"--- Processing: {os.path.basename(athdf_filename)} ---")
    
    # 1. 使用 athena_read 加载全量数据 (原生 level)
    # 建议使用 level=0 或特定 level 保持网格规整
    data = athena_read.athdf(athdf_filename)
    
    gamma = 4.0/3.0 # MAD98 默认绝热指数 [cite: 1299]
    rho = data['rho']
    press = data['press']
    uu = press / (gamma - 1.0) # 内能计算
    
    # 优先使用胞中心磁场 Bcc
    if 'Bcc1' in data:
        b1, b2, b3 = data['Bcc1'], data['Bcc2'], data['Bcc3']
    else:
        b1, b2, b3 = data['B1'], data['B2'], data['B3']
        
    v1, v2, v3 = data['vel1'], data['vel2'], data['vel3']
    
    # 获取网格维度 (athena_read 默认返回 nz, ny, nx)
    nk, nj, ni = rho.shape
    
    # 2. 坐标参数提取
    # 注意：ipole 的 dx 需要对应 log(r)
    r_coords = data['x1f']
    th_coords = data['x2f']
    ph_coords = data['x3f']
    
    startx1 = np.log(r_coords[0])
    startx2 = th_coords[0]
    startx3 = ph_coords[0]
    
    dx1 = np.log(r_coords[1] / r_coords[0])
    dx2 = th_coords[1] - th_coords[0]
    dx3 = ph_coords[1] - ph_coords[0] if nk > 1 else 2 * np.pi

    # 3. 构建 HDF5
    with h5py.File(output_h5, 'w') as f:
        # 顶层元数据
        f.create_dataset('t', data=data.get('Time', 0.0))
        f.create_dataset('dump_cadence', data=1.0)
        
        # Header 组
        hdr = f.create_group('header')
        hdr.create_dataset('n1', data=ni, dtype='i4')
        hdr.create_dataset('n2', data=nj, dtype='i4')
        hdr.create_dataset('n3', data=nk, dtype='i4')
        hdr.create_dataset('n_prim', data=8, dtype='i4')
        hdr.create_dataset('gam', data=gamma)
        hdr.create_dataset('has_electrons', data=0, dtype='i4')
        hdr.create_dataset('metric', data=np.string_("MKS")) # 显式字节流
        
        # 几何组
        geom = hdr.create_group('geom')
        geom.create_dataset('startx1', data=startx1)
        geom.create_dataset('startx2', data=startx2)
        geom.create_dataset('startx3', data=startx3)
        geom.create_dataset('dx1', data=dx1)
        geom.create_dataset('dx2', data=dx2)
        geom.create_dataset('dx3', data=dx3)
        
        # MKS 特定参数 (补齐 R0 以防万一)
        mks = geom.create_group('mks')
        mks.create_dataset('a', data=spin)
        mks.create_dataset('r_in', data=r_coords[0])
        mks.create_dataset('r_out', data=r_coords[-1])
        mks.create_dataset('hslope', data=hslope)
        mks.create_dataset('R0', data=R0) # 显式加入 R0
        mks.create_dataset('r_eh', data=1.0 + np.sqrt(1.0 - spin**2))
        
        # 原始数据平铺 (Transpose 确保顺序为 nx, ny, nz)
        # athena_read: (k, j, i) -> ipole: (i, j, k)
        prims = np.stack([rho, uu, v1, v2, v3, b1, b2, b3], axis=-1)
        prims_final = prims.transpose(2, 1, 0, 3).astype('f4')
        f.create_dataset('prims', data=prims_final)

    print(f"--- Successfully created native H5: {output_h5} ---")

if __name__ == "__main__":
    CPFS_ROOT_PATH = "/cpfs01/projects-HDD/cfff-a7e284de52b3_HDD/cyh_22307110238"
    DATA_PATH = os.path.join(CPFS_ROOT_PATH, "data_test3/")
    # 测试单个文件
    input_file = os.path.join(DATA_PATH, "mad98.prim.00426.athdf")
    base_name = os.path.basename(input_file).replace('.athdf', '')

    output_dir = os.path.join(CPFS_ROOT_PATH, "native_output_run01/")
    os.makedirs(output_dir, exist_ok=True)
    output_h5 = os.path.join(output_dir, f"{base_name}_thermal_test_native.h5")
    if os.path.exists(input_file):
        create_thermal_native_h5(input_file, output_h5)
    else:
        print(f"Error: {input_file} not found.")