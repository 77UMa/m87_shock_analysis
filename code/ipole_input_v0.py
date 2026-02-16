import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

script_dir = os.path.dirname(os.path.abspath(__file__))
pyathena_path = os.path.join(script_dir, '..', 'pyathena')
sys.path.insert(0, pyathena_path)
# 确保能找到 pyathena
# sys.path.insert(0, '/path/to/your/pyathena')
from pyathena import athena_read
from shock_v1 import find_shocks_in_roi_mhd
from nt_electron_v1 import calculate_nonthermal_electrons

def create_ipole_input_dsa_native(athdf_filename, output_h5, config):
    # 1. 沿用 Native 模式读取数据
    data = athena_read.athdf(athdf_filename, level=4)
    # ... (提取 rho, press, vel, b 字段，同 Native 脚本) ...
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
    # 2. 运行 3D MHD 激波探测 (使用之前为您提供的 MHD 升级版函数)
    # 注意：确保传递正确的物理坐标以计算梯度
    shock_props = find_shocks_in_roi_mhd(data, **config["shock_params"])
    
    # 3. 计算非热电子性质 (使用您修正过的 N_inj 逻辑)
    # 确保此处返回的是 N_inj 而非归一化常数 C
    nonthermal_props = calculate_nonthermal_electrons(shock_props, **config["nt_params"])

    # 4. 生成兼容 ipole-DSA 的 HDF5
    with h5py.File(output_h5, 'w') as f:
        # --- 写入 Native 模式的 Header, Geom, Prims ---
        # ... (此处省略，代码完全拷贝自您的 create_thermal_native_h5) ...
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
        # --- 注入 DSA 物理量 ---
        mask = shock_props['mask']
        q_grid = nonthermal_props['q_grid']
        p_grid = q_grid - 1.0
        p_grid[~mask] = 0.0 # 非激波区设为 0
        
        # 写入关键数据集 (注意维度顺序 nx, ny, nz)
        f.create_dataset('KEL', data=mask.transpose(2, 1, 0).astype('f8'))
        f.create_dataset('UNTH', data=nonthermal_props['C_grid'].transpose(2, 1, 0).astype('f8')) # 这里其实是 N_inj
        f.create_dataset('p', data=p_grid.transpose(2, 1, 0).astype('f8'))

    print(f"--- Native-DSA Input Created: {output_h5} ---")


def plot_ipole_output(h5_filename, fov_muas, output_png_filename):
    """
    读取 ipole 输出的 HDF5 文件并可视化 Stokes I 图像。
    [从 workflowFull_v1.4.0.py 恢复]
    """
    print(f"--- Step G: Plotting final image from {os.path.basename(h5_filename)} ---")
    try:
        with h5py.File(h5_filename, 'r') as f:
            if 'pol' not in f:
                print(f"  Error: Could not find 'pol' dataset in {h5_filename}.")
                return
            # IPOLE V4 格式: pol(nx, ny, 4) [I, Q, U, V]
            image_I_raw = np.copy(f['pol'])
            # 转置为 (ny, nx) 以便 imshow
            image_I = image_I_raw[:, :, 0].T
            
        extent = [-fov_muas / 2.0, fov_muas / 2.0, -fov_muas / 2.0, fov_muas / 2.0]
        fig, ax = plt.subplots(figsize=(10, 8))
        
        max_val = np.max(image_I)
        min_val = 1e-8 * max_val if max_val > 0 else 1e-20
        plot_data = np.log10(np.maximum(image_I, min_val))
        
        vmax = np.max(plot_data)
        vmin = vmax - 5 # 动态范围为 5 个数量级
        
        im = ax.imshow(plot_data, extent=extent, origin='lower', cmap='afmhot', vmin=vmin, vmax=vmax)
        ax.set_xlabel("Relative RA (μas)")
        ax.set_ylabel("Relative Dec (μas)")
        ax.set_title(f"Stokes I - {os.path.basename(h5_filename)} (Log Scale, fov={fov_muas})")
        cbar = fig.colorbar(im)
        cbar.set_label("log10(Intensity / [Jy/pixel])")
        ax.set_aspect('equal')
        plt.savefig(output_png_filename, dpi=300, bbox_inches='tight')
        print(f"--- Successfully generated image: {os.path.basename(output_png_filename)} ---")
        plt.close(fig)
    except Exception as e:
        print(f"  An error occurred during plotting: {e}")




# --- 主程序入口 ---
if __name__ == '__main__':
    # 定义您的检查点文件和希望生成的IPOLE输入文件名
    checkpoint_file = 'F:\Research\Shockwave\data_test\mad98.checkpoint.00100.npz' # 示例
    ipole_input_file = 'ipole_input.00100.h5' # 示例
    
    if os.path.exists(checkpoint_file):
        create_ipole_input_h5(checkpoint_file, ipole_input_file)
    else:
        print(f"Error: Checkpoint file '{checkpoint_file}' not found.")