import numpy as np
import h5py
import os
import matplotlib.pyplot as plt

def create_ipole_input_h5(ipole_h5_filename, roi_data, shock_properties, nonthermal_props, spin):
    """
    为新版 ipole 生成完全兼容的 HDF5 输入文件。
    [从 workflowFull_v1.4.0.py 恢复]
    """
    print(f"--- Step E: Creating IPOLE input file: {os.path.basename(ipole_h5_filename)} ---")
    gamma = 4.0/3.0 # 根据Yang et. al 2024选取 [cite: 6003-6824]
    
    # 确保我们有 'Bcc' 变量
    if 'Bcc1' not in roi_data or 'Bcc2' not in roi_data or 'Bcc3' not in roi_data:
        print("  Error: 'Bcc' (cell-centered B field) not found in roi_data. Using 'B' (face-centered) as fallback.")
        # 注意：这在物理上不完全准确，但作为备用方案
        b1, b2, b3 = roi_data.get('B1', np.zeros_like(roi_data['press'])), \
                     roi_data.get('B2', np.zeros_like(roi_data['press'])), \
                     roi_data.get('B3', np.zeros_like(roi_data['press']))
    else:
        b1, b2, b3 = roi_data['Bcc1'], roi_data['Bcc2'], roi_data['Bcc3']

    rho, press = roi_data['rho'], roi_data['press']
    uu = press / (gamma - 1.0)
    vel1, vel2, vel3 = roi_data['vel1'], roi_data['vel2'], roi_data['vel3']
    nk, nj, ni = rho.shape
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
        header.create_dataset('prim_names', data=np.array(prim_names_list, dtype=string_dt), dtype=string_dt)

        geom = header.create_group('geom')
        geom.create_dataset('startx1', data=np.log(roi_data['x1f'][0]), dtype='float64')
        geom.create_dataset('startx2', data=roi_data['x2f'][0], dtype='float64') # x2 (theta) 通常是线性
        geom.create_dataset('startx3', data=roi_data['x3f'][0], dtype='float64') # x3 (phi) 通常是线性
        
        dx1 = np.log(roi_data['x1f'][1] / roi_data['x1f'][0])
        dx2 = roi_data['x2f'][1] - roi_data['x2f'][0]
        dx3 = roi_data['x3f'][1] - roi_data['x3f'][0] if n3 > 1 else 2 * np.pi
        
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
        prims_final = prims_stack.transpose(2, 1, 0, 3) # (nx, ny, nz, nvar)
        f.create_dataset('prims', data=prims_final, dtype='float32')


        # --- 注入非热电子物理 ---
        q_grid = nonthermal_props['q_grid']
        kel_grid = np.where(shock_properties['mask'], 1, 0) # 1=非热, 0=热
        
        # [cite_start]p = q - 1 [cite: 5606-5609, 5344-5347]
        # IPOLE 需要的是谱指数 p，而 DSA 计算的是 q。
        p_grid = q_grid - 1.0 
        p_grid[~shock_properties['mask']] = 0.0 # 在非激波区设为0
        
        # 确保维度正确 (nx, ny, nz)
        # 建议将最后三行稍微修改为：
        f.create_dataset('KEL', data=np.ascontiguousarray(kel_grid.transpose(2, 1, 0)), dtype='float64')
        f.create_dataset('UNTH', data=np.ascontiguousarray(nonthermal_props['C_grid'].transpose(2, 1, 0)), dtype='float64')
        f.create_dataset('p', data=np.ascontiguousarray(p_grid.transpose(2, 1, 0)), dtype='float64')

    print(f"  Successfully created IPOLE input file.")


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