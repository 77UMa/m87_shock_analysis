import numpy as np
import h5py
import os

def convert_to_ipole_format(checkpoint_file, output_h5_file, gamma=5.0/3.0, spin=0.98):
    """
    将我们的检查点数据转换为 IPOLE 可读的 HDF5 文件格式。
    
    Args:
        checkpoint_file (str): 输入的 .npz 检查点文件名。
        output_h5_file (str): 输出的 .h5 文件名。
        gamma (float): 绝热指数。
        spin (float): 黑洞自旋参数 a。
    """
    print(f"--- Converting {checkpoint_file} to IPOLE format ---")
    
    # --- 步骤 1: 加载我们的数据 ---
    data = np.load(checkpoint_file)
    
    # 提取物理量 (注意：检查点文件中没有vel2, vel3, B2, B3，需要从roi_data中获取)
    # 我们需要 'rho', 'press', 'vel1-3', 'Bcc1-3'
    rho = data['rho']
    press = data['press']
    vel1, vel2, vel3 = data['vel1'], data['vel2'], data['vel3']
    b1, b2, b3 = data['Bcc1'], data['Bcc2'], data['Bcc3']
    
    # 从压力计算内能 UU (uu = press / (gamma - 1))
    uu = press / (gamma - 1.0)
    
    # 获取维度信息 (k, j, i) -> (phi, theta, r)
    nk, nj, ni = rho.shape
    print(f"Input data dimensions (phi, theta, r): ({nk}, {nj}, {ni})")
    
    # --- 步骤 2: 创建并写入新的HDF5文件 ---
    with h5py.File(output_h5_file, 'w') as f:
        print(f"Creating output file: {output_h5_file}")
        
        # --- A. 写入顶层数据集 ---
        # HARM3D 的维度顺序是 (x2, x1, x3) -> (theta, r, phi) -> (j, i, k)
        # 我们的数据维度是 (k, j, i)
        # 因此，我们需要将轴从 (0, 1, 2) 转置为 (1, 2, 0)
        
        print("Writing top-level datasets (RHO, UU, B1/2/3, v1/2/3)...")
        f.create_dataset('rho', data=rho.transpose(1, 2, 0), dtype='float64')
        f.create_dataset('uu', data=uu.transpose(1, 2, 0), dtype='float64')
        f.create_dataset('v1', data=vel1.transpose(1, 2, 0), dtype='float64')
        f.create_dataset('v2', data=vel2.transpose(1, 2, 0), dtype='float64')
        f.create_dataset('v3', data=vel3.transpose(1, 2, 0), dtype='float64')
        f.create_dataset('B1', data=b1.transpose(1, 2, 0), dtype='float64')
        f.create_dataset('B2', data=b2.transpose(1, 2, 0), dtype='float64')
        f.create_dataset('B3', data=b3.transpose(1, 2, 0), dtype='float64')
        
        # 写入 HARM 模板中多余的 _p 变量 (内容可以和原变量一样)
        f.create_dataset('rho_p', data=rho.transpose(1, 2, 0), dtype='float64')
        f.create_dataset('uu_p', data=uu.transpose(1, 2, 0), dtype='float64')
        # ... 以此类推，为所有 v_p 和 B_p 创建数据集 ...

        # --- B. 写入 Header 组和子组 ---
        print("Writing Header group...")
        header = f.create_group('Header')
        grid = header.create_group('Grid')
        
        # 写入关键的模拟参数
        grid.create_dataset('a', data=np.array([spin]))
        grid.create_dataset('gam', data=np.array([gamma]))
        
        # 写入网格维度信息 (以 HARM 的 x1,x2,x3 顺序)
        grid.create_dataset('N1', data=np.array([ni])) # r 方向
        grid.create_dataset('N2', data=np.array([nj])) # theta 方向
        grid.create_dataset('N3', data=np.array([nk])) # phi 方向
        
        # 写入坐标范围等其他重要参数 (从您的 roi_data 中提取)
        grid.create_dataset('Rin', data=np.array([data['x1f'][0]]))
        grid.create_dataset('Rout', data=np.array([data['x1f'][-1]]))
        # ... 根据 HARM3D.h5 中的其他 Header 内容，继续添加必要的信息 ...

    print("--- Conversion to IPOLE format complete! ---")


# --- 主程序入口 ---
if __name__ == '__main__':
    # 定义您的检查点文件和希望生成的IPOLE输入文件名
    checkpoint_file = 'F:\Research\Shockwave\data_test\mad98.checkpoint.00100.npz' # 示例
    ipole_input_file = 'ipole_input.00100.h5' # 示例
    
    if os.path.exists(checkpoint_file):
        convert_to_ipole_format(checkpoint_file, ipole_input_file)
    else:
        print(f"Error: Checkpoint file '{checkpoint_file}' not found.")