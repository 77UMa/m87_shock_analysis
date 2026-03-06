'''
输入原始的.athdf文件，用于生成热辐射图像来检验ipole输入格式是否正确，目标是热辐射需要能在100\muas复现精细的事件视界阴影。
'''

import os
import sys
import h5py
import numpy as np
import subprocess
import matplotlib.pyplot as plt
from astropy.io import fits

try:
    # 尝试导入核心依赖
    import ipole as ipole_api
except ImportError as e:
    print(f"Fatal Error: Could not import a required module. {e}")
    print("Please check the paths to 'ipole-master/scripts' and 'pyathena'.")
    sys.exit(1)

# ================= 配置区域 =================
HOME_PATH = "/home/cyh_22307110238/project/Shockwave"
CPFS_ROOT_PATH = "/cpfs01/projects-HDD/cfff-a7e284de52b3_HDD/cyh_22307110238"

# IPOLE 程序路径
IPOLE_BIN = os.path.join(HOME_PATH, "ipole-DSA/ipole") 

DATA_PATH = os.path.join(CPFS_ROOT_PATH, "workflow_output_DSA_run03/ipole_inputs/")
# 输入 HDF5 文件路径 (由 workflowFull 生成的输入文件)
# 更新为您新生成的 Native H5 文件路径
INPUT_H5 = "/cpfs01/projects-HDD/cfff-a7e284de52b3_HDD/cyh_22307110238/native_output_run01/mad98.prim.00426_thermal_test_native.h5"

PARAMS = {
    "thetacam": 163,     # M87 观测倾角
    "freqcgs": 230e9,    # 【建议】暂时改为 230 GHz 以验证阴影可见性 
    "M_unit": 1e25,      # 质量单位
    "trat_j": 1.0,
    "trat_d": 80.0,
    "sigma_cut": 5.0,
    "fov": 100,          # 【建议】缩小 FOV 到 200 以聚焦视界区域 [cite: 4724]
    "nx": 100,
    "ny": 100
}

OUTPUT_DIR = os.path.join(CPFS_ROOT_PATH, "native_test_results")
# ===========================================

def run_ipole(input_file, output_file, emission_type=None):
    """
    运行 IPOLE
    emission_type 为 None 时使用 ipole 默认逻辑（触发 DSA）
    """
    args = PARAMS.copy()
    args['dump'] = input_file
    args['outfile'] = output_file
    
    # 仅当显式指定时才加入参数
    if emission_type is not None:
        args['emission_type'] = emission_type
    
    print(f"\n>>> Running IPOLE via API for {os.path.basename(output_file)}")
    ipole_api.run(args, exe=IPOLE_BIN, verbose=2)
    return True

def load_intensity(h5_file):
    """读取 IPOLE 输出图像的强度图并确保单位为 Jy"""
    with h5py.File(h5_file, 'r') as f:
        # 建议优先读取全偏振模式下的 Stokes I (index 0)
        if 'pol' in f:
            data = f['pol'][:, :, 0] # 获取 Stokes I
        elif 'unpol' in f:
            data = f['unpol'][:]
        else:
            raise KeyError("No intensity data found.")
            
        # 必须乘以 scale 才能转换到 Jansky
        scale = f['scale'][()]
        
    return data * scale, scale

def prepare_input(source_h5, target_h5, mode="shock"):
    """
    创建临时的 IPOLE 输入文件
    mode='shock': 保留原始 KEL
    mode='none': 将 KEL 清零，强制走热辐射/重联逻辑
    """
    import shutil
    shutil.copy2(source_h5, target_h5)
    if mode == "none":
        with h5py.File(target_h5, 'r+') as f:
            if 'KEL' in f:
                data = f['KEL'][:]
                f['KEL'][...] = np.zeros_like(data)
                print(f"Modified {target_h5}: KEL set to zero.")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 定义实验组文件路径
    h5_thermal_in = os.path.join(OUTPUT_DIR, "input_thermal.h5")
    h5_reconn_in  = os.path.join(OUTPUT_DIR, "input_reconnection.h5")


    out_thermal = os.path.join(OUTPUT_DIR, "img_thermal.h5")
    out_reconn  = os.path.join(OUTPUT_DIR, "img_reconnection.h5")
 

    # --- 实验 A: 纯热背景 (KEL=0, emission=1) ---
    prepare_input(INPUT_H5, h5_thermal_in, mode="none")
    run_ipole(h5_thermal_in, out_thermal, emission_type=1)

    # --- 实验 B: 磁重联模型 (KEL=0, emission=3) ---
    # 这会走原版 IPOLE 基于 B^2 的均分逻辑
    prepare_input(INPUT_H5, h5_reconn_in, mode="none")
    run_ipole(h5_reconn_in, out_reconn, emission_type=3)

    # --- 数据处理与可视化 ---
    img_a, _ = load_intensity(out_thermal)
    img_b, _ = load_intensity(out_reconn)

    # 流量统计 (Jansky)
    flux_a, flux_b = np.sum(img_a), np.sum(img_b)
    
    print("\n" + "="*30)
    print("FLUX STATISTICS (Jy)")
    print(f"Model A (Thermal Only):      {flux_a:.4f}")
    print(f"Model B (Reconnection/B^2):  {flux_b:.4f}")
    print("="*30)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6))
    
    im0 = ax0.imshow(img_a, cmap='afmhot', origin='lower')
    ax0.set_title("A: Thermal Only (Native Input)")
    plt.colorbar(im0, ax=ax0)

    im1 = ax1.imshow(img_b, cmap='afmhot', origin='lower')
    ax1.set_title("B: Reconnection (B^2 model)")
    plt.colorbar(im1, ax=ax1)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "native_comparison_large.png")
    plt.savefig(plot_path)
    print(f"\nResults and plot saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()