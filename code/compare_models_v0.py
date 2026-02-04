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

DATA_PATH = os.path.join(CPFS_ROOT_PATH, "workflow_output_DSA_run02/ipole_inputs/")
# 输入 HDF5 文件路径 (由 workflowFull 生成的输入文件)
INPUT_H5 = os.path.join(DATA_PATH, "mad98.prim.00426_ipole_input.h5")

# 观测参数 (请与你之前的运行参数保持一致)
PARAMS = {
    "thetacam": 163,
    "freqcgs": 86e9,
    "M_unit": 1e25,
    "trat_j": 1.0,
    "trat_d": 80.0,
    "sigma_cut": 5.0,
    "fov": 1000,
    "nx": 160,
    "ny": 160
}

# 输出目录
OUTPUT_DIR = os.path.join(CPFS_ROOT_PATH, "jet_image_comparison")
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
    """读取 IPOLE 输出图像的强度图和元数据 (稳健版)"""
    if not os.path.exists(h5_file):
        print(f"Error: 文件不存在 {h5_file}")
        return np.zeros((PARAMS['ny'], PARAMS['nx'])), 1.0

    with h5py.File(h5_file, 'r') as f:
        # 1. 尝试获取强度数据 (适配不同 IPOLE 输出格式)
        if 'unpol' in f:
            data = f['unpol'][:]
        elif 'pol' in f:
            # IPOLE V4 格式: pol(nx, ny, 4)，提取 Stokes I (index 0)
            data = f['pol'][:, :, 0]
        else:
            available_keys = list(f.keys())
            raise KeyError(f"无法在 {h5_file} 中找到 'unpol' 或 'pol'。可用键: {available_keys}")

        # 2. [核心修复] 稳健获取 scale，若不存在则默认为 1.0
        if 'scale' in f:
            scale = f['scale'][()]
        else:
            # 某些版本的 ipole 可能不输出 scale 数据集
            scale = 1.0
            
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
    h5_shock_in   = os.path.join(OUTPUT_DIR, "input_shock.h5")

    out_thermal = os.path.join(OUTPUT_DIR, "img_thermal.h5")
    out_reconn  = os.path.join(OUTPUT_DIR, "img_reconnection.h5")
    out_shock   = os.path.join(OUTPUT_DIR, "img_shock.h5")

    # --- 实验 A: 纯热背景 (KEL=0, emission=1) ---
    prepare_input(INPUT_H5, h5_thermal_in, mode="none")
    run_ipole(h5_thermal_in, out_thermal, emission_type=1)

    # --- 实验 B: 磁重联模型 (KEL=0, emission=3) ---
    # 这会走原版 IPOLE 基于 B^2 的均分逻辑
    prepare_input(INPUT_H5, h5_reconn_in, mode="none")
    run_ipole(h5_reconn_in, out_reconn, emission_type=3)

    # --- 实验 C: 激波加速模型 (KEL=Original, ANY emission) ---
    # 只要 KEL=1，你的修改版就会优先走 DSA 逻辑
    prepare_input(INPUT_H5, h5_shock_in, mode="shock")
    run_ipole(h5_shock_in, out_shock, emission_type=None)

    # --- 数据处理与可视化 ---
    img_a, _ = load_intensity(out_thermal)
    img_b, _ = load_intensity(out_reconn)
    img_c, _ = load_intensity(out_shock)

    # 流量统计 (Jansky)
    flux_a, flux_b, flux_c = np.sum(img_a), np.sum(img_b), np.sum(img_c)
    
    print("\n" + "="*30)
    print("FLUX STATISTICS (Jy)")
    print(f"Model A (Thermal Only):      {flux_a:.4f}")
    print(f"Model B (Reconnection/B^2):  {flux_b:.4f}")
    print(f"Model C (Shock-DSA/Our):     {flux_c:.4f}")
    print("="*30)

    # 绘图
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 第一行：原始图像
    im0 = axes[0,0].imshow(img_a, cmap='afmhot', origin='lower')
    axes[0,0].set_title("A: Thermal Only")
    plt.colorbar(im0, ax=axes[0,0])

    im1 = axes[0,1].imshow(img_b, cmap='afmhot', origin='lower')
    axes[0,1].set_title("B: Reconnection (Yang+24)")
    plt.colorbar(im1, ax=axes[0,1])

    im2 = axes[0,2].imshow(img_c, cmap='afmhot', origin='lower')
    axes[0,2].set_title("C: Shock-DSA (Xia+25)")
    plt.colorbar(im2, ax=axes[0,2])

    # 第二行：差分图 (Residuals)
    # C - A = 纯粹的激波加速贡献
    diff_shock = img_c - img_a
    im3 = axes[1,0].imshow(diff_shock, cmap='viridis', origin='lower')
    axes[1,0].set_title("C - A: Net Shock Contribution")
    plt.colorbar(im3, ax=axes[1,0])

    # C - B = 激波模型对比重联模型
    diff_model = img_c - img_b
    im4 = axes[1,1].imshow(diff_model, cmap='RdBu_r', origin='lower')
    axes[1,1].set_title("C - B: Shock vs Reconnection")
    plt.colorbar(im4, ax=axes[1,1])

    axes[1,2].axis('off') # 留空

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "comparison_results_mhd.png")
    plt.savefig(plot_path)
    print(f"\nResults saved to {OUTPUT_DIR}")
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    main()