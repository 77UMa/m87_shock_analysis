'''
输入`workflowFull`生成的ipole输入文件，比较3种模型下的辐射形态。
'''
import os
import sys
import h5py
import numpy as np
import subprocess
import matplotlib.pyplot as plt
from astropy.io import fits
import matplotlib.colors as mcolors

try:
    # 尝试导入核心依赖
    import ipole as ipole_api
except ImportError as e:
    print(f"Fatal Error: Could not import a required module. {e}")
    print("Please check the paths to 'ipole-master/scripts' and 'pyathena'.")
    sys.exit(1)

# ================= 配置区域 =================
HOME_PATH = "/home/cyh_22307110238/project/Shockwave"
CPFS_ROOT_PATH = "/cpfs01/projects-HDD/cfff-a7e284de52b3_HDD/cyh_22307110238/DSA"

# IPOLE 程序路径
IPOLE_BIN = os.path.join(HOME_PATH, "ipole-DSA/ipole") 

DATA_PATH = os.path.join(CPFS_ROOT_PATH, "workflowV2_run01/ipole_inputs/")
# 输入 HDF5 文件路径 (由 workflowFull 生成的输入文件)
INPUT_H5 = os.path.join(DATA_PATH, "mad98.prim.00455_dsa_input.h5")

FOV = 500
# 观测参数 (请与你之前的运行参数保持一致)
PARAMS = {
    "thetacam": 163,
    "freqcgs": 230e9,
    "M_unit": 1e25,
    "trat_j": 1.0,
    "trat_d": 80.0,
    "sigma_cut": 5.0,
    "fov": FOV,
    "nx": 500,
    "ny": 500
}


# 输出目录
OUTPUT_DIR = os.path.join(CPFS_ROOT_PATH, "Radiation_run01")
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
    """
    读取 IPOLE 输出图像的强度图并确保单位为 Jy (稳健修复版)
    """
    if not os.path.exists(h5_file):
        print(f"Warning: 文件不存在 {h5_file}")
        # 返回全零矩阵，避免程序中断
        return np.zeros((PARAMS['ny'], PARAMS['nx'])), 1.0

    with h5py.File(h5_file, 'r') as f:
        # 1. 尝试获取强度数据 (适配 pol 和 unpol 格式)
        if 'pol' in f:
            # 如果是全偏振输出 (nx, ny, 4)，提取 Stokes I (index 0)
            data = f['pol'][:, :, 0]
        elif 'unpol' in f:
            data = f['unpol'][:]
        else:
            print(f"Error: 在 {h5_file} 中找不到强度数据。可用键: {list(f.keys())}")
            return np.zeros((PARAMS['ny'], PARAMS['nx'])), 1.0

        # 2. 稳健获取转换系数 scale
        # 尝试多个可能的位置，如果都找不到，则使用 1.0 并打印警告
        scale = 1.0
        if 'scale' in f:
            scale = f['scale'][()]
        elif 'header/scale' in f:
            scale = f['header/scale'][()]
        elif 'header' in f and 'scale' in f['header'].attrs:
            scale = f['header'].attrs['scale']
        else:
            print(f"Warning: 在 {os.path.basename(h5_file)} 中未找到 'scale'，流量统计将保持代码单位。")

    # 注意：某些 ipole 输出的数据形状是 (nx, ny)，绘图时通常需要转置为 (ny, nx)
    # 但由于 compare_models 之前运行正常，这里保持原有的返回逻辑，仅修正 scale
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

    # ================= 统一对数颜色标度处理 =================
    vmax_global = max(np.max(img_a), np.max(img_b), np.max(img_c))
    # 对于对数标度，最小值不能为0。通常设置为全局最大值的 10^-4 或 10^-5 作为一个合理的本底下限
    vmin_log = vmax_global * 1e-4 

    # 创建统一的对数归一化器
    norm_log = mcolors.LogNorm(vmin=vmin_log, vmax=vmax_global)

    # 绘图
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 第一行：原始图像 (使用 LogNorm)
    im0 = axes[0,0].imshow(img_a, cmap='afmhot', origin='lower', norm=norm_log)
    axes[0,0].set_title("A: Thermal Only")
    plt.colorbar(im0, ax=axes[0,0])

    im1 = axes[0,1].imshow(img_b, cmap='afmhot', origin='lower', norm=norm_log)
    axes[0,1].set_title("B: Reconnection Model")
    plt.colorbar(im1, ax=axes[0,1])

    im2 = axes[0,2].imshow(img_c, cmap='afmhot', origin='lower', norm=norm_log)
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
    plot_path = os.path.join(OUTPUT_DIR, f"comparison_results_mhd_native_fov{FOV}.png")
    plt.savefig(plot_path)
    print(f"\nResults saved to {OUTPUT_DIR}")
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()