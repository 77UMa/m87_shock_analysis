'''
输入`workflowFull`生成的ipole输入文件，比较3种模型下的辐射形态。
'''
import os
import sys
import time
import logging
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 路径配置：确保直接运行时也能找到 utils
_script_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_script_dir)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from utils.logging_config import setup_logging

try:
    import ipole as ipole_api
except ImportError as e:
    print(f"Fatal Error: Could not import ipole. {e}")
    sys.exit(1)

# ================= 配置区域 =================
HOME_PATH = "/home/cyh_22307110238/project/Shockwave"
CPFS_ROOT_PATH = "/cpfs01/projects-HDD/cfff-a7e284de52b3_HDD/cyh_22307110238/DSA"

IPOLE_BIN = os.path.join(HOME_PATH, "ipole-DSA/ipole")
DATA_PATH = os.path.join(CPFS_ROOT_PATH, "workflowV2_advection02/ipole_inputs/")
INPUT_H5 = os.path.join(DATA_PATH, "mad98.prim.00469_dsa_input.h5")

FOV = 300
PARAMS = {
    "thetacam": 163,
    "freqcgs": 86e9,
    "M_unit": 1e25,
    "trat_j": 1.0,
    "trat_d": 80.0,
    "sigma_cut": 5.0,
    "fov": FOV,
    "nx": FOV,
    "ny": FOV
}

OUTPUT_DIR = os.path.join(CPFS_ROOT_PATH, "Radiation_run04_86GHz")
# ===========================================


def run_ipole(input_file, output_file, emission_type=None, logger=None):
    """
    运行 IPOLE，返回耗时（秒）。
    emission_type 为 None 时使用 ipole 默认逻辑（触发 DSA）
    """
    args = PARAMS.copy()
    args['dump'] = input_file
    args['outfile'] = output_file
    if emission_type is not None:
        args['emission_type'] = emission_type

    label = os.path.basename(output_file)
    if logger:
        logger.info(f"Starting IPOLE: {label} (emission_type={emission_type})")
    else:
        print(f"\n>>> Running IPOLE: {label}")

    t0 = time.time()
    ipole_api.run(args, exe=IPOLE_BIN, verbose=2)
    elapsed = time.time() - t0

    if logger:
        logger.info(f"Finished IPOLE: {label} ({elapsed:.1f}s)")
    return elapsed


def load_intensity(h5_file, logger=None):
    """
    读取 IPOLE 输出图像的强度图并确保单位为 Jy
    """
    if not os.path.exists(h5_file):
        msg = f"Output file not found: {h5_file}"
        if logger:
            logger.warning(msg)
        else:
            print(f"Warning: {msg}")
        return np.zeros((PARAMS['ny'], PARAMS['nx'])), 1.0

    with h5py.File(h5_file, 'r') as f:
        if 'pol' in f:
            data = f['pol'][:, :, 0]
        elif 'unpol' in f:
            data = f['unpol'][:]
        else:
            msg = f"No intensity data in {h5_file}. Keys: {list(f.keys())}"
            if logger:
                logger.error(msg)
            else:
                print(f"Error: {msg}")
            return np.zeros((PARAMS['ny'], PARAMS['nx'])), 1.0

        scale = 1.0
        if 'scale' in f:
            scale = f['scale'][()]
        elif 'header/scale' in f:
            scale = f['header/scale'][()]
        elif 'header' in f and 'scale' in f['header'].attrs:
            scale = f['header'].attrs['scale']
        else:
            msg = f"'scale' not found in {os.path.basename(h5_file)}, flux in code units."
            if logger:
                logger.warning(msg)
            else:
                print(f"Warning: {msg}")

    return data * scale, scale


def prepare_input(source_h5, target_h5, mode="shock", logger=None):
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
                f['KEL'][...] = np.zeros_like(f['KEL'][:])
                msg = f"KEL zeroed in {os.path.basename(target_h5)}"
                if logger:
                    logger.info(msg)
                else:
                    print(f"  {msg}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_dir = os.path.join(OUTPUT_DIR, 'logs')
    logger, log_file = setup_logging(
        log_dir=log_dir,
        log_level=logging.INFO,
        log_name=f"compare_models_{time.strftime('%Y%m%d_%H%M%S')}.log",
        logger_name='CompareModels'
    )

    # --- 运行信息头 ---
    logger.info("=" * 60)
    logger.info("Compare Models Run Started")
    logger.info(f"  Input H5  : {INPUT_H5}")
    logger.info(f"  Output Dir: {OUTPUT_DIR}")
    logger.info(f"  IPOLE bin : {IPOLE_BIN}")
    logger.info(f"  Params    : {PARAMS}")
    logger.info(f"  Log file  : {log_file}")
    logger.info("=" * 60)

    t_total = time.time()

    # 定义文件路径
    h5_thermal_in = os.path.join(OUTPUT_DIR, "input_thermal.h5")
    h5_reconn_in  = os.path.join(OUTPUT_DIR, "input_reconnection.h5")
    h5_shock_in   = os.path.join(OUTPUT_DIR, "input_shock.h5")
    out_thermal   = os.path.join(OUTPUT_DIR, "img_thermal.h5")
    out_reconn    = os.path.join(OUTPUT_DIR, "img_reconnection.h5")
    out_shock     = os.path.join(OUTPUT_DIR, "img_shock.h5")

    elapsed = {}

    # --- 实验 A: 纯热背景 ---
    logger.info("--- Model A: Thermal Only (emission=1) ---")
    try:
        prepare_input(INPUT_H5, h5_thermal_in, mode="none", logger=logger)
        elapsed['A'] = run_ipole(h5_thermal_in, out_thermal, emission_type=1, logger=logger)
    except Exception as e:
        logger.error(f"Model A failed: {e}", exc_info=True)
        elapsed['A'] = None

    # --- 实验 B: 磁重联模型 ---
    logger.info("--- Model B: Reconnection (emission=3) ---")
    try:
        prepare_input(INPUT_H5, h5_reconn_in, mode="none", logger=logger)
        elapsed['B'] = run_ipole(h5_reconn_in, out_reconn, emission_type=3, logger=logger)
    except Exception as e:
        logger.error(f"Model B failed: {e}", exc_info=True)
        elapsed['B'] = None

    # --- 实验 C: 激波加速模型 ---
    logger.info("--- Model C: Shock-DSA (default emission) ---")
    try:
        prepare_input(INPUT_H5, h5_shock_in, mode="shock", logger=logger)
        elapsed['C'] = run_ipole(h5_shock_in, out_shock, emission_type=None, logger=logger)
    except Exception as e:
        logger.error(f"Model C failed: {e}", exc_info=True)
        elapsed['C'] = None

    # --- 耗时汇总 ---
    logger.info("=" * 60)
    logger.info("IPOLE Runtime Summary:")
    for model, t in elapsed.items():
        logger.info(f"  Model {model}: {f'{t:.1f}s' if t is not None else 'FAILED'}")
    logger.info(f"  Total elapsed: {time.time() - t_total:.1f}s")
    logger.info("=" * 60)

    # --- 数据读取 ---
    img_a, _ = load_intensity(out_thermal, logger=logger)
    img_b, _ = load_intensity(out_reconn,  logger=logger)
    img_c, _ = load_intensity(out_shock,   logger=logger)

    # --- 流量统计 ---
    flux_a, flux_b, flux_c = np.sum(img_a), np.sum(img_b), np.sum(img_c)
    logger.info("FLUX STATISTICS (Jy):")
    logger.info(f"  Model A (Thermal Only)     : {flux_a:.4f}")
    logger.info(f"  Model B (Reconnection/B^2) : {flux_b:.4f}")
    logger.info(f"  Model C (Shock-DSA/Ours)   : {flux_c:.4f}")

    # --- 绘图 ---
    logger.info("Generating comparison plot...")
    vmax_global = max(np.max(img_a), np.max(img_b), np.max(img_c))
    vmin_log = vmax_global * 1e-4
    norm_log = mcolors.LogNorm(vmin=vmin_log, vmax=vmax_global)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    im0 = axes[0,0].imshow(img_a, cmap='afmhot', origin='lower', norm=norm_log)
    axes[0,0].set_title("A: Thermal Only")
    plt.colorbar(im0, ax=axes[0,0])

    im1 = axes[0,1].imshow(img_b, cmap='afmhot', origin='lower', norm=norm_log)
    axes[0,1].set_title("B: Reconnection Model")
    plt.colorbar(im1, ax=axes[0,1])

    im2 = axes[0,2].imshow(img_c, cmap='afmhot', origin='lower', norm=norm_log)
    axes[0,2].set_title("C: Shock-DSA (Xia+25)")
    plt.colorbar(im2, ax=axes[0,2])

    diff_shock = img_c - img_a
    im3 = axes[1,0].imshow(diff_shock, cmap='viridis', origin='lower')
    axes[1,0].set_title("C - A: Net Shock Contribution")
    plt.colorbar(im3, ax=axes[1,0])

    diff_model = img_c - img_b
    im4 = axes[1,1].imshow(diff_model, cmap='RdBu_r', origin='lower')
    axes[1,1].set_title("C - B: Shock vs Reconnection")
    plt.colorbar(im4, ax=axes[1,1])

    axes[1,2].axis('off')

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, f"comparison_results_mhd_native_fov{FOV}.png")
    plt.savefig(plot_path)
    plt.close(fig)

    logger.info(f"Plot saved to: {plot_path}")
    logger.info(f"All results saved to: {OUTPUT_DIR}")
    logger.info(f"Log saved to: {log_file}")


if __name__ == "__main__":
    main()
