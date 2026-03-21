#!/usr/bin/env python3
"""
run_sigma_sweep.py — σ 参数扫描实验脚本

对单个快照文件，顺序运行 N 组 sigma_crit 的完整 DSA pipeline，
并生成涵盖所有模型的对比图。

优化策略
--------
- Model A（热辐射）和 Model B（磁重联 B²）与 sigma_crit 完全无关，
  只在 shared/ 目录下计算一次。
- 仅 Model C（激波 DSA）对每个 sigma_crit 值分别重跑。

目录结构
--------
<output_dir>/
├── shared/
│   ├── input_AB_kel0.h5               # KEL 清零的共享输入
│   ├── img_A_thermal_<snap>.h5        # ipole 输出
│   ├── img_B_reconnection_<snap>.h5
│   └── logs/models_AB_<snap>.log
├── sigma_0.010/
│   ├── ipole_inputs/<snap>_dsa_input.h5
│   ├── input_C_shock.h5
│   ├── img_C_shock_<snap>_sigma0.010.h5
│   └── logs/generate_<snap>.log
├── sigma_0.030/  ...
├── sigma_0.100/  ...
├── sweep_<snap>_<date>.png             # 最终对比图
└── sweep_<snap>_<date>.log             # 结构化 Python 日志

日志说明
--------
- sweep_*.log：Python logger 输出（含时间戳、模块名、级别）
- screen_*.log：shell tee 捕获的全部终端输出（含 ipole verbose 打印）
  → 由 launch_sigma_sweep.sh 创建

用法
----
python run_sigma_sweep.py \\
    --snapshot /path/to/mad98.prim.00469.athdf \\
    --sigma-values 0.01 0.03 0.1 \\
    --alpha-sigma 2 \\
    --output-dir /cpfs01/.../sigma_sweep/
"""

import os
import sys
import copy
import time
import shutil
import logging
import argparse

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── 路径配置 ──────────────────────────────────────────────────────────────────
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)
sys.path.insert(0, os.path.join(_script_dir, 'src'))
sys.path.insert(0, os.path.join(_script_dir, '..', 'pyathena'))

from utils.logging_config import setup_logging
from workflows.workflowFull_v2 import process_snapshot

try:
    import ipole as ipole_api
except ImportError as e:
    print(f"Fatal Error: Could not import ipole module. {e}")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# 用户配置区 — 按服务器环境修改
# ══════════════════════════════════════════════════════════════════════════════
HOME_PATH = "/home/cyh_22307110238/project/Shockwave"
CPFS_PATH = "/cpfs01/projects-HDD/cfff-a7e284de52b3_HDD/cyh_22307110238/DSA"

IPOLE_DSA = os.path.join(HOME_PATH, "ipole-DSA/ipole")    # Model C 专用
IPOLE_STD = os.path.join(HOME_PATH, "ipole-master/ipole") # Model A/B 使用

IPOLE_PARAMS = dict(
    thetacam=163,
    freqcgs=86e9,
    M_unit=1e25,
    trat_j=1.0,
    trat_d=80.0,
    sigma_cut=5.0,
    fov=300,
    nx=300,
    ny=300,
)

def _base_pipeline_config():
    """返回基础 pipeline 配置，output_directory 和 nt_params 由外部覆盖。"""
    return {
        "ipole_executable_path": IPOLE_STD,
        "roi_params": {
            'r_min': 10, 'r_max': 1200,
            'theta_min': 0.0, 'theta_max': 3.14159,
            'phi_min':  0.0,  'phi_max':  6.28319,
        },
        "shock_params": {
            "gamma": 4.0 / 3.0,
            "mach_threshold_loose": 1.05,
            "min_physical_mach": 1.7,
            "grad_p_filter_quantile": 0.20,
            "march_cells": 6,
        },
        "nt_params": {
            "gamma": 4.0 / 3.0,
            "x_inj": 3.5,
            "xi_max": 0.05,
            "sigma_crit": 0.1,   # 由外部覆盖
            "alpha_sigma": 2,    # 由外部覆盖
        },
        "physics": {
            "spin": 0.98,
            "hslope": 1.0,
            "R0": 0.0,
            "enable_advection": True,
            "cooling_factor": 50.0,
            "advection_steps": 2000,
            "M_unit": 1e25,        # 代码单位质量标度 [g]（用于 RHO_unit 换算）
            "MBH_solar": 6.2e9,    # M87 黑洞质量 [太阳质量]（用于 L_unit 换算）
        },
        "max_concurrent_tasks": 1,
    }
# ══════════════════════════════════════════════════════════════════════════════


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def _log(msg, logger=None, level=logging.INFO):
    """同时打印到控制台（用于 tee 捕获）和 Python logger。"""
    print(msg, flush=True)
    if logger:
        logger.log(level, msg)


def run_ipole(input_h5, output_h5, ipole_bin, emission_type=None, logger=None):
    """调用 ipole，返回耗时（秒）。ipole 的 verbose 输出会打印到 stdout（被 tee 捕获）。"""
    params = IPOLE_PARAMS.copy()
    params['dump']    = input_h5
    params['outfile'] = output_h5
    if emission_type is not None:
        params['emission_type'] = emission_type

    label = os.path.basename(output_h5)
    _log(f"  [ipole] START: {label}  (emission_type={emission_type})", logger)
    t0 = time.time()
    ipole_api.run(params, exe=ipole_bin, verbose=2)
    elapsed = time.time() - t0
    _log(f"  [ipole] DONE : {label}  ({elapsed:.1f}s)", logger)
    return elapsed


def prepare_input(source_h5, target_h5, zero_kel=False, logger=None):
    """复制 h5 输入文件；zero_kel=True 时将 KEL 数组清零（禁用 DSA）。"""
    shutil.copy2(source_h5, target_h5)
    if zero_kel:
        with h5py.File(target_h5, 'r+') as f:
            if 'KEL' in f:
                f['KEL'][...] = np.zeros_like(f['KEL'][:])
        _log(f"  KEL zeroed → {os.path.basename(target_h5)}", logger)


def load_intensity(h5_file, logger=None):
    """读取 ipole 输出图像，返回强度数组（Jy）。文件缺失时返回全零数组。"""
    ny, nx = IPOLE_PARAMS['ny'], IPOLE_PARAMS['nx']
    if not os.path.exists(h5_file):
        _log(f"  Warning: output not found: {h5_file}", logger, logging.WARNING)
        return np.zeros((ny, nx))

    with h5py.File(h5_file, 'r') as f:
        if 'pol' in f:
            data = f['pol'][:, :, 0]
        elif 'unpol' in f:
            data = f['unpol'][:]
        else:
            _log(f"  Warning: no intensity data in {os.path.basename(h5_file)}", logger, logging.WARNING)
            return np.zeros((ny, nx))

        scale = 1.0
        for key in ('scale', 'header/scale'):
            if key in f:
                scale = float(f[key][()])
                break

    return data * scale


# ── Phase 1: 生成每个 sigma 对应的 h5 ────────────────────────────────────────

def phase1_generate_h5(snapshot_file, sigma_values, alpha_sigma, output_dir, logger):
    """
    为每个 sigma_crit 值调用完整 DSA pipeline，生成对应的 ipole 输入 h5。
    返回 {sigma_crit: h5_path} 字典。
    """
    snap_base = os.path.basename(snapshot_file).replace('.athdf', '')
    sigma_h5_map = {}

    for sc in sigma_values:
        sigma_tag  = f"sigma_{sc:.3f}"
        sigma_dir  = os.path.join(output_dir, sigma_tag)
        os.makedirs(sigma_dir, exist_ok=True)

        # 每个 sigma 有独立的子日志
        sub_logger, sub_log = setup_logging(
            log_dir=os.path.join(sigma_dir, 'logs'),
            log_level=logging.INFO,
            log_name=f"generate_{snap_base}.log",
            logger_name=f'Gen_{sigma_tag}',
        )
        _log(f"\n[Phase 1] Generating h5 for {sigma_tag}  (alpha={alpha_sigma})...", logger)

        cfg = copy.deepcopy(_base_pipeline_config())
        cfg['output_directory']          = sigma_dir
        cfg['nt_params']['sigma_crit']   = sc
        cfg['nt_params']['alpha_sigma']  = alpha_sigma

        try:
            success = process_snapshot(snapshot_file, cfg, sub_logger)
            if not success:
                raise RuntimeError("process_snapshot returned False")

            expected_h5 = os.path.join(sigma_dir, 'ipole_inputs', f"{snap_base}_dsa_input.h5")
            if not os.path.exists(expected_h5):
                raise FileNotFoundError(f"Expected h5 not found: {expected_h5}")

            sigma_h5_map[sc] = expected_h5
            _log(f"  [Phase 1] OK: {sigma_tag} → {os.path.basename(expected_h5)}", logger)

        except Exception as e:
            _log(f"  [Phase 1] FAILED {sigma_tag}: {e}", logger, logging.ERROR)
            if logger:
                logger.exception(e)

    return sigma_h5_map


# ── Phase 2: 共享的 Model A & B（只跑一次）───────────────────────────────────

def phase2_shared_models(any_h5, snap_base, output_dir, logger):
    """
    用 KEL=0 的副本运行 Model A（热）和 Model B（重联），结果存入 shared/。
    返回 (img_A_path, img_B_path, elapsed_dict)。
    """
    shared_dir = os.path.join(output_dir, 'shared')
    os.makedirs(shared_dir, exist_ok=True)

    h5_ab = os.path.join(shared_dir, 'input_AB_kel0.h5')
    prepare_input(any_h5, h5_ab, zero_kel=True, logger=logger)

    img_A = os.path.join(shared_dir, f'img_A_thermal_{snap_base}.h5')
    img_B = os.path.join(shared_dir, f'img_B_reconnection_{snap_base}.h5')
    elapsed = {}

    _log("\n[Phase 2] Model A: Thermal Only (emission_type=1)", logger)
    try:
        elapsed['A'] = run_ipole(h5_ab, img_A, IPOLE_STD, emission_type=1, logger=logger)
    except Exception as e:
        _log(f"  [Phase 2] Model A FAILED: {e}", logger, logging.ERROR)
        if logger: logger.exception(e)
        elapsed['A'] = None

    _log("\n[Phase 2] Model B: Reconnection B² (emission_type=3)", logger)
    try:
        elapsed['B'] = run_ipole(h5_ab, img_B, IPOLE_STD, emission_type=3, logger=logger)
    except Exception as e:
        _log(f"  [Phase 2] Model B FAILED: {e}", logger, logging.ERROR)
        if logger: logger.exception(e)
        elapsed['B'] = None

    return img_A, img_B, elapsed


# ── Phase 3: Model C 每个 sigma 各跑一次 ─────────────────────────────────────

def phase3_dsa_models(sigma_h5_map, snap_base, output_dir, logger):
    """
    对每个 sigma_crit，用保留 KEL 的 h5 运行 Model C（ipole-DSA）。
    返回 {sigma_crit: img_C_path}，elapsed_dict。
    """
    sigma_img_map = {}
    elapsed = {}

    _log("\n[Phase 3] Model C: DSA Shock (ipole-DSA, emission_type=None)", logger)

    for sc, src_h5 in sigma_h5_map.items():
        sigma_tag = f"sigma_{sc:.3f}"
        sigma_dir = os.path.join(output_dir, sigma_tag)

        h5_c   = os.path.join(sigma_dir, f'input_C_shock_{sigma_tag}.h5')
        img_c  = os.path.join(sigma_dir, f'img_C_shock_{snap_base}_{sigma_tag}.h5')

        _log(f"\n  [Phase 3] Model C [{sigma_tag}]", logger)
        try:
            prepare_input(src_h5, h5_c, zero_kel=False, logger=logger)
            elapsed[f'C_{sigma_tag}'] = run_ipole(h5_c, img_c, IPOLE_DSA,
                                                   emission_type=None, logger=logger)
            sigma_img_map[sc] = img_c
        except Exception as e:
            _log(f"  [Phase 3] FAILED [{sigma_tag}]: {e}", logger, logging.ERROR)
            if logger: logger.exception(e)
            sigma_img_map[sc] = None
            elapsed[f'C_{sigma_tag}'] = None

    return sigma_img_map, elapsed


# ── Phase 4: 最终对比图 ────────────────────────────────────────────────────────

def phase4_comparison_figure(img_A, img_B, sigma_img_map, sigma_values,
                             alpha_sigma, snap_base, output_dir, timestamp, logger):
    """
    生成 2 行对比图：
      Row 1: A | B | C_σ1 | C_σ2 | C_σ3   （对数色标，统一范围）
      Row 2: (C_σ1-A)/A | (C_σ2-A)/A | (C_σ3-A)/A | (C_σ1-B)/B   （相对差分）
    """
    _log("\n[Phase 4] Generating comparison figure...", logger)

    n_sigma  = len(sigma_values)
    n_cols   = 2 + n_sigma          # A, B, C×N
    fig, axes = plt.subplots(2, n_cols, figsize=(4.5 * n_cols, 9))

    # ── 读取图像 ──────────────────────────────────────────────────────────────
    img_a = load_intensity(img_A, logger)
    img_b = load_intensity(img_B, logger)
    imgs_c = []
    for sc in sigma_values:
        path = sigma_img_map.get(sc)
        imgs_c.append(load_intensity(path, logger) if path else
                      np.zeros_like(img_a))

    # ── Row 1: 绝对强度（对数归一化，统一色标）────────────────────────────────
    all_imgs = [img_a, img_b] + imgs_c
    valid_max = [np.max(im) for im in all_imgs if np.max(im) > 0]
    vmax = max(valid_max) if valid_max else 1.0
    vmin = vmax * 1e-4
    norm_log = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    row1_imgs   = all_imgs
    row1_titles = (['A: Thermal Only', 'B: Reconnection (B²)'] +
                   [f'C: DSA  σ_crit={sc}' for sc in sigma_values])

    for ax, img, title in zip(axes[0], row1_imgs, row1_titles):
        im = ax.imshow(img, cmap='afmhot', origin='lower', norm=norm_log)
        ax.set_title(title, fontsize=9, pad=4)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)

    # ── Row 2: 相对差分 ────────────────────────────────────────────────────────
    # 前 N 列：(C_σi - A) / A   →  shows net DSA contribution relative to thermal
    # 最后一列（若 N < n_cols-2 补空白）：(C_σ1 - B) / B
    diff_titles = [f'(C σ={sc} − A) / A' for sc in sigma_values]
    diff_imgs   = [np.divide(imgs_c[i] - img_a, img_a + 1e-30 * vmax)
                   for i in range(n_sigma)]

    # 最后一列：C_σ1 对比 B
    if n_sigma > 0:
        diff_imgs.append(np.divide(imgs_c[0] - img_b, img_b + 1e-30 * vmax))
        diff_titles.append(f'(C σ={sigma_values[0]} − B) / B')
    # 剩余空白列（若 n_sigma < n_cols - 1）
    while len(diff_imgs) < n_cols:
        diff_imgs.append(None)
        diff_titles.append('')

    diff_abs_max = max((np.nanpercentile(np.abs(d), 99) for d in diff_imgs if d is not None),
                       default=1.0)
    norm_div = mcolors.TwoSlopeNorm(vmin=-diff_abs_max, vcenter=0, vmax=diff_abs_max)

    for ax, d, title in zip(axes[1], diff_imgs, diff_titles):
        if d is None:
            ax.axis('off')
            continue
        im = ax.imshow(d, cmap='RdBu_r', origin='lower', norm=norm_div)
        ax.set_title(title, fontsize=9, pad=4)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)

    # ── 保存 ────────────────────────────────────────────────────────────────────
    fig.suptitle(
        f'Sigma Sweep: {snap_base}  |  α={alpha_sigma}  |  '
        f'σ_crit ∈ {sigma_values}',
        fontsize=11, y=1.01
    )
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'sweep_{snap_base}_{timestamp}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    _log(f"  Figure saved: {plot_path}", logger)
    return plot_path


# ── 主函数 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='σ参数扫描：单快照 × N 组 sigma_crit × 3 种辐射模型'
    )
    parser.add_argument('--snapshot',     required=True,
                        help='输入 .athdf 文件路径')
    parser.add_argument('--sigma-values', nargs='+', type=float,
                        default=[0.01, 0.03, 0.1],
                        help='要测试的 sigma_crit 列表（默认: 0.01 0.03 0.1）')
    parser.add_argument('--alpha-sigma',  type=float, default=2,
                        help='压低函数陡峭指数 alpha，须 >= 2（默认: 2）')
    parser.add_argument('--output-dir',   required=True,
                        help='输出根目录')
    args = parser.parse_args()

    snapshot     = args.snapshot
    sigma_values = sorted(args.sigma_values)
    alpha_sigma  = args.alpha_sigma
    output_dir   = args.output_dir
    snap_base    = os.path.basename(snapshot).replace('.athdf', '')
    timestamp    = time.strftime('%Y%m%d_%H%M%S')

    os.makedirs(output_dir, exist_ok=True)

    # 主日志：同时写文件 + 控制台（控制台部分被 tee 捕获进 screen log）
    logger, log_file = setup_logging(
        log_dir=output_dir,
        log_level=logging.INFO,
        log_name=f"sweep_{snap_base}_{timestamp}.log",
        logger_name='SigmaSweep',
    )

    _log("=" * 70, logger)
    _log(f"SIGMA SWEEP START", logger)
    _log(f"  Snapshot     : {snapshot}", logger)
    _log(f"  Sigma values : {sigma_values}", logger)
    _log(f"  Alpha        : {alpha_sigma}", logger)
    _log(f"  Output dir   : {output_dir}", logger)
    _log(f"  Structured log: {log_file}", logger)
    _log("=" * 70, logger)

    t_global = time.time()
    elapsed_all = {}

    # ── Phase 1 ────────────────────────────────────────────────────────────────
    _log("\n" + "─" * 60, logger)
    _log("Phase 1 / 4 — Generate h5 files for each sigma", logger)
    _log("─" * 60, logger)
    sigma_h5_map = phase1_generate_h5(snapshot, sigma_values, alpha_sigma,
                                      output_dir, logger)

    if not sigma_h5_map:
        _log("ABORT: No h5 files generated successfully.", logger, logging.ERROR)
        sys.exit(1)

    # ── Phase 2 ────────────────────────────────────────────────────────────────
    _log("\n" + "─" * 60, logger)
    _log("Phase 2 / 4 — Shared models A & B (computed once)", logger)
    _log("─" * 60, logger)
    any_h5 = next(iter(sigma_h5_map.values()))
    img_A, img_B, elapsed_AB = phase2_shared_models(any_h5, snap_base,
                                                    output_dir, logger)
    elapsed_all.update(elapsed_AB)

    # ── Phase 3 ────────────────────────────────────────────────────────────────
    _log("\n" + "─" * 60, logger)
    _log("Phase 3 / 4 — Model C (DSA) for each sigma", logger)
    _log("─" * 60, logger)
    sigma_img_map, elapsed_C = phase3_dsa_models(sigma_h5_map, snap_base,
                                                 output_dir, logger)
    elapsed_all.update(elapsed_C)

    # ── Phase 4 ────────────────────────────────────────────────────────────────
    _log("\n" + "─" * 60, logger)
    _log("Phase 4 / 4 — Comparison figure", logger)
    _log("─" * 60, logger)
    plot_path = phase4_comparison_figure(
        img_A, img_B, sigma_img_map, sigma_values,
        alpha_sigma, snap_base, output_dir, timestamp, logger
    )

    # ── 汇总 ──────────────────────────────────────────────────────────────────
    total_time = time.time() - t_global
    _log("\n" + "=" * 70, logger)
    _log("SIGMA SWEEP COMPLETE", logger)
    _log(f"  Total wall time  : {total_time:.1f}s  ({total_time/60:.1f} min)", logger)
    _log("  Timing breakdown :", logger)
    for k, v in elapsed_all.items():
        status = f"{v:.1f}s" if v is not None else "FAILED"
        _log(f"    {k:<35} {status}", logger)
    _log(f"\n  Comparison figure: {plot_path}", logger)
    _log(f"  Structured log   : {log_file}", logger)
    _log(f"  Output directory : {output_dir}", logger)
    _log("=" * 70, logger)

    # Flux 汇总（方便快速查阅）
    _log("\nFLUX STATISTICS (Jy):", logger)
    img_a = load_intensity(img_A, logger)
    img_b = load_intensity(img_B, logger)
    _log(f"  Model A (Thermal)       : {np.sum(img_a):.4e}", logger)
    _log(f"  Model B (Reconnection)  : {np.sum(img_b):.4e}", logger)
    for sc in sigma_values:
        path = sigma_img_map.get(sc)
        img_c = load_intensity(path, logger) if path else None
        flux  = f"{np.sum(img_c):.4e}" if img_c is not None else "N/A"
        _log(f"  Model C (σ_crit={sc:<5}) : {flux}", logger)


if __name__ == '__main__':
    main()
