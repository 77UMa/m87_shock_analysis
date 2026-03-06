'''
科学计算模块的一部分，负责非热电子物理，按照Xia et al. (2025) 附录A进行计算
'''
import numpy as np
import h5py
import sys
import os
# 您需要在脚本开头增加这个导入，用于计算不完全贝塔函数
from scipy.special import betainc
import pdb
# --------------------------------------------------------------------------

def calculate_nonthermal_electrons(shock_properties, gamma=4.0/3.0, x_inj=3.5, xi_max=0.05):
    """
    严格按照 Xia et al. (2025) 附录A 的公式计算非热电子能谱参数。
    注入效率不再是固定值，而是由激波物理动态决定。
    
    Args:
        shock_properties (dict): find_shocks_in_roi 函数返回的字典。
        gamma (float): 绝热指数。
        x_inj (float): 注入参数 (论文中为 3.3 到 3.6)。
        xi_max (float): 允许非热电子占总能量增加的最高比例 (论文中为 0.05)。
        
    Returns:
        dict: 包含 'q_grid' 和 'C_grid' 的字典。
    """
    print("Starting non-thermal electron calculation (Full Physics Model)...")
    
    # --- 步骤 0: 解包输入数据 ---
    mask = shock_properties["mask"]
    M1 = shock_properties["upstream_mach"]
    T2 = shock_properties["downstream_temp"]
    n_e2 = shock_properties["downstream_n_e"]
    
    # 物理常数 (cgs units)
    M_E, C_LIGHT, K_B = 9.1094e-28, 2.9979e10, 1.3806e-16

    q_grid = np.zeros_like(mask, dtype=float)
    C_grid = np.zeros_like(mask, dtype=float)
    
    if np.any(mask):
        M1_shocks = M1[mask]
        T2_shocks = T2[mask]
        n_e2_shocks = n_e2[mask]
        
        # --- 步骤 1: 计算谱指数 q (与之前相同) ---
        # 根据论文公式(A.5)计算压缩比τ
        inv_tau = (gamma - 1.0) / (gamma + 1.0) + (2.0 / (gamma + 1.0)) / M1_shocks**2
        tau = 1.0 / inv_tau
        # 根据论文公式(A.4)计算谱指数q
        q = (tau + 2.0) / (tau - 1.0)
        q_grid[mask] = q
        print("  Power-law index 'q' calculated.")

        # --- 步骤 2: 计算注入动量 p_min (论文公式 A.2) ---
        thermal_term = 2.0 * K_B * T2_shocks / (M_E * C_LIGHT**2)
        p_min_sq = x_inj**2 * thermal_term
        p_min = np.sqrt(p_min_sq)
        
        # --- 步骤 3: 计算线性注入效率 η_lin (论文公式 A.10) ---
        eta_lin = (4.0 / np.sqrt(np.pi)) * (x_inj**3 / (q - 1.0)) * np.exp(-x_inj**2)
        print("  Linear injection fraction 'eta_lin' calculated.")
        
        # --- 步骤 4: 计算注入粒子的平均动能 K_inj (论文公式 A.8) ---
        # K_inj 包含一个不完全贝塔函数 B_x(a,b)
        # scipy.special.betainc(a, b, x) * beta(a,b)
        # 论文中的 B_x(a,b) 对应 scipy.special.betainc(a,b,x)
        # 注意：这个公式只在 q > 2 和 q < 3 时严格有效
        valid_q_mask = (q > 2.0) & (q < 3.0)
        K_inj = np.zeros_like(q)
        if np.any(valid_q_mask):
            q_valid = q[valid_q_mask]
            p_min_sq_valid = p_min_sq[valid_q_mask]
            
            x_beta = 1.0 / (1.0 + p_min_sq_valid)
            a_beta = (q_valid - 2.0) / 2.0
            b_beta = (3.0 - q_valid) / 2.0
            
            incomplete_beta_func = betainc(a_beta, b_beta, x_beta)
            
            term1 = (p_min[valid_q_mask]**(q_valid - 1.0)) / 2.0
            term2 = np.sqrt(1.0 + p_min_sq_valid) - 1.0
            K_inj[valid_q_mask] = term1 * incomplete_beta_func + term2
        print("  Mean kinetic energy 'K_inj' calculated.")
        
        # --- 步骤 5: 计算线性总能量比 ξ_lin (论文公式 A.11) ---
        # 假设 T1 << T2，则 ΔE_th ≈ E_th_downstream
        E_nonthermal = eta_lin * K_inj * n_e2_shocks * M_E * C_LIGHT**2
        E_thermal_increase = (3.0) * n_e2_shocks * K_B * T2_shocks # 简化近似
        
        # 为避免除以零
        xi_lin = np.divide(E_nonthermal, E_thermal_increase, out=np.zeros_like(E_nonthermal), where=E_thermal_increase!=0)
        
        # --- 步骤 6: 计算归一化常数 C (论文公式 A.1, A.12) ---
        delta = xi_lin / xi_max
        
        # f_e(p) 是麦克斯韦-玻尔兹曼动量分布的概率密度函数
        f_e_p_min = (4.0 * np.pi * n_e2_shocks * p_min**2 * (M_E * C_LIGHT**2 / (2.0 * np.pi * K_B * T2_shocks))**1.5 * np.exp(-M_E * C_LIGHT**2 * p_min_sq / (2.0 * K_B * T2_shocks)))
        
        norm_factor = np.divide(1.0 - np.exp(-delta), delta, out=np.zeros_like(delta), where=delta!=0)
        # 处理delta=0的极限情况，此时 norm_factor -> 1
        norm_factor[delta == 0] = 1.0
        
        C = norm_factor * f_e_p_min * p_min**q
        # 计算总数密度 N_inj (论文 A.6 式)
        N_inj = (C * p_min**(1.0 - q)) / (q - 1.0)
        C_grid[mask] = N_inj  # 现在 C_grid 存储的是真正的电子密度N_inj

        print("  Final normalization 'C' calculated using full physics model.")

    nonthermal_properties = {"q_grid": q_grid,"C_grid": C_grid,"mask": mask}
    return nonthermal_properties

import matplotlib
matplotlib.use('Agg') # 必须在 pyplot 导入前设置，用于无GUI的服务器
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LogNorm

def plot_diagnostic_histograms(shock_properties, nonthermal_props, snapshot_name, output_filename):

    """
    绘制1D统计直方图，用于诊断 M1, q 和 log(C) 的分布。
    """
    print(f"Generating non-thermal 1D diagnostic histograms for {snapshot_name}...")
    mask = shock_properties["mask"]
    if np.sum(mask) == 0:
        print("  No shock cells found. Skipping 1D histograms.")
        return

    M1 = shock_properties["upstream_mach"][mask]
    q = nonthermal_props["q_grid"][mask]
    C = nonthermal_props["C_grid"][mask]
    logC = np.log10(C[C > 0]) # 仅对 C > 0 的值取对数

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle(f"Non-Thermal Electron Diagnostics (1D Histograms) for {snapshot_name}", fontsize=16)

    # 1. 马赫数 M1 直方图
    ax1.hist(M1, bins=50, color='blue', alpha=0.7, log=True)
    ax1.set_title("Upstream Mach Number ($M_1$) Distribution")
    ax1.set_xlabel("Mach Number ($M_1$)")
    ax1.set_ylabel("Count (log scale)")
    ax1.axvline(M1.mean(), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {M1.mean():.2f}')
    ax1.legend()

    # 2. 谱指数 q 直方图
    ax2.hist(q, bins=50, color='green', alpha=0.7, log=True)
    ax2.set_title("Power-law Index (q) Distribution")
    ax2.set_xlabel("Index (q)")
    ax2.set_ylabel("Count (log scale)")
    ax2.axvline(q.mean(), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {q.mean():.2f}')
    ax2.axvline(1.5, color='black', linestyle='dotted', linewidth=2, label='q_min (M->inf) = 1.5') # [cite: 5344-5347]
    ax2.legend()

    # 3. 归一化 log(C) 直方图
    if len(logC) > 0:
        ax3.hist(logC, bins=50, color='purple', alpha=0.7, log=True)
        ax3.set_title("Nonthermal Electron Density (log10 N) Distribution")
        ax3.set_xlabel("log10(N)")
        ax3.set_ylabel("Count (log scale)")
        ax3.axvline(logC.mean(), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {logC.mean():.2f}')
        ax3.legend()
    else:
        ax3.set_title("Normalization (log10 N) Distribution")
        ax3.text(0.5, 0.5, "No C > 0 values found", horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  1D diagnostic histograms saved to {output_filename}")

def plot_diagnostic_correlations(shock_properties, nonthermal_props, snapshot_name, output_filename):
    """
    绘制2D相关性图 (hexbin)，用于验证 M1 vs q 和 M1 vs log(C) 的物理联系。
    """
    print(f"Generating non-thermal 2D diagnostic correlations for {snapshot_name}...")
    mask = shock_properties["mask"]
    if np.sum(mask) == 0:
        print("  No shock cells found. Skipping 2D correlations.")
        return

    M1 = shock_properties["upstream_mach"][mask]
    q = nonthermal_props["q_grid"][mask]
    C = nonthermal_props["C_grid"][mask]
    logC = np.log10(C[C > 0])
    M1_for_C = M1[C > 0] # 确保 M1 和 logC 数组对齐

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"Nonthermal Electron Diagnostics (2D Correlations) for {snapshot_name}", fontsize=16)

    # [cite_start]1. M1 vs q (物理检验) [cite: 5344-5347]
    # 预期：M1 越大，q 越小，且 q 趋近于 1.5
    hb1 = ax1.hexbin(M1, q, gridsize=50, cmap='viridis', norm=LogNorm())
    fig.colorbar(hb1, ax=ax1, label='Count (log scale)')
    ax1.set_title("Physical Check: $M_1$ vs. $q$")
    ax1.set_xlabel("Upstream Mach Number ($M_1$)")
    ax1.set_ylabel("Power-law Index ($q$)")
    ax1.set_ylim(bottom=1.4) # 聚焦于物理最小值 1.5 附近
    ax1.axhline(1.5, color='red', linestyle='dashed', linewidth=2, label='q_min (M->inf) = 1.5')
    ax1.legend()

    # 2. M1 vs log(C) (注入效率检验)
    if len(logC) > 0:
        hb2 = ax2.hexbin(M1_for_C, logC, gridsize=50, cmap='inferno', norm=LogNorm())
        fig.colorbar(hb2, ax=ax2, label='Count (log scale)')
        ax2.set_title("Injection Check: $M_1$ vs. $log_{10}(N)$")
        ax2.set_xlabel("Upstream Mach Number ($M_1$)")
        ax2.set_ylabel("Log10(Normalization N)")
    else:
        ax2.set_title("Injection Check: $M_1$ vs. $log_{10}(N)$")
        ax2.text(0.5, 0.5, "No C > 0 values found", horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  2D diagnostic correlations saved to {output_filename}")
# --- 主程序入口 ---
if __name__ == '__main__':

 
    test_file = 'F:\\Research\\Shockwave\\data_test\\mad98.prim.00200.athdf' 
