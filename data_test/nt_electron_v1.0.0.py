import numpy as np
import h5py
import sys
import os
# 您需要在脚本开头增加这个导入，用于计算不完全贝塔函数
from scipy.special import betainc
# --- 在您的主脚本顶部，除了之前的导入，还需要导入 read_data ---
import pdb
# --------------------------------------------------------------------------


def calculate_nonthermal_electrons(shock_properties, gamma=4.0/3.0, x_inj=3.5, xi_max=0.05):
    """
    【完整物理版】严格按照 Xia et al. (2010) 附录A 的公式计算非热电子能谱参数。
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
        E_thermal_increase = (3.0/2.0) * n_e2_shocks * K_B * T2_shocks # 简化近似
        
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
        C_grid[mask] = C
        print("  Final normalization 'C' calculated using full physics model.")

    nonthermal_properties = {"q_grid": q_grid,"C_grid": C_grid,"mask": mask}
    return nonthermal_properties


# --- 主程序入口 ---
if __name__ == '__main__':

 
    test_file = 'F:\\Research\\Shockwave\\data_test\\mad98.prim.00200.athdf' 
