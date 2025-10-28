import numpy as np
import h5py
import sys
import os
# 您需要在脚本开头增加这个导入，用于计算不完全贝塔函数
from scipy.special import betainc
# --- 在您的主脚本顶部，除了之前的导入，还需要导入 read_data ---
from pyathena.read_data import read_data
from pyathena.metric import kerr_schild
import pdb
from sklearn.cluster import DBSCAN
# --------------------------------------------------------------------------
# 步骤 0: 将 pyathena 文件夹的路径添加到Python的搜索路径中
# 假设您的 pyathena 文件夹与您的分析脚本在同一个父目录下
# 请根据您的实际文件结构修改
script_dir = os.path.dirname(os.path.abspath(__file__))
pyathena_dir = os.path.join(script_dir, '..', 'pyathena') 
sys.path.insert(0, pyathena_dir)

from pyathena import athena_read # 现在可以成功导入了
# --------------------------------------------------------------------------
#q在2到3之间
def find_shocks_in_roi(roi_data, gamma=4.0/3.0, mach_threshold=1.1, grad_p_filter_quantile=0.20):
    """
    在验证激波后，记录并返回其物理性质。(旧版本，牛顿框架下)
    """
    print("Starting Hybrid shock detection (L&H + Entropy Condition)...")
    
    # --- 准备工作 (与之前相同) ---
    press = roi_data['press']
    nk, nj, ni = press.shape
    rho = roi_data['rho']
    vel1, vel2, vel3 = roi_data['vel1'], roi_data['vel2'], roi_data['vel3']
    r_coords = (roi_data['x1f'][:-1] + roi_data['x1f'][1:]) / 2.0
    theta_coords = (roi_data['x2f'][:-1] + roi_data['x2f'][1:]) / 2.0
    phi_coords = (roi_data['x3f'][:-1] + roi_data['x3f'][1:]) / 2.0
    phi_grid, theta_grid, r_grid = np.meshgrid(phi_coords, theta_coords, r_coords, indexing='ij')

    print(f"grid: {ni}×{nj}×{nk}")
    # ... (计算 sound_speed, mach_vec, grad_P 的代码不变) ...
    print("  Step 1 & 2: Calculating base quantities and pressure gradient...")
    sound_speed = np.sqrt(gamma * press / rho)
    mach_vec_r, mach_vec_theta, mach_vec_phi = vel1/sound_speed, vel2/sound_speed, vel3/sound_speed
    grad_P_phi_comp, grad_P_theta_comp, grad_P_r_comp = np.gradient(press, phi_coords, theta_coords, r_coords)
    grad_P_r = grad_P_r_comp
    grad_P_theta = (1.0 / r_grid) * grad_P_theta_comp
    grad_P_phi = (1.0 / (r_grid * np.sin(theta_grid))) * grad_P_phi_comp
    grad_P_mag = np.sqrt(grad_P_r**2 + grad_P_theta**2 + grad_P_phi**2) + 1e-30
    
    # --- 步骤 3: L&H方法快速筛选 (不变) ---
    print("  Step 3: Pre-screening candidates with L&H method...")
    dot_product = mach_vec_r * grad_P_r + mach_vec_theta * grad_P_theta + mach_vec_phi * grad_P_phi
    normal_mach = dot_product / grad_P_mag
    candidate_mask = (normal_mach >= mach_threshold) & (dot_product > 0)
    if grad_p_filter_quantile > 0:
        grad_p_threshold = np.quantile(grad_P_mag, grad_p_filter_quantile)
        candidate_mask &= (grad_P_mag > grad_p_threshold)
    num_candidates = np.sum(candidate_mask)
    print(f"  Found {num_candidates} candidate shock cells for verification.")

    # --- 步骤 4: 熵增验证并记录物理性质 ---
    print("  Step 4: Verifying candidates and storing properties...")
    final_shock_mask = np.zeros_like(press, dtype=bool)
    # 【新增】: 初始化用于存储物理性质的数组
    compression_ratio_grid = np.zeros_like(press)
    downstream_temp_grid = np.zeros_like(press)
    downstream_ne_grid = np.zeros_like(press)
    # 【新增】: 初始化用于存储物理性质的数组
    upstream_mach_grid = np.zeros_like(press) # 上游法向马赫数 M₁
  
    # 物理常数 (cgs units)
    M_P = 1.6726e-24 # 质子质量 (g)
    K_B = 1.3806e-16 # 玻尔兹曼常数 (erg/K)

    candidate_indices = np.argwhere(candidate_mask)
    for k, j, i in candidate_indices:
        if not (0 < k < nk-1 and 0 < j < nj-1 and 0 < i < ni-1): continue
            
        P2, rho2, v2_vec = press[k,j,i], rho[k,j,i], np.array([vel1[k,j,i], vel2[k,j,i], vel3[k,j,i]])
        
        min_pressure, upstream_neighbor = P2, None
        for dk, dj, di in [(0,0,-1), (0,0,1), (0,-1,0), (0,1,0), (-1,0,0), (1,0,0)]:
            p_neighbor = press[k+dk, j+dj, i+di]
            if p_neighbor < min_pressure:
                min_pressure = p_neighbor
                upstream_neighbor = (k+dk, j+dj, i+di)
        
        if upstream_neighbor is None: continue
            
        ku, ju, iu = upstream_neighbor
        P1, rho1, v1_vec = press[ku,ju,iu], rho[ku,ju,iu], np.array([vel1[ku,ju,iu], vel2[ku,ju,iu], vel3[ku,ju,iu]])
        
        n_vec = np.array([grad_P_r[k,j,i], grad_P_theta[k,j,i], grad_P_phi[k,j,i]]) / grad_P_mag[k,j,i]
        a2, a1 = np.sqrt(gamma * P2 / rho2), np.sqrt(gamma * P1 / rho1)
        u2, u1 = np.dot(v2_vec, n_vec), np.dot(v1_vec, n_vec)

        if u1 <= u2: continue

        pressure_ratio = P2 / P1
        u_sh = u1 + a1 * np.sqrt(((gamma + 1)/(2*gamma))*pressure_ratio + (gamma - 1)/(2*gamma))
        
        if (u1 + a1) < u_sh < (u2 + a2):
            final_shock_mask[k,j,i] = True

            # 【核心修正】: 记录上游法向马赫数 M₁
            mach_1 = np.abs(u1 / a1)
            upstream_mach_grid[k,j,i] = mach_1
                        
            mu = 0.5 # 对完全电离的氢等离子体 存疑？
            downstream_temp_grid[k,j,i] = (P2 * mu * M_P) / (rho2 * K_B)
            downstream_ne_grid[k,j,i] = rho2 / M_P

    print(f"Shock detection finished. Verified {np.sum(final_shock_mask)} shock cells.")
    
    # 【新增】: 返回一个包含所有结果的字典
    shock_properties = {
        "mask": final_shock_mask,
        "upstream_mach": upstream_mach_grid,
        "downstream_temp": downstream_temp_grid,
        "downstream_n_e": downstream_ne_grid
    }
    return shock_properties

# 在 nt_electron_v2.py 中
def find_shocks_in_roi_classic(roi_data, gamma=4.0/3.0, mach_threshold=1.7, grad_p_filter_quantile=0.20):
    """
    【最终优化版 v2】
    - 返回值中新增了激波位置的压力梯度大小 'grad_p_mag'，用于更精确的可视化诊断。
    """
    # ... (从函数开始到 candidate_indices = np.argwhere(candidate_mask) 的代码完全不变) ...
    print("Starting Final Classic shock detection (Velocity-based screening, Pressure-based calculation)...")
    press, rho = roi_data['press'], roi_data['rho']
    nk, nj, ni = press.shape
    vel1, vel2, vel3 = roi_data['vel1'], roi_data['vel2'], roi_data['vel3']
    r_coords = (roi_data['x1f'][:-1] + roi_data['x1f'][1:]) / 2.0
    theta_coords = (roi_data['x2f'][:-1] + roi_data['x2f'][1:]) / 2.0
    phi_coords = (roi_data['x3f'][:-1] + roi_data['x3f'][1:]) / 2.0
    phi_grid, theta_grid, r_grid = np.meshgrid(phi_coords, theta_coords, r_coords, indexing='ij')
    print("  Step 1: Screening candidates with local normal Mach number...")
    sound_speed = np.sqrt(gamma * press / rho)
    mach_vec_r, mach_vec_theta, mach_vec_phi = vel1/(sound_speed+1e-30), vel2/(sound_speed+1e-30), vel3/(sound_speed+1e-30)
    grad_P_phi_comp, grad_P_theta_comp, grad_P_r_comp = np.gradient(press, phi_coords, theta_coords, r_coords)
    grad_P_r, grad_P_theta = grad_P_r_comp, (1.0 / r_grid) * grad_P_theta_comp
    grad_P_phi = (1.0 / (r_grid * np.sin(theta_grid) + 1e-30)) * grad_P_phi_comp
    grad_P_mag = np.sqrt(grad_P_r**2 + grad_P_theta**2 + grad_P_phi**2) + 1e-30
    dot_product = mach_vec_r * grad_P_r + mach_vec_theta * grad_P_theta + mach_vec_phi * grad_P_phi
    normal_mach = dot_product / grad_P_mag
    candidate_mask = (normal_mach >= mach_threshold) & (dot_product > 0)
    if grad_p_filter_quantile > 0 and np.any(grad_P_mag > 0):
        grad_p_threshold = np.quantile(grad_P_mag[grad_P_mag > 0], grad_p_filter_quantile)
        candidate_mask &= (grad_P_mag > grad_p_threshold)
    candidate_indices = np.argwhere(candidate_mask)
    print(f"  Found {len(candidate_indices)} candidate shock cells for verification.")

    print("  Step 2: Verifying candidates and calculating properties...")
    final_shock_mask = np.zeros_like(press, dtype=bool)
    upstream_mach_grid = np.zeros_like(press)
    downstream_temp_grid = np.zeros_like(press)
    downstream_ne_grid = np.zeros_like(press)
    # --- 【新增】: 初始化用于存储压力梯度大小的数组 ---
    shock_grad_p_mag_grid = np.zeros_like(press)
    
    M_P, K_B = 1.6726e-24, 1.3806e-16

    for k, j, i in candidate_indices:
        # ... (验证逻辑不变) ...
        if not (0 < k < nk-1 and 0 < j < nj-1 and 0 < i < ni-1): continue
        P2, rho2, v2_vec = press[k,j,i], rho[k,j,i], np.array([vel1[k,j,i], vel2[k,j,i], vel3[k,j,i]])
        min_pressure, upstream_neighbor = P2, None
        for dk, dj, di in [(0,0,-1), (0,0,1), (0,-1,0), (0,1,0), (-1,0,0), (1,0,0)]:
            if not (0 <= k+dk < nk and 0 <= j+dj < nj and 0 <= i+di < ni): continue
            p_neighbor = press[k+dk, j+dj, i+di]
            if p_neighbor < min_pressure:
                min_pressure, upstream_neighbor = p_neighbor, (k+dk, j+dj, i+di)
        if upstream_neighbor is None: continue
        ku, ju, iu = upstream_neighbor
        P1, rho1, v1_vec = press[ku,ju,iu], rho[ku,ju,iu], np.array([vel1[ku,ju,iu], vel2[ku,ju,iu], vel3[ku,ju,iu]])
        n_vec = np.array([grad_P_r[k,j,i], grad_P_theta[k,j,i], grad_P_phi[k,j,i]]) / grad_P_mag[k,j,i]
        a2, a1 = sound_speed[k,j,i], sound_speed[ku,ju,iu]
        u2, u1 = np.dot(v2_vec, n_vec), np.dot(v1_vec, n_vec)
        if u1 <= u2: continue
        try:
            pressure_ratio = P2 / P1
            if pressure_ratio <= 1.0: continue
            u_sh = u1 + a1 * np.sqrt(((gamma + 1)/(2*gamma))*pressure_ratio + (gamma - 1)/(2*gamma))
            if (u1 + a1) < u_sh < (u2 + a2):
                final_shock_mask[k,j,i] = True
                mach_1_sq = 1.0 + (pressure_ratio - 1.0) * (gamma + 1.0) / (2.0 * gamma)
                upstream_mach_grid[k,j,i] = np.sqrt(mach_1_sq)
                mu = 0.5 
                downstream_temp_grid[k,j,i] = (P2 * mu * M_P) / (rho2 * K_B)
                downstream_ne_grid[k,j,i] = rho2 / M_P
                # --- 【新增】: 记录该激波点的压力梯度大小 ---
                shock_grad_p_mag_grid[k,j,i] = grad_P_mag[k,j,i]

        except (ValueError, FloatingPointError):
            continue

    print(f"Final Classic detection finished. Verified {np.sum(final_shock_mask)} shock cells.")
    
    # --- 【新增】: 将压力梯度大小添加到返回的字典中 ---
    return {
        "mask": final_shock_mask,
        "upstream_mach": upstream_mach_grid,
        "downstream_temp": downstream_temp_grid,
        "downstream_n_e": downstream_ne_grid,
        "grad_p_mag": shock_grad_p_mag_grid # 新增的返回项
    }


def find_shocks_in_roi_grmhd(roi_data, gamma=4.0/3.0, mach_threshold=1.7, grad_p_filter_quantile=0.20, spin=0.98):
    """
    【v6】使用“L&H判据”寻找激波位置，在验证激波后，记录并返回其物理性质。
    【考虑GRMHD】: 在克尔度规中处理速度和激波判据
    （旧版本，内积等定义不完善，已弃用）
    """
    print("Starting Hybrid shock detection (L&H + Entropy Condition)...")
    
    # --- 物理常数 (cgs units) ---
    C_LIGHT = 2.99792458e10  # 光速 (cm/s)
    M_P = 1.6726e-24         # 质子质量 (g)
    K_B = 1.3806e-16         # 玻尔兹曼常数 (erg/K)
    M_E = 9.1094e-28         # 电子质量 (g)
    # --- 准备工作 (与之前相同) ---

    press = roi_data['press']
    nk, nj, ni = press.shape
    rho = roi_data['rho']
    vel1, vel2, vel3 = roi_data['vel1'], roi_data['vel2'], roi_data['vel3']
    r_coords = (roi_data['x1f'][:-1] + roi_data['x1f'][1:]) / 2.0
    theta_coords = (roi_data['x2f'][:-1] + roi_data['x2f'][1:]) / 2.0
    phi_coords = (roi_data['x3f'][:-1] + roi_data['x3f'][1:]) / 2.0
    phi_grid, theta_grid, r_grid = np.meshgrid(phi_coords, theta_coords, r_coords, indexing='ij')
    
    # --- 步骤 1: 使用 read_data 类封装数据以进行GRMHD计算 ---
    print("  Step 1: Wrapping data in GRMHD analysis object...")
    # 创建坐标网格
    r_coords_centers = (roi_data['x1f'][:-1] + roi_data['x1f'][1:]) / 2.0
    theta_coords_centers = (roi_data['x2f'][:-1] + roi_data['x2f'][1:]) / 2.0
    phi_coords_centers = (roi_data['x3f'][:-1] + roi_data['x3f'][1:]) / 2.0
    phi_grid, theta_grid, r_grid = np.meshgrid(phi_coords_centers, theta_coords_centers, r_coords_centers, indexing='ij')

    # 实例化度规对象
    metric_obj = kerr_schild(r_grid, theta_grid, phi_grid, a=spin)

    # 创建一个临时的、与 read_data 类兼容的对象
    class TmpData:
        def __init__(self):
            self.rho = roi_data['rho']
            self.prs = roi_data['press']
            self.v1, self.v2, self.v3 = roi_data['vel1'], roi_data['vel2'], roi_data['vel3']
            self.metric = metric_obj
            # 伪造一些 read_data 需要的属性
            self.params = {'dict': {'<hydro>': {'gamma': gamma}}}

    data_obj = TmpData()
    # 将方法绑定到对象上，使其行为像一个 read_data 实例
    data_obj.lorentz_gamma = read_data.lorentz_gamma.__get__(data_obj)
    data_obj.ucon = read_data.ucon.__get__(data_obj) 
    data_obj.ucov = read_data.ucov.__get__(data_obj)

    # 现在我们可以方便地计算GRMHD物理量
    lorentz_gamma = data_obj.lorentz_gamma() #γ=dt/dτ
    u_con_raw = data_obj.ucon() #协变速度
    
    # 【核心修正】: 使用 squeeze() 移除由 read_data.py 引入的多余维度
    u_con = np.squeeze(u_con_raw)

    # --- 步骤 2: 计算压力梯度 (不变) ---
    print("  Step 2: Calculating pressure gradient...")
    sound_speed = np.sqrt(gamma * press / rho) #声速，只对理想气体适用
    speed_sq = C_LIGHT**2 * (1.0 - 1.0 / lorentz_gamma**2)
    # 构造3D马赫数向量 (这是一个近似，但比之前好)？
    mach_vec_r = u_con[1] / sound_speed # u^r / c_s
    mach_vec_theta = u_con[2] / sound_speed # u^θ / c_s
    mach_vec_phi = u_con[3] / sound_speed # u^φ / c_s
    grad_P_phi_comp, grad_P_theta_comp, grad_P_r_comp = np.gradient(press, phi_coords, theta_coords, r_coords) #梯度需要再Kerr下算
    grad_P_r = grad_P_r_comp
    grad_P_theta = (1.0 / r_grid) * grad_P_theta_comp
    grad_P_phi = (1.0 / (r_grid * np.sin(theta_grid))) * grad_P_phi_comp
    grad_P_mag = np.sqrt(grad_P_r**2 + grad_P_theta**2 + grad_P_phi**2) + 1e-30
    
    # --- 步骤 3: L&H方法快速筛选 (不变) ---
    print("  Step 3: Pre-screening candidates with L&H method...")
    
    dot_product = mach_vec_r * grad_P_r + mach_vec_theta * grad_P_theta + mach_vec_phi * grad_P_phi
    normal_mach = dot_product / grad_P_mag
    candidate_mask = (normal_mach >= mach_threshold) & (dot_product > 0)
    if grad_p_filter_quantile > 0:
        grad_p_threshold = np.quantile(grad_P_mag, grad_p_filter_quantile)
        candidate_mask &= (grad_P_mag > grad_p_threshold)
    num_candidates = np.sum(candidate_mask)
    print(f"  Found {num_candidates} candidate shock cells for verification.")

    # --- 步骤 4: 熵增验证并记录物理性质 ---
    print("  Step 4: Verifying candidates and storing properties...")
    final_shock_mask = np.zeros_like(press, dtype=bool)
    # 【新增】: 初始化用于存储物理性质的数组
    compression_ratio_grid = np.zeros_like(press)
    downstream_temp_grid = np.zeros_like(press)
    downstream_ne_grid = np.zeros_like(press)
    # 【新增】: 初始化用于存储物理性质的数组
    upstream_mach_grid = np.zeros_like(press) # 上游法向马赫数 M₁
  
    # 物理常数 (cgs units)
    M_P = 1.6726e-24 # 质子质量 (g)
    K_B = 1.3806e-16 # 玻尔兹曼常数 (erg/K)

    candidate_indices = np.argwhere(candidate_mask)
    
    for k, j, i in candidate_indices:
        if not (0 < k < nk-1 and 0 < j < nj-1 and 0 < i < ni-1): continue
            
        P2, rho2, v2_vec = press[k,j,i], rho[k,j,i], np.array([vel1[k,j,i], vel2[k,j,i], vel3[k,j,i]])
        
        min_pressure, upstream_neighbor = P2, None
        for dk, dj, di in [(0,0,-1), (0,0,1), (0,-1,0), (0,1,0), (-1,0,0), (1,0,0)]:
            p_neighbor = press[k+dk, j+dj, i+di]
            if p_neighbor < min_pressure:
                min_pressure = p_neighbor
                upstream_neighbor = (k+dk, j+dj, i+di)
        
        if upstream_neighbor is None: continue
            
        ku, ju, iu = upstream_neighbor
        P1, rho1, v1_vec = press[ku,ju,iu], rho[ku,ju,iu], np.array([vel1[ku,ju,iu], vel2[ku,ju,iu], vel3[ku,ju,iu]])
        # 使用四维速度的空间分量
        v2_con = u_con[1:, k, j, i] # v^i_2
        v1_con = u_con[1:, ku, ju, iu] # v^i_1
        # 协变法线向量 n_i (近似)
        n_cov = np.array([grad_P_r[k,j,i], grad_P_theta[k,j,i], grad_P_phi[k,j,i]])
        n_cov_mag = np.sqrt(np.sum(n_cov**2)) # 欧几里得模长?仅用于归一化方向
        n_cov /= n_cov_mag
        a2, a1 = np.sqrt(gamma * P2 / rho2), np.sqrt(gamma * P1 / rho1)
     # 正确的速度投影：u = v^i * n_i
        u2 = np.dot(v2_con, n_cov)
        u1 = np.dot(v1_con, n_cov)

        if u1 <= u2: continue

        pressure_ratio = P2 / P1
        u_sh = u1 + a1 * np.sqrt(((gamma + 1)/(2*gamma))*pressure_ratio + (gamma - 1)/(2*gamma)) #Rankine-Hugoniot jump conditions，cited in Xia et. al (13)
        
        if (u1 + a1) < u_sh < (u2 + a2):
            final_shock_mask[k,j,i] = True

            # 【核心修正】: 记录上游法向马赫数 M₁
            mach_1 = np.abs(u1 / a1)
            upstream_mach_grid[k,j,i] = mach_1
                        
            mu = 0.5 # 对完全电离的氢等离子体 存疑？
            downstream_temp_grid[k,j,i] = (P2 * mu * M_P) / (rho2 * K_B)
            downstream_ne_grid[k,j,i] = rho2 / M_P

    print(f"Shock detection finished. Verified {np.sum(final_shock_mask)} shock cells.")
    
    # 【新增】: 返回一个包含所有结果的字典
    shock_properties = {
        "mask": final_shock_mask,
        "upstream_mach": upstream_mach_grid,
        "downstream_temp": downstream_temp_grid,
        "downstream_n_e": downstream_ne_grid
    }
    return shock_properties


def find_shocks_in_roi_grmhd_corrected(roi_data, gamma=4.0/3.0, mach_threshold=1.7, grad_p_filter_quantile=0.20,search_depth=5,spin=0.98):
    """
    【物理修正与完善版 v2】修正了einsum调用中的维度不匹配问题。
    【鲁棒版GRMHD探测函数】通过在激波法线方向上搜索来对抗数值耗散
    """
    print("Starting GRMHD shock detection (Physics Corrected & Completed)...")

    # --- 物理常数 (cgs units) ---
    M_P = 1.6726e-24  # 质子质量 (g)
    K_B = 1.3806e-16  # 玻尔兹曼常数 (erg/K)

    # --- 1. 数据和度规准备 ---
    press = roi_data['press']
    rho = roi_data['rho']
    nk, nj, ni = press.shape
    
    r_coords = (roi_data['x1f'][:-1] + roi_data['x1f'][1:]) / 2.0
    theta_coords = (roi_data['x2f'][:-1] + roi_data['x2f'][1:]) / 2.0
    phi_coords = (roi_data['x3f'][:-1] + roi_data['x3f'][1:]) / 2.0
    phi_grid, theta_grid, r_grid = np.meshgrid(phi_coords, theta_coords, r_coords, indexing='ij')

    # 注意：这里的 metric.py 实现会返回 (4, 4, nk, nj, ni) 形状的张量
    metric_obj = kerr_schild(r_grid, theta_grid, phi_grid, a=spin)
    gcov = metric_obj.gcov()
    gcon = metric_obj.gcon() #g_{\mu \nu}协变度规张量
    gdet = metric_obj.gdet()

    # --- 2. 计算相对论流体性质 ---
    print("  Step 1: Calculating relativistic fluid quantities...")
    eps = press / (rho * (gamma - 1.0))
    h = 1.0 + eps + press / rho #计算比焓
    sound_speed_sq = gamma * press / (rho * h) #声速平方
    sound_speed = np.sqrt(sound_speed_sq)

    class TmpData:
        def __init__(self):
            self.rho, self.prs = roi_data['rho'], roi_data['press']
            self.v1, self.v2, self.v3 = roi_data['vel1'], roi_data['vel2'], roi_data['vel3']
            self.metric, self.params = metric_obj, {'dict': {'<hydro>': {'gamma': gamma}}}
    
    data_obj = TmpData()
    data_obj.lorentz_gamma = read_data.lorentz_gamma.__get__(data_obj) #计算Lorentz gamma = dt/dτ
    data_obj.ucon = read_data.ucon.__get__(data_obj)
    
    u_con = np.squeeze(data_obj.ucon()) # (4, k, j, i)

    # --- 3. 计算压强梯度并构造激波法向量 ---
    print("  Step 2: Calculating pressure gradient and shock normal...")
    grad_P_coord = np.gradient(press, phi_coords, theta_coords, r_coords) #计算压强梯度
    
    n_cov = np.zeros_like(u_con)
    n_cov[3,:,:,:] = grad_P_coord[0]
    n_cov[2,:,:,:] = grad_P_coord[1]
    n_cov[1,:,:,:] = grad_P_coord[2]

    # --- 【核心修正】 ---
    # 修正einsum下标以匹配gcon的 (4, 4, nk, nj, ni) 形状
    # 原: 'kjiab,akji->bkji' -> 现: 'abkji,akji->bkji'
    n_con = np.einsum('abkji,akji->bkji', gcon, n_cov) #法向量升指标
    n_sq = np.einsum('akji,akji->kji', n_cov, n_con)
    
    n_sq_safe = np.maximum(n_sq, 1e-40)
    n_cov /= np.sqrt(n_sq_safe)[np.newaxis, :, :, :] #归一化
    
    # --- 4. L&H 判据的相对论推广 ---
    print("  Step 3: Applying relativistic L&H criterion...")
    u_n = np.einsum('akji,akji->kji', u_con, n_cov) #四维速度在激波法线四维向量上的投影
    
    denominator = sound_speed_sq * (1.0 + u_n**2)
    mach_sq = u_n**2 / np.maximum(denominator, 1e-40)
    # 2. 核心判据：法向速度是否超声速
    candidate_mask = (mach_sq > mach_threshold**2) #激波定义，筛出强激波
    # 计算压力梯度模长的平方 (洛伦兹不变量)
    grad_P_sq = np.einsum('akji,akji->kji', n_cov, n_con)
    # 设置一个动态阈值，例如所有梯度值中的前 80% (可以调整)
    grad_p_threshold = np.quantile(grad_P_sq[grad_P_sq > 0], grad_p_filter_quantile) 
    candidate_mask &= (grad_P_sq > grad_p_threshold)

    grad_P_n_spatial = grad_P_coord[2] * n_con[1] + grad_P_coord[1] * n_con[2] + grad_P_coord[0] * n_con[3]
    candidate_mask &= (u_n < 0.0) & (grad_P_n_spatial > 0.0) #流体的运动方向（u_con）与法向量方向相反 #压力确实是沿着法线方向增加的。这是激波作为压缩波的基本定义

    print(f"  Found {np.sum(candidate_mask)} candidate shock cells for verification.")

    # --- 5. 激波验证 (Relativistic Jump Conditions) ---
    print("  Step 4: Verifying candidates and storing properties...")
    final_shock_mask = np.zeros_like(press, dtype=bool)
    upstream_mach_grid = np.zeros_like(press)
    downstream_temp_grid = np.zeros_like(press)
    downstream_ne_grid = np.zeros_like(press)

    candidate_indices = np.argwhere(candidate_mask)
    for k_c, j_c, i_c in candidate_indices: # c for candidate center
        
        # 1. 确定搜索方向 (沿着压力梯度的反方向寻找上游)
        # 我们需要逆变法向量 n^mu 来确定在 (phi, theta, r) 坐标下的移动方向
        n_con_center = n_con[:, k_c, j_c, i_c]
        
        # 找到空间分量中最大的一个作为搜索的主轴 (简化搜索，避免插值)
        # [1,2,3] -> [r, theta, phi]
        search_axis = np.argmax(np.abs(n_con_center[1:])) + 1
        search_direction = -1 * np.sign(n_con_center[search_axis]) # 沿着压力梯度的反方向

        # 2. 搜索上游点 (Upstream Search)
        found_upstream = False
        min_pressure_upstream = press[k_c, j_c, i_c]
        ku, ju, iu = k_c, j_c, i_c # 初始化为中心点

        for step in range(1, search_depth + 1):
            k_s, j_s, i_s = k_c, j_c, i_c # s for search
            if search_axis == 1:   # r-direction
                i_s += int(step * search_direction)
            elif search_axis == 2: # theta-direction
                j_s += int(step * search_direction)
            elif search_axis == 3: # phi-direction
                k_s += int(step * search_direction)
            
            # 边界检查
            if not (0 <= k_s < nk and 0 <= j_s < nj and 0 <= i_s < ni):
                break

            current_pressure = press[k_s, j_s, i_s]
            if current_pressure < min_pressure_upstream:
                min_pressure_upstream = current_pressure
                ku, ju, iu = k_s, j_s, i_s # 更新最佳上游点
                found_upstream = True

        if not found_upstream:
            continue

        # 3. 验证 (使用找到的最佳上游点 vs 候选点作为下游)
        # 此时, "1" 代表最佳上游点, "2" 代表候选点 (下游)
        u_con1 = u_con[:, ku, ju, iu]
        u_con2 = u_con[:, k_c, j_c, i_c]
        
        # 使用下游点的法向量进行投影
        n_cov_shock = n_cov[:, k_c, j_c, i_c]
        u_n1 = np.dot(u_con1, n_cov_shock)
        u_n2 = np.dot(u_con2, n_cov_shock)
        
        if u_n1 >= u_n2: continue
        
        cs1_sq = sound_speed_sq[ku, ju, iu]
        cs2_sq = sound_speed_sq[k_c, j_c, i_c]

        if cs1_sq * (1.0 + u_n1**2) <= 0 or cs2_sq * (1.0 + u_n2**2) <= 0: continue

        mach1_sq = u_n1**2 / (cs1_sq * (1.0 + u_n1**2))
        mach2_sq = u_n2**2 / (cs2_sq * (1.0 + u_n2**2))

        if mach1_sq > mach_threshold**2 and mach2_sq < 1.0:
             # 将候选点标记为激波下游
             final_shock_mask[k_c, j_c, i_c] = True
             upstream_mach_grid[k_c, j_c, i_c] = np.sqrt(mach1_sq)

             P2 = press[k_c, j_c, i_c]
             rho2_val = rho[k_c, j_c, i_c]
             mu = 0.5 
             downstream_temp_grid[k_c, j_c, i_c] = (P2 * mu * M_P) / (rho2_val * K_B)
             downstream_ne_grid[k_c, j_c, i_c] = rho2_val / M_P

    print(f"Robust GRMHD detection finished. Verified {np.sum(final_shock_mask)} shock cells.")
    
    shock_properties = {
        "mask": final_shock_mask,
        "upstream_mach": upstream_mach_grid,
        "downstream_temp": downstream_temp_grid,
        "downstream_n_e": downstream_ne_grid
    }
    return shock_properties




import matplotlib.pyplot as plt

def calculate_nonthermal_electrons(shock_properties, gamma=4.0/3.0, injection_fraction=1e-4, x_inj=3.5):
    """
    【修正版 v2】严格按照论文附录A的公式计算非热电子能谱参数。
    """
    print("Starting non-thermal electron calculation...")
    
    # --- 步骤 0: 解包输入数据 ---
    mask = shock_properties["mask"]
    M1 = shock_properties["upstream_mach"] # 现在输入的是上游马赫数
    T2 = shock_properties["downstream_temp"]
    n_e2 = shock_properties["downstream_n_e"]

    M_E, C_LIGHT, K_B = 9.1094e-28, 2.9979e10, 1.3806e-16 # cgs units

    q_grid = np.zeros_like(mask, dtype=float)
    C_grid = np.zeros_like(mask, dtype=float)
    
    # --- 步骤 A: 计算幂律指数 q ---
    # 只在激波位置进行计算
    if np.any(mask):
        M1_shocks = M1[mask]
        
        # 【核心修正 A】: 根据论文公式(A.5)从马赫数M1计算压缩比τ
        # 1/τ = (γ-1)/(γ+1) + 2/((γ+1) * M₁²)
        inv_tau = (gamma - 1.0) / (gamma + 1.0) + (2.0 / (gamma + 1.0)) / M1_shocks**2
        tau = 1.0 / inv_tau
        
        # 【核心修正 B】: 根据论文公式(A.4)计算谱指数q
        # q = (τ+2)/(τ-1)
        q = (tau + 2.0) / (tau - 1.0)
        q_grid[mask] = q
        print("  Power-law index 'q' calculated correctly using Appendix A.")

        # --- 步骤 B: 计算归一化常数 C (这部分逻辑之前是正确的，现在保持不变) ---
        thermal_term = 2.0 * K_B * T2[mask] / (M_E * C_LIGHT**2)
        p_min = x_inj * np.sqrt(thermal_term)
        N_inj = injection_fraction * n_e2[mask]
        C_grid[mask] = N_inj * (q - 1.0) * p_min**(q - 1.0)
        print("  Normalization 'C' calculated.")

    nonthermal_properties = {
        "q_grid": q_grid,
        "C_grid": C_grid,
        "mask": mask
    }
    return nonthermal_properties

def calculate_nonthermal_electrons_full(shock_properties, gamma=4.0/3.0, x_inj=3.5, xi_max=0.05):
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

def visualize_shock_slice_v1(roi_data, shock_properties, snapshot_name, shock_plot_filename):
    """
    原始版本的可视化激波函数，只画激波和压强背景。
    """
    print("Generating visualization with dynamic color range...")

    shock_mask_roi= shock_properties['mask']
    # --- 步骤 1: 选择切片 ---
    k_slice_index = shock_mask_roi.shape[0] // 2
    pressure_slice = roi_data['press'][k_slice_index, :, :]
    shock_mask_slice = shock_mask_roi[k_slice_index, :, :]
    num_shocks_in_slice = np.sum(shock_mask_slice)
    print(f"Diagnostic: Found {num_shocks_in_slice} shock cells on this 2D slice.")
    
    # --- 步骤 2: 准备坐标 ---
    r_faces = roi_data['x1f']
    theta_faces = roi_data['x2f']
    r_grid_faces, theta_grid_faces = np.meshgrid(r_faces, theta_faces, indexing='ij')
    R_cyl_faces = r_grid_faces * np.sin(theta_grid_faces)
    Z_cyl_faces = r_grid_faces * np.cos(theta_grid_faces)
    
    # --- 步骤 3: 绘图准备 ---
    fig, ax = plt.subplots(figsize=(8, 10))
    pressure_slice_log = np.log10(pressure_slice + 1e-30)

    vmin, vmax = -8, 0 # 默认范围
    if num_shocks_in_slice > 0:
        # 1. 提取所有激波位置的压力值
        shock_pressures = pressure_slice[shock_mask_slice]
        
        # 2. 计算这些压力值在对数空间的中位数
        median_log_pressure = np.median(np.log10(shock_pressures + 1e-30))
        
        log_p_min = -6.5 #median_log_pressure - 2.0 
        log_p_max = -1.5#median_log_pressure + 2.0
        
        vmin, vmax = log_p_min, log_p_max
        print(f"Dynamic color range set: vmin={vmin:.2f}, vmax={vmax:.2f}")

    # --- 步骤 4: 绘制背景图 (应用新的vmin, vmax) ---
    im = ax.pcolormesh(R_cyl_faces.T, Z_cyl_faces.T, pressure_slice_log, 
                       cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
    fig.colorbar(im, label='log10(Pressure)')
    
    # --- 步骤 5: 叠加激波 (不变) ---
    if num_shocks_in_slice > 0:
        # (这部分代码和之前完全一样)
        r_centers = (r_faces[:-1] + r_faces[1:]) / 2.0
        theta_centers = (theta_faces[:-1] + theta_faces[1:]) / 2.0
        shock_indices_j, shock_indices_i = np.where(shock_mask_slice)
        r_shocks_sph = r_centers[shock_indices_i]
        theta_shocks_sph = theta_centers[shock_indices_j]
        R_shocks_cyl = r_shocks_sph * np.sin(theta_shocks_sph)
        Z_shocks_cyl = r_shocks_sph * np.cos(theta_shocks_sph)
        ax.scatter(R_shocks_cyl.T, Z_shocks_cyl.T, s=15, c='red', marker='.',alpha=0.15, label=f'Shock Fronts ({num_shocks_in_slice} cells)')
        ax.legend()

        # --- 美化图像 (不变) ---
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        ax.set_title(f"Shock Fronts in {snapshot_name}")
        ax.set_xlabel("R (Cylindrical Radius) [$r_g$]")
        ax.set_ylabel("Z (Height) [$r_g$]")
        ax.set_aspect('equal', 'box')
        
        # 为对数坐标设置一个合理的显示范围，避免从0开始
        # 找到大于0的最小坐标作为下限
        min_coord_r = np.min(r_faces[r_faces > 0]) 
        ax.set_xlim(left=min_coord_r)
        ax.set_ylim(bottom=min_coord_r)
        
    
        plt.savefig(shock_plot_filename, dpi=200)
        print(f"Visualization saved to {shock_plot_filename}")
        plt.close(fig)

# --- 确保在文件顶部导入这些模块 ---
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
import numpy as np
# -------------------------------------
"""
    激波可视化函数
    除了生成二维激波分布图，还会自动找到最强的激波点，
    并沿着其法线方向生成一维的压力剖面图，以供详细分析。
"""

def visualize_shock_slice_profile(roi_data, shock_mask_roi, snapshot_name, shock_plot_filename, file_type='gr'):
    """
    【修正版】修复了由于meshgrid索引不匹配导致的IndexError。
    """
    print("Generating enhanced visualization with 1D pressure profile...")

    # --- 步骤 1: 选择切片 ---
    k_slice_index = shock_mask_roi.shape[0] // 2
    pressure_slice = roi_data['press'][k_slice_index, :, :]
    shock_mask_slice = shock_mask_roi[k_slice_index, :, :]
    num_shocks_in_slice = np.sum(shock_mask_slice)
    print(f"Diagnostic: Found {num_shocks_in_slice} shock cells on this 2D slice.")
    
    # --- 步骤 2: 准备坐标 ---
    r_faces = roi_data['x1f']
    theta_faces = roi_data['x2f']
    r_centers = (r_faces[:-1] + r_faces[1:]) / 2.0
    theta_centers = (theta_faces[:-1] + theta_faces[1:]) / 2.0
    
    # --- 【核心修正 1】: 将 meshgrid 的索引方式改为 'xy' ---
    # 这将确保生成的网格数组形状为 (num_j, num_i)，与 pressure_slice 等数据数组保持一致。
    r_grid_faces, theta_grid_faces = np.meshgrid(r_faces, theta_faces, indexing='xy')
    R_cyl_faces = r_grid_faces * np.sin(theta_grid_faces)
    Z_cyl_faces = r_grid_faces * np.cos(theta_grid_faces)
    
    r_grid_centers, theta_grid_centers = np.meshgrid(r_centers, theta_centers, indexing='xy')
    R_cyl_centers = r_grid_centers * np.sin(theta_grid_centers)
    Z_cyl_centers = r_grid_centers * np.cos(theta_grid_centers)
    
    # --- 步骤 3: 绘图准备 ---
    fig, (ax1, ax2) = plt.subplots(
        2, 1, 
        figsize=(10, 14), 
        gridspec_kw={'height_ratios': [3, 1]},
        tight_layout=True
    )
    fig.suptitle(f"Shock Analysis for {snapshot_name}", fontsize=16)
    
    pressure_slice_log = np.log10(pressure_slice + 1e-30)
    
    # ... (动态色条范围计算，不变) ...
    vmin, vmax = -8, 0
    if num_shocks_in_slice > 0:
        shock_pressures = pressure_slice[shock_mask_slice]
        median_log_pressure = np.median(np.log10(shock_pressures + 1e-30))
        log_p_min, log_p_max = -6.5, -1.5
        vmin, vmax = log_p_min, log_p_max

    # --- 步骤 4: 绘制上半部分的二维图 ---
    ax1.set_title("2D Shock Fronts Distribution")
    # --- 【核心修正 2】: 移除 pcolormesh 中的 .T 转置 ---
    # 因为 R_cyl_faces 和 Z_cyl_faces 的形状现在已经与 pressure_slice_log 匹配
    im = ax1.pcolormesh(R_cyl_faces, Z_cyl_faces, pressure_slice_log, 
                       cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax1, label='log10(Pressure)')
    
    if num_shocks_in_slice > 0:
        shock_indices_j, shock_indices_i = np.where(shock_mask_slice)
        # 现在这里的索引将是安全的，因为 R_cyl_centers 的形状是 (num_j, num_i)
        R_shocks_cyl = R_cyl_centers[shock_indices_j, shock_indices_i]
        Z_shocks_cyl = Z_cyl_centers[shock_indices_j, shock_indices_i]
        ax1.scatter(R_shocks_cyl, Z_shocks_cyl, s=15, c='red', marker='.', alpha=0.3, label=f'Shock Fronts ({num_shocks_in_slice} cells)')
        ax1.legend()
    
    # ... (美化二维图像，不变) ...
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel("R (Cylindrical Radius) [$r_g$]")
    ax1.set_ylabel("Z (Height) [$r_g$]")
    ax1.set_aspect('equal', 'box')
    min_coord_r = np.min(r_faces[r_faces > 0])
    ax1.set_xlim(left=min_coord_r)
    ax1.set_ylim(bottom=min_coord_r)

    # --- 步骤 5: 生成并绘制下半部分的一维压力剖面图 (不变) ---
    if num_shocks_in_slice > 0:
        # (这部分代码逻辑是正确的，无需修改)
        # ... (找到最强激波点，计算梯度，插值，绘图) ...
        shock_pressures = pressure_slice[shock_mask_slice]
        max_pressure_shock_idx = np.argmax(shock_pressures)
        j_center, i_center = shock_indices_j[max_pressure_shock_idx], shock_indices_i[max_pressure_shock_idx]
        grad_j, grad_i = np.gradient(pressure_slice)
        grad_vec = np.array([grad_j[j_center, i_center], grad_i[j_center, i_center]])
        grad_norm = np.linalg.norm(grad_vec)
        if grad_norm > 0: direction_vec = grad_vec / grad_norm
        else: direction_vec = np.array([0, 1])
        profile_length, num_points = 20, 100
        s_values = np.linspace(-profile_length, profile_length, num_points)
        j_coords, i_coords = j_center + s_values * direction_vec[0], i_center + s_values * direction_vec[1]
        profile_coords = np.vstack([j_coords, i_coords])
        pressure_profile = map_coordinates(pressure_slice, profile_coords, order=1)
        r_profile = map_coordinates(R_cyl_centers, profile_coords, order=1)
        z_profile = map_coordinates(Z_cyl_centers, profile_coords, order=1)
        distances = np.sqrt((r_profile - r_profile[0])**2 + (z_profile - z_profile[0])**2)
        center_point_idx = np.argmin(np.abs(s_values))
        distances -= distances[center_point_idx]
        ax1.plot(r_profile, z_profile, 'w--', lw=2, label='Profile Line')
        ax1.scatter(R_cyl_centers[j_center, i_center], Z_cyl_centers[j_center, i_center],
                   s=100, facecolors='none', edgecolors='white', lw=2, label='Profile Center (Max P Shock)')
        ax1.legend()
        ax2.set_title("1D Pressure Profile Across Strongest Shock")
        ax2.plot(distances, pressure_profile)
        ax2.set_yscale('log')
        ax2.set_xlabel("Distance along Profile [$r_g$] (0 = Shock Location)")
        ax2.set_ylabel("Pressure")
        ax2.grid(True, which='both', linestyle=':')
        ax2.axvline(x=0, color='red', linestyle='--', lw=2, label='Detected Shock Front')
        ax2.legend()

    # --- 步骤 6: 保存最终图像 ---
    plt.savefig(shock_plot_filename, dpi=200)
    print(f"Enhanced visualization saved to {shock_plot_filename}")
    plt.close(fig)

def visualize_shock_slice(roi_data, shock_properties, snapshot_name, shock_plot_filename, file_type='gr', profile_min_radius=10.0):
    """
    【最终升级版】
    - 修复了剖面线出界导致绘图失败的bug。
    - 新增 profile_min_radius 参数，只在 r > profile_min_radius 的区域寻找最强激波并绘制剖面。
    """
    print(f"Generating enhanced visualization (Profile search radius > {profile_min_radius} r_g)...")

    # --- 步骤 1 & 2:  ---
    shock_mask_roi = shock_properties['mask']
    k_slice_index = shock_mask_roi.shape[0] // 2
    pressure_slice = roi_data['press'][k_slice_index, :, :]
    shock_mask_slice = shock_mask_roi[k_slice_index, :, :]
    num_shocks_in_slice = np.sum(shock_mask_slice)
    print(f"Diagnostic: Found {num_shocks_in_slice} total shock cells on this 2D slice.")
    
    r_faces, theta_faces = roi_data['x1f'], roi_data['x2f']
    r_centers, theta_centers = (r_faces[:-1] + r_faces[1:]) / 2.0, (theta_faces[:-1] + theta_faces[1:]) / 2.0
    
    r_grid_faces, theta_grid_faces = np.meshgrid(r_faces, theta_faces, indexing='xy')
    R_cyl_faces, Z_cyl_faces = r_grid_faces * np.sin(theta_grid_faces), r_grid_faces * np.cos(theta_grid_faces)
    
    r_grid_centers, theta_grid_centers = np.meshgrid(r_centers, theta_centers, indexing='xy')
    R_cyl_centers, Z_cyl_centers = r_grid_centers * np.sin(theta_grid_centers), r_grid_centers * np.cos(theta_grid_centers)
    
    # --- 步骤 3 & 4: (不变, 绘制2D背景图和所有激波点) ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 14), gridspec_kw={'height_ratios': [3, 1]}, tight_layout=True)
    fig.suptitle(f"Shock Analysis for {snapshot_name}", fontsize=16)
    pressure_slice_log = np.log10(pressure_slice + 1e-30)
    vmin, vmax = -8, 0
    if num_shocks_in_slice > 0:
        shock_pressures = pressure_slice[shock_mask_slice]
        if len(shock_pressures) > 0:
            median_log_pressure = np.median(np.log10(shock_pressures + 1e-30))
            vmin, vmax = -6.5, -1.5
    ax1.set_title("2D Shock Fronts Distribution")
    im = ax1.pcolormesh(R_cyl_faces, Z_cyl_faces, pressure_slice_log, cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax1, label='log10(Pressure)')
    if num_shocks_in_slice > 0:
        shock_indices_j, shock_indices_i = np.where(shock_mask_slice)
        R_shocks_cyl = R_cyl_centers[shock_indices_j, shock_indices_i]
        Z_shocks_cyl = Z_cyl_centers[shock_indices_j, shock_indices_i]
        ax1.scatter(R_shocks_cyl, Z_shocks_cyl, s=15, c='red', marker='.', alpha=0.3, label=f'Shock Fronts ({num_shocks_in_slice} cells)')
        ax1.legend()
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel("R (Cylindrical Radius) [$r_g$]")
    ax1.set_ylabel("Z (Height) [$r_g$]")
    ax1.set_aspect('equal', 'box')
    if len(r_faces[r_faces > 0]) > 0:
      min_coord_r = np.min(r_faces[r_faces > 0])
      ax1.set_xlim(left=min_coord_r)
      ax1.set_ylim(bottom=min_coord_r)

    # --- 【核心修改】步骤 5: 在指定半径范围外生成并绘制一维压力剖面图 ---
    if num_shocks_in_slice > 0:
        shock_grad_p_mag_roi = shock_properties['grad_p_mag']
        shock_grad_p_mag_slice = shock_grad_p_mag_roi[k_slice_index, :, :]

        # 1. 获取所有激波点的索引和物理坐标
        shock_indices_j, shock_indices_i = np.where(shock_mask_slice)
        r_shocks_sph = r_grid_centers[shock_indices_j, shock_indices_i] # 球坐标半径r
        
        # 2. 【新增】根据半径进行筛选
        radius_filter = (r_shocks_sph > profile_min_radius)
        
        j_indices_filtered = shock_indices_j[radius_filter]
        i_indices_filtered = shock_indices_i[radius_filter]
        
        # 检查筛选后是否还有激波点
        if len(j_indices_filtered) == 0:
            print(f"  Warning: No shock cells found outside r = {profile_min_radius} r_g. Skipping 1D profile.")
        else:
            # 3.在筛选后的激波点中，寻找压力梯度最大的点
            shock_grad_p_mags_filtered = shock_grad_p_mag_slice[j_indices_filtered, i_indices_filtered]
            max_grad_p_idx_in_filtered = np.argmax(shock_grad_p_mags_filtered)
            
            j_center = j_indices_filtered[max_grad_p_idx_in_filtered]
            i_center = i_indices_filtered[max_grad_p_idx_in_filtered]
            
            grad_j, grad_i = np.gradient(pressure_slice)
            grad_vec = np.array([grad_j[j_center, i_center], grad_i[j_center, i_center]])
            grad_norm = np.linalg.norm(grad_vec)
            if grad_norm > 0: direction_vec = grad_vec / grad_norm
            else: direction_vec = np.array([0, 1])
            profile_length, num_points = 20, 100
            s_values = np.linspace(-profile_length, profile_length, num_points)
            j_coords, i_coords = j_center + s_values * direction_vec[0], i_center + s_values * direction_vec[1]
            profile_coords = np.vstack([j_coords, i_coords])
            pressure_profile = map_coordinates(pressure_slice, profile_coords, order=1, cval=np.nan)
            r_profile = map_coordinates(R_cyl_centers, profile_coords, order=1, cval=np.nan)
            z_profile = map_coordinates(Z_cyl_centers, profile_coords, order=1, cval=np.nan)
            valid_indices = ~np.isnan(pressure_profile)

            if np.sum(valid_indices) < 2:
                print("  Warning: Profile line is mostly off-grid even after filtering. Skipping 1D plot.")
            else:
                distances = np.sqrt((r_profile - r_profile[0])**2 + (z_profile - z_profile[0])**2)
                center_point_idx = np.argmin(np.abs(s_values))
                distances -= distances[center_point_idx]
                distances_valid, pressure_profile_valid = distances[valid_indices], pressure_profile[valid_indices]
                r_profile_valid, z_profile_valid = r_profile[valid_indices], z_profile[valid_indices]
                
                ax1.plot(r_profile_valid, z_profile_valid, color='magenta', linestyle='--', lw=2, zorder=10, label='Profile Line')
                ax1.scatter(R_cyl_centers[j_center, i_center], Z_cyl_centers[j_center, i_center],
                           s=100, facecolors='none', edgecolors='magenta', lw=2, zorder=10, label='Profile Center (Max P Shock)')
                ax1.legend()
                
                ax2.set_title(f"1D Pressure Profile (r > {profile_min_radius} $r_g$)")
                ax2.plot(distances_valid, pressure_profile_valid)
                ax2.set_yscale('log')
                ax2.set_xlabel(f"Distance along Profile [$r_g$] (0 = Detected Shock Location)")
                ax2.set_ylabel("Pressure")
                ax2.grid(True, which='both', linestyle=':')
                ax2.axvline(x=0, color='red', linestyle='--', lw=2, label='Detected Shock Front')
                ax2.legend()

    # --- 步骤 6: 保存最终图像 ---
    plt.savefig(shock_plot_filename, dpi=200)
    print(f"Enhanced visualization saved to {shock_plot_filename}")
    plt.close(fig)

# --- 【新增】更智能的质量检测辅助函数 v2 ---
def _is_profile_shock_like_v2(pressure_profile, shock_index, min_plateau_width=10):
    """
    辅助函数 v2，用于判断一维压力剖面是否像一个激波。
    标准:
    1. 在激波点必须有压力上升。
    2. 激波下游必须有一个宽度不小于 min_plateau_width 的高压平台。
    """
    # 确保有足够的数据点进行分析
    if shock_index <= 0 or shock_index >= len(pressure_profile) - 1:
        return False

    pressure_pre_shock = pressure_profile[shock_index - 1]
    pressure_at_shock = pressure_profile[shock_index]

    # 1. 必须是一个压力上升的点
    if pressure_at_shock <= pressure_pre_shock:
        return False

    # 2. 检查下游平台的宽度
    plateau_width = 0
    # 我们将 "高压平台" 定义为压力高于激波前后压力平均值的区域
    threshold = (pressure_pre_shock + pressure_at_shock) / 2.0
    
    for i in range(shock_index, len(pressure_profile)):
        if pressure_profile[i] > threshold:
            plateau_width += 1
        else:
            break # 压力一旦回落，就停止计数
    
    return plateau_width >= min_plateau_width


def _is_profile_shock_like_v3(pressure_profile, shock_index, 
                              jump_ratio_threshold=1.5, plateau_fraction=0.75, 
                              window_size=15):
    """
    辅助函数 v3，用于判断一维压力剖面是否像一个激波。
    标准:
    1. 必须有显著的压力跳变。
    2. 下游压力必须稳定地维持在高位，而不是迅速回落。
    """
    # 确保有足够的分析窗口
    if shock_index < 5 or shock_index + window_size >= len(pressure_profile):
        return False

    # 定义上游区域 (激波前5个点)
    upstream_region = pressure_profile[shock_index - 5 : shock_index]
    if len(upstream_region) == 0: return False
    p_upstream = np.mean(upstream_region)

    # 定义下游区域 (激波后 window_size 个点)
    downstream_region = pressure_profile[shock_index : shock_index + window_size]
    p_peak_downstream = np.max(downstream_region)

    # 1. 检查跳变幅度是否足够大
    if p_peak_downstream / p_upstream < jump_ratio_threshold:
        return False

    # 2. 检查下游压力是否稳定维持在高位
    #    取下游区域后半部分的平均压力
    plateau_start_index = window_size // 2
    p_plateau = np.mean(downstream_region[plateau_start_index:])
    
    # 要求下游稳定区的压力至少为峰值压力的 plateau_fraction (例如75%)
    if p_plateau / p_peak_downstream < plateau_fraction:
        return False
        
    return True



def visualize_shock_slice_multi_profile(roi_data, shock_properties, snapshot_name, shock_plot_filename, 
                                        num_profiles=3, dbscan_eps=0.08, dbscan_min_samples=6, min_cluster_size=15):
    """
    【多剖面诊断版】
    使用DBSCAN聚类算法自动识别多个激波阵面，并为最大的几个阵面分别绘制压力剖面图。
    【最终诊断版 v2】
    - 增加了对剖面形态的自动质量控制，区分“好激波”与“疑似噪声”。
    - 在2D图上用不同颜色标记不同质量的剖面。
    - 修复了空白剖面图的bug。
    """
    print(f"Generating multi-profile visualization with QUALITY CONTROL for top {num_profiles} shock clusters...")

    # ... (从函数开始到筛选top_clusters的代码与上一版完全相同, 此处省略) ...
    shock_mask_roi = shock_properties['mask']
    k_slice_index = shock_mask_roi.shape[0] // 2
    pressure_slice = roi_data['press'][k_slice_index, :, :]
    shock_mask_slice = shock_mask_roi[k_slice_index, :, :]
    num_shocks_in_slice = np.sum(shock_mask_slice)
    print(f"Diagnostic: Found {num_shocks_in_slice} total shock cells on this 2D slice.")
    if num_shocks_in_slice < dbscan_min_samples:
        print("  Warning: Not enough shock cells for clustering. Skipping.")
        return
    r_faces, theta_faces = roi_data['x1f'], roi_data['x2f']
    r_centers, theta_centers = (r_faces[:-1] + r_faces[1:]) / 2.0, (theta_faces[:-1] + theta_faces[1:]) / 2.0
    r_grid_faces, theta_grid_faces = np.meshgrid(r_faces, theta_faces, indexing='xy')
    R_cyl_faces, Z_cyl_faces = r_grid_faces * np.sin(theta_grid_faces), r_grid_faces * np.cos(theta_grid_faces)
    r_grid_centers, theta_grid_centers = np.meshgrid(r_centers, theta_centers, indexing='xy')
    R_cyl_centers, Z_cyl_centers = r_grid_centers * np.sin(theta_grid_centers), r_grid_centers * np.cos(theta_grid_centers)
    shock_indices_j, shock_indices_i = np.where(shock_mask_slice)
    R_shocks_cyl = R_cyl_centers[shock_indices_j, shock_indices_i]
    Z_shocks_cyl = Z_cyl_centers[shock_indices_j, shock_indices_i]
    log_coords = np.vstack([np.log10(R_shocks_cyl), np.log10(Z_shocks_cyl)]).T
    db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(log_coords)
    labels = db.labels_
    unique_labels = set(labels)
    if -1 in unique_labels: unique_labels.remove(-1)
    if not unique_labels:
        print("  Warning: DBSCAN found no clusters. Skipping profile analysis.")
        return
    clusters = [{'label': l, 'size': np.sum(labels == l), 'j_indices': shock_indices_j[labels == l], 'i_indices': shock_indices_i[labels == l]} for l in unique_labels]
    clusters.sort(key=lambda x: x['size'], reverse=True)
    
    # 【新增】根据最小尺寸过滤簇
    valid_clusters = [c for c in clusters if c['size'] >= min_cluster_size]
    top_clusters = valid_clusters[:num_profiles]
    
    if not top_clusters:
        print(f"  Warning: No clusters found with size >= {min_cluster_size}. Skipping profile analysis.")
        return

    # --- 绘图准备 ---
    num_plots = len(top_clusters)
    fig = plt.figure(figsize=(12, 6 + 3 * num_plots), tight_layout=True)
    gs = fig.add_gridspec(2, num_plots)
    ax_2d = fig.add_subplot(gs[0, :])
    profile_axes = [fig.add_subplot(gs[1, i]) for i in range(num_plots)]
    fig.suptitle(f"Shock Analysis for {snapshot_name}", fontsize=16)
    
    # ... (绘制2D背景图和所有激波点) ...
    pressure_slice_log = np.log10(pressure_slice + 1e-30)
    ax_2d.set_title("2D Shock Fronts Distribution & Selected Profiles")
    im = ax_2d.pcolormesh(R_cyl_faces, Z_cyl_faces, pressure_slice_log, cmap='viridis', shading='auto')
    fig.colorbar(im, ax=ax_2d, label='log10(Pressure)', fraction=0.05, pad=0.01)
    ax_2d.scatter(R_shocks_cyl, Z_shocks_cyl, s=15, c='red', marker='.', alpha=0.1, zorder=1)
    # (美化2D图)
    ax_2d.set_xscale('log'); ax_2d.set_yscale('log'); ax_2d.set_xlabel("R [$r_g$]"); ax_2d.set_ylabel("Z [$r_g$]"); ax_2d.set_aspect('equal', 'box')

    # --- 循环为每个簇绘制剖面 ---
    shock_grad_p_mag_slice = shock_properties['grad_p_mag'][k_slice_index, :, :]
    
    for i, cluster in enumerate(top_clusters):
        ax_1d = profile_axes[i]
        
        # 1. 找到簇内压力梯度最大的点
        grad_p_mags_in_cluster = shock_grad_p_mag_slice[cluster['j_indices'], cluster['i_indices']]
        max_grad_idx = np.argmax(grad_p_mags_in_cluster)
        j_center, i_center = cluster['j_indices'][max_grad_idx], cluster['i_indices'][max_grad_idx]
        
        # 2. 提取剖面数据 (逻辑不变)
        # ... (此处省略与上一版完全相同的剖面数据提取代码) ...
        grad_j, grad_i = np.gradient(pressure_slice)
        grad_vec = np.array([grad_j[j_center, i_center], grad_i[j_center, i_center]])
        grad_norm = np.linalg.norm(grad_vec)
        if grad_norm > 0: direction_vec = grad_vec / grad_norm
        else: direction_vec = np.array([0, 1])
        profile_length, num_points = 20, 100
        s_values = np.linspace(-profile_length, profile_length, num_points)
        j_coords, i_coords = j_center + s_values * direction_vec[0], i_center + s_values * direction_vec[1]
        profile_coords = np.vstack([j_coords, i_coords])
        pressure_profile = map_coordinates(pressure_slice, profile_coords, order=1, cval=np.nan)
        r_profile = map_coordinates(R_cyl_centers, profile_coords, order=1, cval=np.nan)
        z_profile = map_coordinates(Z_cyl_centers, profile_coords, order=1, cval=np.nan)
        
        # 【修正】修复空白剖面图bug
        valid_indices = ~np.isnan(pressure_profile)
        if np.sum(valid_indices) < 2: 
            ax_1d.text(0.5, 0.5, 'Profile Off-Grid', ha='center', va='center', transform=ax_1d.transAxes)
            continue
        
        distances = np.sqrt((r_profile - r_profile[0])**2 + (z_profile - z_profile[0])**2)
        center_point_idx = np.argmin(np.abs(s_values))
        distances -= distances[center_point_idx]
        distances_valid, pressure_profile_valid = distances[valid_indices], pressure_profile[valid_indices]
        r_profile_valid, z_profile_valid = r_profile[valid_indices], z_profile[valid_indices]
        
        # --- 【核心修改】: 调用新的质量检测函数 ---
        shock_index_in_profile = np.argmin(np.abs(distances_valid))
        is_good_shock = _is_profile_shock_like_v3(pressure_profile_valid, shock_index_in_profile)
        
        # 4. 根据评估结果使用不同颜色绘制
        profile_color = 'lime' if is_good_shock else 'orange'
        ax_1d_title = f"Profile {i+1} (Size: {cluster['size']}) - {'GOOD' if is_good_shock else 'NOISY'}"
        
        ax_2d.plot(r_profile_valid, z_profile_valid, color=profile_color, linestyle='--', lw=2, zorder=10, label=f'Profile {i+1} ({ "Good" if is_good_shock else "Noisy"})')
        ax_2d.scatter(R_cyl_centers[j_center, i_center], Z_cyl_centers[j_center, i_center],
                   s=120, facecolors='none', edgecolors=profile_color, lw=2, zorder=10, marker='o')

        ax_1d.set_title(ax_1d_title)
        ax_1d.plot(distances_valid, pressure_profile_valid)
        ax_1d.set_yscale('log')
        ax_1d.grid(True, which='both', linestyle=':')
        ax_1d.axvline(x=0, color='red', linestyle='--', lw=2, label='Detected Shock Front')
        ax_1d.legend()

    ax_2d.legend()
    profile_axes[0].set_ylabel("Pressure")
    if num_plots > 0:
        profile_axes[num_plots//2].set_xlabel("Distance along Profile [$r_g$]")

    plt.savefig(shock_plot_filename, dpi=200)
    print(f"Multi-profile visualization with QC saved to {shock_plot_filename}")
    plt.close(fig)


# --- 主分析函数 ---
def analyze_snapshot(input_filename):
    """
    处理单个时间快照文件：重建、切片、分析。
    """
    print(f"--- Processing: {os.path.basename(input_filename)} ---")

    # ----------------------------------------------------------------------
    # 步骤 1: 在内存中完整重建4级分辨率的网格
    # 这是最消耗内存和CPU的步骤，但只是暂时的
    # ----------------------------------------------------------------------
    print("Reconstructing full grid in memory (level 4)...")
    full_data = athena_read.athdf(input_filename, level=4)
    print("Reconstruction complete.")

    # ----------------------------------------------------------------------
    # 步骤 2: 定义您的ROI并找到对应的数组索引
    # ----------------------------------------------------------------------
    # 物理坐标
    r_coords = full_data['x1f']   # 径向 r 的网格边界
    theta_coords = full_data['x2f'] # 极角 theta 的网格边界

    # 定义ROI的物理边界
    r_min, r_max = 1.1, 1200.0
    theta_min, theta_max = 0.0, 0.3 * np.pi # 将 θ/π 转换为弧度

    # 使用 np.searchsorted 快速找到边界对应的索引
    # 'side="left"' 包含r_min，'side="right"' 包含r_max
    i_start = np.searchsorted(r_coords, r_min, side='left')
    i_end = np.searchsorted(r_coords, r_max, side='right')

    j_start = np.searchsorted(theta_coords, theta_min, side='left')
    j_end = np.searchsorted(theta_coords, theta_max, side='right')
    
    # 方位角 k 方向我们取全部范围
    k_start, k_end = 0, len(full_data['x3f']) - 1

    print(f"ROI defined. Index ranges:")
    print(f"  r (i): {i_start} to {i_end}")
    print(f"  θ (j): {j_start} to {j_end}")
    print(f"  φ (k): {k_start} to {k_end}")
    
    # ----------------------------------------------------------------------
    # 步骤 3: 对所有物理量进行切片，获得“瘦身”后的数据
    # ----------------------------------------------------------------------
    roi_data = {}
    for key, value in full_data.items():
        if key.startswith('x'): # 如果是坐标，也进行切片
             # 注意坐标数组是一维的，需要单独处理
            if key == 'x1f': roi_data[key] = value[i_start:i_end+1]
            if key == 'x2f': roi_data[key] = value[j_start:j_end+1]
            if key == 'x3f': roi_data[key] = value[k_start:k_end+1]
        else: # 如果是三维物理量数据
            roi_data[key] = value[k_start:k_end, j_start:j_end, i_start:i_end]
            
    print("Slicing complete. Data now confined to ROI.")

    # ----------------------------------------------------------------------
    # 步骤 4: （可选但推荐）立即释放巨大数组的内存
    # ----------------------------------------------------------------------
    del full_data
    import gc
    gc.collect() # 触发垃圾回收
    print("Memory from full grid has been released.")
    
    # ----------------------------------------------------------------------
    # 步骤 5: 在“瘦身”后的 roi_data 上执行您的科学分析
    # ----------------------------------------------------------------------
    shock_properties = find_shocks_in_roi(roi_data)

    shock_mask_roi = shock_properties['mask']

    # 步骤 6: 调用新的可视化函数
    snapshot_basename = os.path.basename(input_filename)
    visualize_shock_slice(roi_data, shock_mask_roi, snapshot_basename)
    
    # 步骤 7: 计算非热电子性质
    
    # 只有在找到了激波的情况下才进行计算
    if np.any(shock_properties["mask"]):
        nonthermal_props = calculate_nonthermal_electrons(shock_properties)
        
        # nonthermal_props 字典中包含了 q_grid 和 C_grid
        # 这两个三维数组就是您进行步骤四（计算同步辐射）所需的全部输入！
        print("\n--- Non-thermal electron properties ---")
        q_values = nonthermal_props['q_grid'][shock_properties['mask']]
        print(f"  Mean power-law index q: {np.mean(q_values):.2f}")
    else:
        print("No shocks found, skipping non-thermal electron calculation.")
    


# --- 主程序入口 ---
if __name__ == '__main__':
    # 示例：处理单个文件
 
    test_file = 'F:\\Research\\Shockwave\\data_test\\mad98.prim.00200.athdf' 
    analyze_snapshot(test_file)

    # 最终您会在这里写一个循环来处理所有1000多个文件
    # import glob
    # file_list = sorted(glob.glob('path/to/your/data/mad98.prim.*.athdf'))
    # for f in file_list:
    #     analyze_snapshot(f)