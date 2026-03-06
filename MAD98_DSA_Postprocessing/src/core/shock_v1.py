"""
激波物理计算模块

该模块实现了激波探测算法，主要包括：
1. find_shocks_in_roi_robust: 稳健版激波探测，结合Lovely & Haimes (1999)的法向马赫数判据
2. find_shocks_in_roi_mhd: 3D MHD激波探测器，针对M87 GRMHD模拟优化
3. 多种激波可视化函数：3D交互式HTML、2D投影图、X-Z平面图等

核心算法特点：
- 使用宽松阈值初筛 + 严格物理验证的双重机制
- 支持梯度回溯采样真实的上下游状态
- 计算激波的马赫数、下游温度、电子密度等物理量

应用领域：
- 天体物理中的激波探测
- GRMHD数值模拟分析
- 粒子加速机制研究

参考文献：
- Lovely & Haimes (1999): 法向马赫数激波探测算法
- Xia et al. (2025): MHD激波加速理论
- Yang et al. (2024): M87喷流磁重联模型
"""
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LogNorm
# --------------------------------------------------------------------------
def find_shocks_in_roi_robust(roi_data, gamma=4.0/3.0, 
                              mach_threshold_loose=1.05, 
                              min_physical_mach=1.7,
                              grad_p_filter_quantile=0.20, 
                              march_cells=4):
    """
    【稳健版 v2.0 - 双重阈值与梯度回溯】
    使用宽松的局地马赫数(mach_threshold_loose)进行初筛，以捕获被涂抹的激波。
    然后通过梯度回溯(march_cells)采样真实的上下游状态。
    最后，使用严格的物理马赫数(min_physical_mach)对回溯计算出的M1进行最终验证。
    """
    print(f"Starting ROBUST shock detection (Loose M_local >= {mach_threshold_loose}, Strict M_physical >= {min_physical_mach}, March={march_cells})...")
    
    # --- 步骤 1: 筛选候选点 (使用宽松阈值) ---
    press, rho = roi_data['press'], roi_data['rho']
    nk, nj, ni = press.shape
    if nk == 0 or nj == 0 or ni == 0:
        print("  Error: ROI data has zero dimension. Skipping.")
        return { "mask": np.zeros_like(press, dtype=bool), "upstream_mach": np.zeros_like(press) }

    vel1, vel2, vel3 = roi_data['vel1'], roi_data['vel2'], roi_data['vel3']
    r_coords = (roi_data['x1f'][:-1] + roi_data['x1f'][1:]) / 2.0
    theta_coords = (roi_data['x2f'][:-1] + roi_data['x2f'][1:]) / 2.0
    phi_coords = (roi_data['x3f'][:-1] + roi_data['x3f'][1:]) / 2.0
    
    if r_coords.size == 0 or theta_coords.size == 0 or phi_coords.size == 0:
         print("  Error: Coordinate arrays are empty. Skipping.")
         return { "mask": np.zeros_like(press, dtype=bool), "upstream_mach": np.zeros_like(press) }

    phi_grid, theta_grid, r_grid = np.meshgrid(phi_coords, theta_coords, r_coords, indexing='ij')
    
    print("  Step 1: Screening candidates with local normal Mach number...")
    sound_speed = np.sqrt(gamma * press / rho)
    mach_vec_r = vel1 / (sound_speed + 1e-30)
    mach_vec_theta = vel2 / (sound_speed + 1e-30)
    mach_vec_phi = vel3 / (sound_speed + 1e-30)
    
    grad_P_phi_comp, grad_P_theta_comp, grad_P_r_comp = np.gradient(press, phi_coords, theta_coords, r_coords)
    grad_P_r = grad_P_r_comp
    grad_P_theta = (1.0 / (r_grid + 1e-30)) * grad_P_theta_comp
    grad_P_phi = (1.0 / (r_grid * np.sin(theta_grid) + 1e-30)) * grad_P_phi_comp
    
    grad_P_mag = np.sqrt(grad_P_r**2 + grad_P_theta**2 + grad_P_phi**2) + 1e-30
    dot_product = mach_vec_r * grad_P_r + mach_vec_theta * grad_P_theta + mach_vec_phi * grad_P_phi
    normal_mach = dot_product / grad_P_mag
    
    # 使用宽松的阈值来捕获所有可能的候选点
    candidate_mask = (normal_mach >= mach_threshold_loose) & (dot_product > 0)
    
    if grad_p_filter_quantile > 0 and np.any(grad_P_mag > 0):
        grad_p_threshold = np.quantile(grad_P_mag[grad_P_mag > 0], grad_p_filter_quantile)
        candidate_mask &= (grad_P_mag > grad_p_threshold)
        
    candidate_indices = np.argwhere(candidate_mask)
    print(f"  Found {len(candidate_indices)} candidate shock cells for verification.")

    # 步骤 1.5: 计算用于索引回溯的“逻辑”梯度
    grad_P_k_raw, grad_P_j_raw, grad_P_i_raw = np.gradient(press)

    # --- 步骤 2: 稳健验证 ---
    print("  Step 2: Verifying candidates and calculating properties via marching...")
    final_shock_mask = np.zeros_like(press, dtype=bool)
    upstream_mach_grid = np.zeros_like(press)
    downstream_temp_grid = np.zeros_like(press)
    downstream_ne_grid = np.zeros_like(press)
    shock_grad_p_mag_grid = np.zeros_like(press)
    
    M_P, K_B = 1.6726e-24, 1.3806e-16

    for k, j, i in candidate_indices:
        
        # 2a. 确定法向向量 (在候选点)
        n_vec_r = grad_P_r[k,j,i] / grad_P_mag[k,j,i]
        n_vec_theta = grad_P_theta[k,j,i] / grad_P_mag[k,j,i]
        n_vec_phi = grad_P_phi[k,j,i] / grad_P_mag[k,j,i]
        n_vec = np.array([n_vec_r, n_vec_theta, n_vec_phi]) # (r, theta, phi) 物理法向

        # 2b. 确定回溯的网格轴 (主导轴)
        abs_G_k_raw = np.abs(grad_P_k_raw[k,j,i])
        abs_G_j_raw = np.abs(grad_P_j_raw[k,j,i])
        abs_G_i_raw = np.abs(grad_P_i_raw[k,j,i])

        dominant_axis = np.argmax([abs_G_k_raw, abs_G_j_raw, abs_G_i_raw])
        
        dk_s, dj_s, di_s = 0, 0, 0
        if dominant_axis == 0:
            dk_s = -int(np.sign(grad_P_k_raw[k,j,i]))
        elif dominant_axis == 1:
            dj_s = -int(np.sign(grad_P_j_raw[k,j,i]))
        else:
            di_s = -int(np.sign(grad_P_i_raw[k,j,i]))
            
        if dk_s == 0 and dj_s == 0 and di_s == 0:
            continue 

        # 2c. 采样上游 (State 1) 和下游 (State 2)
        ku = np.clip(k + dk_s * march_cells, 0, nk-1)
        ju = np.clip(j + dj_s * march_cells, 0, nj-1)
        iu = np.clip(i + di_s * march_cells, 0, ni-1)
        
        kd = np.clip(k - dk_s * march_cells, 0, nk-1)
        jd = np.clip(j - dj_s * march_cells, 0, nj-1)
        id = np.clip(i - di_s * march_cells, 0, ni-1)

        # 2d. 提取稳健的物理状态
        P1, rho1 = press[ku,ju,iu], rho[ku,ju,iu]
        v1_vec = np.array([vel1[ku,ju,iu], vel2[ku,ju,iu], vel3[ku,ju,iu]])
        a1 = sound_speed[ku,ju,iu]

        P2, rho2 = press[kd,jd,id], rho[kd,jd,id]
        v2_vec = np.array([vel1[kd,jd,id], vel2[kd,jd,id], vel3[kd,jd,id]])
        a2 = sound_speed[kd,jd,id]

        # 2e. Rankine-Hugoniot 验证
        u1 = np.dot(v1_vec, n_vec)
        u2 = np.dot(v2_vec, n_vec)
        
        if u1 <= u2: continue 

        try:
            pressure_ratio = P2 / P1
            if pressure_ratio <= 1.0: continue 
            
            u_sh = u1 + a1 * np.sqrt(((gamma + 1)/(2*gamma))*pressure_ratio + (gamma - 1)/(2*gamma))
            
            if (u1 + a1) < u_sh < (u2 + a2):
                
                # 计算物理马赫数 M1
                mach_1_sq = 1.0 + (pressure_ratio - 1.0) * (gamma + 1.0) / (2.0 * gamma)
                current_physical_mach = np.sqrt(mach_1_sq)

                # --- 【新增的严格验证】 ---
                # 只有当计算出的物理马赫数也大于我们的严格阈值时，才接受它
                if current_physical_mach < min_physical_mach:
                    continue
                # --- 【验证结束】 ---
                
                final_shock_mask[k,j,i] = True 
                upstream_mach_grid[k,j,i] = current_physical_mach
                
                mu = 0.5 
                downstream_temp_grid[k,j,i] = (P2 * mu * M_P) / (rho2 * K_B)
                downstream_ne_grid[k,j,i] = rho2 / M_P
                shock_grad_p_mag_grid[k,j,i] = grad_P_mag[k,j,i]

        except (ValueError, FloatingPointError):
            continue 

    print(f"Final Robust detection finished. Verified {np.sum(final_shock_mask)} shock cells.")
    
    return {
        "mask": final_shock_mask,
        "upstream_mach": upstream_mach_grid,
        "downstream_temp": downstream_temp_grid,
        "downstream_n_e": downstream_ne_grid,
        "grad_p_mag": shock_grad_p_mag_grid
    }

def find_shocks_in_roi_mhd(roi_data, gamma=4.0/3.0, 
                           mach_threshold_loose=1.05, 
                           min_physical_mach=1.7,
                           grad_p_filter_quantile=0.10, 
                           march_cells=5):
    """
    【3D MHD 稳健版激波探测器】
    针对 M87 GRMHD (球面坐标) 优化：
    1. 使用总压 P_tot = P_gas + 0.5*B^2 进行梯度计算 [cite: 7836, 8412]。
    2. 使用快磁声速 v_fast 作为特征速度 [cite: 7836, 8417]。
    3. 支持球面坐标系下的梯度修正 (r, theta, phi) 。
    """
    print(f"Starting 3D MHD shock detection (Loose M_local >= {mach_threshold_loose}, Strict M_physical >= {min_physical_mach})...")
    
    # --- 1. 数据准备 ---
    press = roi_data['press']
    rho = roi_data['rho']
    vel1, vel2, vel3 = roi_data['vel1'], roi_data['vel2'], roi_data['vel3']
    b1, b2, b3 = roi_data['Bcc1'], roi_data['Bcc2'], roi_data['Bcc3']
    
    nk, nj, ni = press.shape
    
    # 坐标准备 (用于梯度修正)
    r_coords = (roi_data['x1f'][:-1] + roi_data['x1f'][1:]) / 2.0
    theta_coords = (roi_data['x2f'][:-1] + roi_data['x2f'][1:]) / 2.0
    phi_coords = (roi_data['x3f'][:-1] + roi_data['x3f'][1:]) / 2.0
    phi_grid, theta_grid, r_grid = np.meshgrid(phi_coords, theta_coords, r_coords, indexing='ij')

    # --- 2. 计算 MHD 核心物理量 ---
    # a. 计算磁压与总压 [cite: 7836, 8412]
    b_sq = b1**2 + b2**2 + b3**2
    p_mag = 0.5 * b_sq
    p_tot = press + p_mag
    
    # b. 计算快磁声速 v_fast [cite: 7836, 8414, 8417]
    cs_sq = gamma * press / rho
    va_sq = b_sq / rho
    v_fast = np.sqrt(cs_sq + va_sq) # 采用垂直近似以获得最大鲁棒性

    # --- 3. 计算 3D 梯度 (球面坐标系修正) ---
    # np.gradient 返回顺序对应 (dim0, dim1, dim2) -> (phi, theta, r)
    grad_P_phi_raw, grad_P_theta_raw, grad_P_r_raw = np.gradient(p_tot)
    
    # 物理距离缩放 
    dr = np.gradient(r_coords)
    dtheta = np.gradient(theta_coords)
    dphi = np.gradient(phi_coords)
    
    # 广播坐标差
    _, _, dR_grid = np.meshgrid(phi_coords, theta_coords, dr, indexing='ij')
    _, dT_grid, _ = np.meshgrid(phi_coords, dtheta, r_coords, indexing='ij')
    dP_grid, _, _ = np.meshgrid(dphi, theta_coords, r_coords, indexing='ij')

    grad_P_r = grad_P_r_raw / dR_grid
    grad_P_theta = grad_P_theta_raw / (r_grid * dT_grid + 1e-30)
    grad_P_phi = grad_P_phi_raw / (r_grid * np.sin(theta_grid) * dP_grid + 1e-30)
    
    grad_P_mag = np.sqrt(grad_P_r**2 + grad_P_theta**2 + grad_P_phi**2) + 1e-30

    # --- 4. 候选点筛选 (局部法向马赫数) ---
    # 法向量 n = grad(P_tot) / |grad(P_tot)| [cite: 7833]
    n_r, n_theta, n_phi = grad_P_r/grad_P_mag, grad_P_theta/grad_P_mag, grad_P_phi/grad_P_mag
    v_dot_n = vel1 * n_r + vel2 * n_theta + vel3 * n_phi
    
    local_mach = v_dot_n / (v_fast + 1e-30)
    
    # 初筛条件：马赫数阈值 + 压强梯度阈值 [cite: 7832]
    candidate_mask = (local_mach >= mach_threshold_loose) & (v_dot_n > 0)
    if grad_p_filter_quantile > 0:
        p_threshold = np.quantile(grad_P_mag, grad_p_filter_quantile)
        candidate_mask &= (grad_P_mag > p_threshold)
    
    candidate_indices = np.argwhere(candidate_mask)
    print(f"  Found {len(candidate_indices)} candidate cells.")

    # --- 5. 稳健验证 (Gradient Marching) ---
    final_shock_mask = np.zeros_like(press, dtype=bool)
    upstream_mach_grid = np.zeros_like(press)
    
    # 逻辑梯度用于索引回溯
    gk, gj, gi = np.gradient(p_tot)

    for k, j, i in candidate_indices:
        # 确定主导轴进行回溯 [cite: 8440]
        dominant_axis = np.argmax([np.abs(gk[k,j,i]), np.abs(gj[k,j,i]), np.abs(gi[k,j,i])])
        
        dk, dj, di = 0, 0, 0
        if dominant_axis == 0: dk = -int(np.sign(gk[k,j,i]))
        elif dominant_axis == 1: dj = -int(np.sign(gj[k,j,i]))
        else: di = -int(np.sign(gi[k,j,i]))

        # 采样上下游 (State 1: Upstream, State 2: Downstream)
        ku, ju, iu = np.clip([k + dk*march_cells, j + dj*march_cells, i + di*march_cells], 0, [nk-1, nj-1, ni-1])
        kd, jd, id_ = np.clip([k - dk*march_cells, j - dj*march_cells, i - di*march_cells], 0, [nk-1, nj-1, ni-1])

        p1_tot = p_tot[ku, ju, iu]
        p2_tot = p_tot[kd, jd, id_]

        # 压缩性验证：下游总压必须大于上游 [cite: 7837, 8514]
        if p2_tot <= p1_tot * 1.05: continue 

        # 使用总压跳变反推物理马赫数 (MHD 激波近似公式) [cite: 8412]
        press_ratio = p2_tot / p1_tot
        m_phys_sq = 1.0 + (press_ratio - 1.0) * (gamma + 1.0) / (2.0 * gamma)
        m_phys = np.sqrt(m_phys_sq)

        if m_phys >= min_physical_mach:
            final_shock_mask[k,j,i] = True
            upstream_mach_grid[k,j,i] = m_phys

    print(f"  Verified {np.sum(final_shock_mask)} MHD shock cells.")
    
    # 填充返回结构 (保持与 nt_electron 兼容)
    # 计算下游温度与电子密度 (用于非热电子计算)
    M_P, K_B = 1.6726e-24, 1.3806e-16
    mu = 0.5 # 完全电离气体的平均分子量 [cite: 8513]
    downstream_temp = (press * mu * M_P) / (rho * K_B)
    downstream_ne = rho / M_P

    return {
        "mask": final_shock_mask,
        "upstream_mach": upstream_mach_grid,
        "downstream_temp": downstream_temp,
        "downstream_n_e": downstream_ne,
        "grad_p_mag": grad_P_mag
    }


def visualize_shock_3d_interactive_html(roi_data, shock_properties, snapshot_name, html_plot_filename,
                                        x_lim=260.0, y_lim=260.0, z_lim=1200.0):
    """【Interactive HTML 3D版 v3】"""
    print(f"Generating INTERACTIVE 3D shock visualization (X,Y < {x_lim}, Z < {z_lim} r_g)...")
    # ... (代码来自 shock_v1.py / workflowShock_processpool.py) ...
    shock_mask_roi = shock_properties['mask']
    upstream_mach_roi = shock_properties['upstream_mach']
    num_shocks_total = np.sum(shock_mask_roi)
    if num_shocks_total == 0:
        print("  No shocks found to visualize in 3D. Skipping.")
        return
    k_indices, j_indices, i_indices = np.where(shock_mask_roi)
    r_centers = (roi_data['x1f'][:-1] + roi_data['x1f'][1:]) / 2.0
    theta_centers = (roi_data['x2f'][:-1] + roi_data['x2f'][1:]) / 2.0
    phi_centers = (roi_data['x3f'][:-1] + roi_data['x3f'][1:]) / 2.0
    r_shocks = r_centers[i_indices]
    theta_shocks = theta_centers[j_indices]
    phi_shocks = phi_centers[k_indices]
    mach_shocks = upstream_mach_roi[k_indices, j_indices, i_indices]
    x_shocks_all = r_shocks * np.sin(theta_shocks) * np.cos(phi_shocks)
    y_shocks_all = r_shocks * np.sin(theta_shocks) * np.sin(phi_shocks)
    z_shocks_all = r_shocks * np.cos(theta_shocks)
    coord_filter = (np.abs(x_shocks_all) < x_lim) & \
                   (np.abs(y_shocks_all) < y_lim) & \
                   (z_shocks_all < z_lim) & \
                   (z_shocks_all >= 0)
    if np.sum(coord_filter) == 0:
        print(f"  No shocks found within the specified X, Y, Z limits. Skipping 3D visualization.")
        return
    x_shocks_filtered = x_shocks_all[coord_filter]
    y_shocks_filtered = y_shocks_all[coord_filter]
    z_shocks_filtered = z_shocks_all[coord_filter]
    mach_shocks_filtered = mach_shocks[coord_filter]
    print(f"  Visualizing {len(x_shocks_filtered)} shock cells within the specified limits.")
    fig = go.Figure(data=[go.Scatter3d(
        x=x_shocks_filtered, y=y_shocks_filtered, z=z_shocks_filtered,
        mode='markers',
        marker=dict(
            size=2, color=mach_shocks_filtered, colorscale='Plasma',
            opacity=0.7, colorbar=dict(title='Upstream Mach ($M_1$)'),
            cmin=1.0, cmax=max(2.0, np.quantile(mach_shocks_filtered, 0.95)) # 自动调整色阶上限
        )
    )])
    fig.update_layout(
        title=f"Interactive 3D Shock Distribution (X,Y<±{x_lim}, Z<{z_lim}) for {snapshot_name}",
        scene=dict(
            xaxis_title='X [$r_g$]', yaxis_title='Y [$r_g$]', zaxis_title='Z (Height) [$r_g$]',
            aspectmode='data',
            xaxis=dict(range=[-x_lim, x_lim], backgroundcolor="rgb(50, 50, 50)"),
            yaxis=dict(range=[-y_lim, y_lim], backgroundcolor="rgb(50, 50, 50)"),
            zaxis=dict(range=[0, z_lim], backgroundcolor="rgb(50, 50, 50)")
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.write_html(html_plot_filename)
    print(f"Interactive 3D visualization saved to {html_plot_filename}")
    
def visualize_shock_overview(roi_data, shock_properties, snapshot_name, shock_plot_filename, config):
    """【全局概览版】"""
    print("Generating shock overview visualization...")
    # ... (代码来自 workflowShock_processpool.py) ...
    shock_mask_roi = shock_properties['mask']
    k_slice_index = shock_mask_roi.shape[0] // 2
    pressure_slice = roi_data['press'][k_slice_index, :, :]
    shock_mask_slice = shock_mask_roi[k_slice_index, :, :]
    num_shocks_in_slice = np.sum(shock_mask_slice)
    print(f"Diagnostic: Found {num_shocks_in_slice} shock cells on this 2D slice.")
    r_faces, theta_faces = roi_data['x1f'], roi_data['x2f']
    r_centers = (r_faces[:-1] + r_faces[1:]) / 2.0
    theta_centers = (theta_faces[:-1] + theta_faces[1:]) / 2.0
    r_grid_centers, theta_grid_centers = np.meshgrid(r_centers, theta_centers, indexing='xy')
    R_cyl_centers = r_grid_centers * np.sin(theta_grid_centers)
    Z_cyl_centers = r_grid_centers * np.cos(theta_grid_centers)
    fig, ax = plt.subplots(figsize=(10, 10))
    pressure_slice_log = np.log10(pressure_slice + 1e-30)
    vmin_fixed, vmax_fixed = config['v_min'], config['v_max'] 
    im = ax.pcolormesh(R_cyl_centers, Z_cyl_centers, pressure_slice_log, 
                       cmap='magma', shading='auto', vmin=vmin_fixed, vmax=vmax_fixed)
    fig.colorbar(im, ax=ax, label='log10(Pressure)', extend='both')
    if num_shocks_in_slice > 0:
        shock_overlay = np.zeros((pressure_slice.shape[0], pressure_slice.shape[1], 4))
        shock_overlay[shock_mask_slice] = [0.2, 1.0, 0.2, 0.7] # 亮绿色
        ax.imshow(shock_overlay, origin='lower', 
                  extent=[R_cyl_centers.min(), R_cyl_centers.max(), Z_cyl_centers.min(), Z_cyl_centers.max()],
                  aspect='auto', interpolation='none')
    ax.set_title(f"Shock Fronts in {snapshot_name}")
    ax.set_xlabel("R (Cylindrical Radius) [$r_g$]")
    ax.set_ylabel("Z (Height) [$r_g$]")
    ax.set_aspect('equal', 'box')
    plt.savefig(shock_plot_filename, dpi=200, bbox_inches='tight')
    print(f"Overview visualization saved to {shock_plot_filename}")
    plt.close(fig)

def visualize_shock_projection_dual_range(roi_data, shock_properties, snapshot_name, shock_plot_filename):
    """【俯视图版 - 双范围】"""
    print("Generating dual-range shock projection (top-down view)...")
    # ... (代码来自 shock_v1.py / workflowShock_processpool.py) ...
    shock_mask_roi = shock_properties['mask']
    num_shocks_total = np.sum(shock_mask_roi)
    if num_shocks_total == 0:
        print("  No shocks found to project. Skipping visualization.")
        return
    k_indices, j_indices, i_indices = np.where(shock_mask_roi)
    r_centers = (roi_data['x1f'][:-1] + roi_data['x1f'][1:]) / 2.0
    theta_centers = (roi_data['x2f'][:-1] + roi_data['x2f'][1:]) / 2.0
    phi_centers = (roi_data['x3f'][:-1] + roi_data['x3f'][1:]) / 2.0
    r_shocks = r_centers[i_indices]
    theta_shocks = theta_centers[j_indices]
    phi_shocks = phi_centers[k_indices]
    x_shocks = r_shocks * np.sin(theta_shocks) * np.cos(phi_shocks)
    y_shocks = r_shocks * np.sin(theta_shocks) * np.sin(phi_shocks)
    fig, (ax_inner, ax_outer) = plt.subplots(1, 2, figsize=(18, 9))
    fig.suptitle(f"Shock Density Projection (Top-Down View) for {snapshot_name}", fontsize=16)
    cmap = 'inferno'
    inner_mask = (r_shocks < 50)
    x_shocks_inner, y_shocks_inner = x_shocks[inner_mask], y_shocks[inner_mask]
    plot_range_inner, bins_inner = 100, 256
    vmin_inner, vmax_inner = 1, 50
    norm_in = LogNorm(vmin=vmin_inner, vmax=vmax_inner)
    if len(x_shocks_inner) > 0:
        h_inner = ax_inner.hist2d(x_shocks_inner, y_shocks_inner, bins=bins_inner, 
                                  range=[[-plot_range_inner, plot_range_inner], [-plot_range_inner, plot_range_inner]],
                                  cmap=cmap, cmin=1, norm=norm_in)
        fig.colorbar(h_inner[3], ax=ax_inner, label='Shock Cells per Bin (Inner)', extend='max')
    ax_inner.set_title(f"Inner Region (r < 50 $r_g$)")
    ax_inner.set_xlabel("X [$r_g$]"); ax_inner.set_ylabel("Y [$r_g$]")
    ax_inner.set_facecolor('black'); ax_inner.set_aspect('equal', 'box')
    ax_inner.set_xlim(-plot_range_inner, plot_range_inner); ax_inner.set_ylim(-plot_range_inner, plot_range_inner)
    outer_mask = (r_shocks >= 50)
    x_shocks_outer, y_shocks_outer = x_shocks[outer_mask], y_shocks[outer_mask]
    plot_range_outer, bins_outer = 800, 128
    vmin_outer, vmax_outer = 1, 10
    norm_out = LogNorm(vmin=vmin_outer, vmax=vmax_outer)
    if len(x_shocks_outer) > 0:
        h_outer = ax_outer.hist2d(x_shocks_outer, y_shocks_outer, bins=bins_outer, 
                                  range=[[-plot_range_outer, plot_range_outer], [-plot_range_outer, plot_range_outer]],
                                  cmap=cmap, cmin=1, norm=norm_out)
        fig.colorbar(h_outer[3], ax=ax_outer, label='Shock Cells per Bin Outer (log)', extend='max')
    ax_outer.set_title(f"Outer Region (r >= 50 $r_g$)")
    ax_outer.set_xlabel("X [$r_g$]"); ax_outer.set_ylabel("Y [$r_g$]")
    ax_outer.set_facecolor('black'); ax_outer.set_aspect('equal', 'box')
    ax_outer.set_xlim(-plot_range_outer, plot_range_outer); ax_outer.set_ylim(-plot_range_outer, plot_range_outer)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(shock_plot_filename, dpi=200, bbox_inches='tight')
    print(f"Dual-range projection visualization saved to {shock_plot_filename}")
    plt.close(fig)

def visualize_shock_xz_plane(roi_data, shock_properties, snapshot_name, shock_plot_filename, config):
    """
    【X-Z平面全景侧视图版】
    - 选取一个phi角切片，并将其与phi+pi的镜像部分一同绘制，构成完整的X-Z平面。
    - 仅绘制北半球 (theta: 0 to pi/2)。
    - 使用线性和固定的颜色范围。
    - 使用鲜绿色高亮激波区域。

    【X-Z平面全景侧视图版 v2】
    - 修正了'phi_centers'变量在使用前未定义的错误。

    【X-Z平面全景侧视图版 v3】
    - 修正了只显示右半平面的问题，通过手动设置X轴范围来确保左右对称显示。
    """
    print("Generating shock X-Z plane visualization (full side-view)...")

    # --- 1. 数据准备 ---
    shock_mask_roi = shock_properties['mask']
    k_slice_index = shock_mask_roi.shape[0] // 2 
    pressure_slice = roi_data['press'][k_slice_index, :, :]
    shock_mask_slice = shock_mask_roi[k_slice_index, :, :]
    num_shocks_in_slice = np.sum(shock_mask_slice)
    
    # --- 2. 准备坐标 ---
    r_centers = (roi_data['x1f'][:-1] + roi_data['x1f'][1:]) / 2.0
    theta_centers = (roi_data['x2f'][:-1] + roi_data['x2f'][1:]) / 2.0
    phi_centers = (roi_data['x3f'][:-1] + roi_data['x3f'][1:]) / 2.0
    
    # 打印诊断信息
    print(f"Diagnostic: Found {num_shocks_in_slice} shock cells on the phi={phi_centers[k_slice_index]:.2f} slice.")
    
    phi_slice_val = phi_centers[k_slice_index]
    r_grid_centers, theta_grid_centers = np.meshgrid(r_centers, theta_centers, indexing='xy')
    Z_coords = r_grid_centers * np.cos(theta_grid_centers)
    X_coords = r_grid_centers * np.sin(theta_grid_centers) * np.cos(phi_slice_val)

    # --- 3. 绘图 ---
    fig, ax = plt.subplots(figsize=(10, 10))
    pressure_slice_log = np.log10(pressure_slice + 1e-30)

    vmin_fixed, vmax_fixed = config['v_min'], config['v_max']  
    cmap = 'magma'
    
    # a & b. 绘制左右两个半平面
    ax.pcolormesh(X_coords, Z_coords, pressure_slice_log, 
                  cmap=cmap, shading='auto', vmin=vmin_fixed, vmax=vmax_fixed)
    ax.pcolormesh(-X_coords, Z_coords, pressure_slice_log, 
                  cmap=cmap, shading='auto', vmin=vmin_fixed, vmax=vmax_fixed)

    # c. 叠加激波区域
    if num_shocks_in_slice > 0:
        shock_overlay = np.zeros((pressure_slice.shape[0], pressure_slice.shape[1], 4))
        shock_overlay[shock_mask_slice] = [0.2, 1.0, 0.2, 0.7] # 半透明鲜绿色
        
        # extent 定义了图像的坐标范围 [xmin, xmax, ymin, ymax]
        extent_right = [np.min(X_coords), np.max(X_coords), np.min(Z_coords), np.max(Z_coords)]
        extent_left = [-np.max(X_coords), -np.min(X_coords), np.min(Z_coords), np.max(Z_coords)]
        
        ax.imshow(shock_overlay, origin='lower', extent=extent_right, aspect='auto', interpolation='none')
        ax.imshow(shock_overlay, origin='lower', extent=extent_left, aspect='auto', interpolation='none')

    # --- 4. 美化图像 ---
    ax.set_title(f"Shock Fronts in X-Z Plane for {snapshot_name}")
    ax.set_xlabel("X [$r_g$]")
    ax.set_ylabel("Z (Height) [$r_g$]")
    ax.set_aspect('equal', 'box')
    
    # --- 【核心修正】: 手动设置X轴的显示范围 ---
    # 找到X坐标的最大绝对值
    x_max_abs = np.max(np.abs(X_coords))
    # 设置X轴范围为对称的 [-max, +max]，并增加5%的留白
    ax.set_xlim(-x_max_abs * 1.05, x_max_abs * 1.05)
    # --- 【修正结束】 ---
    
    # 添加 colorbar
    norm = Normalize(vmin=vmin_fixed, vmax=vmax_fixed)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='log10(Pressure)', extend='both')

    plt.savefig(shock_plot_filename, dpi=200, bbox_inches='tight')
    print(f"X-Z plane visualization saved to {shock_plot_filename}")
    plt.close(fig)

# --- 主程序入口 ---
if __name__ == '__main__':
    # 示例：处理单个文件
 
    test_file = 'F:\\Research\\Shockwave\\data_test\\mad98.prim.00200.athdf' 

    # 最终您会在这里写一个循环来处理所有1000多个文件
    # import glob
    # file_list = sorted(glob.glob('path/to/your/data/mad98.prim.*.athdf'))
    # for f in file_list:
    #     analyze_snapshot(f)