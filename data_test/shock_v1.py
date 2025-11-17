import numpy as np
import h5py
import sys
import os
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.colors import LogNorm
# --------------------------------------------------------------------------
# 步骤 0: 将 pyathena 文件夹的路径添加到Python的搜索路径中
# 假设您的 pyathena 文件夹与您的分析脚本在同一个父目录下
# 请根据您的实际文件结构修改
script_dir = os.path.dirname(os.path.abspath(__file__))
pyathena_dir = os.path.join(script_dir, '..', 'pyathena') 
sys.path.insert(0, pyathena_dir)

from pyathena import athena_read # 现在可以成功导入了
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

import matplotlib.pyplot as plt
import numpy as np

import plotly.graph_objects as go
import pandas as pd # Optional, but often convenient

def visualize_shock_3d_interactive_html(roi_data, shock_properties, snapshot_name, html_plot_filename,
                                        x_lim=260.0, y_lim=260.0, z_lim=1200.0): # 新增范围参数
    """
    【Interactive HTML 3D版 v2】
    - 绘制指定半径范围内 (r < r_max_vis) 所有激波格点的3D散点图。
    - 【新增】散点颜色根据上游马赫数 (upstream_mach) 变化。
    - 保存为可交互旋转的 HTML 文件。

    【Interactive HTML 3D版 v3】
    - 绘制指定笛卡尔坐标范围内的激波3D散点图。
    - 散点颜色根据上游马赫数变化。
    - 保存为可交互旋转的 HTML 文件。
    """
    print(f"Generating INTERACTIVE 3D shock visualization (X,Y < {x_lim}, Z < {z_lim} r_g)...")

    shock_mask_roi = shock_properties['mask']
    upstream_mach_roi = shock_properties['upstream_mach']
    
    num_shocks_total = np.sum(shock_mask_roi)
    print(f"Diagnostic: Found {num_shocks_total} total shock cells in the 3D ROI.")

    if num_shocks_total == 0:
        print("  No shocks found to visualize in 3D. Skipping.")
        return

    # --- 1. 提取所有三维激波点的坐标和马赫数 ---
    k_indices, j_indices, i_indices = np.where(shock_mask_roi)
    r_centers = (roi_data['x1f'][:-1] + roi_data['x1f'][1:]) / 2.0
    theta_centers = (roi_data['x2f'][:-1] + roi_data['x2f'][1:]) / 2.0
    phi_centers = (roi_data['x3f'][:-1] + roi_data['x3f'][1:]) / 2.0
    
    r_shocks = r_centers[i_indices]
    theta_shocks = theta_centers[j_indices]
    phi_shocks = phi_centers[k_indices]
    mach_shocks = upstream_mach_roi[k_indices, j_indices, i_indices]

    # --- 2. 将所有激波点转换为笛卡尔坐标 ---
    x_shocks_all = r_shocks * np.sin(theta_shocks) * np.cos(phi_shocks)
    y_shocks_all = r_shocks * np.sin(theta_shocks) * np.sin(phi_shocks)
    z_shocks_all = r_shocks * np.cos(theta_shocks)

    # --- 3. 【核心修改】根据笛卡尔坐标范围进行过滤 ---
    coord_filter = (np.abs(x_shocks_all) < x_lim) & \
                   (np.abs(y_shocks_all) < y_lim) & \
                   (z_shocks_all < z_lim) & \
                   (z_shocks_all >= 0) # 仅保留北半球 Z>=0

    if np.sum(coord_filter) == 0:
        print(f"  No shocks found within the specified X, Y, Z limits. Skipping 3D visualization.")
        return
        
    x_shocks_filtered = x_shocks_all[coord_filter]
    y_shocks_filtered = y_shocks_all[coord_filter]
    z_shocks_filtered = z_shocks_all[coord_filter]
    mach_shocks_filtered = mach_shocks[coord_filter]
    
    print(f"  Visualizing {len(x_shocks_filtered)} shock cells within the specified limits.")

    # --- 4. 创建 Plotly 3D 散点图 (不变) ---
    fig = go.Figure(data=[go.Scatter3d(
        x=x_shocks_filtered,
        y=y_shocks_filtered,
        z=z_shocks_filtered,
        mode='markers',
        marker=dict(
            size=2,
            color=mach_shocks_filtered,
            colorscale='Plasma',
            opacity=0.7,
            colorbar=dict(title='Upstream Mach ($M_1$)'),
            cmin=1.0,
            cmax=2.0
        )
    )])

    # --- 5. 【核心修改】配置布局以匹配新的范围 ---
    fig.update_layout(
        title=f"Interactive 3D Shock Distribution (X,Y<±{x_lim}, Z<{z_lim}) for {snapshot_name}",
        scene=dict(
            xaxis_title='X [$r_g$]',
            yaxis_title='Y [$r_g$]',
            zaxis_title='Z (Height) [$r_g$]',
            aspectmode='data', # 使用 'data' 让 Z 轴可以拉伸
            xaxis=dict(range=[-x_lim, x_lim], backgroundcolor="rgb(50, 50, 50)"),
            yaxis=dict(range=[-y_lim, y_lim], backgroundcolor="rgb(50, 50, 50)"),
            zaxis=dict(range=[0, z_lim], backgroundcolor="rgb(50, 50, 50)") # Z 从 0 开始
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # --- 6. 保存为 HTML (不变) ---
    fig.write_html(html_plot_filename)
    print(f"Interactive 3D visualization saved to {html_plot_filename}")


def visualize_shock_projection_dual_range(roi_data, shock_properties, snapshot_name, shock_plot_filename):
    """
    【俯视图版 - 双范围】可视化函数。
    将激波投影分为两个面板显示：
    - 左面板: 内区 (r < 50 rg)，高分辨率，色阶 [0, 30]。
    - 右面板: 外区 (r >= 50 rg)，低分辨率，色阶 [0, 15]。
    均使用线性均匀分箱。
    """
    print("Generating dual-range shock projection (top-down view)...")

    shock_mask_roi = shock_properties['mask']
    num_shocks_total = np.sum(shock_mask_roi)
    print(f"Diagnostic: Found {num_shocks_total} total shock cells in the 3D ROI.")

    if num_shocks_total == 0:
        print("  No shocks found to project. Skipping visualization.")
        return

    # --- 1. 提取并转换坐标 (不变) ---
    k_indices, j_indices, i_indices = np.where(shock_mask_roi)
    r_centers = (roi_data['x1f'][:-1] + roi_data['x1f'][1:]) / 2.0
    theta_centers = (roi_data['x2f'][:-1] + roi_data['x2f'][1:]) / 2.0
    phi_centers = (roi_data['x3f'][:-1] + roi_data['x3f'][1:]) / 2.0
    r_shocks = r_centers[i_indices]
    theta_shocks = theta_centers[j_indices]
    phi_shocks = phi_centers[k_indices]
    x_shocks = r_shocks * np.sin(theta_shocks) * np.cos(phi_shocks)
    y_shocks = r_shocks * np.sin(theta_shocks) * np.sin(phi_shocks)

    # --- 2. 准备绘图 (两个子图) ---
    fig, (ax_inner, ax_outer) = plt.subplots(1, 2, figsize=(18, 9)) # 1行2列
    fig.suptitle(f"Shock Density Projection (Top-Down View) for {snapshot_name}", fontsize=16)

    cmap = 'inferno'

    # --- 3. 绘制左面板 (内区: r < 50 rg) ---
    inner_mask = (r_shocks < 50)
    x_shocks_inner = x_shocks[inner_mask]
    y_shocks_inner = y_shocks[inner_mask]

    plot_range_inner = 100
    bins_inner = 256
    vmin_inner, vmax_inner = 1, 50
    norm_in = LogNorm(vmin=vmin_inner, vmax=vmax_inner)

    if len(x_shocks_inner) > 0:
        h_inner = ax_inner.hist2d(x_shocks_inner, y_shocks_inner, 
                                  bins=bins_inner, 
                                  range=[[-plot_range_inner, plot_range_inner], [-plot_range_inner, plot_range_inner]],
                                  cmap=cmap,
                                  cmin=1,
                                  norm=norm_in)
        fig.colorbar(h_inner[3], ax=ax_inner, label='Shock Cells per Bin (Inner)', extend='max')
    else:
        ax_inner.text(0.5, 0.5, 'No shocks in inner region', ha='center', va='center', transform=ax_inner.transAxes)

    ax_inner.set_title(f"Inner Region (r < 50 $r_g$)")
    ax_inner.set_xlabel("X [$r_g$]")
    ax_inner.set_ylabel("Y [$r_g$]")
    ax_inner.set_facecolor('black')
    ax_inner.set_aspect('equal', 'box')
    ax_inner.set_xlim(-plot_range_inner, plot_range_inner)
    ax_inner.set_ylim(-plot_range_inner, plot_range_inner)

    # --- 4. 绘制右面板 (外区: r >= 50 rg) ---
    outer_mask = (r_shocks >= 50)
    x_shocks_outer = x_shocks[outer_mask]
    y_shocks_outer = y_shocks[outer_mask]

    plot_range_outer = 800
    bins_outer = 128
    vmin_outer, vmax_outer = 1, 10

    norm = LogNorm(vmin=vmin_outer, vmax=vmax_outer)

    if len(x_shocks_outer) > 0:
        h_outer = ax_outer.hist2d(x_shocks_outer, y_shocks_outer, 
                                  bins=bins_outer, 
                                  range=[[-plot_range_outer, plot_range_outer], [-plot_range_outer, plot_range_outer]],
                                  cmap=cmap,
                                  cmin=1,
                                  norm = norm)
        fig.colorbar(h_outer[3], ax=ax_outer, label='Shock Cells per Bin Outer (log)', extend='max')
    else:
         ax_outer.text(0.5, 0.5, 'No shocks in outer region', ha='center', va='center', transform=ax_outer.transAxes)

    ax_outer.set_title(f"Outer Region (r >= 50 $r_g$)")
    ax_outer.set_xlabel("X [$r_g$]")
    ax_outer.set_ylabel("Y [$r_g$]") # Keep label for clarity, though ticks might be removed later if needed
    ax_outer.set_facecolor('black')
    ax_outer.set_aspect('equal', 'box')
    ax_outer.set_xlim(-plot_range_outer, plot_range_outer)
    ax_outer.set_ylim(-plot_range_outer, plot_range_outer)

    # --- 5. 最终调整 ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

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