#! /usr/bin/env python3
'''
实现GRMHD模拟的MAD98磁囚禁盘模拟结果的激波可视化，支持批处理。
批处理使用进程池，内存溢出会直接报错结束而不是卡住。
'''


# ==============================================================================
# 导入所需模块
# ==============================================================================
import os
import sys
import glob
import time
import subprocess
import multiprocessing as mp
from functools import partial # 导入 partial
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.colors import LogNorm


# 将项目根目录添加到Python路径中，以便能找到 pyathena 包
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# --- 确保可以找到 ipole 和 athena 的脚本 ---
# 请根据你的文件结构修改这些路径
# 假设 ipole-master 文件夹与此脚本位于同一父目录下
# 修改：需要ipole脚本和本文件在同一目录之下
script_dir = os.path.dirname(os.path.abspath(__file__))
ipole_dir = os.path.join(script_dir, '..', 'ipole-master')
ipole_scripts_path = os.path.join(ipole_dir, 'scripts')
sys.path.insert(0, ipole_scripts_path)

# 需要 athena_read.py 位于 pyathena 目录下
pyathena_path = os.path.join(script_dir, '..', 'pyathena')
sys.path.insert(0, pyathena_path)

try:
    import ipole as ipole_api
    from pyathena import athena_read
    # 从你的科学计算文件中导入函数
    from shock_v1 import find_shocks_in_roi_robust
except ImportError as e:
    print(f"Fatal Error: Could not import a required module. {e}")
    print("Please check the paths to 'ipole-master/scripts' and 'pyathena'.")
    sys.exit(1)


# ==============================================================================
# 最终版工作流核心函数,只进行到激波
# ==============================================================================
def visualize_shock_overview(roi_data, shock_properties, snapshot_name, shock_plot_filename,config):
    """
    【全局概览版】可视化函数，旨在模仿 Xia et al. Fig. 1 的风格。
    1. 使用线性坐标轴。
    2. 使用 'magma' 色图和固定的压力范围。
    3. 绘制完整的 r-theta 平面。
    4. 使用半透明亮绿色“涂抹”的方式高亮激波区域。
    """
    print("Generating shock overview visualization...")

    shock_mask_roi = shock_properties['mask']
    k_slice_index = shock_mask_roi.shape[0] // 2
    pressure_slice = roi_data['press'][k_slice_index, :, :]
    shock_mask_slice = shock_mask_roi[k_slice_index, :, :]
    num_shocks_in_slice = np.sum(shock_mask_slice)
    print(f"Diagnostic: Found {num_shocks_in_slice} shock cells on this 2D slice.")
    
    # --- 准备坐标 ---
    r_faces, theta_faces = roi_data['x1f'], roi_data['x2f']
    # 使用网格中心点坐标来计算 R 和 Z
    r_centers = (r_faces[:-1] + r_faces[1:]) / 2.0
    theta_centers = (theta_faces[:-1] + theta_faces[1:]) / 2.0
    r_grid_centers, theta_grid_centers = np.meshgrid(r_centers, theta_centers, indexing='xy')
    R_cyl_centers = r_grid_centers * np.sin(theta_grid_centers)
    Z_cyl_centers = r_grid_centers * np.cos(theta_grid_centers)
    
    # --- 绘图准备 ---
    fig, ax = plt.subplots(figsize=(10, 10))
    pressure_slice_log = np.log10(pressure_slice + 1e-30)

    # --- 1. 绘制压力背景 ---
    # 【要求2】使用 'magma' 色图和固定的压力范围，方便对比
    vmin_fixed, vmax_fixed = config['v_min'], config['v_max'] 
    im = ax.pcolormesh(R_cyl_centers, Z_cyl_centers, pressure_slice_log, 
                       cmap='magma', shading='auto', vmin=vmin_fixed, vmax=vmax_fixed)
    fig.colorbar(im, ax=ax, label='log10(Pressure)', extend='both')
    
    # --- 2. 【要求4】使用半透明黑色图层“涂抹”激波 ---
    if num_shocks_in_slice > 0:
        # 创建一个 RGBA 图像 (高度, 宽度, 4个颜色通道)
        # 初始时所有像素都是完全透明的
        shock_overlay = np.zeros((pressure_slice.shape[0], pressure_slice.shape[1], 4))
        
        # 将激波位置的像素设置为半透明的黑色
        # [R, G, B, Alpha]: [0.2, 1.0, 0.2, 0.7] ->亮绿色, 70% 不透明度
        shock_overlay[shock_mask_slice] = [0.2, 1.0, 0.2, 0.7]
        
        # 将这个 RGBA 图像叠加在压力背景图之上
        ax.imshow(shock_overlay, origin='lower', 
                  extent=[R_cyl_centers.min(), R_cyl_centers.max(), Z_cyl_centers.min(), Z_cyl_centers.max()],
                  aspect='auto', interpolation='none')

    # --- 3. 【要求1】美化图像，使用线性坐标 ---
    ax.set_title(f"Shock Fronts in {snapshot_name}")
    ax.set_xlabel("R (Cylindrical Radius) [$r_g$]")
    ax.set_ylabel("Z (Height) [$r_g$]")
    ax.set_aspect('equal', 'box') # 保持R和Z的比例为1:1
    
    # 设置显示范围，可以根据需要调整
    # ax.set_xlim(0, 500)
    # ax.set_ylim(-500, 500)

    plt.savefig(shock_plot_filename, dpi=200, bbox_inches='tight')
    print(f"Overview visualization saved to {shock_plot_filename}")
    plt.close(fig)


def visualize_shock_projection(roi_data, shock_properties, snapshot_name, shock_plot_filename):
    """
    【俯视图版】可视化函数。
    将3D ROI内的所有激波点投影到X-Y平面，以二维直方图的形式展示其密度分布，
    用于观察喷流关于Z轴的对称性。

    【俯视图版 v2】可视化函数。
    - 激波密度色阶范围固定为 [0, 40]。
    - 使用对数间隔的分箱（Logarithmic Bins），以更公平地展示不同半径处的激波密度。

    【俯视图版 v3】可视化函数。
    - 使用线性和均匀的分箱（Bins）来避免坐标转换伪影。
    - 使用对数色阶（LogNorm）来同时凸显中心亮区和外围暗区。
    - 激波密度色阶范围的上限固定为30（为了与上一张图对比）。
    """
    print("Generating shock projection (top-down view) with Logarithmic Color Scale...")

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

    # --- 2. 绘制二维直方图 ---
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # --- 【核心修改】 ---
    # 1. 回归均匀的线性分箱
    plot_range = 100
    bins = 256
    
    # 2. 使用对数色阶 (LogNorm)
    #    vmin=1 是因为对数不能取0
    #    vmax=30 是为了与您上一张图的颜色范围大致对应，可以调整
    norm = LogNorm(vmin=1, vmax=50)
    
    # 3. 在 hist2d 中应用新的 norm
    h = ax.hist2d(x_shocks, y_shocks, 
                  bins=bins, 
                  range=[[-plot_range, plot_range], [-plot_range, plot_range]],
                  cmap='inferno',
                  cmin=1,          # 只显示至少有一个激波点的格子
                  norm=norm)       # 应用对数色阶
    # --- 【修改结束】 ---

    fig.colorbar(h[3], ax=ax, label='Number of Shock Cells per Bin (Log Scale)', extend='max')

    # --- 3. 美化图像 (不变) ---
    ax.set_title(f"Shock Density Projection (Top-Down View) for {snapshot_name}")
    ax.set_xlabel("X [$r_g$]")
    ax.set_ylabel("Y [$r_g$]")
    ax.set_facecolor('black')
    ax.set_aspect('equal', 'box')

    plt.savefig(shock_plot_filename, dpi=200, bbox_inches='tight')
    print(f"Projection visualization saved to {shock_plot_filename}")
    plt.close(fig)

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

# nt_electron_v2.py

# --- (确保文件顶部有 import matplotlib.pyplot as plt 和 import numpy as np) ---
# --- 【新增】导入3D绘图工具 ---
from mpl_toolkits.mplot3d import Axes3D
# -----------------------------
# --- (Ensure imports: import numpy as np) ---
# --- 【New】Import Plotly ---
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


def analyze_snapshot_full_pipeline(filename, config):
    """
    【升级调试版】增加了对完整重建数据(full_data)的检查点功能，
    将耗时的数据重建与快速的算法调试分离开。
    """
    input_athdf_file = filename
    print(f"\n==============================================================================")
    print(f"Processing snapshot: {os.path.basename(input_athdf_file)}")
    print(f"==============================================================================")
    
    base_name = os.path.basename(input_athdf_file).replace('.athdf', '')
    
    # --- 定义两类检查点文件的路径 ---
    # 1. 预处理检查点（存储巨大的 full_data）
    dir_full_data = os.path.join(config['output_directory'], 'full_data_checkpoints')
    os.makedirs(dir_full_data, exist_ok=True)
    full_data_checkpoint_filename = os.path.join(dir_full_data, f"{base_name}_full_data.npz")
    
    # 2. 分析结果检查点（存储 roi_data, shock_properties 等）
    dir_analysis_checkpoints = os.path.join(config['output_directory'], 'analysis_checkpoints')
    os.makedirs(dir_analysis_checkpoints, exist_ok=True)
    analysis_checkpoint_filename = os.path.join(dir_analysis_checkpoints, f"{base_name}_analysis.npz")
    
    # 输出目录
    dir_shock_plots = os.path.join(config['output_directory'], 'shock_visuals_overview')
    os.makedirs(dir_shock_plots, exist_ok=True)
    shock_plot_filename = os.path.join(dir_shock_plots, f"{base_name}_shock_slice.png")

    # --- 阶段一：数据加载/重建 (可跳过) ---
    full_data = None
    if config.get('load_full_data_checkpoint', False) and os.path.exists(full_data_checkpoint_filename):
        print(f"--- Loading full reconstructed data from checkpoint: {os.path.basename(full_data_checkpoint_filename)} ---")
        # 使用 allow_pickle=True 来加载包含字典的对象数组
        with np.load(full_data_checkpoint_filename, allow_pickle=True) as data:
            # .item() 用于从0维数组中提取字典对象
            full_data = data['full_data'].item()
        print("--- Full data loaded successfully. Skipping reconstruction. ---")
    
    if full_data is None:
        print("--- Running full data reconstruction from .athdf file... (This may take a while) ---")
        full_data = athena_read.athdf(input_athdf_file, level=4)

        if config.get('save_full_data_checkpoint', False):
            print(f"--- Saving full reconstructed data to checkpoint: {os.path.basename(full_data_checkpoint_filename)} ---")
            # 使用 savez_compressed 提高存储效率
            np.savez_compressed(full_data_checkpoint_filename, full_data=full_data)
            print("--- Full data checkpoint saved. ---")

    # --- 阶段二：ROI 切片与科学分析 ---
    # 这一阶段现在可以快速、反复地执行

    # 1. 定义并执行切片
    print("  Step A: Slicing ROI from full data...")
    r_coords = full_data['x1f']
    theta_coords = full_data['x2f']
    phi_coords = full_data['x3f']
    # 您可以在这里方便地修改 r_min, theta_min 等参数来探索不同的ROI
    r_min, r_max = 10, 1200
    theta_min, theta_max = 0.0,  np.pi/2
    phi_min, phi_max = 0.0, 2*np.pi
    
    i_start = np.searchsorted(r_coords, r_min, side='left')
    i_end = np.searchsorted(r_coords, r_max, side='right')
    j_start = np.searchsorted(theta_coords, theta_min, side='left')
    j_end = np.searchsorted(theta_coords, theta_max, side='right')
    k_start = np.searchsorted(phi_coords,phi_min, side='left')
    k_end = np.searchsorted(phi_coords,phi_max, side='right')
    #k_start, k_end = 0, len(full_data['x3f']) - 1

    roi_data = {}
    for key, value in full_data.items():
        if key.startswith('x'):
            if key == 'x1f': roi_data[key] = value[i_start:i_end+1]
            if key == 'x2f': roi_data[key] = value[j_start:j_end+1]
            if key == 'x3f': roi_data[key] = value[k_start:k_end+1]
        else:
            roi_data[key] = value[k_start:k_end, j_start:j_end, i_start:i_end]
    
    if 'Time' not in roi_data and 'Time' in full_data:
        roi_data['Time'] = full_data['Time']
    
    # (可选) 如果不再需要 full_data，可以释放内存
    del full_data
    import gc; gc.collect()

    # 2. 执行激波探测和后续分析 (与您之前的代码相同)
    print("  Step B: Finding shocks and visualization...")
   
    shock_properties = find_shocks_in_roi_robust(roi_data) 

    # a. 调用“单phi切片侧视图” (R-Z平面，仅右半)
    overview_plot_filename = os.path.join(dir_shock_plots, f"{base_name}_shock_overview_slice.png")
    visualize_shock_overview(roi_data, shock_properties, os.path.basename(input_athdf_file), overview_plot_filename,config)

    # b. 【核心修改】调用新的“双范围俯视图”函数
    projection_plot_filename = os.path.join(dir_shock_plots, f"{base_name}_shock_projection_dual.png") # 文件名可区分
    visualize_shock_projection_dual_range(roi_data, shock_properties, os.path.basename(filename), projection_plot_filename)

    # # c. 【新增】调用“X-Z全景切片图”
    xz_plane_plot_filename = os.path.join(dir_shock_plots, f"{base_name}_shock_plane_xz.png")
    visualize_shock_xz_plane(roi_data, shock_properties, os.path.basename(input_athdf_file), xz_plane_plot_filename,config)

    # d.激波3D视图(测试版)
# --- Call the new interactive 3D visualization ---
    vis_3d_filename_html = os.path.join(dir_shock_plots, f"{base_name}_shock_3d_r{int(100)}.html") # Note the .html extension
    visualize_shock_3d_interactive_html(roi_data, shock_properties, os.path.basename(filename), vis_3d_filename_html)
    return #非热电子也不算了

    print("  Step C: Calculating non-thermal electrons...")
    # ... (后续的非热电子计算、诊断输出和 return 语句保持不变)
    if np.any(shock_properties["mask"]):
        nonthermal_props = calculate_nonthermal_electrons_full(shock_properties)
    else:
        nonthermal_props = {
            'q_grid': np.zeros_like(roi_data['press']),
            'C_grid': np.zeros_like(roi_data['press']),
            'mask': np.zeros_like(roi_data['press'], dtype=bool)
        }
        print("  No shocks found, non-thermal properties initialized to zero.")

    print("\n--- DEBUGGING OUTPUT ---")
    num_shock_cells = np.sum(shock_properties['mask'])
    print(f"  Total shock cells detected: {num_shock_cells}")
    
    if num_shock_cells > 0:
        mach_values = shock_properties['upstream_mach'][shock_properties['mask']]
        q_values = nonthermal_props['q_grid'][shock_properties['mask']]
        print(f"  Upstream Mach Number (M1) at shocks: Min={np.min(mach_values):.2f}, Mean={np.mean(mach_values):.2f}, Max={np.max(mach_values):.2f}")
        print(f"  Power-law index (q) at shocks:      Min={np.min(q_values):.2f}, Mean={np.mean(q_values):.2f}, Max={np.max(q_values):.2f}")
    
    print("--- Workflow paused for debugging after non-thermal electron calculation. ---")
    print("==============================================================================\n")
    
    return



# ==============================================================================
# 【新】为多进程设置的全局变量和初始化函数
# ==============================================================================
# 定义一个全局变量，用于在每个子进程中存储信号量
g_semaphore = None

def init_worker(semaphore):
    """
    这个函数会在每个子进程启动时被调用一次，用于接收主进程传递过来的信号量。
    """
    global g_semaphore
    g_semaphore = semaphore

def worker_task_wrapper(args):
    """
    这是一个包装函数，它的作用是在真正执行您的科学计算任务前后，
    获取和释放信号量“通行证”，从而控制内存密集型操作的并发数。
    """
    # args 是一个元组: (filename, config)
    filename, config = args
    
    # 在进入内存密集型操作之前，必须先获取一个信号量
    print(f"[Process {os.getpid()}] Waiting for memory slot to process {os.path.basename(filename)}...")
    g_semaphore.acquire()
    print(f"[Process {os.getpid()}] Memory slot acquired. Starting processing for {os.path.basename(filename)}.")
    
    try:
        # 调用您项目中原有的核心分析函数
        # 整个 analyze_snapshot_full_pipeline 函数都将在信号量的保护下运行
        analyze_snapshot_full_pipeline(args)
    finally:
        # 无论任务成功或失败，都必须在最后释放信号量，否则会造成死锁
        g_semaphore.release()
        print(f"[Process {os.getpid()}] Memory slot released for {os.path.basename(filename)}.")


if __name__ == '__main__':
    
    config = {
        # --- 路径配置 ---
        "data_directory": "/cpfs01/projects-HDD/cfff-a7e284de52b3_HDD/cyh_22307110238/data_test3/",
        "output_directory": "/cpfs01/projects-HDD/cfff-a7e284de52b3_HDD/cyh_22307110238/workflow_output/",
        "ipole_executable_path": "/home/cyh_22307110238/project/Shockwave/ipole-master/ipole",
        
        # --- 工作流控制 ---
        "save_checkpoint": True, #检查点2，非热电子后
        "load_from_checkpoint": False,
        "auto_cleanup": True,
        
        # --- 【新增】用于调试的预处理检查点开关 ---
        "save_full_data_checkpoint": False,   # 是否保存重建后的 full_data 数组
        "load_full_data_checkpoint": False,  # 是否跳过重建，直接从 full_data 检查点开始

        # --- 物理参数 ---
        "spin": 0.98,

        # --- 画图压强范围配置 ---
        "v_min": -8,
        "v_max": -1,
        
        # --- 并行计算配置 ---
        # 充分利用CPU核心数
        "num_processes": 30
    }
    
    # --- 【核心配置】: 设置内存并发度 ---
    # 这是最关键的参数。它限制了可以同时加载和重建.athdf文件的进程数量。
    # 估算: 您的服务器内存为128GB，每个任务峰值约6-8GB。
    # 128GB / 8GB ≈ 16。我们设置一个更保守的值，比如 12。
    SAFE_MAX_WORKERS = 12
    
    # --- 准备文件和任务列表 ---
    file_list = sorted(glob.glob(os.path.join(config['data_directory'], 'mad98.prim.*.athdf')))
    tasks = [(filename, config) for filename in file_list]
    if not file_list:
        sys.exit(f"Error: No files found in the data directory.")

    print(f"Found {len(file_list)} files to process.")
    print(f"Using multiprocessing.Pool with {SAFE_MAX_WORKERS} concurrent workers.")
    
    # 使用 partial 来固定 config 参数
    task_func = partial(analyze_snapshot_full_pipeline, config=config)
    # 准备一个只包含文件名的列表
    filenames = [f for f, _ in tasks] # 假设 tasks = [(filename, config), ...] 依然存在

    start_total_time = time.time()
    
    # --- 2. 设置多进程启动方式 ---
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass 

    # --- 3. 【核心修改】使用 multiprocessing.Pool 并加入 maxtasksperchild=1 ---
    try:
        with mp.Pool(processes=SAFE_MAX_WORKERS, maxtasksperchild=1) as pool:
            print(f"--- Submitting {len(filenames)} tasks with process recycling enabled ---")
            
            # pool.map 会阻塞直到所有任务完成
            # 它现在能够稳定处理任意数量的文件
            results = pool.map(task_func, filenames)

            # (可选) 进度显示
            # 注意：pool.map是阻塞的，所以这里的循环只有在所有任务完成后才会执行
            print(f"--- All {len(filenames)} tasks have been processed. ---")

    except Exception as e:
        print(f"\n---!!! An error occurred during parallel execution: {e} ---")
        import traceback
        traceback.print_exc()

    end_total_time = time.time()
    
    print("\n==============================================================================")
    print("--- WORKFLOW SCRIPT FINISHED ---")
    print(f"Total execution time for {len(file_list)} files: {end_total_time - start_total_time:.2f} seconds.")
    print(f"==============================================================================\n")

