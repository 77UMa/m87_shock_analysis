import os
import sys
import glob
import time
import numpy as np
import h5py
import multiprocessing as mp
import matplotlib.pyplot as plt

# 将项目根目录添加到Python路径中，以便能找到 pyathena 包
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# --- 确保可以找到您的 pyathena 模块 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
pyathena_path = os.path.join(script_dir, '..', 'pyathena') # 请根据您的目录结构进行调整
sys.path.insert(0, pyathena_path)
from pyathena import athena_read
# ---------------------------------------------------------

def calculate_verified_shocks(roi_data, gamma=4.0/3.0, mach_threshold=1.1):
    """
    【升级版】一个完整的激波计数函数。
    它执行初步筛选和精细验证，并返回最终被验证的激波格点总数。
    """
    # --- 准备工作 ---
    press, rho = roi_data['press'], roi_data['rho']
    nk, nj, ni = press.shape
    vel1, vel2, vel3 = roi_data['vel1'], roi_data['vel2'], roi_data['vel3']
    r_coords = (roi_data['x1f'][:-1] + roi_data['x1f'][1:]) / 2.0
    theta_coords = (roi_data['x2f'][:-1] + roi_data['x2f'][1:]) / 2.0
    phi_coords = (roi_data['x3f'][:-1] + roi_data['x3f'][1:]) / 2.0
    phi_grid, theta_grid, r_grid = np.meshgrid(phi_coords, theta_coords, r_coords, indexing='ij')

    # --- 阶段一：初步筛选 (L&H 判据) ---
    sound_speed = np.sqrt(gamma * press / np.maximum(rho, 1e-30))
    mach_vec_r, mach_vec_theta, mach_vec_phi = vel1/(sound_speed+1e-30), vel2/(sound_speed+1e-30), vel3/(sound_speed+1e-30)
    grad_P_phi, grad_P_theta, grad_P_r = np.gradient(press, phi_coords, theta_coords, r_coords)
    grad_P_theta *= (1.0 / r_grid)
    grad_P_phi *= (1.0 / (r_grid * np.sin(theta_grid) + 1e-30))
    grad_P_mag = np.sqrt(grad_P_r**2 + grad_P_theta**2 + grad_P_phi**2) + 1e-30
    dot_product = mach_vec_r * grad_P_r + mach_vec_theta * grad_P_theta + mach_vec_phi * grad_P_phi
    normal_mach = dot_product / grad_P_mag
    candidate_mask = (normal_mach >= mach_threshold) & (dot_product > 0)
    candidate_indices = np.argwhere(candidate_mask)

    if len(candidate_indices) == 0:
        return 0

    # --- 阶段二：精细验证 (熵增条件) ---
    verified_shock_count = 0
    for k, j, i in candidate_indices:
        if not (0 < k < nk-1 and 0 < j < nj-1 and 0 < i < ni-1): continue
        P2, rho2, v2_vec = press[k,j,i], rho[k,j,i], np.array([vel1[k,j,i], vel2[k,j,i], vel3[k,j,i]])
        min_pressure, upstream_neighbor = P2, None
        for dk, dj, di in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]:
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
                verified_shock_count += 1
        except (ValueError, FloatingPointError):
            continue
            
    return verified_shock_count

# ==============================================================================
# 多进程相关设置
# ==============================================================================
g_semaphore = None
g_roi_def = None

def init_worker(semaphore, roi_def):
    global g_semaphore, g_roi_def
    g_semaphore = semaphore
    g_roi_def = roi_def

def survey_single_file(filename):
    """
    用于并行处理单个文件的工作函数。
    """
    g_semaphore.acquire()
    print(f"[{os.getpid()}] Acquired slot. Loading & slicing {os.path.basename(filename)}...")
    try:
        # 使用原生的 athena_read 一次性加载和切片，速度更快
        full_data = athena_read.athdf(filename, level=4)
        r_coords, theta_coords = full_data['x1f'], full_data['x2f']
        i_start = np.searchsorted(r_coords, g_roi_def['x1_min'], side='left')
        i_end = np.searchsorted(r_coords, g_roi_def['x1_max'], side='right')
        j_start = np.searchsorted(theta_coords, g_roi_def['x2_min'], side='left')
        j_end = np.searchsorted(theta_coords, g_roi_def['x2_max'], side='right')
        k_start, k_end = 0, len(full_data['x3f']) - 1

        roi_data = {}
        for key in ['rho', 'press', 'vel1', 'vel2', 'vel3']: # 只提取需要的变量
            roi_data[key] = full_data[key][k_start:k_end, j_start:j_end, i_start:i_end]
        roi_data['x1f'] = r_coords[i_start:i_end+1]
        roi_data['x2f'] = theta_coords[j_start:j_end+1]
        roi_data['x3f'] = full_data['x3f']
        
        del full_data
        import gc; gc.collect()

        print(f"[{os.getpid()}] Data loaded. Calculating verified shocks...")
        verified_count = calculate_verified_shocks(roi_data)
        
        snapshot_num = int(os.path.basename(filename).split('.')[2])
        print(f"  -> Snapshot {snapshot_num:05d}: Found {verified_count} VERIFIED shock cells.")
        return (snapshot_num, verified_count)

    finally:
        g_semaphore.release()
        print(f"[{os.getpid()}] Released slot for {os.path.basename(filename)}.")

# ==============================================================================
# 主程序入口
# ==============================================================================
if __name__ == '__main__':
    
    # --- 配置 ---
    DATA_DIRECTORY = "/home/cyh_22307110238/project/Shockwave/data_test3/"
    OUTPUT_PLOT_FILENAME = "shock_activity_survey_verified.png"
    NUM_PROCESSES = max(1, mp.cpu_count() - 2)
    MEMORY_CONCURRENCY = 20 # 根据128GB内存和6-8GB/任务的估算设置

    ROI_DEFINITION = { 'x1_min': 1.1, 'x1_max': 1200.0, 'x2_min': 0.0, 'x2_max': 0.3 * np.pi }

    # --- 开始普查 ---
    file_list = sorted(glob.glob(os.path.join(DATA_DIRECTORY, 'mad98.prim.*.athdf')))
    if not file_list: sys.exit(f"Error: No files found in {DATA_DIRECTORY}")

    print(f"Starting VERIFIED shock survey for {len(file_list)} snapshots.")
    print(f"Total CPU processes: {NUM_PROCESSES}, Max concurrent memory tasks: {MEMORY_CONCURRENCY}")

    try: mp.set_start_method('spawn', force=True)
    except RuntimeError: pass 

    sema = mp.Semaphore(MEMORY_CONCURRENCY)
    
    results = []
    start_time = time.time()
    with mp.Pool(processes=NUM_PROCESSES, initializer=init_worker, initargs=(sema, ROI_DEFINITION)) as pool:
        results = pool.map(survey_single_file, file_list)
    end_time = time.time()
    print(f"\nSurvey complete. Total time: {end_time - start_time:.2f} seconds.")

    # --- 绘制并保存结果 ---
    if results:
        results.sort() # 按快照编号排序
        snapshot_numbers, shock_counts = zip(*results)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(snapshot_numbers, shock_counts, '-o', label='Verified Shock Count')
        ax.set_yscale('log')
        ax.set_xlabel("Snapshot Number")
        ax.set_ylabel("Number of Verified Shock Cells (Log Scale)")
        ax.set_title("Survey of Verified Shock Activity Over Time")
        ax.grid(True, which='both', linestyle=':')
        ax.legend()
        
        if len(shock_counts) > 5:
            top_indices = np.argsort(shock_counts)[-5:]
            for idx in top_indices:
                ax.axvline(snapshot_numbers[idx], color='r', linestyle='--', alpha=0.5)
                ax.text(snapshot_numbers[idx], shock_counts[idx], f' {snapshot_numbers[idx]}', color='r', rotation=90, va='bottom')

        plt.savefig(OUTPUT_PLOT_FILENAME, dpi=150)
        print(f"\nSurvey plot saved to: {OUTPUT_PLOT_FILENAME}")