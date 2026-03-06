import numpy as np
import copy
from numba import njit, prange

# ==============================================================================
# JIT 编译的核心计算内核 (C++ 级速度)
# ==============================================================================
@njit(fastmath=True)
def estimate_cooling_time_jit(B_mag, cooling_factor):
    """JIT 加速的冷却时间计算"""
    # 避免除零
    B_safe = np.maximum(B_mag, 1e-6)
    return cooling_factor / (B_safe**2)

@njit(parallel=True, fastmath=True)
def advection_step_jit(N_curr, Q_curr, S_density, q_source, 
                       v1, v2, v3, 
                       r, th, ph, 
                       dt, tau_cool):
    """
    执行一步时间积分：平流 + 冷却 + 源项
    使用 OpenMP 风格的多核并行 (parallel=True)
    """
    nk, nj, ni = N_curr.shape
    
    # 创建下一步的数组 (在内存中分配一次)
    N_next = np.zeros_like(N_curr)
    Q_next = np.zeros_like(Q_curr)
    
    # 并行循环：外层循环 (phi, theta) 并行化
    for k in prange(nk):
        for j in range(nj):
            # 预计算 theta 方向几何因子 (简化)
            # 完整球坐标散度很复杂，这里主要关注径向平流带来的延伸
            
            for i in range(ni):
                # --- 1. 径向平流 (Radial Advection) ---
                # 迎风格式 (Upwind Scheme): v * dN/dr
                # 如果 v > 0, 使用 N[i] - N[i-1]
                # 如果 v < 0, 使用 N[i+1] - N[i]
                
                vel_r = v1[k, j, i]
                
                # 计算 dr (局部网格间距)
                # 处理边界：内部使用向后差分，边界特殊处理
                if i == 0:
                    dr = r[1] - r[0]
                    dN_dr = 0.0 # 内边界假设无梯度或由源项控制
                    dQ_dr = 0.0
                elif i == ni - 1:
                    dr = r[ni-1] - r[ni-2]
                    dN_dr = (N_curr[k, j, i] - N_curr[k, j, i-1]) / dr
                    dQ_dr = (Q_curr[k, j, i] - Q_curr[k, j, i-1]) / dr
                else:
                    # 迎风判断
                    if vel_r > 0:
                        dr = r[i] - r[i-1]
                        dN_dr = (N_curr[k, j, i] - N_curr[k, j, i-1]) / dr
                        dQ_dr = (Q_curr[k, j, i] - Q_curr[k, j, i-1]) / dr
                    else:
                        dr = r[i+1] - r[i]
                        dN_dr = (N_curr[k, j, i+1] - N_curr[k, j, i]) / dr
                        dQ_dr = (Q_curr[k, j, i+1] - Q_curr[k, j, i]) / dr

                # Advection term: v * grad(N)
                # 忽略球坐标几何扩张项 (N/r * ...) 以保持稳健性，
                # 重点在于将电子推向下游
                adv_N = vel_r * dN_dr
                adv_Q = vel_r * dQ_dr

                # --- 2. 冷却与源项 (Cooling & Source) ---
                # Sink: N / tau
                # Source: S
                
                cool_rate = 1.0 / (tau_cool[k, j, i] + 1e-10)
                
                # 源项
                src_N = S_density[k, j, i]
                src_Q = S_density[k, j, i] * q_source[k, j, i]
                
                # --- 3. 更新 ---
                # dN/dt = S - Adv - Cool
                val_N = N_curr[k, j, i] + dt * (src_N - adv_N - N_curr[k, j, i] * cool_rate)
                val_Q = Q_curr[k, j, i] + dt * (src_Q - adv_Q - Q_curr[k, j, i] * cool_rate)
                
                # 物理约束：非负且不低于源项(可选)
                # 这里只约束非负
                if val_N < 0: val_N = 0.0
                if val_Q < 0: val_Q = 0.0
                
                # 边界锁定：内边界 (i=0) 强制为源项注入值
                if i == 0:
                    val_N = src_N
                    val_Q = src_Q
                
                N_next[k, j, i] = val_N
                Q_next[k, j, i] = val_Q

    return N_next, Q_next

# ==============================================================================
# Python 驱动函数
# ==============================================================================
def solve_steady_advection(roi_data, nonthermal_props, config):
    print(">>> [Advection-JIT] Solving steady-state transport equation using Numba...")
    
    # 1. 数据准备 (转为 Numpy 数组，确保类型为 float64 或 float32)
    rho = np.ascontiguousarray(roi_data['rho'], dtype=np.float64)
    nk, nj, ni = rho.shape
    
    # 坐标 (1D 数组)
    r = np.ascontiguousarray(roi_data['x1v'], dtype=np.float64)
    th = np.ascontiguousarray(roi_data['x2v'], dtype=np.float64) # 之前报错的地方，现在需要确保传入
    ph = np.ascontiguousarray(roi_data['x3v'], dtype=np.float64)
    
    # 速度
    v1 = np.ascontiguousarray(roi_data['vel1'], dtype=np.float64)
    v2 = np.ascontiguousarray(roi_data['vel2'], dtype=np.float64)
    v3 = np.ascontiguousarray(roi_data['vel3'], dtype=np.float64)
    
    # 磁场
    if 'Bcc1' in roi_data:
        Bsq = roi_data['Bcc1']**2 + roi_data['Bcc2']**2 + roi_data['Bcc3']**2
    else:
        Bsq = roi_data['B1']**2 + roi_data['B2']**2 + roi_data['B3']**2
    B_mag = np.sqrt(Bsq).astype(np.float64)
    
    # 冷却参数
    cool_fac = float(config.get('physics', {}).get('cooling_factor', 50.0))
    tau_cool = estimate_cooling_time_jit(B_mag, cool_fac)
    
    # 源项
    S_density = np.ascontiguousarray(nonthermal_props['C_grid'], dtype=np.float64)
    q_source = np.ascontiguousarray(nonthermal_props['q_grid'], dtype=np.float64)
    
    # 2. 迭代参数
    # 自动计算 dt
    dr_min = np.min(r[1:] - r[:-1])
    v_max = np.max(np.abs(v1)) + 1e-5
    dt = 0.4 * dr_min / v_max # CFL 0.4 for stability
    
    max_iter = int(config.get('physics', {}).get('advection_steps', 2000))
    min_iter = 200
    tol = 1e-4

    print(f"    Grid: {ni}x{nj}x{nk}, dt={dt:.2e}, Max Steps={max_iter}")
    
    # 初始化
    N_curr = S_density.copy()
    Q_curr = S_density * q_source
    
    # 3. 时间步循环
    for step in range(max_iter):
        # 调用 JIT 函数 (第一次调用会编译，耗时约1-2秒，之后极快)
        N_next, Q_next = advection_step_jit(N_curr, Q_curr, S_density, q_source,
                                            v1, v2, v3, 
                                            r, th, ph, 
                                            dt, tau_cool)
        
        # 4. 收敛检查 (每 100 步检查一次，在 Python 层做，不影响 JIT 内部效率)
        if step % 100 == 0:
            # 只计算有效区域的误差
            mask_active = N_next > 1e-20
            if np.sum(mask_active) > 0:
                diff = np.max(np.abs(N_next[mask_active] - N_curr[mask_active]))
                max_val = np.max(N_next)
                rel_err = diff / max_val
            else:
                rel_err = 0.0
                
            status = "Evolving" if step < min_iter else "Checking"
            print(f"    Step {step:04d} [{status}]: Max Val={np.max(N_next):.2e}, Rel Err={rel_err:.2e}")
            
            if step > min_iter and rel_err < tol:
                print(f"    >>> Converged at step {step}")
                N_curr = N_next
                Q_curr = Q_next
                break
        
        # 更新数组引用
        N_curr = N_next
        Q_curr = Q_next

    # 4. 重建结果
    mask_valid = N_curr > 1e-20
    q_evolved = np.zeros_like(q_source)
    # 避免除以零
    np.divide(Q_curr, N_curr, out=q_evolved, where=mask_valid)
    q_evolved[~mask_valid] = 3.0 # Default p
    
    evolved_props = copy.deepcopy(nonthermal_props)
    evolved_props['C_grid'] = N_curr
    evolved_props['q_grid'] = q_evolved
    
    return evolved_props