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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LogNorm

try:
    import plotly.graph_objects as go
except ImportError:
    go = None
# --------------------------------------------------------------------------

def find_shocks_in_roi_mhd(
    roi_data,
    gamma=4.0 / 3.0,
    grad_p_filter_quantile=0.10,
    march_cells=5,
    rho_unit=1.0,
    sr_mach_min=1.2,
    jump_residual_max=0.4,
    compressibility_gate=True,
    discontinuity_rel_jump_min=0.05,
    smeared_sr_mach_min=1.2,
    logger=None,
    **deprecated_controls,
):
    """
    【3D MHD 稳健版激波探测器】
    针对 M87 GRMHD (球面坐标) 优化：
    1. 使用总压 P_tot = P_gas + 0.5*B^2 进行梯度计算 [cite: 7836, 8412]。
    2. 使用快磁声速 v_fast 作为特征速度 [cite: 7836, 8417]。
    3. 支持球面坐标系下的梯度修正 (r, theta, phi) 。
    """
    removed_controls = [key for key in ("mach_threshold_loose", "min_physical_mach") if key in deprecated_controls]
    if removed_controls:
        raise ValueError(
            "Removed classical shock controls detected: "
            + ", ".join(removed_controls)
            + ". Candidate screening is now SRMHD-mainline only."
        )

    print(
        "Starting 3D MHD shock detection "
        f"(three-gate candidate scaffold: compressibility={compressibility_gate}, "
        f"discontinuity_rel_jump>={discontinuity_rel_jump_min}, "
        f"smeared_sr_mach>={smeared_sr_mach_min}, "
        f"sr_mach_min={sr_mach_min}, jump_residual_max={jump_residual_max})..."
    )
    if logger:
        logger.ai.func_enter(
            "find_shocks_in_roi_mhd",
            {
                "gamma": gamma,
                "march_cells": march_cells,
                "rho_unit": rho_unit,
                "compressibility_gate": compressibility_gate,
                "discontinuity_rel_jump_min": discontinuity_rel_jump_min,
                "smeared_sr_mach_min": smeared_sr_mach_min,
                "sr_mach_min": sr_mach_min,
                "jump_residual_max": jump_residual_max,
                "rho_shape": roi_data["rho"].shape,
            },
        )
        if deprecated_controls:
            logger.ai.debug(f"Ignored extra controls={sorted(deprecated_controls.keys())}")
    
    # --- 1. 数据准备 ---
    press = roi_data['press']
    rho = roi_data['rho']
    vel1, vel2, vel3 = roi_data['vel1'], roi_data['vel2'], roi_data['vel3']
    b1, b2, b3 = roi_data['Bcc1'], roi_data['Bcc2'], roi_data['Bcc3']

    def _safe_axis_gradient(array):
        gradients = []
        for axis in range(array.ndim):
            if array.shape[axis] < 2:
                gradients.append(np.zeros_like(array))
            else:
                gradients.append(np.gradient(array, axis=axis))
        return gradients
    
    nk, nj, ni = press.shape
    
    # 坐标准备 (用于梯度修正)
    r_coords = (roi_data['x1f'][:-1] + roi_data['x1f'][1:]) / 2.0
    theta_coords = (roi_data['x2f'][:-1] + roi_data['x2f'][1:]) / 2.0
    phi_coords = (roi_data['x3f'][:-1] + roi_data['x3f'][1:]) / 2.0
    r_grid = r_coords[np.newaxis, np.newaxis, :]
    theta_grid = theta_coords[np.newaxis, :, np.newaxis]
    sin_theta_grid = np.sin(theta_grid)

    # --- 2. 计算 MHD 核心物理量 ---
    # a. 计算磁压与总压 [cite: 7836, 8412]
    b_sq = b1**2 + b2**2 + b3**2
    p_mag = 0.5 * b_sq
    p_tot = press + p_mag
    

    # --- 3. 计算 3D 梯度 (球面坐标系修正) ---
    # np.gradient 返回顺序对应 (dim0, dim1, dim2) -> (phi, theta, r)
    grad_P_phi_raw, grad_P_theta_raw, grad_P_r_raw = _safe_axis_gradient(p_tot)

    # 物理距离缩放
    dr = np.gradient(r_coords) if r_coords.size > 1 else np.ones_like(r_coords)
    dtheta = np.gradient(theta_coords) if theta_coords.size > 1 else np.ones_like(theta_coords)
    dphi = np.gradient(phi_coords) if phi_coords.size > 1 else np.ones_like(phi_coords)

    # 广播坐标差
    dR_grid = dr[np.newaxis, np.newaxis, :]
    dT_grid = dtheta[np.newaxis, :, np.newaxis]
    dP_grid = dphi[:, np.newaxis, np.newaxis]

    grad_P_r = grad_P_r_raw / dR_grid
    grad_P_theta = grad_P_theta_raw / (r_grid * dT_grid + 1e-30)
    grad_P_phi = grad_P_phi_raw / (r_grid * sin_theta_grid * dP_grid + 1e-30)

    grad_P_mag = np.sqrt(grad_P_r**2 + grad_P_theta**2 + grad_P_phi**2) + 1e-30


    # --- 4. 三道门候选点筛选 ---
    # 法向量 n = grad(P_tot) / |grad(P_tot)| [cite: 7833]
    n_r, n_theta, n_phi = grad_P_r/grad_P_mag, grad_P_theta/grad_P_mag, grad_P_phi/grad_P_mag
    v_dot_n = vel1 * n_r + vel2 * n_theta + vel3 * n_phi

    # Gate 1: 汇聚流门槛 (Compressibility Gate)

    # 局部网格尺度 (沿法向的近似)
    dl_eff = np.abs(n_r * dR_grid + n_theta * r_grid * dT_grid + n_phi * r_grid * sin_theta_grid * dP_grid) + 1e-30
    # 流体必须顺着压强梯度方向撞向高压
    compressibility_mask = v_dot_n > 0
    compressibility_count = int(np.sum(compressibility_mask))

    # Gate 2: 间断强度门槛 (Discontinuity Gate)
    # 压强梯度跨越网格的相对跳跃 > threshold
    # relative_jump = |grad_P| * dl_eff / P_tot
    relative_jump = grad_P_mag * dl_eff / (p_tot + 1e-30)
    discontinuity_mask = relative_jump >= discontinuity_rel_jump_min
    discontinuity_count = int(np.sum(discontinuity_mask & compressibility_mask))
    # Gate 3:  SRMHD-inspired局地马赫数初筛 (Smeared Local SR-Mach)
    # 假定我们这里找到的马赫数是经过数值涂抹的局地马赫数，要求其超过一定阈值
    pre_mach_mask = compressibility_mask & discontinuity_mask
    local_sr_mach = np.zeros_like(press)
    if np.any(pre_mach_mask):
        k_pre, j_pre, i_pre = np.where(pre_mach_mask)
        r_pre = r_coords[i_pre]
        sin_theta_pre = np.sin(theta_coords[j_pre])
        press_pre = press[k_pre, j_pre, i_pre]
        rho_pre = rho[k_pre, j_pre, i_pre]
        p_mag_pre = p_mag[k_pre, j_pre, i_pre]
        b_sq_pre = b_sq[k_pre, j_pre, i_pre]
        n_r_pre = n_r[k_pre, j_pre, i_pre]
        n_theta_pre = n_theta[k_pre, j_pre, i_pre]
        n_phi_pre = n_phi[k_pre, j_pre, i_pre]
        utilde_n_pre = (
            vel1[k_pre, j_pre, i_pre] * n_r_pre
            + (r_pre * vel2[k_pre, j_pre, i_pre]) * n_theta_pre
            + (r_pre * sin_theta_pre * vel3[k_pre, j_pre, i_pre]) * n_phi_pre
        )
        w_local_pre = rho_pre + press_pre / (gamma - 1.0) + press_pre + p_mag_pre
        cs_sq_local_pre = np.divide(gamma * press_pre, w_local_pre, out=np.zeros_like(press_pre), where=w_local_pre > 0)
        va_sq_local_pre = np.divide(
            b_sq_pre,
            w_local_pre + b_sq_pre,
            out=np.zeros_like(b_sq_pre),
            where=(w_local_pre + b_sq_pre) > 0,
        )
        b_mag_pre = np.sqrt(b_sq_pre)
        b_dot_n_pre = (
            b1[k_pre, j_pre, i_pre] * n_r_pre
            + b2[k_pre, j_pre, i_pre] * n_theta_pre
            + b3[k_pre, j_pre, i_pre] * n_phi_pre
        )
        cos_theta_bn_pre = np.divide(np.abs(b_dot_n_pre), b_mag_pre, out=np.zeros_like(b_mag_pre), where=b_mag_pre > 0)
        cos_theta_bn_pre = np.clip(cos_theta_bn_pre, 0.0, 1.0)
        sin_theta_sq_pre = np.maximum(0.0, 1.0 - cos_theta_bn_pre**2)
        cfast_sq_local_pre = cs_sq_local_pre + va_sq_local_pre * sin_theta_sq_pre - cs_sq_local_pre * va_sq_local_pre * sin_theta_sq_pre
        cfast_sq_local_pre = np.clip(cfast_sq_local_pre, 0.0, 1.0 - 1e-12)
        cfast_local_pre = np.sqrt(cfast_sq_local_pre)
        u_fast_local_pre = np.divide(cfast_local_pre, np.sqrt(np.maximum(1.0 - cfast_sq_local_pre, 1e-12)))
        local_sr_mach[pre_mach_mask] = np.divide(np.abs(utilde_n_pre), u_fast_local_pre, out=np.zeros_like(utilde_n_pre), where=u_fast_local_pre > 0)
    smeared_mach_mask = local_sr_mach >= smeared_sr_mach_min
    smeared_mach_count = int(np.sum(smeared_mach_mask & pre_mach_mask))

    # 综合三门
    candidate_mask = compressibility_mask & discontinuity_mask & smeared_mach_mask

    geom_candidate_mask = candidate_mask.copy()
    geom_candidate_count = int(np.sum(geom_candidate_mask))

    # 打印筛选统计
    gate_stats = {
        "compressibility_gate": compressibility_count,
        "discontinuity_gate": discontinuity_count,
        "smeared_sr_mach_gate": smeared_mach_count,
        "final_candidates": geom_candidate_count,
    }
    print(f"  Gate 1 (compressibility): {compressibility_count} cells")
    print(f"  Gate 2 (discontinuity, rel_jump>={discontinuity_rel_jump_min}): {discontinuity_count} cells")
    print(f"  Gate 3 (smeared SR-Mach>={smeared_sr_mach_min}): {smeared_mach_count} cells")
    print(f"  Final candidates: {geom_candidate_count} cells")
    if logger:
        logger.ai.data("shock.candidate_gate_stats", gate_stats)

    candidate_indices = np.argwhere(candidate_mask)
    if logger:
        logger.ai.debug(f"Shock candidate marching: {len(candidate_indices)} cells to sample")

    # --- 5. 稳健验证 (Gradient Marching) ---
    mainline_mach_grid = np.zeros_like(press)
    rho1_code_grid = np.zeros_like(press)
    press1_code_grid = np.zeros_like(press)
    bsq1_code_grid = np.zeros_like(press)
    beta1_grid = np.zeros_like(press)
    rho2_code_grid = np.zeros_like(press)
    press2_code_grid = np.zeros_like(press)
    press2_over_rho2_grid = np.zeros_like(press)
    bsq2_code_grid = np.zeros_like(press)
    beta2_grid = np.zeros_like(press)
    sigma2_grid = np.zeros_like(press)
    verified_shock_mask = np.zeros_like(press, dtype=bool)
    refined_shock_mask = np.zeros_like(press, dtype=bool)
    h_rel_upstream_grid = np.zeros_like(press)
    w_rel_upstream_grid = np.zeros_like(press)
    v_n_upstream_grid = np.zeros_like(press)
    u_n_upstream_grid = np.zeros_like(press)
    utilde_sq_upstream_grid = np.zeros_like(press)
    gamma_lorentz_upstream_grid = np.zeros_like(press)
    utilde_n_upstream_grid = np.zeros_like(press)
    theta_bn_grid = np.zeros_like(press)
    cfast_n_upstream_grid = np.zeros_like(press)
    sr_sonic_mach_grid = np.zeros_like(press)
    sr_mach_normal_grid = np.zeros_like(press)
    ptot_jump_grid = np.zeros_like(press)
    entropy_jump_grid = np.zeros_like(press)
    jump_residual_light_grid = np.zeros_like(press)

    def _beta_from_press_bsq(local_press, local_bsq):
        return np.divide(2.0 * local_press, local_bsq, out=np.full((), np.inf, dtype=float), where=local_bsq > 0)

    def _sigma_from_state(local_rho, local_press, local_bsq):
        local_uu = local_press / (gamma - 1.0)
        local_denom = local_rho + local_uu + local_press
        return np.divide(local_bsq, 2.0 * local_denom, out=np.zeros((), dtype=float), where=local_denom > 0)

    sample_k2_grid = np.full_like(press, -1, dtype=int)
    sample_j2_grid = np.full_like(press, -1, dtype=int)
    sample_i2_grid = np.full_like(press, -1, dtype=int)
    sample_boundary_clipped_grid = np.zeros_like(press, dtype=bool)

    boundary_clip_count = 0
    accepted_boundary_clip_count = 0
    sr_reject_low_mach_count = 0
    sr_reject_jump_count = 0
    sr_reject_entropy_count = 0
    sr_accepted_count = 0

    # 逻辑梯度用于索引回溯
    gk, gj, gi = grad_P_phi_raw, grad_P_theta_raw, grad_P_r_raw

    for k, j, i in candidate_indices:
        dominant_axis = np.argmax([np.abs(gk[k, j, i]), np.abs(gj[k, j, i]), np.abs(gi[k, j, i])])

        dk, dj, di = 0, 0, 0
        if dominant_axis == 0:
            dk = -int(np.sign(gk[k, j, i]))
        elif dominant_axis == 1:
            dj = -int(np.sign(gj[k, j, i]))
        else:
            di = -int(np.sign(gi[k, j, i]))

        ku_raw, ju_raw, iu_raw = k + dk * march_cells, j + dj * march_cells, i + di * march_cells
        kd_raw, jd_raw, id_raw = k - dk * march_cells, j - dj * march_cells, i - di * march_cells
        ku, ju, iu = np.clip([ku_raw, ju_raw, iu_raw], 0, [nk - 1, nj - 1, ni - 1])
        kd, jd, id_ = np.clip([kd_raw, jd_raw, id_raw], 0, [nk - 1, nj - 1, ni - 1])
        boundary_clipped = (
            (ku != ku_raw)
            or (ju != ju_raw)
            or (iu != iu_raw)
            or (kd != kd_raw)
            or (jd != jd_raw)
            or (id_ != id_raw)
        )
        if boundary_clipped:
            boundary_clip_count += 1

        p1_tot = p_tot[ku, ju, iu]
        p2_tot = p_tot[kd, jd, id_]
        if p2_tot <= p1_tot * 1.05:
            continue

        rho2 = rho[kd, jd, id_]
        press2 = press[kd, jd, id_]
        bsq2 = b_sq[kd, jd, id_]

        verified_shock_mask[k, j, i] = True
        rho2_code_grid[k, j, i] = rho2
        press2_code_grid[k, j, i] = press2
        press2_over_rho2_grid[k, j, i] = np.divide(
            press2,
            rho2,
            out=np.array(0.0, dtype=float),
            where=rho2 > 0,
        )
        bsq2_code_grid[k, j, i] = bsq2
        beta2_grid[k, j, i] = _beta_from_press_bsq(press2, bsq2)
        sigma2_grid[k, j, i] = _sigma_from_state(rho2, press2, bsq2)
        sample_k2_grid[k, j, i] = kd
        sample_j2_grid[k, j, i] = jd
        sample_i2_grid[k, j, i] = id_
        sample_boundary_clipped_grid[k, j, i] = boundary_clipped
        if boundary_clipped:
            accepted_boundary_clip_count += 1

        rho1 = rho[ku, ju, iu]
        press1 = press[ku, ju, iu]
        bsq1 = b_sq[ku, ju, iu]
        rho1_code_grid[k, j, i] = rho1
        press1_code_grid[k, j, i] = press1
        bsq1_code_grid[k, j, i] = bsq1
        beta1_grid[k, j, i] = _beta_from_press_bsq(press1, bsq1)

        local_r = r_coords[i]
        local_theta = theta_coords[j]
        sin_theta = np.sin(local_theta)
        g11 = 1.0
        g22 = local_r**2
        g33 = (local_r * sin_theta) ** 2
        sqrt_g11 = 1.0
        sqrt_g22 = local_r
        sqrt_g33 = np.abs(local_r * sin_theta)

        utilde_sq1 = (
            g11 * vel1[ku, ju, iu] ** 2
            + g22 * vel2[ku, ju, iu] ** 2
            + g33 * vel3[ku, ju, iu] ** 2
        )
        utilde_sq2 = (
            g11 * vel1[kd, jd, id_] ** 2
            + g22 * vel2[kd, jd, id_] ** 2
            + g33 * vel3[kd, jd, id_] ** 2
        )

        w1 = rho1 + press1 / (gamma - 1.0) + press1
        h1 = np.divide(w1, rho1, out=np.zeros((), dtype=float), where=rho1 > 0)

        utilde_r1 = sqrt_g11 * vel1[ku, ju, iu]
        utilde_theta1 = sqrt_g22 * vel2[ku, ju, iu]
        utilde_phi1 = sqrt_g33 * vel3[ku, ju, iu]
        utilde_r2 = sqrt_g11 * vel1[kd, jd, id_]
        utilde_theta2 = sqrt_g22 * vel2[kd, jd, id_]
        utilde_phi2 = sqrt_g33 * vel3[kd, jd, id_]

        utilde_n1 = (
            utilde_r1 * n_r[k, j, i]
            + utilde_theta1 * n_theta[k, j, i]
            + utilde_phi1 * n_phi[k, j, i]
        )
        utilde_n2 = (
            utilde_r2 * n_r[k, j, i]
            + utilde_theta2 * n_theta[k, j, i]
            + utilde_phi2 * n_phi[k, j, i]
        )
        gamma_lorentz_1 = np.sqrt(1.0 + utilde_sq1)
        gamma_lorentz_2 = np.sqrt(1.0 + utilde_sq2)
        v_n1 = np.divide(utilde_n1, gamma_lorentz_1, out=np.zeros((), dtype=float), where=gamma_lorentz_1 > 0)
        v_n2 = np.divide(utilde_n2, gamma_lorentz_2, out=np.zeros((), dtype=float), where=gamma_lorentz_2 > 0)
        u_n1 = utilde_n1
        u_n2 = utilde_n2

        utilde_sq_upstream_grid[k, j, i] = utilde_sq1
        gamma_lorentz_upstream_grid[k, j, i] = gamma_lorentz_1
        utilde_n_upstream_grid[k, j, i] = utilde_n1

        cs_sq_sr = np.divide(gamma * press1, w1, out=np.zeros((), dtype=float), where=w1 > 0)
        cs_sq_sr = np.clip(cs_sq_sr, 0.0, 1.0 - 1e-12)
        csonic = np.sqrt(cs_sq_sr)
        u_sonic = np.divide(csonic, np.sqrt(np.maximum(1.0 - cs_sq_sr, 1e-12)))
        sr_sonic_mach = np.divide(np.abs(u_n1), u_sonic, out=np.zeros((), dtype=float), where=u_sonic > 0)
        va_sq_sr = np.divide(bsq1, w1 + bsq1, out=np.zeros((), dtype=float), where=(w1 + bsq1) > 0)
        b_mag1 = np.sqrt(bsq1)
        b_dot_n = b1[ku, ju, iu] * n_r[k, j, i] + b2[ku, ju, iu] * n_theta[k, j, i] + b3[ku, ju, iu] * n_phi[k, j, i]
        cos_theta_bn = np.divide(np.abs(b_dot_n), b_mag1, out=np.zeros((), dtype=float), where=b_mag1 > 0)
        cos_theta_bn = np.clip(cos_theta_bn, 0.0, 1.0)
        sin_theta_sq = np.maximum(0.0, 1.0 - cos_theta_bn**2)
        theta_bn = np.arccos(cos_theta_bn)

        cfast_sq = cs_sq_sr + va_sq_sr * sin_theta_sq - cs_sq_sr * va_sq_sr * sin_theta_sq
        cfast_sq = np.clip(cfast_sq, 0.0, 1.0 - 1e-12)
        cfast_n = np.sqrt(cfast_sq)
        u_fast = np.divide(cfast_n, np.sqrt(np.maximum(1.0 - cfast_sq, 1e-12)))
        sr_mach = np.divide(np.abs(u_n1), u_fast, out=np.zeros((), dtype=float), where=u_fast > 0)

        rho_flux_up = rho1 * u_n1
        rho_flux_down = rho2 * u_n2
        entropy_up = np.divide(press1, np.power(rho1, gamma), out=np.zeros((), dtype=float), where=rho1 > 0)
        entropy_down = np.divide(press2, np.power(rho2, gamma), out=np.zeros((), dtype=float), where=rho2 > 0)
        entropy_jump = entropy_down - entropy_up
        ptot_jump = p2_tot - p1_tot
        jump_residual_light = np.divide(np.abs(rho_flux_down - rho_flux_up), np.abs(rho_flux_up) + 1e-30)
        jump_residual_light += np.divide(np.maximum(0.0, -ptot_jump), p1_tot + 1e-30)

        h_rel_upstream_grid[k, j, i] = h1
        w_rel_upstream_grid[k, j, i] = w1
        v_n_upstream_grid[k, j, i] = v_n1
        u_n_upstream_grid[k, j, i] = u_n1
        theta_bn_grid[k, j, i] = theta_bn
        cfast_n_upstream_grid[k, j, i] = cfast_n
        sr_sonic_mach_grid[k, j, i] = sr_sonic_mach
        sr_mach_normal_grid[k, j, i] = sr_mach
        ptot_jump_grid[k, j, i] = ptot_jump
        entropy_jump_grid[k, j, i] = entropy_jump
        jump_residual_light_grid[k, j, i] = jump_residual_light

        sr_accept = True
        if sr_mach <= sr_mach_min:
            sr_reject_low_mach_count += 1
            sr_accept = False
        elif jump_residual_light >= jump_residual_max:
            sr_reject_jump_count += 1
            sr_accept = False
        elif entropy_jump <= 0:
            sr_reject_entropy_count += 1
            sr_accept = False

        if sr_accept:
            refined_shock_mask[k, j, i] = True
            mainline_mach_grid[k, j, i] = sr_mach
            sr_accepted_count += 1

    verified_cells = int(np.sum(verified_shock_mask))
    refined_cells = int(np.sum(refined_shock_mask))
    sampling_stats = {
        "gate_stats": gate_stats,
        "geom_candidate_count": int(geom_candidate_count),
        "candidate_count": int(len(candidate_indices)),
        "verified_count": verified_cells,
        "boundary_clipped_candidate_count": int(boundary_clip_count),
        "boundary_clipped_verified_count": int(accepted_boundary_clip_count),
        "sr_refined_count": int(sr_accepted_count),
        "sr_rejected_low_mach_count": int(sr_reject_low_mach_count),
        "sr_rejected_jump_count": int(sr_reject_jump_count),
        "sr_rejected_entropy_count": int(sr_reject_entropy_count),
    }
    print(f"  Verified {verified_cells} candidate shock cells.")
    print(
        "  Mainline SRMHD shock selection: "
        f"accepted={refined_cells}, low_mach={sr_reject_low_mach_count}, "
        f"jump={sr_reject_jump_count}, entropy={sr_reject_entropy_count}"
    )
    print(
        "  Downstream sampling diagnostics: "
        f"candidate_clipped={boundary_clip_count}, accepted_clipped={accepted_boundary_clip_count}"
    )
    if logger:
        logger.ai.debug(f"Verified shock cells={verified_cells}")
        logger.ai.debug(f"Sampling diagnostics={sampling_stats}")
        if np.any(candidate_mask):
            logger.ai.data("shock.relative_jump.candidates", relative_jump[candidate_mask])
            logger.ai.data("shock.local_sr_mach.candidates", local_sr_mach[candidate_mask])
        if refined_cells > 0:
            logger.ai.data("shock.mainline_mach.active", mainline_mach_grid[refined_shock_mask])
            logger.ai.data("shock.rho1_code.active", rho1_code_grid[refined_shock_mask])
            logger.ai.data("shock.press1_code.active", press1_code_grid[refined_shock_mask])
            logger.ai.data("shock.beta1.active", beta1_grid[refined_shock_mask])
            logger.ai.data("shock.rho2_code.active", rho2_code_grid[refined_shock_mask])
            logger.ai.data("shock.press2_code.active", press2_code_grid[refined_shock_mask])
            logger.ai.data("shock.bsq2_code.active", bsq2_code_grid[refined_shock_mask])
            logger.ai.data("shock.beta2.active", beta2_grid[refined_shock_mask])
            logger.ai.data("shock.sigma2.active", sigma2_grid[refined_shock_mask])
            logger.ai.data("shock.sr_mach_normal.active", sr_mach_normal_grid[refined_shock_mask])
            logger.ai.data("shock.utilde_sq_upstream.active", utilde_sq_upstream_grid[refined_shock_mask])
            logger.ai.data("shock.gamma_lorentz_upstream.active", gamma_lorentz_upstream_grid[refined_shock_mask])
            logger.ai.data("shock.utilde_n_upstream.active", utilde_n_upstream_grid[refined_shock_mask])
            logger.ai.data("shock.theta_bn.active", theta_bn_grid[refined_shock_mask])
            logger.ai.data("shock.sr_sonic_mach.active", sr_sonic_mach_grid[refined_shock_mask])
            logger.ai.data("shock.jump_residual_light.active", jump_residual_light_grid[refined_shock_mask])
            logger.ai.codepath(
                "Shock detection branch",
                f"SRMHD mainline shock cells present, verified={verified_cells}, accepted={refined_cells}",
            )
        else:
            logger.ai.codepath("Shock detection branch", "no SRMHD mainline shock cells")

    result = {
        "mask": refined_shock_mask,
        "verified_mask": verified_shock_mask,
        "mainline_mach": mainline_mach_grid,
        "rho1_code_grid": rho1_code_grid,
        "press1_code_grid": press1_code_grid,
        "bsq1_code_grid": bsq1_code_grid,
        "beta1_grid": beta1_grid,
        "rho2_code_grid": rho2_code_grid,
        "press2_code_grid": press2_code_grid,
        "press2_over_rho2_grid": press2_over_rho2_grid,
        "bsq2_code_grid": bsq2_code_grid,
        "beta2_grid": beta2_grid,
        "sigma2_grid": sigma2_grid,
        "h_rel_upstream": h_rel_upstream_grid,
        "w_rel_upstream": w_rel_upstream_grid,
        "v_n_upstream": v_n_upstream_grid,
        "u_n_upstream": u_n_upstream_grid,
        "utilde_sq_upstream": utilde_sq_upstream_grid,
        "gamma_lorentz_upstream": gamma_lorentz_upstream_grid,
        "utilde_n_upstream": utilde_n_upstream_grid,
        "theta_Bn": theta_bn_grid,
        "cfast_n_upstream": cfast_n_upstream_grid,
        "sr_sonic_mach": sr_sonic_mach_grid,
        "sr_mach_normal": sr_mach_normal_grid,
        "ptot_jump": ptot_jump_grid,
        "entropy_jump": entropy_jump_grid,
        "jump_residual_light": jump_residual_light_grid,
        "sample_k2_grid": sample_k2_grid,
        "sample_j2_grid": sample_j2_grid,
        "sample_i2_grid": sample_i2_grid,
        "sample_boundary_clipped_grid": sample_boundary_clipped_grid,
        "sampling_stats": sampling_stats,
        "grad_p_mag": grad_P_mag,
    }
    if logger:
        logger.ai.func_exit(
            "find_shocks_in_roi_mhd",
            {"verified_cells": verified_cells, "result_keys": sorted(result.keys())},
        )
    return result


def visualize_shock_3d_interactive_html(roi_data, shock_properties, snapshot_name, html_plot_filename,
                                        x_lim=260.0, y_lim=260.0, z_lim=1200.0):
    """【Interactive HTML 3D版 v3】"""
    if go is None:
        raise ImportError("plotly is required for visualize_shock_3d_interactive_html")
    print(f"Generating INTERACTIVE 3D shock visualization (X,Y < {x_lim}, Z < {z_lim} r_g)...")
    # ... (代码来自 shock_v1.py / workflowShock_processpool.py) ...
    shock_mask_roi = shock_properties['mask']
    upstream_mach_roi = shock_properties['mainline_mach']
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
            opacity=0.7, colorbar=dict(title='Mainline SR Mach'),
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
    


# --- 主程序入口 ---
if __name__ == '__main__':
    # 示例：处理单个文件
 
    test_file = 'F:\\Research\\Shockwave\\data_test\\mad98.prim.00200.athdf' 

    # 最终您会在这里写一个循环来处理所有1000多个文件
    # import glob
    # file_list = sorted(glob.glob('path/to/your/data/mad98.prim.*.athdf'))
    # for f in file_list:
    #     analyze_snapshot(f)
