#!/usr/bin/env python3
"""Quantify why shock-downstream Dirichlet advection is too weak.

This script is intended to run on the server with real Athena++ data.
It measures:
1. Retained vs rejected shock injection strength bias
2. Shock-cell UNTH vs downstream-cell UNTH vs local-max UNTH
3. Whether cooling is the dominant suppression mechanism

Default output:
    OUTPUT/diagnostics/advection_dirichlet/<snapshot>_diagnostics.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from run_dsa_pipeline import create_default_config
from src.core.advection_v0 import (
    _build_dirichlet_face_map,
    _compute_face_speed_line_py,
    estimate_cooling_time_jit,
    solve_steady_advection,
)
from src.core.shock_v1 import compute_comoving_magnetic_geometry
from src.workflows.base_workflow import calculate_dsa_physics, load_and_slice_data


def _summary(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"count": 0}
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"count": int(arr.size), "finite_count": 0}
    return {
        "count": int(arr.size),
        "finite_count": int(finite.size),
        "sum": float(np.sum(finite)),
        "min": float(np.min(finite)),
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90.0)),
        "p95": float(np.percentile(finite, 95.0)),
        "p99": float(np.percentile(finite, 99.0)),
        "max": float(np.max(finite)),
    }


def _prepare_physics(snapshot: str):
    config = create_default_config()
    athdf = os.path.join(config["data_directory"], f"{snapshot}.athdf")
    if not os.path.exists(athdf):
        raise FileNotFoundError(f"Input snapshot not found: {athdf}")

    roi_data = load_and_slice_data(athdf, config["roi_params"])
    shock_sr_cfg = dict(config.get("shock_sr", {}))
    shock_params = dict(config["shock_params"])
    shock_params.update(
        {
            "sr_mach_min": shock_sr_cfg.get("sr_mach_min", 1.7),
            "jump_residual_max": shock_sr_cfg.get("jump_residual_max", 0.8),
        }
    )

    physics_cfg = config["physics"]
    g_cgs = 6.674e-8
    m_sun = 1.989e33
    c_light = 2.998e10
    m_unit = physics_cfg.get("M_unit", 1.0e25)
    mbh_solar = physics_cfg.get("MBH_solar", 6.2e9)
    l_unit = g_cgs * (mbh_solar * m_sun) / c_light**2
    rho_unit = m_unit / l_unit**3

    nt_params = dict(config["nt_params"])
    nt_params["rho_unit"] = rho_unit
    nt_params["u_unit"] = rho_unit * c_light**2

    shock_props, nonthermal_props = calculate_dsa_physics(roi_data, shock_params, nt_params)
    return config, roi_data, shock_props, nonthermal_props


def _collect_dirichlet_bias(roi_data, shock_props, nonthermal_props, config):
    n_init = np.asarray(nonthermal_props["unth_code_grid"], dtype=float)
    vel1 = np.asarray(roi_data["vel1"], dtype=float)
    mask = np.asarray(shock_props["mask"], dtype=bool)
    sample_k2 = np.asarray(shock_props["sample_k2_grid"], dtype=int)
    sample_j2 = np.asarray(shock_props["sample_j2_grid"], dtype=int)
    sample_i2 = np.asarray(shock_props["sample_i2_grid"], dtype=int)

    face_mask, face_value, face_stats = _build_dirichlet_face_map(shock_props, n_init, vel1)

    magnetic_geom = compute_comoving_magnetic_geometry(roi_data)
    b_sq = np.asarray(magnetic_geom["b_sq_comoving"], dtype=float)
    gamma_min = np.asarray(nonthermal_props["gamma_min_grid"], dtype=float)
    gamma_char = np.maximum(gamma_min, 1.0 + 1.0e-3)
    tau_cool = estimate_cooling_time_jit(
        b_sq,
        gamma_char,
        float(config["physics"].get("cooling_factor", 50.0)),
    )
    dr = np.diff(np.asarray(roi_data["x1f"], dtype=float))

    shock_vals = []
    downstream_vals = []
    localmax_vals = []

    retained_shock_vals = []
    retained_face_vals = []
    retained_downstream_vals = []
    retained_localmax_vals = []
    retained_tau_vals = []
    retained_tau_over_crossing = []

    rejected_nonradial_vals = []
    rejected_samecell_vals = []
    rejected_flow_vals = []
    rejected_nonpositive_vals = []

    for k, j, i in np.argwhere(mask):
        v = float(n_init[k, j, i])
        shock_vals.append(v)

        kd = int(sample_k2[k, j, i])
        jd = int(sample_j2[k, j, i])
        id_ = int(sample_i2[k, j, i])

        if 0 <= kd < n_init.shape[0] and 0 <= jd < n_init.shape[1] and 0 <= id_ < n_init.shape[2]:
            downstream_vals.append(float(n_init[kd, jd, id_]))
        else:
            downstream_vals.append(0.0)

        lo = max(i - 1, 0)
        hi = min(i + 2, n_init.shape[2])
        localmax_vals.append(float(np.max(n_init[k, j, lo:hi])))

        if kd != k or jd != j:
            rejected_nonradial_vals.append(v)
            continue
        if id_ == i or id_ < 0 or id_ >= n_init.shape[2]:
            rejected_samecell_vals.append(v)
            continue
        if v <= 0.0:
            rejected_nonpositive_vals.append(v)
            continue

        face_speed = _compute_face_speed_line_py(vel1[k, j, :])
        if id_ > i:
            face_idx = i + 1
            consistent = face_speed[face_idx] > 0.0
        else:
            face_idx = i
            consistent = face_speed[face_idx] < 0.0

        if not consistent:
            rejected_flow_vals.append(v)
            continue

        retained_shock_vals.append(v)
        retained_face_vals.append(float(face_value[k, j, face_idx]))
        retained_downstream_vals.append(float(n_init[kd, jd, id_]))
        retained_localmax_vals.append(float(np.max(n_init[k, j, lo:hi])))
        retained_tau_vals.append(float(tau_cool[k, j, i]))

        speed = abs(float(face_speed[face_idx]))
        crossing = dr[min(max(face_idx - 1, 0), dr.size - 1)] / max(speed, 1.0e-30)
        retained_tau_over_crossing.append(float(tau_cool[k, j, i] / crossing))

    tau_ratio_arr = np.asarray(retained_tau_over_crossing, dtype=float)
    cooling_flags = {
        "fraction_tau_lt_crossing": float(np.mean(tau_ratio_arr < 1.0)) if tau_ratio_arr.size else 0.0,
        "fraction_tau_lt_10crossing": float(np.mean(tau_ratio_arr < 10.0)) if tau_ratio_arr.size else 0.0,
    }

    return {
        "face_stats": face_stats,
        "retained_vs_rejected_bias": {
            "all_shock_unth": _summary(shock_vals),
            "retained_shock_unth": _summary(retained_shock_vals),
            "rejected_nonradial_unth": _summary(rejected_nonradial_vals),
            "rejected_samecell_unth": _summary(rejected_samecell_vals),
            "rejected_flow_unth": _summary(rejected_flow_vals),
            "rejected_nonpositive_unth": _summary(rejected_nonpositive_vals),
        },
        "shock_vs_downstream_vs_localmax": {
            "all_shock_cell": _summary(shock_vals),
            "all_downstream_cell": _summary(downstream_vals),
            "all_localmax_3cell": _summary(localmax_vals),
            "retained_face_value": _summary(retained_face_vals),
            "retained_downstream_value": _summary(retained_downstream_vals),
            "retained_localmax_3cell": _summary(retained_localmax_vals),
        },
        "cooling_scale_diagnostics": {
            "retained_tau_cool": _summary(retained_tau_vals),
            "retained_tau_over_crossing": _summary(retained_tau_over_crossing),
            **cooling_flags,
        },
        "dirichlet_face_count": int(np.count_nonzero(face_mask)),
    }


def _collect_cooling_impact(roi_data, shock_props, nonthermal_props, config):
    cfg_nominal = json.loads(json.dumps(config))
    cfg_nocool = json.loads(json.dumps(config))
    cfg_nocool["physics"]["cooling_factor"] = 1.0e30

    result_nominal = solve_steady_advection(roi_data, shock_props, nonthermal_props, cfg_nominal)
    result_nocool = solve_steady_advection(roi_data, shock_props, nonthermal_props, cfg_nocool)

    n_init = np.asarray(nonthermal_props["unth_code_grid"], dtype=float)
    n_nom = np.asarray(result_nominal["unth_code_grid"], dtype=float)
    n_nocool = np.asarray(result_nocool["unth_code_grid"], dtype=float)

    return {
        "initial_unth": _summary(n_init),
        "nominal_cooling_unth": _summary(n_nom),
        "no_cooling_unth": _summary(n_nocool),
        "suppression_ratios": {
            "nominal_over_initial_sum": float(np.sum(n_nom) / max(np.sum(n_init), 1.0e-30)),
            "nocool_over_initial_sum": float(np.sum(n_nocool) / max(np.sum(n_init), 1.0e-30)),
            "nominal_over_nocool_sum": float(np.sum(n_nom) / max(np.sum(n_nocool), 1.0e-30)),
            "nominal_over_nocool_max": float(np.max(n_nom) / max(np.max(n_nocool), 1.0e-30)),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", default="mad98.prim.00405", help="Snapshot basename without .athdf")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Default: OUTPUT/diagnostics/advection_dirichlet/<snapshot>_diagnostics.json",
    )
    args = parser.parse_args()

    output_path = (
        Path(args.output)
        if args.output
        else REPO_ROOT / "OUTPUT" / "diagnostics" / "advection_dirichlet" / f"{args.snapshot}_diagnostics.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config, roi_data, shock_props, nonthermal_props = _prepare_physics(args.snapshot)
    results = {
        "snapshot": args.snapshot,
        "output_path": str(output_path),
        "dirichlet_bias": _collect_dirichlet_bias(roi_data, shock_props, nonthermal_props, config),
        "cooling_impact": _collect_cooling_impact(roi_data, shock_props, nonthermal_props, config),
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)

    print(f"Saved diagnostics to: {output_path}")
    print(json.dumps(results["dirichlet_bias"]["face_stats"], ensure_ascii=False, indent=2))
    print(json.dumps(results["cooling_impact"]["suppression_ratios"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
