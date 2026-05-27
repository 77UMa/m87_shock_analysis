#!/usr/bin/env python3
"""Generate lightweight report figures from a DSA HDF5 radiation interface.

This script is intended to run on the server where real HDF5 outputs live.
It reads one ``*_dsa_input.h5`` file and writes PNG/CSV/JSON artifacts that
can be synced back to the local ``Report`` workflow.

Required datasets:
    KEL, UNTH, p, GAMMA_MIN

Optional diagnostic datasets are used when present:
    P_EFF, SR_MACH_NORMAL, SONIC_MACH, sigma, sigma2, THETA_BN,
    JUMP_RESIDUAL_LIGHT, INJ_GATE, INJ_LIMIT_MODE, N_NTH,
    N_NTH_ETA, N_NTH_EPS, ETA_INJ_E, EPS_NTH_E, E_DISS_E,
    E_DISS_TOT, U_NTH_BUDGET, C_ETA, C_EPS, GAMMA_MIN_FAILURE_CODE
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from src.utils.ai_event_log import AIEventLogger


REQUIRED_DATASETS = ("KEL", "UNTH", "p", "GAMMA_MIN")
OPTIONAL_DATASETS = (
    "P_EFF",
    "SR_MACH_NORMAL",
    "SONIC_MACH",
    "sigma",
    "sigma2",
    "THETA_BN",
    "JUMP_RESIDUAL_LIGHT",
    "INJ_GATE",
    "INJ_LIMIT_MODE",
    "N_NTH",
    "N_NTH_ETA",
    "N_NTH_EPS",
    "ETA_INJ_E",
    "EPS_NTH_E",
    "E_DISS_E",
    "E_DISS_TOT",
    "U_NTH_BUDGET",
    "C_ETA",
    "C_EPS",
    "GAMMA_MIN_FAILURE_CODE",
    "SOURCE_MASK",
    "ACTIVE_MASK",
    "SOURCE_DENSITY",
    "UNTH_INITIAL",
    "RESIDUAL_ABS",
    "TAU_COOL_EFF",
    "N_NTH_OVER_N_E",
)


def _as_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _finite_positive(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr) & (arr > 0.0)]


def _finite(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def _summary(values: np.ndarray, positive_only: bool = False) -> dict[str, Any]:
    arr = _finite_positive(values) if positive_only else _finite(values)
    if arr.size == 0:
        return {
            "count": int(np.size(values)),
            "finite_count": 0,
        }
    return {
        "count": int(np.size(values)),
        "finite_count": int(arr.size),
        "sum": float(np.sum(arr)),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5.0)),
        "p16": float(np.percentile(arr, 16.0)),
        "p84": float(np.percentile(arr, 84.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "p99": float(np.percentile(arr, 99.0)),
        "max": float(np.max(arr)),
    }


def _load_h5(input_h5: Path) -> tuple[dict[str, np.ndarray], dict[str, Any], list[str]]:
    with h5py.File(input_h5, "r") as handle:
        missing_required = [name for name in REQUIRED_DATASETS if name not in handle]
        if missing_required:
            raise KeyError(f"Missing required datasets in {input_h5}: {missing_required}")

        datasets: dict[str, np.ndarray] = {}
        for name in REQUIRED_DATASETS + OPTIONAL_DATASETS:
            if name in handle:
                datasets[name] = handle[name][...]

        attrs = {key: _as_jsonable(value) for key, value in handle.attrs.items()}
        present = sorted(datasets.keys())

    reference_shape = datasets["UNTH"].shape
    for name in REQUIRED_DATASETS:
        if datasets[name].shape != reference_shape:
            raise ValueError(f"Dataset {name} has shape {datasets[name].shape}, expected {reference_shape}")
        if not np.all(np.isfinite(datasets[name])):
            raise ValueError(f"Dataset {name} contains non-finite values")
    for name, values in datasets.items():
        if name in REQUIRED_DATASETS:
            continue
        if values.shape != reference_shape:
            raise ValueError(f"Optional dataset {name} has shape {values.shape}, expected {reference_shape}")

    return datasets, attrs, present


def _projection(values: np.ndarray, axis: int, mode: str, positive_only: bool = False) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if positive_only:
        arr = np.where(arr > 0.0, arr, np.nan)

    with np.errstate(invalid="ignore", divide="ignore"):
        if mode == "sum":
            projected = np.nansum(np.nan_to_num(arr, nan=0.0), axis=axis)
        elif mode == "max":
            projected = np.nanmax(arr, axis=axis)
        elif mode == "mean":
            projected = np.nanmean(arr, axis=axis)
        elif mode == "median":
            projected = np.nanmedian(arr, axis=axis)
        else:
            raise ValueError(f"Unsupported projection mode: {mode}")

    return np.nan_to_num(projected, nan=0.0, posinf=0.0, neginf=0.0)


def _imshow(ax: plt.Axes, image: np.ndarray, title: str, cmap: str = "viridis", log: bool = False) -> None:
    data = np.asarray(image, dtype=float)
    if log:
        positive = data[data > 0.0]
        if positive.size:
            vmin = max(float(np.percentile(positive, 1.0)), 1.0e-300)
            vmax = float(np.percentile(positive, 99.5))
            norm = LogNorm(vmin=vmin, vmax=max(vmax, vmin * 1.01))
            shown = np.where(data > 0.0, data, np.nan)
            im = ax.imshow(shown.T, origin="lower", aspect="auto", cmap=cmap, norm=norm)
        else:
            im = ax.imshow(data.T, origin="lower", aspect="auto", cmap=cmap)
    else:
        im = ax.imshow(data.T, origin="lower", aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("grid index")
    ax.set_ylabel("grid index")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _sample_pair(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    valid = mask & np.isfinite(x) & np.isfinite(y)
    idx = np.flatnonzero(valid.ravel())
    if idx.size == 0:
        return np.array([]), np.array([])
    if idx.size > max_points:
        idx = rng.choice(idx, size=max_points, replace=False)
    return x.ravel()[idx], y.ravel()[idx]


def _write_stats_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "name",
        "count",
        "finite_count",
        "positive_count",
        "sum",
        "min",
        "median",
        "p05",
        "p16",
        "p84",
        "p95",
        "p99",
        "max",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _write_metrics_csv(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for key in sorted(metrics):
            writer.writerow({"metric": key, "value": metrics[key]})


def _safe_label(value: str | None, fallback: str) -> str:
    label = value or fallback
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in label)
    return safe.strip("_") or fallback


def _savefig_atomic(fig: plt.Figure, path: Path, **kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.tmp{path.suffix}")
    fig.savefig(tmp, **kwargs)
    with tmp.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _fraction(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _count_mask(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def _attr_number(attrs: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = attrs.get(key, default)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return default


def _build_report_metrics(datasets: dict[str, np.ndarray], attrs: dict[str, Any]) -> dict[str, Any]:
    kel = datasets["KEL"] > 0.0
    unth = np.asarray(datasets["UNTH"], dtype=float)
    shock_count = _count_mask(kel)
    total_cells = int(kel.size)
    unth_shock_positive = kel & (unth > 0.0)
    unth_all_positive = unth > 0.0

    metrics: dict[str, Any] = {
        "roi_cell_count": total_cells,
        "accepted_shock_count": shock_count,
        "accepted_shock_fraction": _fraction(shock_count, total_cells),
        "unth_positive_shock_count": _count_mask(unth_shock_positive),
        "unth_positive_shock_fraction": _fraction(_count_mask(unth_shock_positive), shock_count),
        "unth_positive_volume_count": _count_mask(unth_all_positive),
        "unth_positive_volume_fraction": _fraction(_count_mask(unth_all_positive), total_cells),
        "unth_shock_sum": float(np.sum(unth[kel])),
        "unth_all_positive_sum": float(np.sum(unth[unth_all_positive])),
        "unth_shock_max": float(np.max(unth[kel])) if shock_count else 0.0,
        "unth_all_max": float(np.max(unth)) if unth.size else 0.0,
        "gamma_min_fallback_risk_count": int(_attr_number(attrs, "gamma_min_fallback_risk_count", 0.0)),
        "gamma_min_fallback_risk_fraction": _attr_number(attrs, "gamma_min_fallback_risk_fraction", 0.0),
        "shock_sampling_candidate_count": int(_attr_number(attrs, "shock_sampling_candidate_count", 0.0)),
        "shock_sampling_verified_count": int(_attr_number(attrs, "shock_sampling_verified_count", 0.0)),
        "shock_sampling_sr_refined_count": int(_attr_number(attrs, "shock_sampling_sr_refined_count", shock_count)),
        "shock_sampling_sr_rejected_low_mach_count": int(
            _attr_number(attrs, "shock_sampling_sr_rejected_low_mach_count", 0.0)
        ),
        "shock_sampling_sr_rejected_jump_count": int(_attr_number(attrs, "shock_sampling_sr_rejected_jump_count", 0.0)),
        "shock_sampling_sr_rejected_entropy_count": int(
            _attr_number(attrs, "shock_sampling_sr_rejected_entropy_count", 0.0)
        ),
    }

    if "GAMMA_MIN_FAILURE_CODE" in datasets:
        ok = datasets["GAMMA_MIN_FAILURE_CODE"][kel] == int(attrs.get("gamma_failure_code_ok", 0))
        metrics["valid_gamma_min_count"] = _count_mask(ok)
        metrics["valid_gamma_min_fraction"] = _fraction(_count_mask(ok), shock_count)

    if "INJ_GATE" in datasets:
        inj_positive = kel & (datasets["INJ_GATE"] > 0.0)
        metrics["positive_inj_gate_count"] = _count_mask(inj_positive)
        metrics["positive_inj_gate_fraction"] = _fraction(_count_mask(inj_positive), shock_count)
        metrics["inj_gate_median_on_shocks"] = float(np.median(datasets["INJ_GATE"][kel])) if shock_count else 0.0

    if "INJ_LIMIT_MODE" in datasets:
        values = datasets["INJ_LIMIT_MODE"][kel].astype(int)
        total = max(values.size, 1)
        for label, attr_key in (
            ("eta_cap", "inj_limit_mode_eta_cap"),
            ("eps_cap", "inj_limit_mode_eps_cap"),
            ("quenched_by_gate", "inj_limit_mode_quenched_by_gate"),
            ("invalid_or_boundary", "inj_limit_mode_invalid_or_boundary"),
        ):
            code = attrs.get(attr_key)
            if isinstance(code, int):
                count = int(np.count_nonzero(values == code))
                metrics[f"inj_limit_{label}_count"] = count
                metrics[f"inj_limit_{label}_fraction"] = _fraction(count, total)

    for name in (
        "SR_MACH_NORMAL",
        "SONIC_MACH",
        "sigma",
        "sigma2",
        "THETA_BN",
        "JUMP_RESIDUAL_LIGHT",
        "P_EFF",
        "GAMMA_MIN",
        "N_NTH",
        "N_NTH_ETA",
        "N_NTH_EPS",
        "ETA_INJ_E",
        "EPS_NTH_E",
        "E_DISS_E",
        "E_DISS_TOT",
        "U_NTH_BUDGET",
        "RESIDUAL_ABS",
        "TAU_COOL_EFF",
        "N_NTH_OVER_N_E",
    ):
        if name not in datasets:
            continue
        values = _finite(datasets[name][kel])
        if values.size == 0:
            continue
        prefix = name.lower()
        metrics[f"{prefix}_median_on_shocks"] = float(np.median(values))
        metrics[f"{prefix}_p95_on_shocks"] = float(np.percentile(values, 95.0))
        metrics[f"{prefix}_max_on_shocks"] = float(np.max(values))
        if name in {"E_DISS_E", "E_DISS_TOT", "U_NTH_BUDGET", "N_NTH", "N_NTH_ETA", "N_NTH_EPS"}:
            metrics[f"{prefix}_sum_on_shocks"] = float(np.sum(values))

    if "SOURCE_MASK" in datasets:
        source_count = _count_mask(datasets["SOURCE_MASK"] > 0.0)
        metrics["advection_source_cell_count"] = source_count
        metrics["advection_source_cell_fraction"] = _fraction(source_count, total_cells)
    if "ACTIVE_MASK" in datasets:
        active_count = _count_mask(datasets["ACTIVE_MASK"] > 0.0)
        metrics["advection_active_cell_count"] = active_count
        metrics["advection_active_cell_fraction"] = _fraction(active_count, total_cells)
    if "SOURCE_DENSITY" in datasets:
        metrics["advection_source_density_positive_sum"] = float(np.sum(datasets["SOURCE_DENSITY"][datasets["SOURCE_DENSITY"] > 0.0]))
    if "UNTH_INITIAL" in datasets:
        initial = datasets["UNTH_INITIAL"]
        metrics["unth_initial_positive_sum"] = float(np.sum(initial[initial > 0.0]))
        metrics["unth_final_over_initial_positive_sum"] = _fraction(
            metrics["unth_all_positive_sum"],
            metrics["unth_initial_positive_sum"],
        )

    return metrics


def _make_core_projection(datasets: dict[str, np.ndarray], output_dir: Path, axis: int) -> Path:
    kel = datasets["KEL"] > 0.0
    unth = datasets["UNTH"]
    p_grid = np.where(kel, datasets["p"], np.nan)
    gamma_min = np.where(kel, datasets["GAMMA_MIN"], np.nan)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.5), constrained_layout=True)
    _imshow(axes[0, 0], _projection(kel.astype(float), axis, "max"), "Shock mask KEL", cmap="gray")
    _imshow(axes[0, 1], _projection(unth, axis, "sum", positive_only=True), "Projected UNTH", log=True)
    _imshow(axes[1, 0], _projection(p_grid, axis, "median", positive_only=True), "Median p on shock cells")
    _imshow(
        axes[1, 1],
        _projection(gamma_min, axis, "median", positive_only=True),
        "Median gamma_min on shock cells",
        log=True,
    )
    fig.suptitle("Core DSA radiation-interface fields")
    out = output_dir / "fig_core_projection.png"
    _savefig_atomic(fig, out, dpi=220)
    plt.close(fig)
    return out


def _make_injection_projection(datasets: dict[str, np.ndarray], output_dir: Path, axis: int) -> Path | None:
    available = [name for name in ("P_EFF", "INJ_GATE", "N_NTH", "N_NTH_ETA", "N_NTH_EPS") if name in datasets]
    if not available:
        return None

    kel = datasets["KEL"] > 0.0
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.5), constrained_layout=True)
    panels = [
        ("P_EFF", "Median P_EFF on shock cells", "median", False),
        ("INJ_GATE", "Median injection gate", "median", False),
        ("N_NTH", "Projected N_NTH", "sum", True),
        ("N_NTH_EPS", "Projected N_NTH_EPS", "sum", True),
    ]
    for ax, (name, title, mode, use_log) in zip(axes.ravel(), panels):
        if name not in datasets:
            ax.axis("off")
            ax.set_title(f"{name} unavailable")
            continue
        values = np.where(kel, datasets[name], np.nan)
        _imshow(ax, _projection(values, axis, mode, positive_only=True), title, log=use_log)
    fig.suptitle("Nonthermal injection diagnostics")
    out = output_dir / "fig_injection_projection.png"
    _savefig_atomic(fig, out, dpi=220)
    plt.close(fig)
    return out


def _make_phase_space(
    datasets: dict[str, np.ndarray],
    output_dir: Path,
    max_points: int,
    rng: np.random.Generator,
) -> Path | None:
    needed_any = ("SR_MACH_NORMAL", "SONIC_MACH")
    mach_name = next((name for name in needed_any if name in datasets), None)
    sigma_name = "sigma2" if "sigma2" in datasets else "sigma" if "sigma" in datasets else None
    if mach_name is None or sigma_name is None:
        return None

    kel = datasets["KEL"] > 0.0
    mach = np.asarray(datasets[mach_name], dtype=float)
    sigma = np.asarray(datasets[sigma_name], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), constrained_layout=True)

    x, y = _sample_pair(mach, sigma, kel & (sigma > 0.0), max_points, rng)
    if x.size:
        axes[0].scatter(x, y, s=2, alpha=0.25, linewidths=0)
        axes[0].set_yscale("log")
    axes[0].set_xlabel(mach_name)
    axes[0].set_ylabel(sigma_name)
    axes[0].set_title("Mach-sigma phase space")

    if "THETA_BN" in datasets:
        theta = np.degrees(np.asarray(datasets["THETA_BN"], dtype=float))
        x, y = _sample_pair(theta, sigma, kel & (sigma > 0.0), max_points, rng)
        if x.size:
            axes[1].scatter(x, y, s=2, alpha=0.25, linewidths=0)
            axes[1].set_yscale("log")
        axes[1].set_xlabel("theta_Bn [deg]")
        axes[1].set_ylabel(sigma_name)
        axes[1].set_title("Obliquity-sigma phase space")
    else:
        axes[1].axis("off")
        axes[1].set_title("THETA_BN unavailable")

    if "JUMP_RESIDUAL_LIGHT" in datasets:
        residual = _finite(np.asarray(datasets["JUMP_RESIDUAL_LIGHT"])[kel])
        axes[2].hist(residual, bins=80, histtype="step", color="tab:blue")
        axes[2].set_xlabel("jump residual")
        axes[2].set_ylabel("shock-cell count")
        axes[2].set_title("Jump-residual distribution")
    else:
        axes[2].axis("off")
        axes[2].set_title("JUMP_RESIDUAL_LIGHT unavailable")

    out = output_dir / "fig_shock_phase_space.png"
    _savefig_atomic(fig, out, dpi=220)
    plt.close(fig)
    return out


def _make_injection_funnel(datasets: dict[str, np.ndarray], attrs: dict[str, Any], output_dir: Path) -> Path:
    kel = datasets["KEL"] > 0.0
    shock_count = _count_mask(kel)
    stages: list[tuple[str, int]] = [
        ("candidate shocks", int(_attr_number(attrs, "shock_sampling_candidate_count", shock_count))),
        ("verified shocks", int(_attr_number(attrs, "shock_sampling_verified_count", shock_count))),
        ("accepted shocks", shock_count),
    ]

    if "GAMMA_MIN_FAILURE_CODE" in datasets:
        ok_code = int(attrs.get("gamma_failure_code_ok", 0))
        stages.append(("valid gamma_min", _count_mask(datasets["GAMMA_MIN_FAILURE_CODE"][kel] == ok_code)))
    if "INJ_GATE" in datasets:
        stages.append(("positive injection gate", _count_mask(kel & (datasets["INJ_GATE"] > 0.0))))
    if "UNTH" in datasets:
        stages.append(("nonzero UNTH on shocks", _count_mask(kel & (datasets["UNTH"] > 0.0))))
    if "N_NTH" in datasets:
        stages.append(("nonzero N_NTH on shocks", _count_mask(kel & (datasets["N_NTH"] > 0.0))))

    labels = [label for label, _ in stages]
    counts = np.asarray([count for _, count in stages], dtype=float)
    denominator = counts[0] if counts.size else 1.0
    fractions = np.divide(counts, denominator, out=np.zeros_like(counts), where=denominator > 0)

    fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    bars = ax.bar(np.arange(len(labels)), counts, color="tab:blue")
    ax.set_yscale("log")
    ax.set_ylabel("cell count")
    ax.set_title("DSA evidence funnel")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
    for bar, count, frac in zip(bars, counts, fractions):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            count,
            f"{int(count):,}\n{100.0 * frac:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    out = output_dir / "fig_injection_funnel.png"
    _savefig_atomic(fig, out, dpi=220)
    plt.close(fig)
    return out


def _make_budget_diagnostics(datasets: dict[str, np.ndarray], attrs: dict[str, Any], output_dir: Path) -> Path | None:
    kel = datasets["KEL"] > 0.0
    has_energy = any(name in datasets for name in ("E_DISS_E", "E_DISS_TOT", "U_NTH_BUDGET"))
    has_caps = "INJ_LIMIT_MODE" in datasets
    has_number = any(name in datasets for name in ("N_NTH", "N_NTH_ETA", "N_NTH_EPS", "UNTH"))
    if not (has_energy or has_caps or has_number):
        return None

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), constrained_layout=True)

    energy_names = [name for name in ("E_DISS_TOT", "E_DISS_E", "U_NTH_BUDGET") if name in datasets]
    if energy_names:
        values = [float(np.sum(_finite(datasets[name][kel]))) for name in energy_names]
        axes[0].bar(energy_names, values, color="tab:blue")
        axes[0].set_yscale("log")
        axes[0].set_title("Energy budget proxies")
        axes[0].set_ylabel("sum on shock cells")
        axes[0].tick_params(axis="x", rotation=20)
    else:
        axes[0].axis("off")
        axes[0].set_title("Energy fields unavailable")

    if has_caps:
        labels = _code_labels_from_attrs(attrs, "inj_limit_mode_")
        codes, counts = np.unique(datasets["INJ_LIMIT_MODE"][kel].astype(int), return_counts=True)
        total = max(int(np.sum(counts)), 1)
        xlabels = [labels.get(int(code), str(int(code))) for code in codes]
        axes[1].bar(xlabels, counts, color="tab:orange")
        axes[1].set_title("Injection limiting branch")
        axes[1].set_ylabel("shock-cell count")
        axes[1].tick_params(axis="x", rotation=25)
        for idx, count in enumerate(counts):
            axes[1].text(idx, count, f"{100.0 * count / total:.1f}%", ha="center", va="bottom", fontsize=8)
    else:
        axes[1].axis("off")
        axes[1].set_title("INJ_LIMIT_MODE unavailable")

    number_names = [name for name in ("N_NTH_ETA", "N_NTH_EPS", "N_NTH", "UNTH") if name in datasets]
    if number_names:
        values = [float(np.sum(_finite(datasets[name][kel]))) for name in number_names]
        axes[2].bar(number_names, values, color="tab:green")
        axes[2].set_yscale("log")
        axes[2].set_title("Number-density budget proxies")
        axes[2].set_ylabel("sum on shock cells")
        axes[2].tick_params(axis="x", rotation=20)
    else:
        axes[2].axis("off")
        axes[2].set_title("Number fields unavailable")

    out = output_dir / "fig_budget_diagnostics.png"
    _savefig_atomic(fig, out, dpi=220)
    plt.close(fig)
    return out


def _make_transport_budget(datasets: dict[str, np.ndarray], output_dir: Path) -> Path | None:
    available = any(name in datasets for name in ("UNTH_INITIAL", "SOURCE_DENSITY", "RESIDUAL_ABS", "TAU_COOL_EFF"))
    if not available:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0), constrained_layout=True)

    budget_names = [name for name in ("UNTH_INITIAL", "SOURCE_DENSITY", "UNTH") if name in datasets]
    values = [float(np.sum(datasets[name][datasets[name] > 0.0])) for name in budget_names]
    if budget_names:
        axes[0].bar(budget_names, values, color="tab:blue")
        axes[0].set_yscale("log")
        axes[0].set_title("Transport positive-budget sums")
        axes[0].tick_params(axis="x", rotation=20)
    else:
        axes[0].axis("off")

    if "RESIDUAL_ABS" in datasets:
        values = _finite_positive(datasets["RESIDUAL_ABS"])
        if values.size:
            axes[1].hist(np.log10(values), bins=80, histtype="step", color="tab:blue")
            axes[1].set_xlabel("log10(RESIDUAL_ABS)")
        axes[1].set_title("Transport residuals")
        axes[1].set_ylabel("cell count")
    else:
        axes[1].axis("off")
        axes[1].set_title("RESIDUAL_ABS unavailable")

    if "TAU_COOL_EFF" in datasets:
        values = _finite_positive(datasets["TAU_COOL_EFF"])
        if values.size:
            axes[2].hist(np.log10(values), bins=80, histtype="step", color="tab:blue")
            axes[2].set_xlabel("log10(TAU_COOL_EFF)")
        axes[2].set_title("Cooling timescale proxy")
        axes[2].set_ylabel("cell count")
    else:
        axes[2].axis("off")
        axes[2].set_title("TAU_COOL_EFF unavailable")

    out = output_dir / "fig_transport_budget.png"
    _savefig_atomic(fig, out, dpi=220)
    plt.close(fig)
    return out


def _make_advection_topology(datasets: dict[str, np.ndarray], output_dir: Path, axis: int) -> Path | None:
    needed = ("SOURCE_MASK", "ACTIVE_MASK")
    if not all(name in datasets for name in needed):
        return None

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.5), constrained_layout=True)
    _imshow(
        axes[0, 0],
        _projection(datasets["SOURCE_MASK"] > 0.0, axis, "max"),
        "Advection source mask",
        cmap="gray",
    )
    _imshow(
        axes[0, 1],
        _projection(datasets["ACTIVE_MASK"] > 0.0, axis, "max"),
        "Advection active mask",
        cmap="gray",
    )

    if "SOURCE_DENSITY" in datasets:
        _imshow(
            axes[1, 0],
            _projection(datasets["SOURCE_DENSITY"], axis, "sum", positive_only=True),
            "Projected source density",
            log=True,
        )
    else:
        axes[1, 0].axis("off")
        axes[1, 0].set_title("SOURCE_DENSITY unavailable")

    if "UNTH_INITIAL" in datasets:
        final = _projection(datasets["UNTH"], axis, "sum", positive_only=True)
        initial = _projection(datasets["UNTH_INITIAL"], axis, "sum", positive_only=True)
        ratio = np.divide(final, initial, out=np.zeros_like(final), where=initial > 0.0)
        _imshow(axes[1, 1], ratio, "Projected UNTH final / initial", log=True)
    else:
        _imshow(
            axes[1, 1],
            _projection(datasets["UNTH"], axis, "sum", positive_only=True),
            "Projected final UNTH",
            log=True,
        )

    fig.suptitle("Advection source-active-final topology")
    out = output_dir / "fig_advection_topology.png"
    _savefig_atomic(fig, out, dpi=220)
    plt.close(fig)
    return out


def _make_histograms(datasets: dict[str, np.ndarray], output_dir: Path) -> Path:
    kel = datasets["KEL"] > 0.0
    entries = [
        ("UNTH all positive cells", datasets["UNTH"][datasets["UNTH"] > 0.0], True),
        ("UNTH on shock cells", datasets["UNTH"][kel & (datasets["UNTH"] > 0.0)], True),
        ("p on shock cells", datasets["p"][kel], False),
        ("GAMMA_MIN on shock cells", datasets["GAMMA_MIN"][kel & (datasets["GAMMA_MIN"] > 0.0)], True),
    ]
    for name in ("P_EFF", "SR_MACH_NORMAL", "SONIC_MACH", "N_NTH"):
        if name in datasets:
            values = datasets[name][kel]
            if name == "N_NTH":
                values = values[values > 0.0]
            entries.append((name, values, name == "N_NTH"))

    ncols = 3
    nrows = int(np.ceil(len(entries) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.0, 3.3 * nrows), constrained_layout=True)
    axes_1d = np.atleast_1d(axes).ravel()
    for ax, (name, values, log_values) in zip(axes_1d, entries):
        finite = _finite(values)
        if finite.size == 0:
            ax.text(0.5, 0.5, "no finite values", ha="center", va="center", transform=ax.transAxes)
        elif log_values:
            positive = finite[finite > 0.0]
            if positive.size:
                ax.hist(np.log10(positive), bins=80, histtype="step", color="tab:blue")
                ax.set_xlabel(f"log10({name})")
            else:
                ax.text(0.5, 0.5, "no positive values", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.hist(finite, bins=80, histtype="step", color="tab:blue")
            ax.set_xlabel(name)
        ax.set_ylabel("count")
        ax.set_title(name)
    for ax in axes_1d[len(entries) :]:
        ax.axis("off")

    out = output_dir / "fig_distribution_histograms.png"
    _savefig_atomic(fig, out, dpi=220)
    plt.close(fig)
    return out


def _code_labels_from_attrs(attrs: dict[str, Any], prefix: str) -> dict[int, str]:
    labels: dict[int, str] = {}
    for key, value in attrs.items():
        if not key.startswith(prefix):
            continue
        if not isinstance(value, int):
            continue
        name = key[len(prefix) :].replace("_", " ")
        labels[int(value)] = f"{value}: {name}"
    return labels


def _make_failure_and_limit_bars(
    datasets: dict[str, np.ndarray],
    attrs: dict[str, Any],
    output_dir: Path,
) -> Path | None:
    panels = []
    kel = datasets["KEL"] > 0.0
    if "GAMMA_MIN_FAILURE_CODE" in datasets:
        panels.append(
            (
                "GAMMA_MIN_FAILURE_CODE",
                datasets["GAMMA_MIN_FAILURE_CODE"][kel],
                _code_labels_from_attrs(attrs, "gamma_failure_code_"),
            )
        )
    if "INJ_LIMIT_MODE" in datasets:
        panels.append(
            (
                "INJ_LIMIT_MODE",
                datasets["INJ_LIMIT_MODE"][kel],
                _code_labels_from_attrs(attrs, "inj_limit_mode_"),
            )
        )
    if not panels:
        return None

    fig, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 4.0), constrained_layout=True)
    axes_1d = np.atleast_1d(axes)
    for ax, (name, values, labels) in zip(axes_1d, panels):
        finite = values[np.isfinite(values)]
        codes, counts = np.unique(finite.astype(int), return_counts=True)
        total = max(int(np.sum(counts)), 1)
        xlabels = [labels.get(int(code), str(code)) for code in codes]
        ax.bar(xlabels, counts, color="tab:blue")
        for idx, count in enumerate(counts):
            ax.text(idx, count, f"{100.0 * count / total:.1f}%", ha="center", va="bottom", fontsize=8)
        ax.set_title(name)
        ax.set_xlabel("code")
        ax.set_ylabel("shock-cell count")
        ax.tick_params(axis="x", rotation=25)
    out = output_dir / "fig_failure_and_limit_codes.png"
    _savefig_atomic(fig, out, dpi=220)
    plt.close(fig)
    return out


def _collect_stats(datasets: dict[str, np.ndarray]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    manifest_stats: dict[str, Any] = {}
    kel = datasets["KEL"] > 0.0
    for name, values in datasets.items():
        if values.shape == datasets["UNTH"].shape:
            active_values = values[kel] if name != "KEL" else values
        else:
            active_values = values
        positive_count = int(np.count_nonzero(np.asarray(active_values) > 0.0))
        row = {"name": name, "positive_count": positive_count}
        row.update(_summary(active_values, positive_only=False))
        rows.append(row)
        manifest_stats[name] = row
        if name == "UNTH":
            all_positive = values[values > 0.0]
            all_row = {"name": "UNTH_ALL_POSITIVE", "positive_count": int(all_positive.size)}
            all_row.update(_summary(all_positive, positive_only=False))
            rows.append(all_row)
            manifest_stats["UNTH_ALL_POSITIVE"] = all_row
    return rows, manifest_stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5", required=True, help="Path to *_dsa_input.h5 on the server")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: OUTPUT/report_figures/<input-stem>",
    )
    parser.add_argument(
        "--projection-axis",
        type=int,
        default=2,
        choices=(0, 1, 2),
        help="Grid axis to collapse in projection figures. Default: 2.",
    )
    parser.add_argument(
        "--max-scatter-points",
        type=int,
        default=200000,
        help="Maximum points in phase-space scatter panels.",
    )
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for scatter downsampling.")
    parser.add_argument(
        "--ai-log",
        default=None,
        help="Structured JSONL event log path. Default: <output-dir>/logs/ai_events.jsonl.",
    )
    parser.add_argument("--run-label", default=None, help="Run label recorded in ai_events.jsonl.")
    parser.add_argument("--snapshot", default=None, help="Snapshot name recorded in ai_events.jsonl.")
    parser.add_argument(
        "--scratch-dir",
        default=None,
        help="Optional fast local scratch root. Input HDF5 is copied there for reading, then removed.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_h5 = Path(args.input_h5).expanduser().resolve()
    if not input_h5.exists():
        raise FileNotFoundError(f"Input HDF5 not found: {input_h5}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else Path.cwd() / "OUTPUT" / "report_figures" / input_h5.stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    ai_log_path = Path(args.ai_log).expanduser().resolve() if args.ai_log else output_dir / "logs" / "ai_events.jsonl"
    scratch_root = Path(args.scratch_dir).expanduser().resolve() if args.scratch_dir else None

    with AIEventLogger(
        ai_log_path,
        run_label=args.run_label,
        snapshot=args.snapshot or input_h5.stem,
        stage="report_figures",
    ) as ai_log:
        read_input_h5 = input_h5
        scratch_work_dir: Path | None = None
        scratch_h5: Path | None = None
        scratch_copy_time_s: float | None = None
        input_h5_bytes = input_h5.stat().st_size
        ai_log.emit(
            "run_start",
            params={
                "input_h5": str(input_h5),
                "output_dir": str(output_dir),
                "projection_axis": args.projection_axis,
                "max_scatter_points": args.max_scatter_points,
                "seed": args.seed,
                "scratch_dir": str(scratch_root) if scratch_root else None,
            },
            metrics={"input_h5_bytes": input_h5_bytes},
        )
        ai_log.emit_artifact(input_h5, kind="hdf5")

        try:
            if scratch_root is not None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                scratch_label = _safe_label(args.run_label, input_h5.stem)
                scratch_work_dir = scratch_root / f"make_report_figures_{scratch_label}_{timestamp}"
                scratch_work_dir.mkdir(parents=True, exist_ok=False)
                scratch_h5 = scratch_work_dir / input_h5.name
                copy_start = time.perf_counter()
                shutil.copy2(input_h5, scratch_h5)
                scratch_copy_time_s = time.perf_counter() - copy_start
                read_input_h5 = scratch_h5
                ai_log.emit(
                    "scratch_copy_complete",
                    status="ok",
                    params={
                        "source_h5": str(input_h5),
                        "scratch_h5": str(scratch_h5),
                        "scratch_work_dir": str(scratch_work_dir),
                    },
                    metrics={
                        "input_h5_bytes": input_h5_bytes,
                        "scratch_h5_bytes": scratch_h5.stat().st_size,
                        "copy_time_s": scratch_copy_time_s,
                    },
                )
                ai_log.emit_artifact(scratch_h5, kind="hdf5")

            read_start = time.perf_counter()
            datasets, attrs, present = _load_h5(read_input_h5)
            h5_read_time_s = time.perf_counter() - read_start
            ai_log.emit(
                "h5_loaded",
                metrics={
                    "shape": list(datasets["UNTH"].shape),
                    "dataset_count": len(present),
                    "h5_read_time_s": h5_read_time_s,
                    "scratch_copy_time_s": scratch_copy_time_s,
                },
                params={"read_input_h5": str(read_input_h5), "source_input_h5": str(input_h5)},
                present_datasets=present,
            )
            rng = np.random.default_rng(args.seed)

            artifacts: list[str] = []
            artifacts.append(str(_make_core_projection(datasets, output_dir, args.projection_axis)))
            artifacts.append(str(_make_histograms(datasets, output_dir)))
            artifacts.append(str(_make_injection_funnel(datasets, attrs, output_dir)))

            for artifact in (
                _make_injection_projection(datasets, output_dir, args.projection_axis),
                _make_phase_space(datasets, output_dir, args.max_scatter_points, rng),
                _make_budget_diagnostics(datasets, attrs, output_dir),
                _make_advection_topology(datasets, output_dir, args.projection_axis),
                _make_transport_budget(datasets, output_dir),
                _make_failure_and_limit_bars(datasets, attrs, output_dir),
            ):
                if artifact is not None:
                    artifacts.append(str(artifact))

            for artifact in artifacts:
                ai_log.emit_artifact(artifact, kind="png" if str(artifact).endswith(".png") else "file")

            rows, stats = _collect_stats(datasets)
            stats_csv = output_dir / "report_figure_stats.csv"
            _write_stats_csv(stats_csv, rows)
            artifacts.append(str(stats_csv))
            ai_log.emit_artifact(stats_csv, kind="csv")

            metrics = _build_report_metrics(datasets, attrs)
            metrics_csv = output_dir / "report_metrics.csv"
            metrics_json = output_dir / "report_metrics.json"
            _write_metrics_csv(metrics_csv, metrics)
            with metrics_json.open("w", encoding="utf-8") as handle:
                json.dump(metrics, handle, indent=2, ensure_ascii=True)
            artifacts.extend([str(metrics_csv), str(metrics_json)])
            ai_log.emit("report_metrics", metrics=metrics)
            ai_log.emit_artifact(metrics_csv, kind="csv")
            ai_log.emit_artifact(metrics_json, kind="json")

            skipped = [name for name in OPTIONAL_DATASETS if name not in datasets]
            manifest = {
                "input_h5": str(input_h5),
                "read_input_h5": str(read_input_h5),
                "scratch_work_dir": str(scratch_work_dir) if scratch_work_dir else None,
                "scratch_copy_time_s": scratch_copy_time_s,
                "h5_read_time_s": h5_read_time_s,
                "output_dir": str(output_dir),
                "shape": list(datasets["UNTH"].shape),
                "projection_axis": args.projection_axis,
                "present_datasets": present,
                "skipped_optional_datasets": skipped,
                "h5_attrs": attrs,
                "stats": stats,
                "report_metrics": metrics,
                "artifacts": artifacts,
                "ai_events": str(ai_log_path),
            }
            manifest_path = output_dir / "manifest.json"
            with manifest_path.open("w", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2, ensure_ascii=True)
            artifacts.append(str(manifest_path))
            ai_log.emit_artifact(manifest_path, kind="json")
            ai_log.emit("run_complete", status="ok", metrics={"artifact_count": len(artifacts)})
        finally:
            if scratch_work_dir is not None:
                try:
                    shutil.rmtree(scratch_work_dir)
                    ai_log.emit(
                        "scratch_cleanup",
                        status="ok",
                        params={"scratch_work_dir": str(scratch_work_dir)},
                    )
                except Exception as exc:
                    ai_log.emit(
                        "scratch_cleanup",
                        level="error",
                        status="failed",
                        error=f"{exc.__class__.__name__}: {exc}",
                        params={"scratch_work_dir": str(scratch_work_dir)},
                    )

    print(f"Saved report figures to: {output_dir}")
    for artifact in artifacts:
        print(f"  {artifact}")
    if skipped:
        print("Skipped optional datasets:")
        for name in skipped:
            print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
