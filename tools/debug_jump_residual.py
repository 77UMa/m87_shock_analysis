#!/usr/bin/env python3
"""Threshold debugger for the SRMHD jump-residual refinement gate."""

from __future__ import annotations

import argparse
import csv
import sys
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

try:
    import plotly.graph_objects as go
except ImportError:
    go = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_ROI = {
    "r_min": 10.0,
    "r_max": 1200.0,
    "theta_min": 0.0,
    "theta_max": np.pi,
    "phi_min": 0.0,
    "phi_max": 2.0 * np.pi,
}

DEFAULT_SHOCK_PARAMS = {
    "gamma": 4.0 / 3.0,
    "mach_threshold_loose": 1.05,
    "min_physical_mach": 1.7,
    "grad_p_filter_quantile": 0.20,
    "march_cells": 6,
    "enable_sr_refine": True,
    "sr_mach_min": 1.2,
    "jump_residual_max": 0.4,
}

DEFAULT_THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.80]


class TimestampedTee:
    """Mirror stdout/stderr to console and a timestamped AI-friendly log file."""

    def __init__(self, sink, log_path: Path):
        self.sink = sink
        self.handle = log_path.open("w", encoding="utf-8")
        self._buffer = ""

    def write(self, data: str) -> int:
        if not data:
            return 0
        self.sink.write(data)
        self._buffer += data
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.handle.write(f"[{timestamp}] {line}\n")
        return len(data)

    def flush(self) -> None:
        self.sink.flush()
        self.handle.flush()

    def close(self) -> None:
        if self._buffer:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.handle.write(f"[{timestamp}] {self._buffer}\n")
            self._buffer = ""
        self.handle.flush()
        self.handle.close()


@dataclass
class ThresholdSummary:
    jump_residual_max: float
    verified_count: int
    refined_count: int
    refined_fraction: float
    rejected_low_mach_count: int
    rejected_jump_count: int
    rejected_entropy_count: int
    refined_sr_mach_median: float
    refined_sr_mach_p90: float
    refined_jump_residual_median: float
    refined_theta_bn_median: float


def parse_thresholds(value: str | None) -> list[float]:
    if not value:
        return list(DEFAULT_THRESHOLDS)
    thresholds = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        thresholds.append(float(item))
    if not thresholds:
        raise ValueError("No jump thresholds were parsed from --jump-thresholds")
    return sorted(thresholds)


def threshold_label(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def finite_median(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def finite_percentile(values: np.ndarray, percentile: float) -> float:
    values = values[np.isfinite(values)]
    return float(np.percentile(values, percentile)) if values.size else float("nan")


def summarize_distribution(name: str, values: np.ndarray) -> None:
    values = values[np.isfinite(values)]
    if values.size == 0:
        print(f"{name}=empty")
        return
    print(
        f"{name}.count={values.size} "
        f"{name}.min={np.min(values):.6g} "
        f"{name}.median={np.median(values):.6g} "
        f"{name}.max={np.max(values):.6g}"
    )


def build_threshold_masks(
    shock_props: dict,
    sr_mach_min: float,
    jump_residual_max: float,
) -> dict[str, np.ndarray]:
    verified_mask = np.asarray(shock_props["mask"], dtype=bool)
    sr_mach_normal = np.asarray(shock_props["sr_mach_normal"], dtype=float)
    jump_residual_light = np.asarray(shock_props["jump_residual_light"], dtype=float)
    entropy_jump = np.asarray(shock_props["entropy_jump"], dtype=float)

    pass_low_mach = verified_mask & np.isfinite(sr_mach_normal) & (sr_mach_normal > sr_mach_min)
    pass_jump = verified_mask & np.isfinite(jump_residual_light) & (jump_residual_light < jump_residual_max)
    pass_entropy = verified_mask & np.isfinite(entropy_jump) & (entropy_jump > 0.0)
    refined_mask = verified_mask & pass_low_mach & pass_jump & pass_entropy

    rejected_low_mach = verified_mask & (~pass_low_mach)
    rejected_jump = verified_mask & pass_low_mach & (~pass_jump)
    rejected_entropy = verified_mask & pass_low_mach & pass_jump & (~pass_entropy)

    return {
        "verified_mask": verified_mask,
        "pass_low_mach": pass_low_mach,
        "pass_jump": pass_jump,
        "pass_entropy": pass_entropy,
        "refined_mask": refined_mask,
        "rejected_low_mach": rejected_low_mach,
        "rejected_jump": rejected_jump,
        "rejected_entropy": rejected_entropy,
    }


def summarize_threshold(
    shock_props: dict,
    masks: dict[str, np.ndarray],
    jump_residual_max: float,
) -> ThresholdSummary:
    verified_mask = masks["verified_mask"]
    refined_mask = masks["refined_mask"]
    rejected_low_mach = masks["rejected_low_mach"]
    rejected_jump = masks["rejected_jump"]
    rejected_entropy = masks["rejected_entropy"]

    verified_count = int(np.sum(verified_mask))
    refined_count = int(np.sum(refined_mask))
    refined_fraction = float(refined_count / verified_count) if verified_count else 0.0

    sr_mach_normal = np.asarray(shock_props["sr_mach_normal"], dtype=float)
    jump_residual_light = np.asarray(shock_props["jump_residual_light"], dtype=float)
    theta_bn = np.asarray(shock_props["theta_Bn"], dtype=float)

    refined_sr_mach = sr_mach_normal[refined_mask]
    refined_jump = jump_residual_light[refined_mask]
    refined_theta_bn = theta_bn[refined_mask]

    return ThresholdSummary(
        jump_residual_max=jump_residual_max,
        verified_count=verified_count,
        refined_count=refined_count,
        refined_fraction=refined_fraction,
        rejected_low_mach_count=int(np.sum(rejected_low_mach)),
        rejected_jump_count=int(np.sum(rejected_jump)),
        rejected_entropy_count=int(np.sum(rejected_entropy)),
        refined_sr_mach_median=finite_median(refined_sr_mach),
        refined_sr_mach_p90=finite_percentile(refined_sr_mach, 90.0),
        refined_jump_residual_median=finite_median(refined_jump),
        refined_theta_bn_median=finite_median(refined_theta_bn),
    )


def make_output_dirs(base_output_dir: Path, snapshot_name: str) -> dict[str, Path]:
    root = base_output_dir / snapshot_name
    dirs = {
        "root": root,
        "shock3d": root / "shock3d",
        "relations": root / "relations",
        "summary": root / "summary",
        "tables": root / "tables",
        "log": root / "log",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def cell_centers_xyz(roi_data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = np.asarray(roi_data["x1v"], dtype=float)
    theta = np.asarray(roi_data["x2v"], dtype=float)
    phi = np.asarray(roi_data["x3v"], dtype=float)
    phi_grid, theta_grid, r_grid = np.meshgrid(phi, theta, r, indexing="ij")
    x = r_grid * np.sin(theta_grid) * np.cos(phi_grid)
    y = r_grid * np.sin(theta_grid) * np.sin(phi_grid)
    z = r_grid * np.cos(theta_grid)
    return x, y, z


def downsample_indices(count: int, max_points: int | None) -> np.ndarray:
    if max_points is None or max_points <= 0 or count <= max_points:
        return np.arange(count)
    return np.linspace(0, count - 1, max_points, dtype=int)


def plot_3d_subset(
    roi_data: dict,
    subset_mask: np.ndarray,
    color_grid: np.ndarray,
    title: str,
    output_path: Path,
    max_points: int | None,
) -> None:
    if go is None:
        raise ImportError("plotly is required for 3D debug plots")

    x_grid, y_grid, z_grid = cell_centers_xyz(roi_data)
    k_idx, j_idx, i_idx = np.where(subset_mask)
    if k_idx.size == 0:
        fig = go.Figure()
        fig.update_layout(title=f"{title} (no cells)")
        fig.write_html(str(output_path))
        return

    keep = downsample_indices(k_idx.size, max_points)
    x = x_grid[k_idx, j_idx, i_idx][keep]
    y = y_grid[k_idx, j_idx, i_idx][keep]
    z = z_grid[k_idx, j_idx, i_idx][keep]
    color = color_grid[k_idx, j_idx, i_idx][keep]

    finite_color = color[np.isfinite(color)]
    cmin = float(np.min(finite_color)) if finite_color.size else 0.0
    cmax = float(np.max(finite_color)) if finite_color.size else 1.0
    if cmax <= cmin:
        cmax = cmin + 1.0

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker={
                    "size": 2,
                    "color": color,
                    "colorscale": "Plasma",
                    "opacity": 0.75,
                    "colorbar": {"title": "sr_mach_normal"},
                    "cmin": cmin,
                    "cmax": cmax,
                },
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene={
            "xaxis_title": "X [r_g]",
            "yaxis_title": "Y [r_g]",
            "zaxis_title": "Z [r_g]",
            "aspectmode": "data",
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
    )
    fig.write_html(str(output_path))


def relation_plot(
    x_values: np.ndarray,
    y_values: np.ndarray,
    x_label: str,
    y_label: str,
    title: str,
    output_path: Path,
) -> None:
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[mask]
    y_values = y_values[mask]

    fig, ax = plt.subplots(figsize=(8, 6))
    if x_values.size:
        ax.scatter(x_values, y_values, s=6, alpha=0.35, edgecolors="none")
    else:
        ax.text(0.5, 0.5, "No valid verified cells", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def summary_curve_plot(
    thresholds: list[float],
    values: list[float],
    y_label: str,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, values, marker="o", linewidth=1.8)
    ax.set_xlabel("jump_residual_max")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_summary_csv(
    output_path: Path,
    snapshot_name: str,
    sr_mach_min: float,
    summaries: list[ThresholdSummary],
) -> None:
    fieldnames = [
        "snapshot",
        "sr_mach_min",
        "jump_residual_max",
        "verified_count",
        "refined_count",
        "refined_fraction",
        "rejected_low_mach_count",
        "rejected_jump_count",
        "rejected_entropy_count",
        "refined_sr_mach_median",
        "refined_sr_mach_p90",
        "refined_jump_residual_median",
        "refined_theta_bn_median",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "snapshot": snapshot_name,
                    "sr_mach_min": sr_mach_min,
                    "jump_residual_max": summary.jump_residual_max,
                    "verified_count": summary.verified_count,
                    "refined_count": summary.refined_count,
                    "refined_fraction": summary.refined_fraction,
                    "rejected_low_mach_count": summary.rejected_low_mach_count,
                    "rejected_jump_count": summary.rejected_jump_count,
                    "rejected_entropy_count": summary.rejected_entropy_count,
                    "refined_sr_mach_median": summary.refined_sr_mach_median,
                    "refined_sr_mach_p90": summary.refined_sr_mach_p90,
                    "refined_jump_residual_median": summary.refined_jump_residual_median,
                    "refined_theta_bn_median": summary.refined_theta_bn_median,
                }
            )


def log_verified_summary(shock_props: dict) -> None:
    verified_mask = np.asarray(shock_props["mask"], dtype=bool)
    print(f"debug.verified.count={int(np.sum(verified_mask))}")
    summarize_distribution("debug.verified.sr_mach", np.asarray(shock_props["sr_mach_normal"])[verified_mask])
    summarize_distribution("debug.verified.jump_residual", np.asarray(shock_props["jump_residual_light"])[verified_mask])
    summarize_distribution("debug.verified.theta_bn", np.asarray(shock_props["theta_Bn"])[verified_mask])


def log_threshold_summary(
    jump_residual_max: float,
    shock_props: dict,
    masks: dict[str, np.ndarray],
    summary: ThresholdSummary,
) -> None:
    print(f"debug.threshold.value={jump_residual_max:.2f}")
    print(f"debug.threshold.refined.count={summary.refined_count}")
    print(f"debug.threshold.refined.fraction={summary.refined_fraction:.6f}")
    print(f"debug.threshold.rejected_low_mach.count={summary.rejected_low_mach_count}")
    print(f"debug.threshold.rejected_jump.count={summary.rejected_jump_count}")
    print(f"debug.threshold.rejected_entropy.count={summary.rejected_entropy_count}")

    sr_mach = np.asarray(shock_props["sr_mach_normal"])
    jump = np.asarray(shock_props["jump_residual_light"])
    theta_bn = np.asarray(shock_props["theta_Bn"])

    summarize_distribution("debug.threshold.refined.sr_mach", sr_mach[masks["refined_mask"]])
    summarize_distribution("debug.threshold.refined.jump_residual", jump[masks["refined_mask"]])
    summarize_distribution("debug.threshold.refined.theta_bn", theta_bn[masks["refined_mask"]])
    summarize_distribution("debug.threshold.rejected_jump.sr_mach", sr_mach[masks["rejected_jump"]])
    summarize_distribution("debug.threshold.rejected_jump.jump_residual", jump[masks["rejected_jump"]])
    summarize_distribution("debug.threshold.rejected_jump.theta_bn", theta_bn[masks["rejected_jump"]])
    print(
        f"debug.threshold.compact jrmax={jump_residual_max:.2f} "
        f"refined={summary.refined_count}/{summary.verified_count} "
        f"fraction={summary.refined_fraction:.3f} "
        f"rej_low_mach={summary.rejected_low_mach_count} "
        f"rej_jump={summary.rejected_jump_count} "
        f"rej_entropy={summary.rejected_entropy_count} "
        f"refined_sr_median={summary.refined_sr_mach_median:.3f}"
    )


def run_debugger(args: argparse.Namespace) -> None:
    from src.core.shock_v1 import find_shocks_in_roi_mhd
    from src.workflows.base_workflow import load_and_slice_data

    input_path = Path(args.input).resolve()
    snapshot_name = input_path.name.replace(".athdf", "")
    thresholds = parse_thresholds(args.jump_thresholds)
    roi_params = dict(DEFAULT_ROI)
    shock_params = dict(DEFAULT_SHOCK_PARAMS)
    shock_params["sr_mach_min"] = args.sr_mach_min
    shock_params["jump_residual_max"] = 0.4

    output_dirs = make_output_dirs(Path(args.output_dir).resolve(), snapshot_name)
    log_path = output_dirs["log"] / f"{snapshot_name}_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    tee = TimestampedTee(sys.stdout, log_path)

    try:
        with redirect_stdout(tee), redirect_stderr(tee):
            print(f"debug.run.snapshot={snapshot_name}")
            print(f"debug.run.input={input_path}")
            print(f"debug.run.log_path={log_path}")
            print(f"debug.run.thresholds={','.join(f'{value:.2f}' for value in thresholds)}")
            print(f"debug.run.sr_mach_min={args.sr_mach_min:.2f}")
            print(f"debug.run.max_points_3d={args.max_points_3d}")

            roi_data = load_and_slice_data(str(input_path), roi_params, logger=None)
            if roi_data is None:
                raise RuntimeError(f"Failed to load ROI data from {input_path}")

            shock_props = find_shocks_in_roi_mhd(roi_data, logger=None, **shock_params)
            log_verified_summary(shock_props)

            verified_mask = np.asarray(shock_props["mask"], dtype=bool)
            sr_mach_normal = np.asarray(shock_props["sr_mach_normal"], dtype=float)
            jump_residual = np.asarray(shock_props["jump_residual_light"], dtype=float)
            theta_bn = np.asarray(shock_props["theta_Bn"], dtype=float)

            plot_3d_subset(
                roi_data,
                verified_mask,
                sr_mach_normal,
                f"{snapshot_name} verified shocks colored by sr_mach_normal",
                output_dirs["shock3d"] / f"{snapshot_name}_verified_srmach.html",
                args.max_points_3d,
            )
            relation_plot(
                jump_residual[verified_mask],
                sr_mach_normal[verified_mask],
                "jump_residual_light",
                "sr_mach_normal",
                f"{snapshot_name} verified jump residual vs SR Mach",
                output_dirs["relations"] / f"{snapshot_name}_verified_jump_vs_srmach.png",
            )
            relation_plot(
                jump_residual[verified_mask],
                theta_bn[verified_mask],
                "jump_residual_light",
                "theta_Bn [rad]",
                f"{snapshot_name} verified jump residual vs theta_Bn",
                output_dirs["relations"] / f"{snapshot_name}_verified_jump_vs_theta_bn.png",
            )

            summaries: list[ThresholdSummary] = []
            for threshold in thresholds:
                masks = build_threshold_masks(shock_props, args.sr_mach_min, threshold)
                summary = summarize_threshold(shock_props, masks, threshold)
                summaries.append(summary)
                log_threshold_summary(threshold, shock_props, masks, summary)

                label = threshold_label(threshold)
                plot_3d_subset(
                    roi_data,
                    masks["refined_mask"],
                    sr_mach_normal,
                    f"{snapshot_name} refined shocks jrmax={threshold:.2f}",
                    output_dirs["shock3d"] / f"{snapshot_name}_jrmax_{label}_refined_srmach.html",
                    args.max_points_3d,
                )
                plot_3d_subset(
                    roi_data,
                    masks["rejected_jump"],
                    sr_mach_normal,
                    f"{snapshot_name} rejected-by-jump shocks jrmax={threshold:.2f}",
                    output_dirs["shock3d"] / f"{snapshot_name}_jrmax_{label}_rejected_jump_srmach.html",
                    args.max_points_3d,
                )

            write_summary_csv(
                output_dirs["tables"] / f"{snapshot_name}_jump_residual_threshold_scan.csv",
                snapshot_name,
                args.sr_mach_min,
                summaries,
            )

            summary_curve_plot(
                thresholds,
                [summary.refined_count for summary in summaries],
                "refined_count",
                f"{snapshot_name} threshold vs refined count",
                output_dirs["summary"] / f"{snapshot_name}_threshold_vs_refined_count.png",
            )
            summary_curve_plot(
                thresholds,
                [summary.refined_fraction for summary in summaries],
                "refined_fraction",
                f"{snapshot_name} threshold vs refined fraction",
                output_dirs["summary"] / f"{snapshot_name}_threshold_vs_refined_fraction.png",
            )
            summary_curve_plot(
                thresholds,
                [summary.refined_sr_mach_median for summary in summaries],
                "median refined sr_mach_normal",
                f"{snapshot_name} threshold vs refined SR Mach median",
                output_dirs["summary"] / f"{snapshot_name}_threshold_vs_refined_srmach_median.png",
            )

            print(f"debug.output.root={output_dirs['root']}")
            print(f"debug.output.log={log_path}")
    finally:
        tee.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Debug SRMHD jump-residual refinement thresholds")
    parser.add_argument("--input", required=True, help="Single Athena++ .athdf snapshot")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "OUTPUT" / "debug_jump_residual"),
        help="Base output directory",
    )
    parser.add_argument(
        "--jump-thresholds",
        default=",".join(f"{value:.2f}" for value in DEFAULT_THRESHOLDS),
        help="Comma-separated jump_residual_max values",
    )
    parser.add_argument(
        "--sr-mach-min",
        type=float,
        default=1.2,
        help="Offline SR Mach threshold for refined-mask reconstruction",
    )
    parser.add_argument(
        "--max-points-3d",
        type=int,
        default=50000,
        help="Optional downsampling cap for 3D plots",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_debugger(args)


if __name__ == "__main__":
    main()
