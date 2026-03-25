#!/usr/bin/env python3
"""Dual-channel logging utilities for the DSA workflow."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np


RESULT_LEVEL = 25
TIME_LEVEL = 26
DATA_LEVEL = 15
CODEPATH_LEVEL = 16

logging.addLevelName(RESULT_LEVEL, "RESULT")
logging.addLevelName(TIME_LEVEL, "TIME")
logging.addLevelName(DATA_LEVEL, "DATA")
logging.addLevelName(CODEPATH_LEVEL, "CODEPATH")


def _to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "min": float(np.min(value)) if value.size else None,
            "max": float(np.max(value)) if value.size else None,
        }
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _compact_value(value: Any, max_items: int = 8) -> Any:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return {"shape": list(value.shape), "dtype": str(value.dtype), "size": 0}
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "min": float(np.min(value)),
            "max": float(np.max(value)),
            "median": float(np.median(value)),
        }
    if isinstance(value, dict):
        items = list(value.items())[:max_items]
        return {str(k): _compact_value(v, max_items=max_items) for k, v in items}
    if isinstance(value, (list, tuple)):
        return [_compact_value(v, max_items=max_items) for v in list(value)[:max_items]]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


class HumanLogger:
    """Human-oriented logger for concise physics results and progress."""

    def __init__(self, log_dir: str, run_id: str):
        self.logger = logging.getLogger(f"dsa.human.{run_id}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()
        self.logger.propagate = False

        self.log_file = os.path.join(log_dir, f"human_{run_id}.log")

        file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        self.logger.error(msg)

    def result(self, msg: str) -> None:
        self.logger.log(RESULT_LEVEL, msg)

    def time(self, stage: str, seconds: float) -> None:
        self.logger.log(TIME_LEVEL, f"{stage}: {seconds:.2f}s")

    def log_start(self, snapshot_name: str, config: Optional[Dict[str, Any]] = None) -> None:
        self.info("=" * 60)
        self.info(f"Start processing {snapshot_name}")
        if config:
            roi = config.get("roi_params", {})
            self.info(
                "ROI "
                f"r=[{roi.get('r_min', 'N/A')}, {roi.get('r_max', 'N/A')}], "
                f"theta=[{roi.get('theta_min', 'N/A')}, {roi.get('theta_max', 'N/A')}], "
                f"phi=[{roi.get('phi_min', 'N/A')}, {roi.get('phi_max', 'N/A')}]"
            )

    def log_finish(self, snapshot_name: str, output_file: str, processing_time: float) -> None:
        self.info(f"Finished processing {snapshot_name}")
        self.info(f"Output file: {os.path.basename(output_file)}")
        self.time("Total runtime", processing_time)
        self.info("=" * 60)

    def log_error(self, snapshot_name: str, error_msg: str) -> None:
        self.error("=" * 60)
        self.error(f"Failed processing {snapshot_name}")
        self.error(f"Error: {error_msg}")
        self.error("=" * 60)

    def log_shock_stats(
        self,
        n_shock_cells: int,
        mach_values: np.ndarray,
        sigma_values: Optional[np.ndarray] = None,
        coverage: Optional[float] = None,
    ) -> None:
        if n_shock_cells == 0 or mach_values.size == 0:
            self.result("Shock statistics: no shock cells detected")
            return

        self.result("Shock statistics:")
        self.result(f"Shock cell count: {n_shock_cells}")
        if coverage is not None:
            self.result(f"Shock coverage: {coverage:.2%}")
        self.result(
            "Mach number: "
            f"mean={np.mean(mach_values):.2f}, max={np.max(mach_values):.2f}, min={np.min(mach_values):.2f}"
        )

        if sigma_values is not None and sigma_values.size > 0:
            self.result(
                "Sigma at shocks: "
                f"median={np.median(sigma_values):.3e}, max={np.max(sigma_values):.3e}"
            )
            strong_sigma_fraction = np.mean(sigma_values > 0.1)
            if strong_sigma_fraction > 0.01:
                self.warning(
                    f"Strong sigma suppression region detected: {strong_sigma_fraction:.1%} of shock cells have sigma > 0.1"
                )

        low_mach = int(np.sum(mach_values < 1.5))
        high_mach = int(np.sum(mach_values > 10.0))
        if low_mach:
            self.warning(f"Abnormal Mach values: {low_mach} shock cells with Mach < 1.5")
        if high_mach:
            self.warning(f"Abnormal Mach values: {high_mach} shock cells with Mach > 10")

    def log_nonthermal_stats(
        self,
        n_inj_total: float,
        n_inj_range: tuple[float, float],
        p_values: np.ndarray,
        gamma_min_values: np.ndarray,
        sigma_suppression: Optional[np.ndarray] = None,
    ) -> None:
        self.result("Non-thermal electron statistics:")
        self.result(f"Total N_inj: {n_inj_total:.3e}")
        self.result(f"N_inj range: [{n_inj_range[0]:.3e}, {n_inj_range[1]:.3e}]")
        self.result(
            "p index: "
            f"mean={np.mean(p_values):.2f}, range=[{np.min(p_values):.2f}, {np.max(p_values):.2f}]"
        )
        self.result(f"gamma_min mean: {np.mean(gamma_min_values):.2f}")

        if sigma_suppression is not None and sigma_suppression.size > 0:
            median_suppression = float(np.median(sigma_suppression))
            frac_strong = float(np.mean(sigma_suppression < 0.1))
            self.result(f"Sigma suppression median: {median_suppression:.3f}")
            if frac_strong > 0.0:
                self.warning(f"Strong sigma suppression: {frac_strong:.1%} of shock cells have suppression < 0.1")

        low_p = int(np.sum(p_values < 1.5))
        high_p = int(np.sum(p_values > 5.0))
        if low_p:
            self.warning(f"Abnormal p values: {low_p} shock cells with p < 1.5")
        if high_p:
            self.warning(f"Abnormal p values: {high_p} shock cells with p > 5")


class AILogger:
    """AI-oriented logger for debug details, data ranges, and code paths."""

    def __init__(self, log_dir: str, run_id: str):
        self.logger = logging.getLogger(f"dsa.ai.{run_id}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()
        self.logger.propagate = False

        self.log_file = os.path.join(log_dir, f"ai_{run_id}.log")

        file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def debug(self, msg: str) -> None:
        self.logger.debug(msg)

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def exception(self, msg: str) -> None:
        self.logger.exception(msg)

    def data(self, var_name: str, array: Any, unit: str = "") -> None:
        if isinstance(array, np.ndarray):
            if array.size == 0:
                msg = f"{var_name}: empty array shape={array.shape} dtype={array.dtype}"
            else:
                msg = (
                    f"{var_name}"
                    f"{f' [{unit}]' if unit else ''}: "
                    f"shape={array.shape}, dtype={array.dtype}, "
                    f"range=[{np.min(array):.3e}, {np.max(array):.3e}], "
                    f"median={np.median(array):.3e}"
                )
        else:
            msg = f"{var_name}: {_compact_value(array)}"
        self.logger.log(DATA_LEVEL, msg)

    def codepath(self, branch: str, details: str = "") -> None:
        msg = branch if not details else f"{branch} | {details}"
        self.logger.log(CODEPATH_LEVEL, msg)

    def func_enter(self, func_name: str, args: Optional[Dict[str, Any]] = None) -> None:
        if args:
            self.debug(f"ENTER {func_name} | args={_compact_value(args)}")
        else:
            self.debug(f"ENTER {func_name}")

    def func_exit(self, func_name: str, result: Any = None) -> None:
        if result is None:
            self.debug(f"EXIT {func_name}")
        else:
            self.debug(f"EXIT {func_name} | result={_compact_value(result)}")

    def config_snapshot(self, config: Dict[str, Any]) -> None:
        self.debug(f"CONFIG_SNAPSHOT {json.dumps(_to_serializable(config), ensure_ascii=False, sort_keys=True)}")


class ConfigSnapshot:
    """Persist a config snapshot for each run."""

    def __init__(self, log_dir: str, run_id: str):
        self.configs_dir = os.path.join(log_dir, "configs")
        os.makedirs(self.configs_dir, exist_ok=True)
        self.config_file = os.path.join(self.configs_dir, f"{run_id}.json")
        self.run_id = run_id

    def save(self, config: Dict[str, Any]) -> str:
        payload = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "git": self._get_git_info(),
            "config": _to_serializable(config),
        }
        with open(self.config_file, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        return self.config_file

    def _get_git_info(self) -> Dict[str, Any]:
        repo_root = Path(__file__).resolve().parents[2]
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=repo_root,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=repo_root,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            status = subprocess.check_output(
                ["git", "status", "--short"],
                cwd=repo_root,
                stderr=subprocess.DEVNULL,
                text=True,
            ).splitlines()
            return {"commit": commit, "branch": branch, "dirty_files": status}
        except Exception:
            return {"commit": "unknown", "branch": "unknown", "dirty_files": []}


class DUALogger:
    """Dual-channel logger composed of a human logger and an AI logger."""

    def __init__(self, log_dir: str = "logs", run_id: Optional[str] = None):
        self.run_id = run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(log_dir, exist_ok=True)

        self.human = HumanLogger(log_dir, self.run_id)
        self.ai = AILogger(log_dir, self.run_id)
        self.config = ConfigSnapshot(log_dir, self.run_id)

        self.ai.debug(f"DUALogger initialized with run_id={self.run_id}")
        self.human.info(f"Logging initialized with run_id={self.run_id}")

    def save_config(self, config: Dict[str, Any]) -> str:
        path = self.config.save(config)
        self.ai.config_snapshot(config)
        self.ai.debug(f"Config snapshot saved to {path}")
        return path

    def get_log_files(self) -> Dict[str, str]:
        return {
            "human": self.human.log_file,
            "ai": self.ai.log_file,
            "config": self.config.config_file,
        }


def is_dual_logger(logger: Any) -> bool:
    """Duck-typed dual logger detection that is robust to import-path differences."""
    return hasattr(logger, "human") and hasattr(logger, "ai") and hasattr(logger, "save_config")


def _unwrap_logger(logger: Any) -> Any:
    """Unwrap any dual logger instance to its human-facing channel."""
    if is_dual_logger(logger):
        return logger.human
    return logger


def setup_logging(
    log_dir: str = "logs",
    log_level: int = logging.INFO,
    log_name: Optional[str] = None,
    logger_name: str = "DSAWorkflow",
):
    """Legacy single-channel logger retained for compatibility."""
    os.makedirs(log_dir, exist_ok=True)
    if log_name is None:
        log_name = f"dsa_workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    log_file = os.path.join(log_dir, log_name)
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    logger.handlers.clear()
    logger.propagate = False

    file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger, log_file


def log_start(logger: Any, snapshot_name: str, config: Optional[Dict[str, Any]] = None) -> None:
    logger = _unwrap_logger(logger)
    logger.info("=" * 60)
    logger.info(f"Start processing {snapshot_name}")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if config:
        logger.info(f"Config: {_compact_value(config)}")


def log_finish(logger: Any, snapshot_name: str, output_file: str, processing_time: float) -> None:
    logger = _unwrap_logger(logger)
    logger.info("=" * 60)
    logger.info(f"Finished processing {snapshot_name}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Runtime: {processing_time:.2f}s")
    logger.info("=" * 60)


def log_error(logger: Any, snapshot_name: str, error_msg: str, exc_info: bool = True) -> None:
    logger = _unwrap_logger(logger)
    logger.error("=" * 60)
    logger.error(f"Failed processing {snapshot_name}")
    logger.error(f"Error: {error_msg}", exc_info=exc_info)
    logger.error("=" * 60)


def log_summary(
    logger: Any,
    total_files: int,
    successful_files: int,
    failed_files: int,
    total_time: float,
) -> None:
    logger = _unwrap_logger(logger)
    logger.info("=" * 60)
    logger.info("Processing summary")
    logger.info(f"Total files: {total_files}")
    logger.info(f"Successful files: {successful_files}")
    logger.info(f"Failed files: {failed_files}")
    if total_files:
        logger.info(f"Success rate: {successful_files / total_files * 100:.1f}%")
    logger.info(f"Total runtime: {total_time:.2f}s")
    logger.info("=" * 60)
