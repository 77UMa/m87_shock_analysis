"""Structured JSONL event logging for AI-readable pipeline diagnostics."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover - keeps the logger usable in lightweight local environments.
    np = None


def _jsonable(value: Any) -> Any:
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


class AIEventLogger:
    """Append-only JSONL logger for structured scientific run events."""

    def __init__(
        self,
        path: str | Path,
        *,
        run_label: str | None = None,
        snapshot: str | None = None,
        stage: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.run_label = run_label
        self.snapshot = snapshot
        self.stage = stage
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()

    def emit(
        self,
        event: str,
        *,
        level: str = "info",
        stage: str | None = None,
        status: str | None = None,
        params: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        artifacts: list[dict[str, Any]] | None = None,
        error: str | None = None,
        **fields: Any,
    ) -> dict[str, Any]:
        record = {
            "time": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "level": level,
            "event": event,
            "stage": stage or self.stage,
            "run_label": self.run_label,
            "snapshot": self.snapshot,
        }
        if status is not None:
            record["status"] = status
        if params is not None:
            record["params"] = params
        if metrics is not None:
            record["metrics"] = metrics
        if artifacts is not None:
            record["artifacts"] = artifacts
        if error is not None:
            record["error"] = error
        record.update(fields)
        record = _jsonable(record)
        self._handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())
        return record

    def emit_artifact(
        self,
        path: str | Path,
        *,
        kind: str,
        stage: str | None = None,
        fatal_if_missing: bool = True,
        fatal_if_empty: bool = True,
        **fields: Any,
    ) -> dict[str, Any]:
        artifact_path = Path(path)
        exists = artifact_path.exists()
        bytes_size = artifact_path.stat().st_size if exists else 0
        status = "ok" if exists and bytes_size > 0 else "missing" if not exists else "empty"
        artifact = {
            "kind": kind,
            "path": str(artifact_path),
            "exists": exists,
            "bytes": bytes_size,
            "status": status,
        }
        artifact.update(fields)
        record = self.emit("artifact", stage=stage, status=status, artifacts=[artifact])

        if (fatal_if_missing and not exists) or (fatal_if_empty and exists and bytes_size == 0):
            reason = "missing_artifact" if not exists else "zero_byte_artifact"
            self.emit(
                "validation_failed",
                level="fatal",
                stage=stage,
                status="failed",
                error=reason,
                artifacts=[artifact],
            )
            raise ValueError(f"{reason}: {artifact_path}")
        return record

    def __enter__(self) -> "AIEventLogger":
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> None:
        if exc is not None:
            self.emit(
                "exception",
                level="fatal",
                status="failed",
                error=f"{exc.__class__.__name__}: {exc}",
            )
        self.close()
