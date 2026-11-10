"""Metadata written next to every benchmark output so a result can be traced to its code."""

import hashlib
import os
import platform
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

TRACKED_PACKAGES = ("datasets", "openai", "pillow", "pytesseract")


def _run(*args: str) -> str | None:
    try:
        return subprocess.run(
            list(args), capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _git(*args: str) -> str | None:
    return _run("git", *args)


def gpu_info() -> list[dict[str, str]] | None:
    """GPUs visible on this machine (name, memory, driver); None when there is no NVIDIA GPU.

    Latency is a cost in this project, so every run records the hardware it was measured on.
    """
    out = _run(
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version,compute_cap",
        "--format=csv,noheader",
    )
    if not out:
        return None
    keys = ("name", "memory", "driver", "compute_capability")
    return [
        dict(zip(keys, (v.strip() for v in line.split(",")), strict=False))
        for line in out.splitlines()
    ]


def cpu_info() -> str:
    """CPU model and logical core count, e.g. ``Intel(R) Xeon(R) CPU @ 2.20GHz x 12``.

    OCR runs on the CPU and its time is a reported cost, so OCR outputs record where they were
    measured, just as model runs record the GPU.
    """
    name = None
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("model name"):
                name = line.split(":", 1)[1].strip()
                break
    name = name or platform.processor() or platform.machine() or "unknown CPU"
    return f"{name} x {os.cpu_count()}"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_metadata(**extra: Any) -> dict[str, Any]:
    versions = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "date": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
        **extra,
    }
