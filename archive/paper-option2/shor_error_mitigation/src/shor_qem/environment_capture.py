"""Capture a paper-ready software, CUDA, GPU, and host environment snapshot."""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import io
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SAFE_ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "CUPY_ACCELERATORS",
    "CUPY_CACHE_DIR",
    "MPLCONFIGDIR",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "PYTHONHASHSEED",
)


def _run(command: list[str]) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if executable is None:
        return {
            "command": command,
            "available": False,
            "returncode": None,
            "stdout": "",
            "stderr": f"{command[0]} was not found",
        }
    completed = subprocess.run(
        [executable, *command[1:]],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return {
        "command": [executable, *command[1:]],
        "available": True,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _write_command(path: Path, command: list[str]) -> dict[str, Any]:
    result = _run(command)
    text = result["stdout"]
    if result["stderr"]:
        text += ("\n" if text else "") + "[stderr]\n" + result["stderr"]
    path.write_text(text, encoding="utf-8")
    return {
        "command": result["command"],
        "available": result["available"],
        "returncode": result["returncode"],
        "file": path.name,
    }


def _package_records() -> list[dict[str, str]]:
    records = [
        {
            "name": distribution.metadata.get("Name", distribution.name),
            "version": distribution.version,
        }
        for distribution in importlib.metadata.distributions()
    ]
    records.sort(key=lambda item: item["name"].lower())
    return records


def _module_config(module_name: str) -> tuple[dict[str, Any], str]:
    stream = io.StringIO()
    try:
        module = __import__(module_name)
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            show_config = getattr(module, "show_config", None)
            if callable(show_config):
                show_config()
        metadata = {
            "available": True,
            "version": getattr(module, "__version__", "unknown"),
        }
        if module_name == "cupy":
            metadata["cuda_runtime"] = int(
                module.cuda.runtime.runtimeGetVersion()
            )
            metadata["cuda_driver"] = int(
                module.cuda.runtime.driverGetVersion()
            )
            metadata["visible_device_count"] = int(
                module.cuda.runtime.getDeviceCount()
            )
    except Exception as error:  # environment diagnostics must survive failures
        metadata = {
            "available": False,
            "error": f"{type(error).__name__}: {error}",
        }
    return metadata, stream.getvalue()


def _capture_conda_metadata(output: Path) -> dict[str, Any]:
    conda_meta = Path(sys.prefix) / "conda-meta"
    if not conda_meta.is_dir():
        return {"available": False}
    history = conda_meta / "history"
    if history.is_file():
        shutil.copy2(history, output / "conda_history.txt")
    packages = []
    for record_path in sorted(conda_meta.glob("*.json")):
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        packages.append(
            {
                key: record.get(key)
                for key in (
                    "name",
                    "version",
                    "build",
                    "build_number",
                    "channel",
                    "subdir",
                    "url",
                    "md5",
                    "sha256",
                )
            }
        )
    (output / "conda_packages.json").write_text(
        json.dumps(packages, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "available": True,
        "prefix": sys.prefix,
        "package_count": len(packages),
        "history_file": "conda_history.txt" if history.is_file() else None,
        "packages_file": "conda_packages.json",
    }


def capture(output_dir: Path) -> dict[str, Any]:
    output = output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    command_records = [
        _write_command(output / "uname.txt", ["uname", "-a"]),
        _write_command(output / "os_release.txt", ["cat", "/etc/os-release"]),
        _write_command(output / "lscpu.txt", ["lscpu"]),
        _write_command(output / "memory.txt", ["free", "-h"]),
        _write_command(output / "nvidia_smi.txt", ["nvidia-smi"]),
        _write_command(
            output / "nvidia_smi_query.csv",
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,driver_version,pci.bus_id,"
                "memory.total,compute_cap",
                "--format=csv",
            ],
        ),
        _write_command(
            output / "nvidia_topology.txt", ["nvidia-smi", "topo", "-m"]
        ),
        _write_command(output / "nvcc.txt", ["nvcc", "--version"]),
    ]
    packages = _package_records()
    (output / "installed_packages.json").write_text(
        json.dumps(packages, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "pip_freeze.txt").write_text(
        "\n".join(f"{item['name']}=={item['version']}" for item in packages)
        + "\n",
        encoding="utf-8",
    )

    numpy_metadata, numpy_config = _module_config("numpy")
    cupy_metadata, cupy_config = _module_config("cupy")
    conda_metadata = _capture_conda_metadata(output)
    (output / "numpy_config.txt").write_text(numpy_config, encoding="utf-8")
    (output / "cupy_config.txt").write_text(cupy_config, encoding="utf-8")
    snapshot = {
        "schema_version": 1,
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
            "build": platform.python_build(),
        },
        "numpy": numpy_metadata,
        "cupy": cupy_metadata,
        "conda": conda_metadata,
        "environment": {
            key: os.environ[key]
            for key in SAFE_ENVIRONMENT_KEYS
            if key in os.environ
        },
        "commands": command_records,
        "package_count": len(packages),
    }
    (output / "environment_snapshot.json").write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return snapshot


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    snapshot = capture(args.output_dir)
    print(
        f"Captured environment on {snapshot['hostname']} in "
        f"{args.output_dir.resolve()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
