"""System and dependency diagnostics probe for Noema."""

import sys
import shutil
import platform
import subprocess
from pathlib import Path


def check_python() -> dict[str, str | bool]:
    major, minor, micro = sys.version_info[:3]
    ok = (major == 3 and minor >= 12)
    return {
        "component": "Python",
        "version": f"{major}.{minor}.{micro}",
        "status": "PASS" if ok else "FAIL",
        "details": sys.executable,
    }


def check_torch() -> dict[str, str | bool]:
    try:
        import torch
        device = "cpu"
        if torch.cuda.is_available():
            device = f"cuda ({torch.cuda.get_device_name(0)})"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps (Apple Silicon Metal)"
        return {
            "component": "PyTorch",
            "version": torch.__version__,
            "status": "PASS",
            "details": f"Acceleration device: {device}",
        }
    except ImportError as e:
        return {
            "component": "PyTorch",
            "version": "not installed",
            "status": "FAIL",
            "details": str(e),
        }


def check_lean() -> dict[str, str | bool]:
    lean_path = shutil.which("lean")
    if not lean_path:
        return {
            "component": "Lean 4",
            "version": "missing",
            "status": "FAIL",
            "details": "lean binary not found in PATH",
        }
    try:
        out = subprocess.check_output([lean_path, "--version"], text=True).strip()
        return {
            "component": "Lean 4",
            "version": out,
            "status": "PASS",
            "details": lean_path,
        }
    except Exception as e:
        return {
            "component": "Lean 4",
            "version": "error",
            "status": "FAIL",
            "details": str(e),
        }


def check_lake() -> dict[str, str | bool]:
    lake_path = shutil.which("lake")
    if not lake_path:
        return {
            "component": "Lake",
            "version": "missing",
            "status": "FAIL",
            "details": "lake binary not found in PATH",
        }
    try:
        out = subprocess.check_output([lake_path, "--version"], text=True).strip()
        return {
            "component": "Lake",
            "version": out,
            "status": "PASS",
            "details": lake_path,
        }
    except Exception as e:
        return {
            "component": "Lake",
            "version": "error",
            "status": "FAIL",
            "details": str(e),
        }


def run_diagnostics() -> int:
    print("=" * 60)
    print("Noema Environment Diagnostics (make doctor)")
    print("=" * 60)
    print(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
    
    checks = [check_python(), check_torch(), check_lean(), check_lake()]
    all_ok = True
    for c in checks:
        status_symbol = "[OK]" if c["status"] == "PASS" else "[FAIL]"
        print(f"{status_symbol} {c['component']}: {c['version']}")
        print(f"     Details: {c['details']}")
        if c["status"] != "PASS":
            all_ok = False
            
    print("=" * 60)
    if all_ok:
        print("All required core components are available and operational.")
        return 0
    else:
        print("Diagnostic checks failed. Please address missing components.")
        return 1


if __name__ == "__main__":
    sys.exit(run_diagnostics())
