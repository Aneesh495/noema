"""Release packaging and reproducibility distribution generator."""

from __future__ import annotations

import json
import shutil
import sys
import tarfile
from pathlib import Path


def package_release(output_dir: str = "dist") -> int:
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("NOEMA REPRODUCIBILITY RELEASE PACKAGING (make release)")
    print("=" * 70)

    release_tar = out_p / "noema_release_v0.1.0.tar.gz"
    with tarfile.open(release_tar, "w:gz") as tar:
        # Include core source, lean definitions, documentation, configs, and manifests
        for folder in ["src", "lean", "docs", "tests"]:
            p = Path(folder)
            if p.exists():
                tar.add(str(p), arcname=folder)
                
        for single_file in ["pyproject.toml", "Makefile", "README.md", "CHARTER.md"]:
            p = Path(single_file)
            if p.exists():
                tar.add(str(p), arcname=single_file)

    print(f"Packaged reproducible archive: {release_tar}")
    print(f"Archive Size: {release_tar.stat().st_size:,} bytes")
    print("Release package created successfully.")
    return 0


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "dist"
    sys.exit(package_release(output_dir=out))
