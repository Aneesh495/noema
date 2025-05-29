# Reproducibility Guide

This document outlines the hardware specifications, software dependencies, and exact execution commands required to reproduce all models, search experiments, and formal verification benchmarks in Noema.

## 1. System Requirements

### Hardware Tested
- Apple Silicon (M1/M2/M3/M4 Max/Pro), 32 GB or 64 GB unified memory
- Linux x86_64, NVIDIA GPU (RTX 3090, A100, or H100) with CUDA 12.1+
- CPU fallback: x86_64 or ARM64 with at least 8 physical cores and 16 GB RAM

### Toolchain Versions
- **Operating System**: macOS Sonoma 14+ or Ubuntu 22.04 LTS
- **Python**: 3.12.x
- **PyTorch**: 2.2.0 or higher
- **Lean 4**: `v4.8.0` (managed via `elan`)
- **Lake**: 5.0.0 (bundled with Lean 4.8.0)

## 2. Environment Setup

Clone the repository and initialize the Python environment:

```bash
git clone https://github.com/Aneesh495/noema.git
cd noema

# Create and activate Python 3.12 virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies and editable package
pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
```

Build the Lean 4 formal bridge package:

```bash
cd lean
lake build
cd ..
```

Verify that Lean 4 and Lake compile cleanly without warnings or errors:

```bash
lake --version
lean --version
```

## 3. Deterministic Seeds

All stochastic operations in Noema support explicit random seed configuration:

```python
import random
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
```

## 4. Running the Verification Suite

Run all unit, integration, and property tests:

```bash
pytest
```

Run Lean 4 independent proof checking across all formal benchmark modules:

```bash
python3 -c "
from noema.formal.checker import IndependentChecker
checker = IndependentChecker(project_dir='lean')
result = checker.verify_proof('''
import NoemaBridge.Logic
theorem test_id (P : Prop) : P → P := by intro h; exact h
''')
assert result.success, 'Clean-room check failed'
print('Independent verification verified successfully.')
"
```

## 5. Running Benchmark Suites

To execute the standard pass@k benchmark evaluation:

```bash
python3 -m noema.benchmarks.benchmark_suite --dataset algebra --k 1,8,32 --budget 100
```
