"""Performance benchmarking: neural scoring throughput, formal kernel transitions, and memory latency."""

from __future__ import annotations

import os
import time
from pathlib import Path
from pydantic import BaseModel, Field
import torch
from noema.contracts.actions import TacticAction
from noema.contracts.environment import EnvironmentSpec
from noema.contracts.theorems import TheoremSpec
from noema.formal.bridge import FormalBridge
from noema.models.config import TransformerConfig
from noema.models.transformer import NoemaTransformer


class BenchmarkSample(BaseModel):
    sample_id: int
    workload_type: str
    duration_ms: float
    items_processed: int
    throughput_per_sec: float
    memory_rss_mb: float


class SystemBenchmarkReport(BaseModel):
    neural_token_throughput_per_sec: float
    neural_latency_p50_ms: float
    neural_latency_p99_ms: float
    kernel_transition_throughput_per_sec: float
    kernel_latency_p50_ms: float
    kernel_latency_p99_ms: float
    peak_rss_mb: float
    short_samples: list[BenchmarkSample] = Field(default_factory=list)
    long_samples: list[BenchmarkSample] = Field(default_factory=list)


def get_current_rss_mb() -> float:
    try:
        import resource
        # ru_maxrss on macOS is in bytes
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        return float(rusage.ru_maxrss) / (1024.0 * 1024.0)
    except Exception:
        return 0.0


class BenchmarkRunner:
    """Executes throughput, latency, and resource benchmarking (make bench)."""

    def __init__(self, bridge: FormalBridge | None = None):
        self.bridge = bridge or FormalBridge(
            env_spec=EnvironmentSpec(
                toolchain="leanprover/lean4:v4.8.0",
                imported_modules=["Init.Data.Nat.Basic"],
            )
        )

    def benchmark_neural_throughput(
        self,
        num_short: int = 30,
        num_long: int = 10,
    ) -> tuple[list[BenchmarkSample], list[BenchmarkSample]]:
        cfg = TransformerConfig(vocab_size=500, d_model=128, n_layers=4, n_heads=4, d_ff=256)
        model = NoemaTransformer(cfg)
        model.eval()

        short_samples: list[BenchmarkSample] = []
        long_samples: list[BenchmarkSample] = []

        # 1. Short measurements (batch size 4, seq len 32)
        for i in range(num_short):
            x = torch.randint(1, 500, (4, 32))
            mask = torch.ones(4, 32, dtype=torch.bool)
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(x, attention_mask=mask)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            tokens = 4 * 32
            rate = (tokens / dt_ms) * 1000.0 if dt_ms > 0 else 0.0

            short_samples.append(
                BenchmarkSample(
                    sample_id=i,
                    workload_type="neural_short",
                    duration_ms=dt_ms,
                    items_processed=tokens,
                    throughput_per_sec=rate,
                    memory_rss_mb=get_current_rss_mb(),
                )
            )

        # 2. Longer repetitions (batch size 8, seq len 128)
        for i in range(num_long):
            x = torch.randint(1, 500, (8, 128))
            mask = torch.ones(8, 128, dtype=torch.bool)
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(x, attention_mask=mask)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            tokens = 8 * 128
            rate = (tokens / dt_ms) * 1000.0 if dt_ms > 0 else 0.0

            long_samples.append(
                BenchmarkSample(
                    sample_id=i,
                    workload_type="neural_long",
                    duration_ms=dt_ms,
                    items_processed=tokens,
                    throughput_per_sec=rate,
                    memory_rss_mb=get_current_rss_mb(),
                )
            )

        return short_samples, long_samples

    def benchmark_kernel_transitions(self, num_transitions: int = 15) -> list[BenchmarkSample]:
        thm = TheoremSpec(
            theorem_id="bench_trans",
            name="thm_bench_trans",
            module="Test.Basic",
            statement="∀ (x : Nat), x = x",
        )
        init_state, _ = self.bridge.get_initial_state(thm)
        act = TacticAction.from_string("rfl")

        samples = []
        for i in range(num_transitions):
            t0 = time.perf_counter()
            res = self.bridge.step(thm, init_state, act, tactic_history=["intro x"])
            dt_ms = (time.perf_counter() - t0) * 1000.0
            rate = (1.0 / dt_ms) * 1000.0 if dt_ms > 0 else 0.0

            samples.append(
                BenchmarkSample(
                    sample_id=i,
                    workload_type="kernel_transition",
                    duration_ms=dt_ms,
                    items_processed=1,
                    throughput_per_sec=rate,
                    memory_rss_mb=get_current_rss_mb(),
                )
            )
        return samples

    def run_full_benchmark(self, out_dir: Path | str = "runs/benchmarks") -> SystemBenchmarkReport:
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)

        short_samples, long_samples = self.benchmark_neural_throughput(num_short=30, num_long=10)
        kernel_samples = self.benchmark_kernel_transitions(num_transitions=15)

        neural_dts = sorted([s.duration_ms for s in short_samples])
        kernel_dts = sorted([s.duration_ms for s in kernel_samples])

        p50_idx = len(neural_dts) // 2
        p99_idx = int(len(neural_dts) * 0.99)
        k_p50_idx = len(kernel_dts) // 2
        k_p99_idx = int(len(kernel_dts) * 0.99)

        avg_neural_throughput = sum(s.throughput_per_sec for s in short_samples) / max(1, len(short_samples))
        avg_kernel_throughput = sum(s.throughput_per_sec for s in kernel_samples) / max(1, len(kernel_samples))

        report = SystemBenchmarkReport(
            neural_token_throughput_per_sec=avg_neural_throughput,
            neural_latency_p50_ms=neural_dts[p50_idx],
            neural_latency_p99_ms=neural_dts[p99_idx],
            kernel_transition_throughput_per_sec=avg_kernel_throughput,
            kernel_latency_p50_ms=kernel_dts[k_p50_idx],
            kernel_latency_p99_ms=kernel_dts[k_p99_idx],
            peak_rss_mb=get_current_rss_mb(),
            short_samples=short_samples,
            long_samples=long_samples,
        )

        with open(p / "benchmark_report.json", "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))

        return report
