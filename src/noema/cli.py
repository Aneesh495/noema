"""Noema unified command line interface."""

from __future__ import annotations

import sys
import click
from noema.doctor import run_diagnostics
from noema.demo import run_demo


@click.group()
@click.version_option(version="0.1.0")
def main():
    """Noema: Deep neural search for checked mathematical proofs."""
    pass


@main.command()
def doctor():
    """Probe device, precision, memory, data, and toolchain dependencies."""
    sys.exit(run_diagnostics())


@main.command()
def demo():
    """Execute bounded end-to-end trained-model demonstration."""
    sys.exit(run_demo())


@main.command()
@click.option("--output", default="data", help="Output directory for generated datasets")
@click.option("--num-train", default=20000, help="Number of training tasks")
@click.option("--num-eval", default=2000, help="Number of held-out tasks")
def data(output, num_train, num_eval):
    """Generate controlled mathematical task datasets."""
    from pathlib import Path
    from noema.corpus.synthetic import TaskGenerator
    import json
    
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    click.echo(f"Generating {num_train} train and {num_eval} eval tasks...")
    gen = TaskGenerator(seed=42)
    train_tasks, eval_tasks = gen.generate_corpus(num_train=num_train, num_eval=num_eval)
    
    with open(out_dir / "train_controlled.jsonl", "w") as f:
        for t in train_tasks:
            f.write(t.model_dump_json() + "\n")
            
    with open(out_dir / "eval_controlled.jsonl", "w") as f:
        for t in eval_tasks:
            f.write(t.model_dump_json() + "\n")

    click.echo(f"Dataset saved to {out_dir}/")


@main.command()
@click.option("--output", default="runs/benchmarks", help="Output directory for benchmark reports")
def bench(output):
    """Run throughput, latency, and memory resource benchmarks."""
    from noema.benchmarks.throughput import BenchmarkRunner
    runner = BenchmarkRunner()
    click.echo("Running performance and latency benchmarks...")
    rep = runner.run_full_benchmark(out_dir=output)
    click.echo(f"Neural throughput: {rep.neural_token_throughput_per_sec:.1f} tokens/s (p50: {rep.neural_latency_p50_ms:.2f} ms)")
    click.echo(f"Kernel throughput: {rep.kernel_transition_throughput_per_sec:.2f} trans/s (p50: {rep.kernel_latency_p50_ms:.2f} ms)")
    click.echo(f"Peak RSS: {rep.peak_rss_mb:.1f} MB")


@main.command()
@click.option("--run-id", default=None, help="Run ID for acceptance bundle")
@click.option("--out-dir", default=None, help="Output directory for evidence")
def acceptance(run_id, out_dir):
    """Generate fresh complete source-bound acceptance evidence."""
    from noema.acceptance import run_acceptance
    sys.exit(run_acceptance(run_id=run_id, out_dir=out_dir))


@main.command()
@click.option("--evidence", required=True, help="Path to evidence directory to verify")
def verify(evidence):
    """Independently validate an existing evidence bundle without retraining."""
    from noema.verifier import verify_evidence_bundle
    sys.exit(verify_evidence_bundle(evidence_dir=evidence))


@main.command()
@click.option("--theorem", default=None, help="Lean 4 theorem declaration to prove interactively")
def shell(theorem):
    """Launch interactive proof shell with neural assistance."""
    from noema.contracts.theorems import TheoremSpec
    from noema.contracts.states import ProofState, ProofGoal
    from noema.shell import InteractiveProofShell
    
    decl = theorem or "theorem demo_id (P : Prop) : P → P := by"
    th = TheoremSpec(
        theorem_id="cli_interactive",
        formal_statement=decl,
        informal_statement="Interactive demonstration conjecture",
        domain="logic",
        difficulty=1,
    )
    init_state = ProofState(
        state_id="init_0",
        goals=[ProofGoal(goal_id="g0", target_type="P → P")],
    )
    click.echo(f"Starting interactive proof shell for: {decl}")
    sh = InteractiveProofShell(theorem=th, initial_state=init_state)
    click.echo("Type tactics, 'suggest', 'undo', 'status', or 'help'.")


if __name__ == "__main__":
    main()
