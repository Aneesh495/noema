"""Noema unified command line interface."""

from __future__ import annotations

import re
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


def _parse_theorem_declaration(declaration: str) -> tuple[str, str]:
    match = re.fullmatch(r"\s*theorem\s+([\w.']+)\s*(.+?)\s*:=\s*by\s*", declaration, re.DOTALL)
    if match:
        name, signature = match.groups()
        depth = 0
        for index, char in enumerate(signature):
            if char in "([{":
                depth += 1
            elif char in ")]}":
                depth -= 1
            elif char == ":" and depth == 0:
                binders = signature[:index].strip()
                proposition = signature[index + 1:].strip()
                if proposition and (not binders or binders[0] in "([{"):
                    return name, f"∀ {binders}, {proposition}" if binders else proposition
                break
    raise click.BadParameter("expected 'theorem name (optional binders) : proposition := by'", param_hint="--theorem")


@main.command()
@click.option("--theorem", default="theorem demo_id (P : Prop) : P → P := by",
              help="Lean 4 theorem declaration to prove interactively")
def shell(theorem: str):
    """Read tactics and check a proof interactively."""
    from noema.contracts.states import ProofState, ProofGoal
    from noema.contracts.theorems import TheoremSpec
    from noema.shell import InteractiveProofShell, ShellCommandType

    name, proposition = _parse_theorem_declaration(theorem)
    theorem_spec = TheoremSpec(
        theorem_id="cli_interactive",
        name=name,
        module="Init",
        statement=proposition,
        family="logic",
    )
    initial_state = ProofState(
        state_id="init_0",
        goals=[ProofGoal(goal_id="g0", target=proposition, mvar_id="?m.0")],
    )
    proof_shell = InteractiveProofShell(theorem=theorem_spec, initial_state=initial_state)
    click.echo(proof_shell.format_status())
    click.echo("Type a Lean tactic, 'help', or 'quit'.")

    while True:
        try:
            command, argument = proof_shell.parse_command(input("noema> "))
        except EOFError:
            click.echo()
            return

        match command:
            case ShellCommandType.QUIT:
                return
            case ShellCommandType.STATUS:
                click.echo(proof_shell.format_status())
            case ShellCommandType.HELP:
                click.echo("Commands: status, undo, redo, suggest [count], retrieve <query>, export [path], quit")
            case ShellCommandType.UNDO:
                click.echo(proof_shell.undo()[1])
            case ShellCommandType.REDO:
                click.echo(proof_shell.redo()[1])
            case ShellCommandType.SUGGEST:
                try:
                    count = int(argument or "5")
                    if count < 1:
                        raise ValueError
                except ValueError:
                    click.echo("Suggestion count must be a positive integer.")
                    continue
                for tactic, _ in proof_shell.suggest_tactics(top_k=count)[:count]:
                    click.echo(tactic)
            case ShellCommandType.RETRIEVE:
                click.echo("\n".join(proof_shell.retrieve_premises(argument)) or "No matching premises.")
            case ShellCommandType.EXPORT:
                try:
                    click.echo(proof_shell.export_proof_script(argument or None))
                except OSError as error:
                    click.echo(f"Export failed: {error}", err=True)
            case ShellCommandType.TACTIC:
                click.echo(proof_shell.execute_tactic(argument)[1] if argument else "Enter a Lean tactic.")


if __name__ == "__main__":
    main()
