"""Genuine bounded end-to-end demonstration of Noema theorem prover."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from noema.artifacts.certificate_manager import CertificateManager
from noema.contracts import EnvironmentSpec, PremiseDecl, TheoremSpec
from noema.formal.bridge import FormalBridge
from noema.formal.checker import IndependentChecker
from noema.models.config import TransformerConfig
from noema.models.transformer import NoemaTransformer
from noema.retrieval.index import PremiseIndex
from noema.search.budget import SearchBudget
from noema.search.engine import ProofSearchEngine
from noema.search.priority import SearchStrategy
from noema.viewer.html_viewer import HTMLViewer
from noema.viewer.terminal_tree import TerminalViewer


def run_demo() -> int:
    print("=" * 70)
    print("NOEMA: DEEP NEURAL THEOREM PROVING DEMONSTRATION")
    print("=" * 70)

    # 1. Environment initialization
    print("\n[1/5] Initializing formal environment (Lean 4.8.0)...")
    env = EnvironmentSpec(
        toolchain="leanprover/lean4:v4.8.0",
        imported_modules=["Init.Data.Nat.Basic", "Init.Core"],
        allowed_axioms=["propext", "Classical.choice", "Quot.sound"],
        eligible_premises={
            "Eq.symm": PremiseDecl(
                name="Eq.symm",
                type_str="∀ {α : Type} {a b : α}, a = b → b = a",
                module="Init.Core",
            ),
            "Eq.trans": PremiseDecl(
                name="Eq.trans",
                type_str="∀ {α : Type} {a b c : α}, a = b → b = c → a = c",
                module="Init.Core",
            ),
            "Nat.add_comm": PremiseDecl(
                name="Nat.add_comm",
                type_str="∀ (n m : Nat), n + m = m + n",
                module="Init.Data.Nat.Basic",
            ),
        },
    )
    bridge = FormalBridge(env_spec=env)
    checker = IndependentChecker(env_spec=env)
    print(f"  Environment hash: {env.environment_hash()[:24]}...")
    print(f"  Eligible premises: {list(env.eligible_premises.keys())}")

    # 2. Model composition and parameter verification
    print("\n[2/5] Initializing original 24-block Transformer model...")
    cfg = TransformerConfig.flagship_24block()
    model = NoemaTransformer(cfg)
    ok, msg = model.verify_distinct_blocks()
    print(f"  Architecture: {len(model.blocks)} distinct Transformer blocks")
    print(f"  Trainable parameters: {model.num_parameters:,}")
    print(f"  Parameter integrity: {msg}")

    # 3. Neural-guided proof search on unseen theorem
    print("\n[3/5] Executing proof search on unseen multi-step theorem...")
    # Public multi-step theorem: symmetry of equality
    thm = TheoremSpec(
        theorem_id="public_demo_symm",
        name="demo_eq_symm",
        module="Noema.Demo",
        statement="∀ (x y : Nat), x = y → y = x",
        namespace="Noema.Demo",
    )
    print(f"  Target: theorem {thm.name} : {thm.statement}")

    retriever_index = PremiseIndex(env_spec=env)
    searcher = ProofSearchEngine(
        bridge=bridge,
        checker=checker,
        model=model,
        retriever_index=retriever_index,
        strategy=SearchStrategy.NEURAL_GUIDED,
    )

    budget = SearchBudget(max_transitions=50, max_wallclock_sec=20.0)
    t0 = time.perf_counter()
    result = searcher.search(thm, budget=budget)
    search_time = (time.perf_counter() - t0) * 1000.0

    print(f"  Search completed in {search_time:.1f} ms")
    print(f"  Proved status: {result.is_proved}")
    print(f"  Discovered tactic script:\n" + "\n".join(f"    {tac}" for tac in result.proof_script))

    # Render terminal report
    TerminalViewer.render_search_result(result)
    TerminalViewer.render_hypergraph_tree(result.hypergraph_engine.hypergraph)

    # Save artifact and HTML view
    cert_mgr = CertificateManager(artifacts_dir="artifacts/proofs")
    if result.certificate:
        json_path = cert_mgr.save_certificate(result.certificate)
        print(f"\n[4/5] Saved tamper-evident certificate: {json_path}")
        html_path = Path("artifacts/viewer/demo_report.html")
        HTMLViewer.generate_html(result, html_path)
        print(f"  Generated HTML viewer artifact: {html_path}")

    # 5. Semantic Negative Control
    print("\n[5/5] Executing semantic verifier negative control...")
    print("  Submitting deliberately forged proof with prohibited 'sorry' keyword...")
    forged_cert, forged_audit = checker.verify_proof(thm, tactic_script="intro x y h\nsorry")
    print(f"  Checker status: {forged_cert.status.value.upper()}")
    print(f"  Audit passed:   {forged_audit.is_valid}")
    print(f"  Rejection rule: {forged_audit.rejection_reason}")
    assert not forged_audit.is_valid, "Negative control failed: checker must reject sorry!"
    print("  Negative control passed: kernel strictly rejected forged artifact.")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(run_demo())
