"""Standalone HTML and SVG proof hypergraph viewer generator."""

from __future__ import annotations

from pathlib import Path
from noema.contracts.certificates import ProofCertificate
from noema.search.engine import SearchResult


class HTMLViewer:
    """Generates interactive, self-contained HTML visualizer for verified proof search trees."""

    @classmethod
    def generate_html(cls, result: SearchResult, output_path: Path | str) -> Path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        
        thm = result.theorem
        acc = result.accounting
        hg = result.hypergraph_engine.hypergraph
        cert = result.certificate

        badge_color = "#10b981" if result.is_proved else "#ef4444"
        badge_text = "VERIFIED LEAN 4 PROOF" if result.is_proved else "UNSOLVED"

        tactic_rows = "\n".join(
            f"<tr><td style='padding:6px;border-bottom:1px solid #334155;color:#94a3b8;'>{idx}</td>"
            f"<td style='padding:6px;border-bottom:1px solid #334155;font-family:monospace;color:#38bdf8;'>{tac}</td></tr>"
            for idx, tac in enumerate(result.proof_script, 1)
        )

        nodes_html = []
        for n_id, n in hg.nodes.items():
            status_color = "#10b981" if n.is_proved else "#f59e0b"
            nodes_html.append(
                f"<div style='margin:8px 0;padding:10px;background:#1e293b;border-left:4px solid {status_color};border-radius:4px;'>"
                f"<div style='font-size:12px;color:#94a3b8;'>Node: {n_id} (Depth {n.depth})</div>"
                f"<div style='font-family:monospace;color:#f8fafc;margin-top:4px;'>⊢ {n.goal.target}</div>"
                f"</div>"
            )
        nodes_block = "\n".join(nodes_html)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Noema Proof Report: {thm.name}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }}
  .container {{ max-width: 900px; margin: 0 auto; }}
  .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #334155; padding-bottom: 16px; margin-bottom: 24px; }}
  .badge {{ background: {badge_color}; color: #ffffff; padding: 6px 14px; border-radius: 9999px; font-weight: 600; font-size: 13px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px; }}
  .card {{ background: #1e293b; padding: 16px; border-radius: 8px; border: 1px solid #334155; }}
  .label {{ font-size: 12px; color: #94a3b8; text-transform: uppercase; margin-bottom: 4px; }}
  .val {{ font-size: 18px; font-weight: 600; color: #f8fafc; }}
  pre {{ background: #0b1120; padding: 12px; border-radius: 6px; overflow-x: auto; color: #38bdf8; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, monospace; }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <div>
      <h1 style="margin:0 0 6px 0;font-size:22px;">{thm.name}</h1>
      <div style="color:#94a3b8;font-size:13px;">ID: {thm.theorem_id} | Module: {thm.module}</div>
    </div>
    <div class="badge">{badge_text}</div>
  </div>

  <div class="grid">
    <div class="card">
      <div class="label">Mathematical Statement</div>
      <div style="font-family:monospace;color:#f8fafc;">{thm.statement}</div>
    </div>
    <div class="card">
      <div class="label">Search Performance</div>
      <div class="val">{acc.total_time_ms:.1f} ms</div>
      <div style="font-size:12px;color:#94a3b8;margin-top:4px;">{acc.valid_transitions} valid / {acc.attempted_transitions} attempted transitions</div>
    </div>
  </div>

  <div class="card" style="margin-bottom:24px;">
    <div class="label">Verified Lean 4 Proof Script</div>
    <table style="width:100%;border-collapse:collapse;margin-top:8px;">
      {tactic_rows if tactic_rows else "<tr><td style='color:#94a3b8;'>No proof found.</td></tr>"}
    </table>
  </div>

  <div class="card" style="margin-bottom:24px;">
    <div class="label">Proof Hypergraph State Hierarchy</div>
    <div style="margin-top:12px;">
      {nodes_block}
    </div>
  </div>

  <div class="card">
    <div class="label">Trusted Kernel Boundary Verification</div>
    <pre>{cert.model_dump_json(indent=2) if cert else "No certificate emitted"}</pre>
  </div>
</div>
</body>
</html>
"""
        with open(out_p, "w", encoding="utf-8") as f:
            f.write(html)
        return out_p
