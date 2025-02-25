"""Unit tests for acceptance evidence generation and independent verifier."""

import tempfile
from pathlib import Path
import pytest
from noema.acceptance import run_acceptance
from noema.verifier import EvidenceVerifier, run_negative_controls, verify_evidence_bundle


def test_acceptance_bundle_generation_and_verification():
    with tempfile.TemporaryDirectory() as tmpdir:
        evidence_path = Path(tmpdir) / "evidence_test"
        
        # 1. Run acceptance in fast_mode for unit test
        ret = run_acceptance(run_id="test_run", out_dir=str(evidence_path), fast_mode=True)
        assert ret == 0
        assert (evidence_path / "MANIFEST.json").exists()
        assert (evidence_path / "ACCEPTANCE.json").exists()
        assert (evidence_path / "flagship_24block.pt").exists()

        # 2. Verify bundle
        v_ret = verify_evidence_bundle(evidence_path)
        assert v_ret == 0
