"""Tests for Merkle tree certificate audit ledger and cryptographic inclusion proofs."""

import pytest
from noema.artifacts.merkle import MerkleAuditTree, MerkleInclusionProof
from noema.contracts.certificates import ProofCertificate, VerificationStatus


def make_sample_cert(tid: str) -> ProofCertificate:
    cert = ProofCertificate(
        theorem_id=tid,
        theorem_name=f"thm_{tid}",
        statement="x = x",
        statement_hash="hash123",
        environment_hash="env123",
        tactic_script="rfl",
        status=VerificationStatus.VERIFIED,
        timestamp_utc="2026-01-01T00:00:00Z",
    )
    cert.seal()
    return cert


def test_merkle_tree_root_and_verification():
    certs = [make_sample_cert(f"c{i}") for i in range(5)]
    tree = MerkleAuditTree.from_certificates(certs)

    assert len(tree.leaves) == 5
    assert len(tree.root) == 64  # Valid SHA-256 hex string

    # Verify inclusion proof for every single certificate leaf
    for idx in range(len(certs)):
        proof = tree.generate_proof(idx)
        assert proof.leaf_index == idx
        assert proof.expected_root == tree.root
        assert proof.verify(), f"Inclusion proof failed for leaf {idx}"


def test_merkle_tamper_detection():
    certs = [make_sample_cert(f"c{i}") for i in range(4)]
    tree = MerkleAuditTree.from_certificates(certs)
    proof = tree.generate_proof(leaf_index=2)

    assert proof.verify()

    # Tamper with leaf hash
    tampered_proof = MerkleInclusionProof(
        leaf_hash="0" * 64,
        leaf_index=proof.leaf_index,
        audit_path=proof.audit_path,
        expected_root=proof.expected_root,
    )
    assert not tampered_proof.verify()


def test_merkle_tree_incremental_append():
    tree = MerkleAuditTree()
    assert tree.root == ""

    cert0 = make_sample_cert("c0")
    tree.append_certificate(cert0)
    root0 = tree.root
    assert len(root0) == 64

    cert1 = make_sample_cert("c1")
    tree.append_certificate(cert1)
    root1 = tree.root
    assert root1 != root0

    proof1 = tree.generate_proof(1)
    assert proof1.verify()
