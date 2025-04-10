"""Cryptographic Merkle tree audit ledger and inclusion proof verification for proof certificates."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Sequence
from noema.contracts.certificates import ProofCertificate


@dataclass(frozen=True)
class MerkleInclusionProof:
    """Cryptographic audit path proving a certificate's inclusion in the Merkle root."""
    leaf_hash: str
    leaf_index: int
    audit_path: tuple[tuple[str, str], ...]  # Tuple of (direction, sibling_hash)
    expected_root: str

    def verify(self) -> bool:
        """Verify that hashing along the audit path matches the expected root."""
        current = self.leaf_hash
        for direction, sibling in self.audit_path:
            if direction == "left":
                combined = sibling + current
            else:
                combined = current + sibling
            current = hashlib.sha256(combined.encode("utf-8")).hexdigest()
        return current == self.expected_root


class MerkleAuditTree:
    """Binary Merkle tree providing immutable tamper-evidence across verification certificates."""

    def __init__(self, leaf_hashes: Sequence[str] | None = None):
        self.leaves: list[str] = list(leaf_hashes) if leaf_hashes else []
        self.levels: list[list[str]] = []
        if self.leaves:
            self._build_tree()

    @classmethod
    def from_certificates(cls, certificates: Sequence[ProofCertificate]) -> MerkleAuditTree:
        """Construct a Merkle tree from an ordered collection of proof certificates."""
        leaf_hashes = []
        for cert in certificates:
            # Hash certificate's sealed cryptographic hash
            h = hashlib.sha256(cert.certificate_hash.encode("utf-8")).hexdigest()
            leaf_hashes.append(h)
        return cls(leaf_hashes=leaf_hashes)

    def append_certificate(self, certificate: ProofCertificate) -> str:
        """Append a new certificate leaf and rebuild the tree."""
        h = hashlib.sha256(certificate.certificate_hash.encode("utf-8")).hexdigest()
        self.leaves.append(h)
        self._build_tree()
        return h

    def _build_tree(self) -> None:
        """Compute all internal tree levels up to root."""
        if not self.leaves:
            self.levels = []
            return

        current_level = list(self.leaves)
        self.levels = [current_level]

        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                if i + 1 < len(current_level):
                    right = current_level[i + 1]
                else:
                    # Duplicate odd leaf to balance tree
                    right = left
                combined = left + right
                parent = hashlib.sha256(combined.encode("utf-8")).hexdigest()
                next_level.append(parent)

            current_level = next_level
            self.levels.append(current_level)

    @property
    def root(self) -> str:
        """Return root hash of the Merkle tree."""
        if not self.levels or not self.levels[-1]:
            return ""
        return self.levels[-1][0]

    def generate_proof(self, leaf_index: int) -> MerkleInclusionProof:
        """Generate an audit path of sibling hashes for the specified leaf index."""
        if not (0 <= leaf_index < len(self.leaves)):
            raise IndexError(f"Leaf index {leaf_index} out of bounds")

        audit_path: list[tuple[str, str]] = []
        idx = leaf_index

        for level in self.levels[:-1]:
            is_right_child = (idx % 2 == 1)
            if is_right_child:
                sibling_idx = idx - 1
                direction = "left"
            else:
                sibling_idx = idx + 1 if idx + 1 < len(level) else idx
                direction = "right"

            audit_path.append((direction, level[sibling_idx]))
            idx //= 2

        return MerkleInclusionProof(
            leaf_hash=self.leaves[leaf_index],
            leaf_index=leaf_index,
            audit_path=tuple(audit_path),
            expected_root=self.root,
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize tree ledger state to dictionary."""
        return {
            "root": self.root,
            "leaf_count": len(self.leaves),
            "leaves": list(self.leaves),
        }
