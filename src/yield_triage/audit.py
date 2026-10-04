"""Append-only, hash-chained JSONL audit log and its verifier.

What: every policy decision (allow or deny), ticket proposal, approval and
commit becomes one JSON line. Each record carries ``prev_hash`` (the previous
record's ``hash``) and its own ``hash`` = SHA-256 over its canonical JSON
without the ``hash`` field. ``verify_audit`` recomputes the chain.

Why: a reviewer must be able to see what the agent tried, what was blocked and
why, and detect if someone edited, deleted or reordered records afterwards.

Connects to: ``policy/gateway.py`` (tool calls), ``approvals.py`` (approve and
commit), the ``verify-audit`` CLI command, and every security test, which
asserts that the blocked attack shows up here.

Limits (stated in THREAT_MODEL.md T10): a hash chain detects edits, deletions
in the middle and reordering. It cannot detect truncation of the tail or a
whole-file rewrite by someone with write access.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from yield_triage.filelock import exclusive_lock

GENESIS_HASH = "0" * 64


def canonical_json(value: Any) -> str:
    """Deterministic JSON: sorted keys, no spaces, ASCII only."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def sha256_hex(text: str | bytes) -> str:
    data = text.encode("utf-8") if isinstance(text, str) else text
    return hashlib.sha256(data).hexdigest()


def record_hash(record: dict[str, Any]) -> str:
    body = {k: v for k, v in record.items() if k != "hash"}
    return sha256_hex(canonical_json(body))


class AuditLog:
    """Appends records under a cross-process lock so the chain never forks."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, **fields: Any) -> dict[str, Any]:
        """Write one record. Callers pass hashes and sizes, never raw secrets."""
        # SECURITY: T11 - the API takes keyword fields chosen by our code; tool
        # arguments arrive only as args_sha256, so a secret pasted into an
        # argument is never written in clear.
        with exclusive_lock(self.path):
            prev_seq, prev_hash = self._tail()
            record: dict[str, Any] = {
                "seq": prev_seq + 1,
                "ts": datetime.now(UTC).isoformat(),
                **fields,
                "prev_hash": prev_hash,
            }
            record["hash"] = record_hash(record)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(canonical_json(record) + "\n")
        return record

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def _tail(self) -> tuple[int, str]:
        """Last seq and hash, read from disk so other processes' writes count."""
        if not self.path.exists():
            return 0, GENESIS_HASH
        last = ""
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    last = line
        if not last:
            return 0, GENESIS_HASH
        rec = json.loads(last)
        return int(rec["seq"]), str(rec["hash"])


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    n_records: int
    first_bad_seq: int | None = None
    problem: str | None = None


def verify_audit(path: Path) -> VerifyResult:
    """Recompute the chain; report the first record that does not fit."""
    # SECURITY: T10 - any edit changes a record's hash; any deletion or
    # reordering breaks seq continuity or the prev_hash link.
    if not path.exists():
        return VerifyResult(True, 0)
    prev_hash, expected_seq, n = GENESIS_HASH, 1, 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            return VerifyResult(False, n, expected_seq, "unparseable line")
        problem = _check_record(rec, expected_seq, prev_hash)
        if problem:
            return VerifyResult(False, n, expected_seq, problem)
        prev_hash, expected_seq, n = rec["hash"], expected_seq + 1, n + 1
    return VerifyResult(True, n)


def _check_record(rec: Any, expected_seq: int, prev_hash: str) -> str | None:
    if not isinstance(rec, dict):
        return "record is not an object"
    if rec.get("seq") != expected_seq:
        return "sequence gap or reorder"
    if rec.get("prev_hash") != prev_hash:
        return "prev_hash does not link to previous record"
    if rec.get("hash") != record_hash(rec):
        return "record content does not match its hash"
    return None
