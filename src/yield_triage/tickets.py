"""File-backed ticket store: pending drafts and committed tickets.

What: ``propose`` writes ``pending/<id>.json``; ``commit`` (called only by
``approvals.commit_ticket``) moves a ticket to ``committed/<id>.json``.
``content_hash`` is SHA-256 over the canonical JSON of exactly the fields a
human approves: id, title, body and evidence refs.

Why: the agent may draft but never finalise. Keeping drafts as plain files
makes the approval step easy to inspect, and the content hash makes any edit
after approval detectable (THREAT_MODEL.md T5).

Connects to: ``policy/gateway.py`` (``propose_ticket`` tool) and
``approvals.py`` (approve, commit).
"""

from __future__ import annotations

import json
import os
import re
import secrets
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from yield_triage.audit import canonical_json, sha256_hex

TICKET_ID_RE = re.compile(r"^T-[0-9a-f]{12}$")


@dataclass(frozen=True)
class Ticket:
    ticket_id: str
    title: str
    body: str
    evidence_refs: list[str]
    run_id: str
    created_at: str
    status: str  # "pending" or "committed"

    def approved_content(self) -> dict[str, Any]:
        """The exact fields an approval token is bound to."""
        return {
            "ticket_id": self.ticket_id,
            "title": self.title,
            "body": self.body,
            "evidence_refs": list(self.evidence_refs),
        }

    @property
    def content_hash(self) -> str:
        return sha256_hex(canonical_json(self.approved_content()))


def validate_ticket_id(ticket_id: str) -> str:
    # SECURITY: T2 - ticket IDs become file names; a strict pattern rules out
    # traversal like "../../x" before any path is built.
    if not TICKET_ID_RE.fullmatch(ticket_id):
        raise ValueError("invalid ticket id")
    return ticket_id


class TicketStore:
    def __init__(self, root: Path) -> None:
        self.pending_dir = root / "pending"
        self.committed_dir = root / "committed"
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        self.committed_dir.mkdir(parents=True, exist_ok=True)

    def propose(self, *, title: str, body: str, evidence_refs: list[str], run_id: str) -> Ticket:
        ticket = Ticket(
            ticket_id=f"T-{secrets.token_hex(6)}",
            title=title,
            body=body,
            evidence_refs=list(evidence_refs),
            run_id=run_id,
            created_at=datetime.now(UTC).isoformat(),
            status="pending",
        )
        _atomic_write(self.pending_dir / f"{ticket.ticket_id}.json", asdict(ticket))
        return ticket

    def load_pending(self, ticket_id: str) -> Ticket:
        return self._load(self.pending_dir, ticket_id)

    def load_committed(self, ticket_id: str) -> Ticket:
        return self._load(self.committed_dir, ticket_id)

    def is_committed(self, ticket_id: str) -> bool:
        return (self.committed_dir / f"{validate_ticket_id(ticket_id)}.json").exists()

    def mark_committed(self, ticket: Ticket, approval: dict[str, Any]) -> None:
        """Write the committed record, then remove the draft."""
        record = {**asdict(ticket), "status": "committed", "approval": approval}
        _atomic_write(self.committed_dir / f"{ticket.ticket_id}.json", record)
        (self.pending_dir / f"{ticket.ticket_id}.json").unlink()

    def _load(self, directory: Path, ticket_id: str) -> Ticket:
        path = directory / f"{validate_ticket_id(ticket_id)}.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw.pop("approval", None)
        return Ticket(**raw)


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
