"""Human approval: HMAC-signed, single-use, expiring tokens bound to ticket content.

What: ``approve_ticket`` mints a token for the *current* content of a pending
ticket; ``commit_ticket`` verifies that token and finalises the ticket. Token
= ``v1.<payload>.<signature>``, where the payload holds the ticket ID, the
SHA-256 of the canonical ticket content, an expiry time and a random nonce,
and the signature is HMAC-SHA256 under a key only the approval CLI loads.

Why: this is the control against excessive agency (OWASP LLM06). The agent
can draft a ticket but cannot finalise one: commit is not an MCP tool, the
agent never sees the key, and a token cannot be replayed, reused for another
ticket, used after the ticket is edited, or used after it expires.

Connects to: the ``yield-triage approve`` / ``commit`` CLI commands (Phase 3)
and ``tests/security/test_approvals.py``. The MCP server never imports
``load_signing_key``.
"""

from __future__ import annotations

import base64
import hmac
import json
import os
import secrets
import stat
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from yield_triage.audit import AuditLog, canonical_json
from yield_triage.filelock import exclusive_lock
from yield_triage.tickets import Ticket, TicketStore, validate_ticket_id

KEY_ENV = "YIELD_TRIAGE_APPROVAL_KEY"
KEY_FILE_ENV = "YIELD_TRIAGE_KEY_FILE"
DEFAULT_KEY_FILE = Path(".local/keys/approval.key")
MIN_KEY_BYTES = 32
TOKEN_VERSION = "v1"  # noqa: S105 - token format tag, not a secret


class ApprovalError(Exception):
    """Commit refused. ``reason`` is a short machine-readable code."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


# ---------------------------------------------------------------- key handling


def load_signing_key(env: dict[str, str] | None = None) -> bytes:
    """Key from ``YIELD_TRIAGE_APPROVAL_KEY`` (hex) or an owner-only key file.

    If neither exists, a random key file is created with mode 0600.
    """
    # SECURITY: T8 - only the human-run approval CLI calls this. The agent is
    # launched with a scrubbed environment and the MCP stdio client passes the
    # server only a fixed safe set of variables, so neither process sees it.
    env = dict(os.environ) if env is None else env
    if env.get(KEY_ENV):
        key = bytes.fromhex(env[KEY_ENV])
        if len(key) < MIN_KEY_BYTES:
            raise ValueError(f"{KEY_ENV} must be at least {MIN_KEY_BYTES} bytes of hex")
        return key
    path = Path(env.get(KEY_FILE_ENV) or DEFAULT_KEY_FILE)
    if not path.exists():
        _create_key_file(path)
    _check_key_file_permissions(path)
    key = bytes.fromhex(path.read_text(encoding="ascii").strip())
    if len(key) < MIN_KEY_BYTES:
        raise ValueError("key file too short")
    return key


def _create_key_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="ascii") as fh:
        fh.write(secrets.token_hex(MIN_KEY_BYTES) + "\n")


def _check_key_file_permissions(path: Path) -> None:
    # SECURITY: T8 - on POSIX refuse a key readable by group or others. On
    # Windows, POSIX modes are not meaningful; this is documented as a
    # limitation and the container (Linux) is the verified path.
    if sys.platform != "win32" and stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise PermissionError("approval key file must be owner-only (chmod 600)")


# ---------------------------------------------------------------- tokens


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _sign(key: bytes, payload_b64: str) -> str:
    msg = f"{TOKEN_VERSION}.{payload_b64}".encode("ascii")
    return _b64(hmac.new(key, msg, "sha256").digest())


def mint_token(key: bytes, ticket_id: str, content_hash: str, *, now: float, ttl_s: int) -> str:
    payload = {
        "tid": ticket_id,
        "h": content_hash,
        "exp": int(now) + ttl_s,
        "n": secrets.token_hex(16),
    }
    payload_b64 = _b64(canonical_json(payload).encode("ascii"))
    return f"{TOKEN_VERSION}.{payload_b64}.{_sign(key, payload_b64)}"


def verify_token(
    key: bytes, token: str, ticket_id: str, content_hash: str, *, now: float
) -> dict[str, Any]:
    """Return the payload if the token is valid for exactly this content now."""
    parts = token.split(".")
    if len(parts) != 3 or parts[0] != TOKEN_VERSION or len(token) > 1024:
        raise ApprovalError("token_malformed")
    _, payload_b64, sig = parts
    # SECURITY: T7 - constant-time comparison; signature checked before the
    # payload is trusted for anything.
    if not hmac.compare_digest(sig, _sign(key, payload_b64)):
        raise ApprovalError("token_bad_signature")
    try:
        payload: dict[str, Any] = json.loads(_unb64(payload_b64))
    except ValueError:
        raise ApprovalError("token_malformed") from None
    # SECURITY: T5 - bound to ticket ID and to the hash of the approved content.
    if payload.get("tid") != ticket_id:
        raise ApprovalError("token_wrong_ticket")
    if not hmac.compare_digest(str(payload.get("h")), content_hash):
        raise ApprovalError("ticket_content_changed")
    # SECURITY: T6 - expiry.
    if now >= float(payload.get("exp", 0)):
        raise ApprovalError("token_expired")
    return payload


# ---------------------------------------------------------------- single use


class NonceLedger:
    """Append-only record of consumed token nonces."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def consume(self, nonce: str) -> None:
        # SECURITY: T4 - check and record under one exclusive lock, so two
        # concurrent commits with the same token cannot both succeed.
        with exclusive_lock(self.path):
            used = set(self.path.read_text("ascii").split()) if self.path.exists() else set()
            if nonce in used:
                raise ApprovalError("token_replayed")
            with self.path.open("a", encoding="ascii") as fh:
                fh.write(nonce + "\n")


# ---------------------------------------------------------------- workflow


@dataclass
class ApprovalService:
    """Approve and commit, with every outcome written to the audit log."""

    store: TicketStore
    ledger: NonceLedger
    audit: AuditLog
    key: bytes
    ttl_s: int
    clock: Callable[[], float] = time.time

    def approve(self, ticket_id: str) -> tuple[Ticket, str]:
        ticket = self.store.load_pending(validate_ticket_id(ticket_id))
        token = mint_token(
            self.key, ticket.ticket_id, ticket.content_hash, now=self.clock(), ttl_s=self.ttl_s
        )
        self.audit.append(
            actor="approval_cli",
            event="ticket_approved",
            ticket_id=ticket.ticket_id,
            content_hash=ticket.content_hash,
            decision="allow",
        )
        return ticket, token

    def commit(self, ticket_id: str, token: str | None) -> Ticket:
        # SECURITY: T3 - no token, no commit; every refusal is audited.
        try:
            ticket = self._commit(ticket_id, token)
        except ApprovalError as err:
            self.audit.append(
                actor="approval_cli",
                event="ticket_commit",
                ticket_id=_safe_id(ticket_id),
                decision="deny",
                reason=err.reason,
            )
            raise
        self.audit.append(
            actor="approval_cli",
            event="ticket_commit",
            ticket_id=ticket.ticket_id,
            content_hash=ticket.content_hash,
            decision="allow",
        )
        return ticket

    def _commit(self, ticket_id: str, token: str | None) -> Ticket:
        if not token:
            raise ApprovalError("token_missing")
        try:
            ticket = self.store.load_pending(validate_ticket_id(ticket_id))
        except (ValueError, FileNotFoundError):
            raise ApprovalError("ticket_not_pending") from None
        payload = verify_token(
            self.key, token, ticket.ticket_id, ticket.content_hash, now=self.clock()
        )
        self.ledger.consume(str(payload["n"]))
        self.store.mark_committed(
            ticket, {"content_hash": ticket.content_hash, "exp": payload["exp"]}
        )
        return ticket


def _safe_id(ticket_id: str) -> str:
    """Log the ID only if it is well formed; never log attacker-shaped strings."""
    try:
        return validate_ticket_id(ticket_id)
    except ValueError:
        return "<invalid>"
