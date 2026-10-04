"""Result envelopes, untrusted-text marking and size truncation.

What: every tool result goes back to the model as one JSON envelope with
``tool``, ``call_id``, ``dataset``, ``untrusted``, ``provenance``,
``truncated`` and ``data``. Denials use a separate, smaller envelope.
``fit_to_size`` drops trailing list items until the envelope fits the
character budget.

Why: free text from data sources (maintenance notes) can contain instructions
aimed at the model. The server cannot stop a model from reading them, but it
can label them so the agent and any reviewer know they are data, never
instructions (THREAT_MODEL.md T1). Size limits stop a single result from
flooding the context window (T9).

Connects to: ``policy/gateway.py`` builds all envelopes through here.
"""

from __future__ import annotations

import copy
from typing import Any

from yield_triage.audit import canonical_json


def ok_envelope(
    *,
    tool: str,
    call_id: str,
    dataset: str,
    data: dict[str, Any],
    untrusted: bool,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    # SECURITY: T1 - untrusted is set by the tool definition in our code,
    # never inferred from content and never settable by the caller.
    return {
        "tool": tool,
        "call_id": call_id,
        "decision": "allow",
        "dataset": dataset,
        "untrusted": untrusted,
        "provenance": provenance,
        "truncated": False,
        "data": data,
    }


def deny_envelope(
    *, tool: str, call_id: str, reason: str, fields: list[str] | None = None
) -> dict[str, Any]:
    # SECURITY: T2 - the denial names the reason and, for validation errors,
    # only the offending field names. The rejected value is never echoed back,
    # so a hostile argument cannot be reflected into the model's context.
    env: dict[str, Any] = {"tool": tool, "call_id": call_id, "decision": "deny", "reason": reason}
    if fields:
        env["fields"] = fields
    return env


def fit_to_size(envelope: dict[str, Any], max_chars: int) -> dict[str, Any]:
    """Drop items from the end of the longest list in ``data`` until it fits."""
    # SECURITY: T9 - bounded output regardless of what the analysis returns.
    env = copy.deepcopy(envelope)
    data = env.get("data", {})
    while len(canonical_json(env)) > max_chars:
        lists = [k for k, v in data.items() if isinstance(v, list) and v]
        if not lists:
            return {
                **{k: env[k] for k in ("tool", "call_id", "dataset")},
                "decision": "allow",
                "truncated": True,
                "untrusted": env.get("untrusted", True),
                "data": {"error": "result_too_large"},
            }
        longest = max(lists, key=lambda k: len(canonical_json(data[k])))
        data[longest].pop()
        env["truncated"] = True
    return env
