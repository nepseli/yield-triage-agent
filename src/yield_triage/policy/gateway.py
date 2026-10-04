"""The policy gateway: the single path from an MCP tool call to a tool.

What: ``PolicyGateway.call(name, raw_args)`` runs, in order:
  1. budget (call count, wall clock)        T9
  2. default-deny allowlist and scope       T13, T1b
  3. argument size cap                      T2, T9
  4. strict schema validation               T2
  5. the tool, under the remaining time     T9
  6. envelope, untrusted marking, size fit  T1, T9
  7. audit record, for allow and deny       T10, T11
and returns a JSON envelope. Nothing here asks the model anything or depends
on what the model believes.

Why: putting every control in one deterministic function means a reviewer
can check the whole security boundary in one file, and the tests can prove
each control by calling it directly or over MCP.

Connects to: ``server/app.py`` forwards every ``tools/call`` here (the
low-level MCP server is used precisely so unknown tools and malformed
arguments also arrive here and get audited). ``server/tools.py`` holds the
tool bodies.
"""

from __future__ import annotations

import re
from typing import Any

import anyio
import anyio.to_thread
from pydantic import ValidationError

from yield_triage.audit import AuditLog, canonical_json, sha256_hex
from yield_triage.policy.budget import RunBudget
from yield_triage.policy.config import Policy
from yield_triage.policy.models import TOOL_ARG_MODELS
from yield_triage.policy.wrap import deny_envelope, fit_to_size, ok_envelope
from yield_triage.server.tools import HANDLERS, ToolContext, ToolDenied

_TOOL_NAME_RE = re.compile(r"^[a-z_]{1,40}$")


class PolicyGateway:
    def __init__(
        self, *, policy: Policy, profile: str, ctx: ToolContext, audit: AuditLog, budget: RunBudget
    ) -> None:
        if profile not in policy.profiles:
            raise ValueError(f"unknown policy profile {profile!r}")
        self.policy = policy
        self.profile = policy.profiles[profile]
        self.ctx = ctx
        self.audit = audit
        self.budget = budget
        self._seq = 0

    def allowed_tools(self) -> list[str]:
        return list(self.profile.allowed_tools)

    async def call(self, name: str, raw_args: Any) -> dict[str, Any]:
        self._seq += 1
        call_id = f"c{self._seq:04d}"
        # Attacker-chosen tool names are logged only if they look like a name.
        tool = name if isinstance(name, str) and _TOOL_NAME_RE.fullmatch(name) else "<invalid>"
        scope = self.policy.tools[tool].scope if tool in self.policy.tools else None
        audit = {
            "tool": tool,
            "call_id": call_id,
            "scope": scope,
            "args_sha256": _args_hash(raw_args),
        }

        reason, fields = self._precheck(tool, scope, raw_args)
        if reason is None:
            try:
                args = TOOL_ARG_MODELS[tool].model_validate(
                    raw_args, context={"sensor_ids": self.ctx.dataset.sensor_ids}
                )
            except ValidationError as err:
                reason = "invalid_arguments"
                fields = sorted(
                    {".".join(str(p) for p in e["loc"]) or "<root>" for e in err.errors()}
                )
        if reason is not None:
            return self._deny(audit, reason, fields)
        return await self._run(tool, args, audit)

    def _precheck(
        self, tool: str, scope: str | None, raw_args: Any
    ) -> tuple[str | None, list[str] | None]:
        # SECURITY: T9 - budget first, so every attempt is counted.
        budget_reason = self.budget.try_spend()
        if budget_reason:
            return budget_reason, None
        # SECURITY: T13, T1b - default deny. A tool must exist, be listed for
        # this run's profile, and its scope must be granted to the profile.
        if tool not in self.profile.allowed_tools or tool not in HANDLERS:
            return "tool_not_allowed", None
        if scope not in self.profile.scopes:
            return "missing_scope", None
        # SECURITY: T2, T9 - bound argument size before parsing anything.
        if not isinstance(raw_args, dict):
            return "invalid_arguments", ["<root>"]
        if len(canonical_json(raw_args)) > self.policy.limits.max_arg_chars:
            return "arguments_too_large", None
        return None, None

    async def _run(self, tool: str, args: Any, audit: dict[str, Any]) -> dict[str, Any]:
        handler = HANDLERS[tool]
        try:
            # SECURITY: T9 - the tool's work is bounded by what is left of the
            # run's wall-clock budget. A timed-out worker thread is abandoned
            # and its result discarded.
            with anyio.fail_after(max(self.budget.remaining_s(), 0.001)):
                out = await anyio.to_thread.run_sync(
                    handler, self.ctx, args, abandon_on_cancel=True
                )
        except TimeoutError:
            return self._deny(audit, "budget_time_exceeded", None)
        except ToolDenied as err:
            return self._deny(audit, err.reason, None)
        env = ok_envelope(
            tool=tool,
            call_id=audit["call_id"],
            dataset=self.ctx.dataset.name,
            data=out.data,
            untrusted=out.untrusted,
            provenance=out.provenance,
        )
        env = fit_to_size(env, self.policy.limits.max_result_chars)
        self.ctx.successful_call_ids.add(audit["call_id"])
        self.audit.append(
            actor="mcp_server",
            event="tool_call",
            run_id=self.ctx.run_id,
            decision="allow",
            reason=None,
            rows_scanned=out.rows_scanned,
            result_bytes=len(canonical_json(env)),
            truncated=env.get("truncated", False),
            **audit,
        )
        return env

    def _deny(self, audit: dict[str, Any], reason: str, fields: list[str] | None) -> dict[str, Any]:
        env = deny_envelope(
            tool=audit["tool"], call_id=audit["call_id"], reason=reason, fields=fields
        )
        self.audit.append(
            actor="mcp_server",
            event="tool_call",
            run_id=self.ctx.run_id,
            decision="deny",
            reason=reason,
            rows_scanned=0,
            result_bytes=len(canonical_json(env)),
            **audit,
        )
        return env


def _args_hash(raw_args: Any) -> str:
    # SECURITY: T11 - only a hash of the arguments is logged.
    try:
        return sha256_hex(canonical_json(raw_args))
    except (TypeError, ValueError):
        return sha256_hex(repr(raw_args))
