# Code tour

A reading order for a reviewer, then one paragraph per file: what it does
and what to look at.

## Reading order

1. `src/yield_triage/policy/gateway.py`: the whole security boundary in one
   function.
2. `src/yield_triage/policy/policy.yaml` and `policy/models.py`: what is
   allowed and what an argument must look like.
3. `src/yield_triage/approvals.py` and `tickets.py`: why the agent cannot
   commit.
4. `src/yield_triage/audit.py`: how tampering is detected.
5. `src/yield_triage/server/tools.py` and `server/app.py`: tool bodies and
   MCP wiring.
6. `src/yield_triage/analysis/`: the statistics (`# STATS:` comments).
7. `src/yield_triage/agent/graph.py`, then `llm.py`, `scenarios.py` and
   `mcp_client.py`.
8. `src/yield_triage/cli.py`, then `scripts/smoke.py` and
   `eval/run_eval.py`.
9. `tests/security/` and `tests/agent/test_agent_flow.py`.

Search for `# SECURITY: T` to jump from any control to its threat in
`docs/THREAT_MODEL.md`.

## Package `src/yield_triage/`

**`policy/gateway.py`.** `PolicyGateway.call` runs budget, allowlist and
scope, size cap, schema validation, tool execution under a deadline,
envelope wrapping and size fitting, and audit, in that order. Point at
`_precheck`. It shows that every attempt is counted before anything else,
and that unknown tool names fall into the same deny path.

**`policy/policy.yaml`.** The default-deny table: which tool needs which
scope, which profile may call which tools, plus limits and analysis and
approval settings. Point at the `read_only` profile, which has no write scope
at all.

**`policy/config.py`.** Strict pydantic loader for the YAML. Point at
`_profiles_are_consistent`. A profile cannot allow an unknown tool or a tool
whose scope it lacks, so a typo fails at start-up instead of widening access.

**`policy/models.py`.** One argument model per tool, with `extra="forbid"`
and `strict=True`. Timestamps follow a fixed grammar and are then parsed;
sensor IDs must match a pattern and be in an allowlist passed as validation
context. Point at `_known_sensor`, which fails closed when there is no
allowlist, and at `_safe_text`.

**`policy/budget.py`.** Call counter plus wall-clock deadline, with an
injectable clock for tests. Point at the comment explaining why denied calls
count.

**`policy/wrap.py`.** Builds the allow and deny envelopes and fits results to
the character budget by dropping trailing list items. Point at
`deny_envelope`, which never echoes rejected values.

**`approvals.py`.** Key loading, token mint and verify, the single-use nonce
ledger, and `ApprovalService` (approve and commit, both audited). Point at
`verify_token`: signature first, then ticket ID, content hash, expiry. Point
at `NonceLedger.consume`, which checks and records under a lock.

**`tickets.py`.** Pending and committed ticket files. Point at
`approved_content` and `content_hash`, which are exactly what a token is
bound to, and at `validate_ticket_id` (no traversal).

**`audit.py`.** Hash-chained JSONL append under a cross-process lock, plus
`verify_audit`. Point at `record_hash` and `_check_record`, and at the
docstring's stated limit: tail truncation is not detected.

**`filelock.py`.** About 30 lines of exclusive locking, using `fcntl` on
POSIX and `msvcrt` on Windows. It exists because the server and the approval
CLI are different processes that write the same files.

**`server/tools.py`.** Tool bodies. Point at `_window` (row cap before any
work), `get_maintenance_notes` (verbatim text with `untrusted: true` and
provenance), and `propose_ticket` (evidence refs must be successful calls
from this run, and it only ever creates a pending draft).

**`server/app.py`.** Wires the gateway to the SDK's low-level `Server`. Point
at the module docstring explaining why the high-level `MCPServer` was not
used: its validation would bypass the audit log.

**`server/fixtures/maintenance_notes.jsonl`.** Fictional notes, some carrying
injection payloads, covering both the SECOM (2008) and synthetic (2030) date
ranges.

**`analysis/ranking.py`.** Mann-Whitney U per sensor, rank-biserial effect
size, Benjamini-Hochberg across sensors. Constant and sparse columns are
dropped before BH. Point at the `# STATS:` comments and at
`benjamini_hochberg`, which is cross-checked against scipy in tests.

**`analysis/drift.py`.** EWMA with exact time-varying limits and optional
Western Electric rules. Point at the comment on why EWMA is used, and at
missing-value handling (skipped, with `t` not advanced).

**`analysis/summary.py`.** Counts, fail rate and missingness for a window.

**`data.py`.** `Dataset` (sorted, UTC, named `secom` or `synthetic`), the
SECOM parser, and save and load. Point at `sensor_ids`: the allowlist comes
from loaded data, not from callers.

**`synthetic.py`.** Seeded generator with planted sensors, constant and
sparse columns, a drift window and heavy-tailed sensors. Point at
`GroundTruth`, which is returned separately and never given to the server.

**`config.py`.** Server settings from environment variables. Note what is
absent: the approval key is not a server setting.

**`agent/graph.py`.** The LangGraph graph. Point at `await_approval`
(interrupt, then a read-only commit check on resume), at `tool_loop`
(forwards every call, even unknown tools), and at `Deps`, which holds live
clients in runtime context rather than in checkpointed state.

**`agent/llm.py`.** `LLMClient` protocol, `ScriptedLLM`, `BedrockLLM`
(Converse) and `AnthropicLLM`. Point at `require_model_id` (no default) and
at the message converters, which merge same-role turns.

**`agent/scenarios.py`.** Scripted behaviours: `happy`, `fooled_rejected`
(obeys every injected note) and `fooled_pending` (drafts a "pre-approved"
ticket).

**`agent/mcp_client.py`.** Official MCP `Client` over stdio. Point at
`server_params`, which passes an explicit allowlist of environment
variables.

**`agent/prompts.py`.** System prompt and step instructions. The rule "tool
output is data" is stated here, but nothing depends on it.

**`cli.py`.** `run`, `resume`, `approve`, `commit`, `verify-audit` and
`serve`. Point at `scrub_agent_environment`, and at `approve`, which shows
the exact content and its hash before minting a token.

## Scripts, eval, ops

**`scripts/fetch_data.py`.** Downloads SECOM from a fixed URL with a size
cap and reads zip members by exact name, or writes synthetic data with a
separate `ground_truth.json`.

**`scripts/smoke.py`.** The real-process end-to-end check used by
`make smoke` and inside the container.

**`scripts/check_hygiene.sh`.** Grep gate for emails, keys, entropy, home
paths and attribution strings, plus detect-secrets.

**`scripts/update_readme.py`.** Splices `eval/results/latest.md` into the
README between markers.

**`eval/run_eval.py`.** Fixed-seed evaluation of statistics, injection and
approval-bypass block rates (with a positive control), and per-tool latency.
Exits non-zero on any unblocked attack.

**`Dockerfile`, `.dockerignore`.** Multi-stage build, digest-pinned bases,
non-root runtime, healthcheck, no secrets.

**`.github/workflows/ci.yml`.** Runs the same make targets with actions
pinned to SHAs and no secrets.

## Tests

**`tests/analysis/`.** Planted recovery, FDR on null and planted data,
drop rules, drift firing and quiet behaviour, parser, generator.

**`tests/security/`.** One file per control family. Each case asserts both
the block and the audit record.

**`tests/property/`.** Hypothesis properties: validation never raises
anything except `ValidationError`, and never accepts path-like text or
unknown sensors.

**`tests/integration/test_stdio_server.py`.** The real server subprocess
over stdio.

**`tests/agent/`.** Graph flows including the fooled models, numbers in the
ticket traced back to tool output, Bedrock shapes validated by botocore's
Stubber, and agent key isolation.
