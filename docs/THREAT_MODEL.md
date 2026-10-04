# Threat model

Code references these IDs with `# SECURITY: T<n>` comments. The "proving
test" column names real tests. Status shows which phase delivered each test.
Every test listed here runs offline in `pytest`.

## Assets and trust boundaries

- **Assets:** integrity of tickets, approval authority, the audit log, the
  dataset, availability of the server.
- **Untrusted:** model output, free text in tool results (maintenance notes),
  and every argument the model sends.
- **Trusted:** policy code and `policy.yaml`, the approval CLI run by a human,
  and whoever holds the signing key.
- **Out of scope:** an attacker with shell access as the same OS user,
  compromise of the host or of the Python supply chain, side channels.

## Threats, controls, proofs

OWASP LLM Top 10 (2025) mapping: LLM01 prompt injection, LLM02 sensitive
information disclosure, LLM05 improper output handling, LLM06 excessive
agency, LLM09 misinformation, LLM10 unbounded consumption.

| ID | Threat | OWASP | Control (code) | Proving test |
|---|---|---|---|---|
| T1 | Indirect prompt injection in `get_maintenance_notes` text | LLM01 | Text is returned verbatim with `untrusted: true` and provenance (`server/tools.py`, `policy/wrap.py`). The server never interprets data. No security property depends on the model | `tests/security/test_injection.py::test_notes_are_returned_verbatim_and_marked_untrusted`, `::test_reading_injected_notes_causes_no_side_effects`, `::test_injection_blocked_over_real_mcp_protocol` |
| T1b | The model is fooled and obeys the injection | LLM01, LLM06 | Default-deny allowlist, scope check, evidence check, schema validation, call budget (`policy/gateway.py`) | `test_injection.py::test_fooled_model_actions_are_all_blocked_and_audited`, `::test_fooled_model_repeating_calls_hits_budget`. Full-agent ScriptedLLM version: Phase 3 |
| T2 | Malicious arguments: path traversal, unknown sensors, oversized input, malformed timestamps, extra keys, coerced types | LLM05 | Strict pydantic models (`policy/models.py`), sensor allowlist taken from the dataset, size cap before parsing. Denials name fields only | `tests/security/test_arguments.py` (23 parametrised cases plus 3 tests), `tests/property/test_validation_properties.py` (Hypothesis) |
| T3 | Write without an approval token | LLM06 | `propose_ticket` only creates a pending draft. Commit is not an MCP tool and needs a token (`approvals.py`) | `tests/security/test_approvals.py::test_commit_without_token`, `::test_propose_alone_never_commits` |
| T4 | Token replay | LLM06 | Single-use nonce ledger, checked and consumed under an exclusive lock | `test_approvals.py::test_token_replay` |
| T5 | Token reused for a different ticket, or after the ticket is edited | LLM06 | Token bound to the ticket ID and to the SHA-256 of the canonical content | `test_approvals.py::test_token_for_a_different_ticket`, `::test_token_after_ticket_is_edited` |
| T6 | Expired token | LLM06 | `exp` claim (default 900 s) checked against an injectable clock | `test_approvals.py::test_expired_token` |
| T7 | Forged, tampered or malformed token | LLM06 | HMAC-SHA256 with constant-time compare; signature checked before the payload is trusted | `test_approvals.py::test_token_from_another_key_is_rejected`, `::test_tampered_payload_fails_signature`, `::test_malformed_or_injected_tokens` |
| T8 | Agent or server obtains the signing key | LLM06, LLM02 | Only the approval code loads the key. The MCP stdio client passes the server a fixed safe set of environment variables. The key file is created 0600 and refused if group/other-readable (POSIX). The agent environment is scrubbed (Phase 3) | `test_approvals.py::test_mcp_stdio_client_does_not_pass_key_to_server`, `::test_server_code_never_loads_the_signing_key`, `::test_key_file_is_generated_owner_only`, `::test_group_readable_key_file_refused` (POSIX only) |
| T9 | Unbounded consumption: too many calls, huge scans, huge outputs, hung runs | LLM10 | Call budget that counts denials too, wall-clock deadline that also bounds each tool's work, row cap, result truncation, `top_k <= 50` | `tests/security/test_limits.py` (all 6 tests) |
| T10 | Audit tampering: edit, delete, reorder, garbage | n/a | Hash chain plus `verify_audit` (`audit.py`) | `tests/security/test_audit_and_policy.py::test_edited_record_detected`, `::test_rehashed_edit_still_breaks_the_link`, `::test_deleted_record_detected`, `::test_reordered_records_detected`, `::test_garbage_line_detected` |
| T11 | Secrets leaking into logs | LLM02 | The audit log stores `args_sha256`, never raw arguments | `test_audit_and_policy.py::test_audit_never_contains_raw_argument_values`, plus the no-reflection checks in `test_arguments.py` |
| T12 | Fabricated evidence in a ticket | LLM09 | `evidence_refs` must be successful call IDs from this run | `test_injection.py::test_fooled_model_actions_are_all_blocked_and_audited` (the `c9999` case) |
| T13 | Unknown or hidden tools; policy misconfiguration | LLM06 | Default-deny per profile. The policy file is validated strictly (unknown tool, missing scope, unknown key) | `test_injection.py::test_read_only_profile_cannot_propose`, `test_audit_and_policy.py::test_profile_*`, `::test_unknown_policy_key_rejected` |

## Residual risks (stated, not hidden)

- **Tail truncation:** a hash chain cannot detect truncation of the tail or a
  full-file rewrite by someone with write access. A test pins this known
  limitation: `test_known_limitation_tail_truncation_not_detected`.
- **Misleading narrative:** injected text can still steer the model toward a
  poor summary. The controls limit the impact (nothing is written without a
  human) but not the quality of the narrative, so the approver must read the
  ticket and its cited evidence.
- **Windows permissions:** on Windows the key file's permissions are not
  enforced. The Linux container is the verified path.
- **Same-user attacker:** an attacker running as the same OS user can read the
  key file.
- **Advisory locks:** the file locks are advisory, so a process that ignores
  them can still write.
- **Abandoned workers:** a timed-out tool thread is abandoned, not killed. It
  may finish in the background, but its result is discarded.
