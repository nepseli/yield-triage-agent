# Decisions

Short entries: the decision, why, and what it costs.

## D1. uv and `uv.lock` for dependency pinning
Exact versions of everything, including transitive dependencies, are pinned
in `uv.lock`. `make install` uses `uv sync --frozen`.
Trade-off: contributors need uv; there is no plain `requirements.txt`.

## D2. Normalised CSV, not Parquet
`fetch_data.py` writes `units.csv` plus `dataset.json`.
Why: no pyarrow dependency, and a reviewer can open the file directly.
Trade-off: slower to load and larger on disk. Neither matters at about 1,600 rows.

## D3. SECOM timestamps treated as UTC
The source gives `dd/mm/yyyy HH:MM:SS` with no timezone. Parsing uses that
exact format and labels the result UTC, so every comparison is between
timezone-aware values.
Trade-off: if the source was local time, absolute times are offset. Ordering
and window arithmetic are unaffected.

## D4. Sort by timestamp; row index = position after sorting
Windows and drift charts need time order. The raw SECOM file is already in
time order (checked), so in practice the row index equals the source line
number. It has 33 duplicate timestamps. The stable sort keeps their source
order, and timestamps are not used as unique keys.
Trade-off: if the source were ever out of order, row indices would no longer
match source line numbers.

## D5. Mann-Whitney U, rank-biserial effect, Benjamini-Hochberg
Rank test because sensor distributions are skewed and heavy-tailed. BH because
about 590 tests at 0.05 would give about 30 false leads by chance.
Cross-checked against `scipy.stats.false_discovery_control`.
Trade-off: Mann-Whitney detects location-type differences and is weak against
pure variance changes. BH assumes independent or positively dependent tests.

## D6. Missing values: available-case, never imputed
Each sensor is tested on its own non-missing units, and `n_pass` and `n_fail`
are reported per sensor.
Trade-off: different sensors use different subsets of units. If missingness
depends on failure, results can be biased. That is a known limitation.

## D7. Drop rules applied before BH
Columns that are constant, or have fewer than 5 non-missing values in either
group, are dropped before BH. An all-missing column counts as constant.
Why: untestable columns should not inflate the number of tests.
Trade-off: the threshold of 5 is a judgement call. It is a parameter.

## D8. EWMA (lambda 0.2, L 3) with exact time-varying limits; WE rules optional
The baseline mean and sample standard deviation come from a caller-chosen,
assumed-clean window, with at least 20 non-missing points. Missing readings
are skipped.
Trade-off: a dirty baseline hides drift. Autocorrelation raises the
false-alarm rate. All four WE rules together alarm about four times as often
as rule 1 alone.

## D9. Synthetic ground truth kept out of the dataset files
`make_synthetic` returns `(Dataset, GroundTruth)` separately. On disk the
truth goes to `ground_truth.json`, which `load_dataset` never reads.
Why: the server and agent must not be able to see the answer.

## D10. Synthetic dates start in 2030
Synthetic data can never be confused with the 2008 SECOM period. The name
`"synthetic"` is carried in `Dataset.name` and will be echoed in every tool
result.

## D11. Minimum-units guard (agreed in Phase 0)
`rank_failing_sensors` returns `status: insufficient_data` when a window has
fewer than `analysis.min_fail_units` failures (default 10). That is a normal
result, not a policy denial.
Trade-off: small windows get no ranking at all instead of a weak one.

## D12. No `make` on the Windows dev host
The Makefile is real and CI runs it on Linux. Locally the same commands are
run through `uv run`.

## D13. Drift baseline = the N units just before the window
`check_sensor_drift(sensor_id, start, end)` uses the
`analysis.drift_baseline_units` units (default 200) immediately before
`start`. It returns `insufficient_data` if fewer than 20 of them are present.
Why: the caller does not need a second window, and "compared with the recent
past" is easy to explain.
Trade-off: if the recent past already drifted, detection weakens (see D8).

## D14. MCP SDK 2.x low-level `Server`, not `MCPServer`
I read the installed source of mcp 2.3.0. In 2.x, `FastMCP` has become
`MCPServer`. `MCPServer` validates arguments from function signatures and
rejects unknown tool names before user code runs, so those denials would
never reach our audit log. The low-level `Server(on_list_tools=...,
on_call_tool=...)` passes every `tools/call` to our gateway unchanged. The
SDK also has `ServerMiddleware`, but one explicit handler is easier to read.
Trade-off: we build tool schemas ourselves (from pydantic models) and don't
use the high-level decorators.

## D15. One server process = one run
The run ID, call budget, wall-clock deadline and the set of citable call IDs
all live in the server process. The agent starts a fresh server per run.
Trade-off: there is no shared multi-run server. That fits a stdio proof of
concept.

## D16. Denied calls count against the call budget
Otherwise a fooled model could retry blocked calls forever.

## D17. Evidence refs must be successful call IDs from this run
`propose_ticket` refuses citations to calls that never happened or were
denied (T12).
Trade-off: a ticket cannot cite evidence from an earlier run.

## D18. Validation errors return field names only
Rejected values are never echoed in the response or the audit log, so hostile
text cannot be reflected back into the model's context.
Trade-off: the model gets less help fixing a genuine mistake.

## D19. Cross-process file lock for the audit log and nonce ledger
The server and the approval CLI are different processes that write the same
files. The lock uses `fcntl.flock` on POSIX and `msvcrt.locking` on Windows,
on a sidecar `.lock` file.
Trade-off: it is advisory only. A process that ignores the lock can still
write.

## D20. Key file permissions enforced on POSIX only
On POSIX, `load_signing_key` refuses a key file readable by group or others.
On Windows, POSIX modes mean little and ACLs are not checked. That gap is
documented, and the Linux container is the verified path.

## D21. The agent observes commit; the CLI performs it
The graph's last node, `await_approval`, uses LangGraph `interrupt` and, on
resume, only checks whether the ticket file is committed. The commit happens
in `yield-triage commit`, which runs as a separate human-run process that
loads the key.
Trade-off: "commit" is not a graph node with side effects. That is the point:
the agent process never holds the key.

## D22. Live clients are injected through LangGraph runtime context
The MCP client and the LLM are passed as `context=Deps(...)`. They are never
stored in the checkpointed state, so the checkpoint holds only JSON (messages,
tool log, ticket ID). Resume needs neither an LLM nor MCP.

## D23. A provider-neutral message format
Plain dicts with roles user, assistant and tool. Bedrock and Anthropic
adapters convert them, merging adjacent same-role turns because both APIs
require alternation. Tool specs are always sent, because both APIs expect
tools to be declared when the history contains tool calls.
Trade-off: no streaming and no provider-specific features.

## D24. Unknown tool calls from the model are forwarded, not filtered
The agent passes every tool call to the server, which decides. Filtering in
the agent would hide the server-side control from the tests.

## D25. A deterministic triage step before any model call
`get_window_summary` runs first. An empty window ends the run with no model
call at all.

## D26. Bedrock adapter validated offline with botocore's Stubber
Stubber checks request and response shapes against the installed service
model, so the Converse mapping is tested without credentials.
Trade-off: Stubber does not prove that a given model accepts the tools. That
is covered only by the live run (see README "verified versus not verified").

## D27. ScriptedLLM scenarios double as eval fixtures
`happy`, `fooled_rejected` and `fooled_pending` are deterministic. The fooled
ones obey the injected notes, so the controls are proven without needing a
real model to be fooled on cue.

## D28. Docker base images pinned by tag and digest
`python:3.13.7-slim-bookworm` and `ghcr.io/astral-sh/uv:0.12.18`, each with
its digest. Dependencies are installed from `uv.lock` with `--frozen`. The
runtime image holds only the virtual environment and `scripts/`, and runs as
a system user `app` with no home directory. The healthcheck imports the
server and validates the policy file, because a stdio server has no port to
probe. The only key-like variable in the image is `GPG_KEY`, which the
official Python image sets to its public signing-key fingerprint. It is not a
secret.

## D29. CI actions pinned to commit SHAs; no secrets
`actions/checkout` v7.0.1 and `astral-sh/setup-uv` v10.2.0 are pinned to
full SHAs, with `permissions: contents: read` and
`persist-credentials: false`. The Docker job uses the runner's own docker,
with no extra actions.

## D30. Hygiene gate: grep patterns plus detect-secrets
The script checks every file that would be committed (tracked plus untracked,
non-ignored), excluding `uv.lock` (hashes) and itself (it spells out the
patterns). Two dummy test values carry `# pragma: allowlist secret`, the
scanner's standard inline exception, so a reviewer can see each one.
