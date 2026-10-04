#!/usr/bin/env bash
# Repository hygiene gate (run by `make hygiene` and in CI).
#
# Fails if any file that would be committed contains: email-like strings,
# long high-entropy tokens, cloud-key or private-key patterns, absolute home
# directory paths, or assistant/tool attribution strings. Also fails if an
# assistant config file or directory exists in the tree. Then runs the
# detect-secrets scanner when uv is available.
#
# Scope: tracked files plus untracked files that are not git-ignored. The lock
# file (full of hashes) and this script (which must spell the patterns out)
# are excluded.
set -euo pipefail
cd "$(dirname "$0")/.."

mapfile -t FILES < <(git ls-files -co --exclude-standard \
  | grep -v -E '^(uv\.lock|scripts/check_hygiene\.sh)$' \
  | while read -r f; do [ -f "$f" ] && echo "$f"; done)

status=0
check() {  # check <label> <grep flags> <pattern>
  local label="$1" flags="$2" pattern="$3"
  local hits
  hits=$(grep -I -n $flags -e "$pattern" -- "${FILES[@]}" 2>/dev/null || true)
  if [ -n "$hits" ]; then
    echo "HYGIENE FAIL [$label]:"
    echo "$hits" | head -20
    status=1
  fi
}

check "email address"      "-E"  '[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z]{2,}'
check "AWS access key id"  "-E"  'AKIA[0-9A-Z]{16}'
check "API key prefix"     "-E"  'sk-[A-Za-z0-9_-]{16,}'
check "private key block"  "-E"  '-----BEGIN [A-Z ]*PRIVATE KEY'
check "macOS home path"    "-E"  '/Users/[A-Za-z]'
check "Linux home path"    "-E"  '/home/[A-Za-z]'
check "Windows home path"  "-Ei" '[A-Z]:[\\/]+Users[\\/]'
check "attribution"        "-i"  'Co-Authored-By'
check "attribution"        "-i"  'Generated with'
check "assistant config"   "-i"  'CLAUDE\.md'
check "assistant config"   "-i"  '\.claude\b'

# High-entropy tokens: 40+ chars of base64/hex-ish text mixing upper, lower and digits.
# grep prints "file:line:token"; the character-class tests apply to the token only.
entropy=$(grep -I -n -o -E '[A-Za-z0-9+/_=-]{40,}' -- "${FILES[@]}" 2>/dev/null \
  | awk '{ t = $0; sub(/^[^:]*:[0-9]+:/, "", t); if (t ~ /[A-Z]/ && t ~ /[a-z]/ && t ~ /[0-9]/) print }' \
  || true)
if [ -n "$entropy" ]; then
  echo "HYGIENE FAIL [high-entropy token]:"
  echo "$entropy" | head -20
  status=1
fi

# Assistant/tool config files must not exist in the repository tree.
if git ls-files -co --exclude-standard | grep -i -E '(^|/)(CLAUDE\.md|\.claude/)' ; then
  echo "HYGIENE FAIL [assistant config file present]"
  status=1
fi

# Secret scanner (optional dependency: needs uv on PATH).
if command -v uvx >/dev/null 2>&1; then
  report=$(uvx --quiet --from detect-secrets==1.5.0 detect-secrets scan "${FILES[@]}")
  found=$(printf '%s' "$report" | python -c 'import json,sys; r=json.load(sys.stdin)["results"]; print(sum(len(v) for v in r.values()))')
  if [ "$found" != "0" ]; then
    echo "HYGIENE FAIL [detect-secrets found $found candidate(s)]:"
    printf '%s' "$report" | python -c 'import json,sys; [print(f, s["type"], s["line_number"]) for f,v in json.load(sys.stdin)["results"].items() for s in v]'
    status=1
  else
    echo "detect-secrets: 0 findings in ${#FILES[@]} files"
  fi
else
  echo "detect-secrets skipped (uvx not found)"
fi

if [ "$status" -eq 0 ]; then
  echo "hygiene: OK (${#FILES[@]} files checked)"
fi
exit "$status"
