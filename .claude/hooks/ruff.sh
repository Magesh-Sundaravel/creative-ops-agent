#!/usr/bin/env bash
# After Claude edits a Python file: format it, auto-fix lint, and report what's left.
set -uo pipefail

file=$(jq -r '.tool_input.file_path // empty')

# Only Python files
[[ "$file" == *.py ]] || exit 0

uvx ruff format --quiet "$file"
uvx ruff check --fix --quiet "$file"

# Anything Ruff couldn't fix goes back to Claude so it fixes it
if ! output=$(uvx ruff check "$file" 2>&1); then
  echo "Ruff found issues in $file:" >&2
  echo "$output" >&2
  exit 2
fi

exit 0