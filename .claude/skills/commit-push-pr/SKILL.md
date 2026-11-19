---
name: commit-push-pr
description: Commit, push, and open a PR with a body in the pr skill's shape
disable-model-invocation: true
allowed-tools: Bash(git checkout -b:*), Bash(git add:*), Bash(git status:*), Bash(git push:*), Bash(git commit:*), Bash(gh pr create:*), Skill
---

## Context

- Current git status: !`git status`
- Current git diff (staged and unstaged changes): !`git diff HEAD`
- Current branch: !`git branch --show-current`

## Your task

Based on the above changes:

1. If on `main`, create a new branch with `git checkout -b <type>/<short-name>`.
2. Stage the changes. Do not stage `.env` or other secret files; if you see one, stop and tell me.
3. Create one commit using Conventional Commits:
   - Header: `<type>(<scope>): <summary>`, max 72 characters in total
   - Type, pick the one that matches the main change:
     - `feat`: new behaviour
     - `fix`: bug fix
     - `refactor`: code change, same behaviour
     - `test`: tests only
     - `docs`: documentation only
     - `chore`: setup, config, dependencies
     - `ci`: CI pipeline
   - Scope: one of `intake`, `review`, `handoff`, `coordinator`, `loop`, `mcp`, `api`,
     `ui`, `evals`, `claude`. Leave it out if the change spans several areas.
   - Summary: imperative ("add", not "added"), lowercase, no trailing period.
     Describe the outcome, not the files touched.
   - Body (optional): 1–3 lines on what changed and why. Skip it if the header says it all.
   - Breaking change: add `!` after the scope (`feat(api)!: ...`) and explain it in the body.
   - Ticket: if the branch name or the diff refers to an issue number, end with `Closes #<n>`.
   - Use a heredoc for multi-line messages so the formatting is kept.
4. Push the branch with `git push -u origin <branch>`.
5. Load the `pr` skill and write the PR title and body following its shape.
6. Open the PR with `gh pr create --title "<title>" --body "<body>"`.
7. Reply with only the PR URL.

Do not run any other commands.
