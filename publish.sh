#!/usr/bin/env bash
# Refresh the results page. Generated artifacts (results.js, sample PNGs) live on the
# gh-pages branch, not main — a run produces hundreds of them and they would otherwise
# bury the code history. main keeps docs/index.html as the source.
#
# Usage: ./publish.sh            refresh once
#        ./publish.sh --loop 20  refresh every 20 minutes until killed
set -euo pipefail
cd "$(dirname "$0")"

WORKTREE=.gh-pages

publish_once() {
  .venv/bin/python export_results.py
  [ -d "$WORKTREE" ] || git worktree add -B gh-pages "$WORKTREE" origin/gh-pages 2>/dev/null \
    || git worktree add --orphan -B gh-pages "$WORKTREE"
  rsync -a --delete --exclude .git docs/ "$WORKTREE"/
  git -C "$WORKTREE" add -A
  git -C "$WORKTREE" diff --cached --quiet && { echo "no change"; return; }
  git -C "$WORKTREE" commit -qm "results @ $(grep -c . runs/base/metrics.jsonl) log points"
  git -C "$WORKTREE" push -q origin gh-pages
  echo "published"
}

if [ "${1-}" = "--loop" ]; then
  while true; do publish_once || echo "publish failed, retrying next cycle"; sleep $(( ${2:-20} * 60 )); done
else
  publish_once
fi
