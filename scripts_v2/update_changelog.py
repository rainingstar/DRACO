"""scripts_v2/update_changelog.py — append a dated entry to CHANGELOG.md.

Convention: each entry is grouped under a date heading (## YYYY-MM-DD).
Within a date, sub-entries are time-stamped (### HH:MM — type(scope): headline).

Usage:
    # Single-line entry (auto time, auto type detect from headline)
    python3 scripts_v2/update_changelog.py "feat(train): add foo flag"

    # With details bullets
    python3 scripts_v2/update_changelog.py \\
        "bugfix(eval): fix foo" --details "Symptom: ..." "Root cause: ..." "Fix: ..."

    # With auto-attached changed-files list (uses `git status --porcelain`)
    python3 scripts_v2/update_changelog.py "refactor(scripts_v2): X" --git-status

    # Custom location (default: <repo_root>/CHANGELOG.md, found by walking up)
    python3 scripts_v2/update_changelog.py "..." --path /custom/CHANGELOG.md

The script:
- Inserts the new entry at the TOP of the file (newest first).
- If today's date heading already exists, inserts under it.
- If not, creates a new date heading.
- Time is local time (server CST).
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime


def find_changelog(start_dir):
    """Walk up from start_dir looking for CHANGELOG.md."""
    cur = os.path.abspath(start_dir)
    while True:
        candidate = os.path.join(cur, "CHANGELOG.md")
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent


def git_status_porcelain(repo_dir):
    """Return list of (status_code, path) tuples from `git status --porcelain`."""
    try:
        out = subprocess.check_output(
            ["git", "-C", repo_dir, "status", "--porcelain"],
            stderr=subprocess.DEVNULL, text=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    rows = []
    for line in out.splitlines():
        if len(line) < 4:
            continue
        rows.append((line[:2].strip(), line[3:]))
    return rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("headline", help="Commit-style headline, e.g. 'feat(train): add X'")
    p.add_argument("--details", nargs="*", default=[],
                   help="Bullet points under the headline.")
    p.add_argument("--git-status", action="store_true",
                   help="Auto-append changed-files list from `git status --porcelain`.")
    p.add_argument("--path", default=None,
                   help="Path to CHANGELOG.md. Default: walk up from cwd.")
    p.add_argument("--repo", default=None,
                   help="Repo root for git status (default: dir of CHANGELOG.md).")
    p.add_argument("--time", default=None,
                   help="Override time HH:MM (default: now).")
    p.add_argument("--date", default=None,
                   help="Override date YYYY-MM-DD (default: today).")
    return p.parse_args()


def main():
    args = parse_args()

    path = args.path or find_changelog(os.getcwd())
    if not path:
        print("CHANGELOG.md not found. Use --path to specify.", file=sys.stderr)
        return 1

    repo = args.repo or os.path.dirname(os.path.abspath(path))
    now = datetime.now()
    date_str = args.date or now.strftime("%Y-%m-%d")
    time_str = args.time or now.strftime("%H:%M")

    # Build new entry
    lines = [f"### {time_str} — {args.headline}", ""]
    for detail in args.details:
        lines.append(f"- {detail}")
    if args.git_status:
        rows = git_status_porcelain(repo)
        if rows:
            lines.append("- Changed files (`git status --porcelain`):")
            for code, p in rows:
                lines.append(f"  - `{code or '·'}` `{p}`")
    lines.append("")  # blank line after entry
    new_entry = "\n".join(lines)

    # Read current CHANGELOG
    with open(path, encoding="utf-8") as f:
        content = f.read()

    # Find insertion point.
    # Strategy: find the first line starting with "## " (date heading) and insert
    # the new entry right above it OR right after if same date.
    today_heading = f"## {date_str}"
    # Quick scan
    body_idx = None
    for marker in (today_heading, "\n## "):
        idx = content.find(marker)
        if idx != -1:
            body_idx = idx
            break

    if today_heading in content:
        # Insert right after today's heading
        # Find end of today heading line
        idx = content.find(today_heading)
        line_end = content.find("\n", idx)
        if line_end == -1:
            line_end = len(content)
        # Skip blank line(s) right after the heading
        insert_at = line_end + 1
        while insert_at < len(content) and content[insert_at] == "\n":
            insert_at += 1
        new_content = content[:insert_at] + new_entry + "\n" + content[insert_at:]
    else:
        # No today heading; create one. Insert before first existing date heading,
        # or at top if no date heading exists.
        # Find first "## YYYY" heading
        import re
        m = re.search(r"^## \d{4}-\d{2}-\d{2}", content, flags=re.MULTILINE)
        if m:
            insert_at = m.start()
            block = f"## {date_str}\n\n{new_entry}\n"
            new_content = content[:insert_at] + block + content[insert_at:]
        else:
            # No date headings at all; append after preamble.
            # Heuristic: append at end.
            new_content = content.rstrip() + f"\n\n## {date_str}\n\n{new_entry}\n"

    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"[changelog] appended {time_str} entry to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
