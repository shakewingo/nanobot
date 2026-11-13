"""Aggregate-only statistics on a real `memory/history.jsonl`.

Prints counts and lengths only: no entry content, fact text, or session id
leaves this script, so the output is safe to paste into a public issue.

Measures how much of each entry `MemoryStore.build_dream_prompt` can see: it
keeps only the first 1000 characters of every entry, and Dream advances its
cursor past the whole batch afterwards.

Standard library only. Usage (run on a copy so a live gateway cannot rewrite
the file mid-read; adjust the path if your workspace is elsewhere):
    SNAP=$(mktemp -d)
    cp ~/.nanobot/workspace/memory/{history.jsonl,.cursor,.dream_cursor} "$SNAP"/
    python3 -I scripts/dream_history_audit.py "$SNAP"

Overlap compares exact fact lines only, so it does not detect paraphrased
repetition.
"""

import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Mirrors truncate_text(e['content'], 1000) in MemoryStore.build_dream_prompt.
DREAM_ENTRY_CHARS = 1000
FACT_RE = re.compile(r"^\s*-\s*\[(\w+)\]\s*\S")


def group_of(session_key: str | None) -> str:
    if session_key == "heartbeat":
        return "heartbeat"
    if session_key and session_key.startswith("cron:"):
        return "cron"
    return "conversation"


def fact_lines(content: str) -> list[tuple[str, str, int]]:
    """Return (normalized line, mark, end offset) for each `- [mark] fact` line."""
    out, offset = [], 0
    for raw in content.split("\n"):
        end = offset + len(raw)
        m = FACT_RE.match(raw)
        if m:
            out.append((raw.strip(), m.group(1).lower(), end))
        offset = end + 1
    return out


def read_cursor(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return "missing"


def main(memory_dir: Path) -> None:
    entries = []
    for line in (memory_dir / "history.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record.get("content"), str):
            entries.append(record)

    dream_cursor = read_cursor(memory_dir / ".dream_cursor")
    print(f"entries: {len(entries)}")
    print(f"write cursor / Dream cursor: {read_cursor(memory_dir / '.cursor')} / {dream_cursor}")
    if dream_cursor.isdigit():
        pending = sum(1 for e in entries if e.get("cursor", 0) > int(dream_cursor))
        print(f"entries not yet consumed by Dream: {pending}")

    by_group: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        by_group[group_of(e.get("session_key"))].append(e)

    for group, items in sorted(by_group.items()):
        lengths = [len(e["content"]) for e in items]
        sessions: dict[str, list[dict]] = defaultdict(list)
        for e in items:
            sessions[e.get("session_key") or "(none)"].append(e)

        # Share of an entry's fact lines already present verbatim in the previous
        # entry of the same session; pairs where either side has no fact lines
        # (e.g. a [RAW] fallback) are skipped.
        overlaps = []
        for session_entries in sessions.values():
            session_entries.sort(key=lambda e: e.get("cursor", 0))
            for prev, cur in zip(session_entries, session_entries[1:]):
                cur_lines = {line for line, _, _ in fact_lines(cur["content"])}
                prev_lines = {line for line, _, _ in fact_lines(prev["content"])}
                if cur_lines and prev_lines:
                    overlaps.append(len(cur_lines & prev_lines) / len(cur_lines))

        mark_of: dict[str, str] = {}
        visible: set[str] = set()
        for e in items:
            for line, mark, end in fact_lines(e["content"]):
                mark_of.setdefault(line, mark)
                if end <= DREAM_ENTRY_CHARS:
                    visible.add(line)
        hidden = [line for line in mark_of if line not in visible]
        hidden_marks = Counter(mark_of[line] for line in hidden)
        over = sum(1 for n in lengths if n > DREAM_ENTRY_CHARS)
        stamps = sorted(e.get("timestamp", "") for e in items)

        print(f"\n== {group}")
        print(f"time range: {stamps[0]} .. {stamps[-1]}")
        print(f"entries / sessions: {len(items)} / {len(sessions)}")
        print(f"entries per session: {sorted((len(s) for s in sessions.values()), reverse=True)}")
        raw = sum(1 for e in items if e["content"].startswith("[RAW]"))
        print(f"[RAW] fallback entries (no fact lines): {raw}")
        print(f"entries > {DREAM_ENTRY_CHARS} chars: {over} ({over / len(items):.0%})")
        print(f"longest entry: {max(lengths)} chars; median: {int(statistics.median(lengths))}")
        cut = sum(max(0, n - DREAM_ENTRY_CHARS) for n in lengths)
        print(f"characters past the first {DREAM_ENTRY_CHARS} of their entry: {cut / sum(lengths):.1%}")
        if overlaps:
            print(f"adjacent same-session verbatim fact-line overlap: median "
                  f"{statistics.median(overlaps):.2f} over {len(overlaps)} pairs")
        if mark_of:
            print(f"distinct fact lines: {len(mark_of)}")
            print(f"never inside the first {DREAM_ENTRY_CHARS} chars of any entry: {len(hidden)} "
                  f"({len(hidden) / len(mark_of):.1%}) {dict(hidden_marks.most_common())}")


if __name__ == "__main__":
    main(Path(sys.argv[1]).expanduser())
