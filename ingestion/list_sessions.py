"""
List the F1 sessions already ingested into Supabase, with per-category row
counts.

This is the discovery step the (future) webpage needs - "which sessions
can the user pick from?" - and, day to day, the quickest way to see what's
stored and spot a partial ingestion (a category sitting at 0 rows).

    python ingestion/list_sessions.py
    python ingestion/list_sessions.py --year 2024

Reads only. Uses the same SUPABASE_URL / SUPABASE_KEY as the ingestion
pipeline (see supabase_store.py).
"""

import argparse
import sys

import supabase_store

# A session is "complete" when it has a non-zero row count for every one of
# these. FastF1 legitimately can't provide some categories for some
# sessions (e.g. no telemetry for very old seasons), so a partial session
# isn't necessarily broken - it's just flagged for a look.
_EXPECTED_CATEGORIES = len(supabase_store.CATEGORY_TABLES)


def _present_categories(session: dict) -> list[str]:
    counts = session["counts"]
    return [table for table in supabase_store.CATEGORY_TABLES if counts.get(table)]


def _format_session(session: dict) -> str:
    present = _present_categories(session)
    counts = session["counts"]
    ingested = str(session.get("ingested_at", ""))[:10]

    header = (
        f"[{session['id']:>3}] {session['season_year']} "
        f"R{session['round_number']:02d} {session['event_name']} - "
        f"{session['session_label']:<2} | ingested {ingested} | "
        f"{len(present)}/{_EXPECTED_CATEGORIES} categories"
    )
    if len(present) < _EXPECTED_CATEGORIES:
        header += " | PARTIAL"

    if present:
        detail = "      " + "  ".join(f"{table} {counts[table]}" for table in present)
    else:
        detail = "      (no category rows)"
    return f"{header}\n{detail}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List F1 sessions already ingested into Supabase."
    )
    parser.add_argument(
        "--year", type=int, default=None,
        help="Only show sessions from this season (4-digit year).",
    )
    args = parser.parse_args()

    try:
        sessions = supabase_store.list_sessions(year=args.year)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if not sessions:
        scope = "" if args.year is None else f" for {args.year}"
        print(f"No ingested sessions found{scope}.")
        return

    partial = 0
    for session in sessions:
        print(_format_session(session))
        if len(_present_categories(session)) < _EXPECTED_CATEGORIES:
            partial += 1

    summary = f"\n{len(sessions)} session(s)"
    if partial:
        summary += f", {partial} partial"
    print(summary)


if __name__ == "__main__":
    main()
