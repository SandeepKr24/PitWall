"""
Supabase persistence layer for ingested FastF1 session data.

This is also the ingestion pipeline's dedup log: before fetching anything
from FastF1, `find_session()` checks whether a row already exists in the
`sessions` table (see schema.sql) for that (year, round, session_label).
If one does, the data is already durably stored in Supabase and FastF1 is
never hit for the expensive part (loading laps/telemetry/etc.) again -
the `sessions` table itself is the log, so there's no separate log file
that could drift out of sync with what's actually in the database.

Requires two environment variables (e.g. from a .env file, already
covered by .gitignore):
    SUPABASE_URL - the project's API URL
    SUPABASE_KEY - the project's service_role key (NOT the anon key -
        this runs from a trusted backend process and needs to bypass
        Row Level Security to insert data; the anon key will be rejected
        by RLS unless you've added policies permitting these inserts)
"""

import datetime
import os
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()

# Inserting thousands of telemetry rows in a single request risks hitting
# request size/timeout limits, so large frames are sent in chunks instead.
INSERT_CHUNK_SIZE = 1000

# The per-session data tables (everything except `sessions` itself), in the
# order they're exported. `session_info` is a single JSONB row; the rest
# are one-row-per-datapoint. Used by list_sessions() to report coverage.
CATEGORY_TABLES = (
    "results", "laps", "weather", "car_data", "pos_data",
    "track_status", "session_status", "race_control_messages", "session_info",
)

_client: Optional[Client] = None


class SupabaseStoreError(RuntimeError):
    """
    Raised when a session's data could not be durably stored - in
    particular a *partial* write, where the `sessions` row was created but
    one or more categories that FastF1 did return failed to upload.

    Subclasses RuntimeError so the ingestion CLI's existing
    `except RuntimeError` handler surfaces it cleanly (same as the missing-
    env-var error from get_client()).
    """


def get_client() -> Client:
    global _client
    if _client is None:
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        if not url or not key:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_KEY must be set (e.g. in a .env "
                "file) to store or look up ingested session data."
            )
        _client = create_client(url, key)
    return _client


def find_session(year: int, round_number: int, session_label: str) -> Optional[dict]:
    """Return the existing `sessions` row for this (year, round, session), or None."""
    response = (
        get_client()
        .table("sessions")
        .select("id")
        .eq("season_year", _json_safe(year))
        .eq("round_number", _json_safe(round_number))
        .eq("session_label", _json_safe(session_label))
        .limit(1)
        .execute()
    )
    return response.data[0] if response.data else None


def delete_session(session_id: int) -> None:
    """
    Delete a `sessions` row and, via schema.sql's ON DELETE CASCADE, every
    category row that belongs to it. Used to re-ingest a session (--force)
    and to roll back a partial write.
    """
    get_client().table("sessions").delete().eq("id", session_id).execute()


def list_sessions(year: Optional[int] = None) -> list[dict]:
    """
    Return every ingested session (optionally filtered to one season),
    each as its `sessions` row plus a `counts` dict mapping each category
    table to its row count for that session. Ordered by year, round, then
    session label.

    This is the discovery step the future webpage needs ("which sessions
    can I ask about?") and the quickest way to spot a partial ingestion
    (some category at 0).
    """
    client = get_client()
    query = (
        client.table("sessions")
        .select("*")
        .order("season_year")
        .order("round_number")
        .order("session_label")
    )
    if year is not None:
        query = query.eq("season_year", _json_safe(year))
    sessions = query.execute().data

    for session in sessions:
        counts: dict[str, int] = {}
        for table in CATEGORY_TABLES:
            result = (
                client.table(table)
                .select("session_id", count="exact")
                .eq("session_id", session["id"])
                .limit(1)
                .execute()
            )
            counts[table] = result.count or 0
        session["counts"] = counts

    return sessions


def _json_safe(value):
    """
    Make a pandas/numpy/raw-Python value safe to send through Supabase's
    REST API as JSON. pandas Timedelta/Timestamp, plain datetime/timedelta
    (e.g. FastF1's session_info, which holds native datetime objects rather
    than pandas ones), numpy int/float/bool scalars, and NaN/NaT are not
    JSON-serializable as-is: NaN/NaT become SQL NULL, Timedelta/Timestamp/
    datetime become strings that Postgres parses back into INTERVAL/
    TIMESTAMP, and numpy scalars are unwrapped to native types. dicts and
    lists are walked recursively, since session_info nests nested dicts
    (e.g. "Meeting") that can themselves hold such values.

    A whole-number float (e.g. 2.0) is returned as a Python int rather than
    a float: Postgres's PostgREST insert path rejects "2.0" as input for an
    INTEGER column (columns like race_control_messages.Sector are float64
    in pandas only because a NaN elsewhere in the column forced the
    upcast), while an INTEGER value is always accepted for a
    DOUBLE PRECISION column, so this is safe for both. This has to check
    plain `float`/`int`, not just `np.floating`/`np.integer`: DataFrame.
    to_dict() already unwraps numpy scalars to native Python types, so by
    the time a value reaches here it's rarely still a numpy type.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, datetime.datetime)):
        return value.isoformat()
    if isinstance(value, (pd.Timedelta, datetime.timedelta)):
        return str(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        as_float = float(value)
        return int(as_float) if as_float.is_integer() else as_float
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _records_for_insert(frame: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to a list of JSON-safe dicts, one per row."""
    return [
        {column: _json_safe(value) for column, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _insert_in_chunks(table_name: str, records: list[dict]) -> None:
    client = get_client()
    for i in range(0, len(records), INSERT_CHUNK_SIZE):
        client.table(table_name).insert(records[i:i + INSERT_CHUNK_SIZE]).execute()


def _rollback_session(session_id: int, failed_on: str) -> None:
    """
    Delete a partially-written session so a re-run can retry it cleanly.

    find_session() treats the mere existence of a `sessions` row as "this
    session is fully ingested" and skips the FastF1 fetch on that basis, so
    a session whose data only partly uploaded must not be left behind - it
    would be skipped forever. Deleting the `sessions` row cascades to every
    child table (see schema.sql's ON DELETE CASCADE).
    """
    try:
        delete_session(session_id)
        print(f"Rolled back session_id={session_id} after '{failed_on}' failed to store.")
    except Exception as exc:
        print(
            f"WARNING: '{failed_on}' failed to store AND the partial "
            f"session_id={session_id} could not be rolled back ({exc}). "
            f"Delete that `sessions` row manually before re-ingesting."
        )


def store_session(
    year: int,
    round_number: int,
    event_name: str,
    session_label: str,
    data_frames: dict[str, pd.DataFrame],
    session_info: Optional[dict] = None,
) -> int:
    """
    Insert a new `sessions` row plus all of its associated data. Table
    names in `data_frames` must match the tables created by schema.sql
    (results, laps, weather, car_data, pos_data, track_status,
    session_status, race_control_messages). `session_info` is the raw
    nested dict from FastF1 and is stored as-is in the session_info
    table's JSONB column, since it isn't a per-row table like the others.

    All-or-nothing: if any category in `data_frames` (or `session_info`)
    fails to upload, the `sessions` row is deleted again and
    SupabaseStoreError is raised, so a half-ingested session is never left
    for find_session() to mistake for a complete one.

    Returns the new session's id.
    """
    client = get_client()

    try:
        inserted = (
            client.table("sessions")
            .insert({
                # FastF1 event fields (e.g. RoundNumber) are numpy scalars, not
                # native Python types, so this goes through _json_safe same as
                # every other value that reaches the REST API.
                "season_year": _json_safe(year),
                "round_number": _json_safe(round_number),
                "event_name": _json_safe(event_name),
                "session_label": _json_safe(session_label),
            })
            .execute()
        )
    except Exception as exc:
        raise SupabaseStoreError(
            f"Could not create the `sessions` row for {year} {event_name} "
            f"{session_label}: {exc}"
        ) from exc
    session_id = inserted.data[0]["id"]

    # Every key in `data_frames` is a category FastF1 actually returned
    # (the ingestion script drops categories that didn't load before
    # calling this) - so a category here failing to insert is a genuine
    # partial ingestion, not the "FastF1 had no data for it" case. A
    # half-written session can't be left behind: find_session() would treat
    # its `sessions` row as complete and skip re-ingestion forever. So on
    # the first failure, roll the whole session back and raise - a re-run
    # then retries it from scratch.
    for table_name, frame in data_frames.items():
        if frame is None or frame.empty:
            continue
        records = _records_for_insert(frame)
        for record in records:
            record["session_id"] = session_id
        try:
            _insert_in_chunks(table_name, records)
        except Exception as exc:
            _rollback_session(session_id, failed_on=table_name)
            raise SupabaseStoreError(
                f"Failed to store '{table_name}' for {year} {event_name} "
                f"{session_label}; the session was rolled back, re-run the "
                f"ingestion to retry. Underlying error: {exc}"
            ) from exc

    if session_info:
        try:
            client.table("session_info").insert({
                "session_id": session_id,
                "info": _json_safe(session_info),
            }).execute()
        except Exception as exc:
            _rollback_session(session_id, failed_on="session_info")
            raise SupabaseStoreError(
                f"Failed to store 'session_info' for {year} {event_name} "
                f"{session_label}; the session was rolled back, re-run the "
                f"ingestion to retry. Underlying error: {exc}"
            ) from exc

    return session_id
