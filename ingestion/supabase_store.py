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

_client: Optional[Client] = None


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


def _json_safe(value):
    """
    Make a single pandas/numpy scalar safe to send through Supabase's REST
    API as JSON. pandas Timedelta/Timestamp, numpy int/float/bool scalars,
    and NaN/NaT are not JSON-serializable as-is: NaN/NaT become SQL NULL,
    Timedelta/Timestamp become strings that Postgres parses back into
    INTERVAL/TIMESTAMP, and numpy scalars are unwrapped to native types.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
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

    Returns the new session's id.
    """
    client = get_client()

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
    session_id = inserted.data[0]["id"]

    for table_name, frame in data_frames.items():
        if frame is None or frame.empty:
            continue
        records = _records_for_insert(frame)
        for record in records:
            record["session_id"] = session_id
        _insert_in_chunks(table_name, records)

    if session_info:
        client.table("session_info").insert({
            "session_id": session_id,
            "info": {k: _json_safe(v) for k, v in session_info.items()},
        }).execute()

    return session_id
