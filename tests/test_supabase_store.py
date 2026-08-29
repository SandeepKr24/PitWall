"""
Tests for ingestion/supabase_store.py - the JSON-safety conversion (where
several real ingestion bugs have lived) and the all-or-nothing
store_session / rollback behaviour. Nothing here touches a real Supabase;
the client is mocked.
"""

import datetime
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import supabase_store
from supabase_store import _json_safe, _records_for_insert


# ---------------------------------------------------------------------------
# _json_safe
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [None, float("nan"), np.nan, pd.NaT])
def test_json_safe_nullish_becomes_none(value):
    assert _json_safe(value) is None


def test_json_safe_unwraps_numpy_scalars_to_native():
    assert _json_safe(np.int64(5)) == 5
    assert type(_json_safe(np.int64(5))) is int
    assert _json_safe(np.float64(1.5)) == 1.5
    assert type(_json_safe(np.float64(1.5))) is float
    assert _json_safe(np.bool_(True)) is True


def test_json_safe_whole_number_float_becomes_int():
    # PostgREST rejects "2.0" for an INTEGER column; a float64 column that
    # only looks like a float because of a stray NaN must round-trip as int.
    assert _json_safe(2.0) == 2 and type(_json_safe(2.0)) is int
    assert type(_json_safe(np.float64(3.0))) is int
    # genuine fractional values stay float
    assert _json_safe(2.5) == 2.5 and type(_json_safe(2.5)) is float


def test_json_safe_bool_is_checked_before_int():
    # bool is a subclass of int - the bool branch must win.
    assert _json_safe(True) is True
    assert _json_safe(False) is False


def test_json_safe_stringifies_temporal_types():
    assert _json_safe(pd.Timedelta(seconds=5)) == str(pd.Timedelta(seconds=5))
    assert _json_safe(datetime.timedelta(seconds=5)) == "0:00:05"
    ts = pd.Timestamp("2024-03-02T15:00:00")
    assert _json_safe(ts) == ts.isoformat()
    dt = datetime.datetime(2024, 3, 2, 15, 0, 0)
    assert _json_safe(dt) == dt.isoformat()


def test_json_safe_recurses_into_dicts_and_lists():
    out = _json_safe({"a": np.int64(1), "b": [np.nan, pd.NaT, {"c": 2.0}]})
    assert out == {"a": 1, "b": [None, None, {"c": 2}]}


def test_records_for_insert_applies_json_safe_per_cell():
    frame = pd.DataFrame([{"Position": 1.0, "LapTime": pd.Timedelta(seconds=90)}])
    records = _records_for_insert(frame)
    assert records == [{"Position": 1, "LapTime": "0 days 00:01:30"}]


# ---------------------------------------------------------------------------
# store_session - happy path and rollback
# ---------------------------------------------------------------------------

def _fake_client_returning_session_id(session_id):
    client = MagicMock()
    client.table.return_value.insert.return_value.execute.return_value.data = [
        {"id": session_id}
    ]
    return client


def test_store_session_returns_id_and_inserts_each_category(monkeypatch):
    client = _fake_client_returning_session_id(7)
    monkeypatch.setattr(supabase_store, "get_client", lambda: client)

    inserted = []
    monkeypatch.setattr(
        supabase_store, "_insert_in_chunks",
        lambda table, records: inserted.append((table, len(records))),
    )

    frames = {
        "results": pd.DataFrame([{"DriverNumber": "1"}, {"DriverNumber": "16"}]),
        "laps": pd.DataFrame([{"Driver": "VER", "LapNumber": 1.0}]),
    }
    session_id = supabase_store.store_session(
        year=2024, round_number=1, event_name="Test GP",
        session_label="R", data_frames=frames,
    )

    assert session_id == 7
    assert sorted(inserted) == [("laps", 1), ("results", 2)]
    client.table.return_value.delete.assert_not_called()


def test_store_session_skips_empty_and_none_frames(monkeypatch):
    client = _fake_client_returning_session_id(1)
    monkeypatch.setattr(supabase_store, "get_client", lambda: client)
    inserted = []
    monkeypatch.setattr(
        supabase_store, "_insert_in_chunks",
        lambda table, records: inserted.append(table),
    )

    frames = {
        "results": pd.DataFrame([{"DriverNumber": "1"}]),
        "laps": pd.DataFrame(),          # empty
        "weather": None,                 # missing
    }
    supabase_store.store_session(
        year=2024, round_number=1, event_name="Test GP",
        session_label="R", data_frames=frames,
    )
    assert inserted == ["results"]


def test_store_session_rolls_back_when_a_category_fails(monkeypatch):
    client = _fake_client_returning_session_id(42)
    monkeypatch.setattr(supabase_store, "get_client", lambda: client)

    def boom(table, records):
        raise RuntimeError("simulated upload failure")

    monkeypatch.setattr(supabase_store, "_insert_in_chunks", boom)

    frames = {"results": pd.DataFrame([{"DriverNumber": "1"}])}
    with pytest.raises(supabase_store.SupabaseStoreError, match="rolled back"):
        supabase_store.store_session(
            year=2024, round_number=1, event_name="Test GP",
            session_label="R", data_frames=frames,
        )

    # the half-written sessions row must have been deleted
    client.table.assert_any_call("sessions")
    client.table.return_value.delete.return_value.eq.assert_called_with("id", 42)


def test_store_session_wraps_sessions_insert_failure(monkeypatch):
    client = MagicMock()
    client.table.return_value.insert.return_value.execute.side_effect = RuntimeError(
        "supabase down"
    )
    monkeypatch.setattr(supabase_store, "get_client", lambda: client)

    with pytest.raises(supabase_store.SupabaseStoreError, match="Could not create"):
        supabase_store.store_session(
            year=2024, round_number=1, event_name="Test GP",
            session_label="R", data_frames={},
        )


def test_store_session_error_is_a_runtimeerror():
    # the ingestion CLI catches RuntimeError - the subclass must stay under it
    assert issubclass(supabase_store.SupabaseStoreError, RuntimeError)


# ---------------------------------------------------------------------------
# delete_session / list_sessions
# ---------------------------------------------------------------------------

def test_delete_session_deletes_by_id(monkeypatch):
    client = MagicMock()
    monkeypatch.setattr(supabase_store, "get_client", lambda: client)

    supabase_store.delete_session(99)

    client.table.assert_called_with("sessions")
    client.table.return_value.delete.return_value.eq.assert_called_with("id", 99)
    client.table.return_value.delete.return_value.eq.return_value.execute.assert_called_once()


def test_list_sessions_attaches_per_category_counts(monkeypatch):
    session_rows = [
        {"id": 11, "season_year": 2024, "round_number": 1,
         "event_name": "Bahrain Grand Prix", "session_label": "R"},
    ]

    # counts per category table for session_id 11 - everything present.
    per_table_count = {t: 10 for t in supabase_store.CATEGORY_TABLES}

    class FakeTable:
        def __init__(self, name):
            self.name = name

        def select(self, *a, **k):
            return self

        def order(self, *a, **k):
            return self

        def eq(self, *a, **k):
            return self

        def limit(self, *a, **k):
            return self

        def execute(self):
            if self.name == "sessions":
                return MagicMock(data=session_rows)
            return MagicMock(count=per_table_count[self.name])

    client = MagicMock()
    client.table.side_effect = FakeTable
    monkeypatch.setattr(supabase_store, "get_client", lambda: client)

    out = supabase_store.list_sessions()

    assert len(out) == 1
    assert out[0]["id"] == 11
    assert out[0]["counts"] == per_table_count
    assert set(out[0]["counts"]) == set(supabase_store.CATEGORY_TABLES)


def test_list_sessions_filters_by_year(monkeypatch):
    seen_filters = []

    class FakeTable:
        def __init__(self, name):
            self.name = name

        def select(self, *a, **k):
            return self

        def order(self, *a, **k):
            return self

        def eq(self, col, val):
            seen_filters.append((self.name, col, val))
            return self

        def limit(self, *a, **k):
            return self

        def execute(self):
            if self.name == "sessions":
                return MagicMock(data=[])
            return MagicMock(count=0)

    client = MagicMock()
    client.table.side_effect = FakeTable
    monkeypatch.setattr(supabase_store, "get_client", lambda: client)

    supabase_store.list_sessions(year=2024)

    assert ("sessions", "season_year", 2024) in seen_filters


def test_category_tables_matches_query_agent_allowed_tables():
    # query_agent.ALLOWED_TABLES is CATEGORY_TABLES + the sessions index.
    import query_agent
    assert set(supabase_store.CATEGORY_TABLES) | {"sessions"} == query_agent.ALLOWED_TABLES
