"""
Tests for the batch layer of ingestion/fastF1_ingestion_script.py -
target expansion (--weekend all / --session all) and the run_ingestion
orchestrator's tally / abort behaviour. FastF1 and the single-session
ingest are mocked; nothing here hits the network or Supabase.
"""

import pandas as pd
import pytest

import fastF1_ingestion_script as ing


class _FakeEvent:
    def __init__(self, event_format):
        self.EventFormat = event_format


@pytest.fixture(autouse=True)
def _stub_cache(monkeypatch):
    # run_ingestion / the helpers call this; don't touch the real cache dir.
    monkeypatch.setattr(ing, "_ensure_cache_enabled", lambda: None)


# ---------------------------------------------------------------------------
# _season_weekend_names
# ---------------------------------------------------------------------------

def test_season_weekend_names_returns_trimmed_non_blank_names(monkeypatch):
    monkeypatch.setattr(
        ing.fastf1, "get_event_schedule",
        lambda year, **kw: pd.DataFrame({"EventName": ["Bahrain GP", "  ", " Spa GP "]}),
    )
    assert ing._season_weekend_names(2024) == ["Bahrain GP", "Spa GP"]


def test_season_weekend_names_raises_when_schedule_empty(monkeypatch):
    monkeypatch.setattr(
        ing.fastf1, "get_event_schedule",
        lambda year, **kw: pd.DataFrame({"EventName": []}),
    )
    with pytest.raises(ing.InvalidInputError):
        ing._season_weekend_names(2024)


def test_season_weekend_names_wraps_schedule_load_failure(monkeypatch):
    def boom(year, **kw):
        raise ConnectionError("offline")

    monkeypatch.setattr(ing.fastf1, "get_event_schedule", boom)
    with pytest.raises(ing.InvalidInputError, match="season schedule"):
        ing._season_weekend_names(2024)


# ---------------------------------------------------------------------------
# _expand_targets
# ---------------------------------------------------------------------------

def test_expand_targets_plain_pair_never_touches_fastf1(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("no FastF1 lookup expected for an explicit weekend+session")

    monkeypatch.setattr(ing, "get_event", boom)
    monkeypatch.setattr(ing.fastf1, "get_event_schedule", boom)

    assert ing._expand_targets(2024, "  Bahrain ", " R ") == [("Bahrain", "R")]


def test_expand_targets_session_all_uses_conventional_labels(monkeypatch):
    monkeypatch.setattr(ing, "get_event", lambda year, name: _FakeEvent("conventional"))
    assert ing._expand_targets(2024, "Bahrain", "all") == [
        ("Bahrain", label) for label in ing.CONVENTIONAL_SESSIONS
    ]


def test_expand_targets_session_all_uses_sprint_labels_case_insensitive(monkeypatch):
    monkeypatch.setattr(ing, "get_event", lambda year, name: _FakeEvent("sprint_qualifying"))
    assert ing._expand_targets(2024, "China", "ALL") == [
        ("China", label) for label in ing.SPRINT_SESSIONS
    ]


def test_expand_targets_weekend_all(monkeypatch):
    monkeypatch.setattr(
        ing.fastf1, "get_event_schedule",
        lambda year, **kw: pd.DataFrame({"EventName": ["A GP", "B GP"]}),
    )
    assert ing._expand_targets(2024, "all", "R") == [("A GP", "R"), ("B GP", "R")]


def test_expand_targets_weekend_and_session_all_respects_each_format(monkeypatch):
    monkeypatch.setattr(
        ing.fastf1, "get_event_schedule",
        lambda year, **kw: pd.DataFrame({"EventName": ["A GP", "B GP"]}),
    )
    monkeypatch.setattr(
        ing, "get_event",
        lambda year, name: _FakeEvent("conventional" if name == "A GP" else "sprint_qualifying"),
    )
    out = ing._expand_targets(2024, "all", "all")

    assert ("A GP", "P2") in out and ("A GP", "SQ") not in out
    assert ("B GP", "SQ") in out and ("B GP", "P2") not in out
    assert len(out) == len(ing.CONVENTIONAL_SESSIONS) + len(ing.SPRINT_SESSIONS)


# ---------------------------------------------------------------------------
# run_ingestion
# ---------------------------------------------------------------------------

def test_run_ingestion_single_calls_ingest_once_and_returns_zero(monkeypatch):
    calls = []
    monkeypatch.setattr(
        ing, "ingest_session",
        lambda year, weekend, session, force=False: calls.append((year, weekend, session, force)),
    )
    assert ing.run_ingestion(2024, "Bahrain", "R") == 0
    assert calls == [(2024, "Bahrain", "R", False)]


def test_run_ingestion_single_propagates_invalid_input(monkeypatch):
    def bad(*a, **k):
        raise ing.InvalidInputError("nope")

    monkeypatch.setattr(ing, "ingest_session", bad)
    with pytest.raises(ing.InvalidInputError):
        ing.run_ingestion(2024, "Bahrain", "ZZ")


def test_run_ingestion_batch_tallies_stored_skipped_failed(monkeypatch, capsys):
    monkeypatch.setattr(
        ing, "_expand_targets",
        lambda year, weekend, session: [("A", "R"), ("B", "R"), ("C", "R"), ("D", "R")],
    )

    def fake_ingest(year, weekend, session, force=False):
        if weekend == "A":
            return ing.Path("data/a")            # stored
        if weekend == "B":
            return None                          # already present -> skipped
        if weekend == "C":
            raise ValueError("no data for this session yet")   # per-session fail
        return ing.Path("data/d")               # stored

    monkeypatch.setattr(ing, "ingest_session", fake_ingest)

    code = ing.run_ingestion(2024, "all", "R", force=True)

    assert code == 1  # a session failed
    summary = capsys.readouterr().out
    assert "2 stored, 1 already present, 1 failed" in summary


def test_run_ingestion_batch_all_ok_returns_zero(monkeypatch):
    monkeypatch.setattr(
        ing, "_expand_targets",
        lambda year, weekend, session: [("A", "R"), ("B", "R")],
    )
    monkeypatch.setattr(ing, "ingest_session", lambda *a, **k: ing.Path("x"))
    assert ing.run_ingestion(2024, "all", "R") == 0


def test_run_ingestion_batch_aborts_on_runtimeerror(monkeypatch):
    seen = []

    monkeypatch.setattr(
        ing, "_expand_targets",
        lambda year, weekend, session: [("A", "R"), ("B", "R"), ("C", "R")],
    )

    def fake_ingest(year, weekend, session, force=False):
        seen.append(weekend)
        if weekend == "B":
            raise ing.supabase_store.SupabaseStoreError("storage unreachable")
        return ing.Path("x")

    monkeypatch.setattr(ing, "ingest_session", fake_ingest)

    assert ing.run_ingestion(2024, "all", "R") == 1
    assert seen == ["A", "B"]  # did not carry on to C
