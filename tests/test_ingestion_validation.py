"""
Tests for the pure validation / resolution helpers in
ingestion/fastF1_ingestion_script.py - the backend re-checks of what the
future webpage's dropdowns and text input will enforce, plus the small
frame-shaping helpers. No FastF1 network calls here.
"""

import fastf1
import pandas as pd
import pytest

from fastF1_ingestion_script import (
    CONVENTIONAL_SESSIONS,
    SPRINT_SESSIONS,
    SESSION_ALIASES,
    InvalidInputError,
    _session_info_to_frame,
    _slugify,
    _telemetry_dict_to_frame,
    validate_session_label,
    validate_weekend_name,
    validate_year,
)


# ---------------------------------------------------------------------------
# validate_year
# ---------------------------------------------------------------------------

def test_validate_year_accepts_a_normal_season():
    assert validate_year(2024) == 2024


@pytest.mark.parametrize("bad", [2024.0, "2024", None, True, False])
def test_validate_year_rejects_non_int(bad):
    with pytest.raises(InvalidInputError):
        validate_year(bad)


@pytest.mark.parametrize("bad", [999, 10_000, 1949])
def test_validate_year_rejects_out_of_range(bad):
    # 999 / 10000 aren't 4-digit; 1949 is before FastF1's earliest season.
    with pytest.raises(InvalidInputError):
        validate_year(bad)


# ---------------------------------------------------------------------------
# validate_weekend_name
# ---------------------------------------------------------------------------

def test_validate_weekend_name_strips_surrounding_whitespace():
    assert validate_weekend_name("  Bahrain ") == "Bahrain"


@pytest.mark.parametrize("bad", ["", "   ", None, 123])
def test_validate_weekend_name_rejects_empty_or_non_str(bad):
    with pytest.raises(InvalidInputError):
        validate_weekend_name(bad)


# ---------------------------------------------------------------------------
# validate_session_label
# ---------------------------------------------------------------------------

def test_validate_session_label_normalises_case_and_whitespace():
    assert validate_session_label(" r ", is_sprint_weekend=False) == "R"
    assert validate_session_label("q", is_sprint_weekend=False) == "Q"


def test_validate_session_label_enforces_weekend_format():
    # SQ/SR only exist on a sprint weekend; P2/P3 only on a conventional one.
    assert validate_session_label("SQ", is_sprint_weekend=True) == "SQ"
    assert validate_session_label("SR", is_sprint_weekend=True) == "SR"
    with pytest.raises(InvalidInputError):
        validate_session_label("SQ", is_sprint_weekend=False)
    with pytest.raises(InvalidInputError):
        validate_session_label("P2", is_sprint_weekend=True)


def test_validate_session_label_rejects_unknown_label():
    with pytest.raises(InvalidInputError):
        validate_session_label("NOPE", is_sprint_weekend=False)


# ---------------------------------------------------------------------------
# SESSION_ALIASES
# ---------------------------------------------------------------------------

def test_session_aliases_cover_every_ui_label():
    for label in set(CONVENTIONAL_SESSIONS) | set(SPRINT_SESSIONS):
        assert label in SESSION_ALIASES


def test_session_aliases_map_to_fastf1_identifiers():
    assert SESSION_ALIASES["P1"] == "FP1"
    assert SESSION_ALIASES["P2"] == "FP2"
    assert SESSION_ALIASES["P3"] == "FP3"
    assert SESSION_ALIASES["SR"] == "S"      # sprint race
    assert SESSION_ALIASES["SQ"] == "SQ"
    assert SESSION_ALIASES["Q"] == "Q"
    assert SESSION_ALIASES["R"] == "R"


# ---------------------------------------------------------------------------
# _slugify
# ---------------------------------------------------------------------------

def test_slugify_event_name():
    assert _slugify("Bahrain Grand Prix") == "bahrain_grand_prix"
    assert _slugify("  São Paulo Grand Prix!  ") == "s_o_paulo_grand_prix"


# ---------------------------------------------------------------------------
# _telemetry_dict_to_frame
# ---------------------------------------------------------------------------

def test_telemetry_dict_to_frame_tags_driver_and_drops_sessiontime():
    per_driver = {
        "44": pd.DataFrame({"Time": [1, 2], "Speed": [100, 200], "SessionTime": [10, 20]}),
        "1": pd.DataFrame({"Time": [1], "Speed": [150], "SessionTime": [11]}),
    }
    out = _telemetry_dict_to_frame(per_driver)

    assert list(out.columns)[0] == "DriverNumber"
    assert "SessionTime" not in out.columns          # this is what broke uploads before
    assert list(out["DriverNumber"]) == ["44", "44", "1"]
    assert len(out) == 3


def test_telemetry_dict_to_frame_raises_when_no_data():
    with pytest.raises(fastf1.exceptions.DataNotLoadedError):
        _telemetry_dict_to_frame({})


# ---------------------------------------------------------------------------
# _session_info_to_frame
# ---------------------------------------------------------------------------

def test_session_info_to_frame_flattens_nested_dict_to_one_row():
    out = _session_info_to_frame({"Meeting": {"Key": 1234, "Name": "Bahrain"}, "Type": "Race"})
    assert len(out) == 1
    assert out["Meeting.Key"].iloc[0] == 1234
    assert out["Meeting.Name"].iloc[0] == "Bahrain"
    assert out["Type"].iloc[0] == "Race"


def test_session_info_to_frame_raises_when_empty():
    with pytest.raises(fastf1.exceptions.DataNotLoadedError):
        _session_info_to_frame({})
