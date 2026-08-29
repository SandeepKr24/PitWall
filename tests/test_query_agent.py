"""
Tests for query_agent/query_agent.py - the SQL guardrails (fence
stripping + the SELECT-only / known-tables validator), the text-block
extraction, the Anthropic-error wrapping, and the result-row cap fed to
Claude. No database or Anthropic network calls; the SDK is mocked.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import anthropic
import pytest

import query_agent as qa


# ---------------------------------------------------------------------------
# _strip_code_fence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("SELECT 1", "SELECT 1"),
    ("   SELECT 1   ", "SELECT 1"),
    ("```sql\nSELECT * FROM results\n```", "SELECT * FROM results"),
    ("```\nSELECT 1\n```", "SELECT 1"),
    ("```postgresql\nSELECT 1\n```", "SELECT 1"),
])
def test_strip_code_fence(raw, expected):
    assert qa._strip_code_fence(raw) == expected


# ---------------------------------------------------------------------------
# validate_select_only
# ---------------------------------------------------------------------------

def test_validate_select_only_accepts_plain_select():
    out = qa.validate_select_only('SELECT "DriverNumber" FROM results')
    assert "results" in out.lower()


def test_validate_select_only_accepts_union_of_selects():
    qa.validate_select_only(
        "SELECT \"DriverNumber\" FROM results UNION SELECT \"DriverNumber\" FROM laps"
    )


def test_validate_select_only_accepts_join_across_known_tables():
    qa.validate_select_only(
        'SELECT r."FullName", l."LapTime" FROM laps l '
        'JOIN results r ON r."DriverNumber" = l."DriverNumber"'
    )


@pytest.mark.parametrize("sql", [
    "SELECT 1 FROM results; SELECT 2 FROM laps",   # more than one statement
    "UPDATE results SET \"Points\" = 0",           # not a SELECT
    "DELETE FROM results",
    "DROP TABLE results",
    "INSERT INTO results (\"DriverNumber\") VALUES ('1')",
    "SELECT * FROM pg_catalog.pg_tables",          # unknown table
    "SELECT * FROM information_schema.columns",
    "SELECT * FROM secrets",
])
def test_validate_select_only_rejects_dangerous_or_unknown(sql):
    with pytest.raises(qa.QueryAgentError):
        qa.validate_select_only(sql)


def test_validate_select_only_reports_unparseable_sql():
    with pytest.raises(qa.QueryAgentError, match="parse"):
        qa.validate_select_only("SELECT FROM WHERE )(")


# ---------------------------------------------------------------------------
# _extract_text
# ---------------------------------------------------------------------------

def test_extract_text_skips_non_text_blocks():
    response = SimpleNamespace(content=[
        SimpleNamespace(type="thinking", thinking="considering..."),
        SimpleNamespace(type="text", text="the real answer"),
    ])
    assert qa._extract_text(response) == "the real answer"


def test_extract_text_raises_when_no_text_block():
    response = SimpleNamespace(content=[SimpleNamespace(type="thinking", thinking="...")])
    with pytest.raises(qa.QueryAgentError):
        qa._extract_text(response)


# ---------------------------------------------------------------------------
# _claude_text - Anthropic SDK error wrapping
# ---------------------------------------------------------------------------

def _mock_anthropic(monkeypatch, *, side_effect=None, return_value=None):
    client = MagicMock()
    if side_effect is not None:
        client.messages.create.side_effect = side_effect
    else:
        client.messages.create.return_value = return_value
    monkeypatch.setattr(anthropic, "Anthropic", lambda *a, **k: client)
    return client


def test_claude_text_returns_text_on_success(monkeypatch):
    _mock_anthropic(monkeypatch, return_value=SimpleNamespace(
        content=[SimpleNamespace(type="text", text="hello")]
    ))
    assert qa._claude_text(model="m", max_tokens=1, messages=[]) == "hello"


def test_claude_text_wraps_generic_sdk_error(monkeypatch):
    _mock_anthropic(monkeypatch, side_effect=anthropic.AnthropicError("kaboom"))
    with pytest.raises(qa.QueryAgentError, match="Anthropic API call failed"):
        qa._claude_text(model="m", max_tokens=1, messages=[])


def test_claude_text_wraps_auth_error_with_key_hint(monkeypatch):
    import httpx

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(401, request=request)
    auth_error = anthropic.AuthenticationError("invalid key", response=response, body=None)
    _mock_anthropic(monkeypatch, side_effect=auth_error)

    with pytest.raises(qa.QueryAgentError, match="ANTHROPIC_API_KEY"):
        qa._claude_text(model="m", max_tokens=1, messages=[])


# ---------------------------------------------------------------------------
# synthesize_answer - result-row cap handed to Claude
# ---------------------------------------------------------------------------

def test_synthesize_answer_caps_rows_and_flags_truncation(monkeypatch):
    captured = {}

    def fake_claude_text(**kwargs):
        captured.update(kwargs)
        return "answer"

    monkeypatch.setattr(qa, "_claude_text", fake_claude_text)

    rows = [{"n": i} for i in range(qa.MAX_ROWS_TO_CLAUDE + 25)]
    qa.synthesize_answer("how many?", "SELECT n FROM laps", rows)

    content = captured["messages"][0]["content"]
    assert f"{len(rows)} rows" in content
    assert "only the first" in content
    # only MAX_ROWS_TO_CLAUDE row dicts were actually serialised into the prompt
    assert content.count("'n':") == qa.MAX_ROWS_TO_CLAUDE


def test_synthesize_answer_no_truncation_note_for_small_result(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        qa, "_claude_text",
        lambda **kwargs: captured.update(kwargs) or "answer",
    )
    qa.synthesize_answer("q", "SELECT 1", [{"n": 1}, {"n": 2}])
    assert "only the first" not in captured["messages"][0]["content"]
