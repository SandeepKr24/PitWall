"""
Natural-language query agent for a single ingested F1 session.

Stands in for the "chat interface" step of the (future) webpage: after a
user picks a Weekend/Year/Session and it's been ingested (see
ingestion/fastF1_ingestion_script.py), they land on a chat screen where
they can ask anything about that session - "what were the results of this
race", "lap-by-lap comparison between Verstappen and Hamilton", etc.

This script answers one such question at a time:
    1. Resolve the session the user already selected to its session_id in
       Supabase (it must already be ingested; this agent only reads).
    2. Ask Claude to write a single read-only SQL SELECT statement that
       would answer the question, given the database schema.
    3. Validate that SQL (single statement, SELECT/UNION only, only
       touches known tables) before running it anywhere near a database.
    4. Execute it directly against Postgres as `query_agent_ro` - a role
       with SELECT-only grants and no table-owner privileges - inside a
       transaction scoped to this one session_id via Postgres Row Level
       Security (see agent_access.sql). Even if step 3's validation or
       Claude's SQL itself were somehow wrong, RLS makes it structurally
       impossible for the query to see another session's data, and the
       role's grants make it structurally impossible for it to write or
       run DDL, no matter what text was in the request.
    5. Ask Claude to turn the question and the actual query results into
       the final answer - a table, a written comparison, whatever suits
       the question - grounded only in that data.

Requires (in .env):
    SUPABASE_URL, SUPABASE_KEY - already used by the ingestion pipeline;
        used here only for the lightweight `sessions` lookup in step 1.
    SUPABASE_DB_URL - a Postgres connection string authenticating as the
        query_agent_ro role created by agent_access.sql, via Supabase's
        Transaction pooler (port 6543), e.g.:
        postgresql://query_agent_ro.<project-ref>:<role-password>@aws-0-<region>.pooler.supabase.com:6543/postgres
        Use the Transaction pooler, not the direct connection
        (db.<ref>.supabase.co:5432) - the direct endpoint is IPv6-only on
        the free tier, and this script only ever uses SET LOCAL, which is
        transaction-scoped anyway. The username needs the .<project-ref>
        suffix so the pooler routes to the right project; use
        query_agent_ro's own password (ALTER ROLE query_agent_ro WITH
        PASSWORD ...), not the project's main database password.
    ANTHROPIC_API_KEY - for the Claude API calls in steps 2 and 5.

There is no chat UI yet, so this is runnable directly from the CLI:
    python query_agent.py --year 2024 --weekend Bahrain --session R \
        --question "What were the results of this race?"
"""

import argparse
import os
import re
import sys
from pathlib import Path

import anthropic
import psycopg2
import psycopg2.extras
import sqlglot
from dotenv import load_dotenv
from sqlglot import exp

# Reuse the ingestion pipeline's input validation and FastF1 event
# resolution instead of duplicating it - both scripts describe the same
# three inputs the (future) webpage collects.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ingestion"))
import fastF1_ingestion_script as ingestion  # noqa: E402
import supabase_store  # noqa: E402

load_dotenv()

CLAUDE_MODEL = "claude-sonnet-5"

ALLOWED_TABLES = {
    "sessions", "results", "laps", "weather", "car_data", "pos_data",
    "track_status", "session_status", "race_control_messages", "session_info",
}

# Defense against an accidentally (or adversarially) expensive query,
# independent of whether the SQL itself looks reasonable.
QUERY_TIMEOUT_MS = 10_000


class QueryAgentError(Exception):
    """Raised for any agent-specific failure (bad SQL, missing session, etc.)."""


def _db_connection():
    db_url = os.environ.get("SUPABASE_DB_URL")
    if not db_url:
        raise QueryAgentError(
            "SUPABASE_DB_URL must be set (see this script's module docstring) "
            "to run queries."
        )
    return psycopg2.connect(db_url)


def describe_schema(conn) -> str:
    """
    Pull the live column list for each known table from Postgres itself,
    so the prompt can't silently drift out of sync with schema.sql.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = ANY(%s)
            ORDER BY table_name, ordinal_position
            """,
            (list(ALLOWED_TABLES),),
        )
        rows = cur.fetchall()

    tables: dict[str, list[str]] = {}
    for table_name, column_name, data_type in rows:
        tables.setdefault(table_name, []).append(f'"{column_name}" {data_type}')

    return "\n".join(
        f"{table}({', '.join(columns)})" for table, columns in tables.items()
    )


def _extract_text(response) -> str:
    """
    response.content[0] isn't reliably the text block - a model can return
    a ThinkingBlock (or other non-text block) first, so this finds the
    actual text block instead of assuming it's at a fixed index.
    """
    for block in response.content:
        if block.type == "text":
            return block.text
    raise QueryAgentError("Claude's response contained no text block.")


def generate_sql(question: str, schema_description: str) -> str:
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=(
            "You write a single read-only PostgreSQL SELECT statement to "
            "answer a question about one F1 session's data.\n\n"
            "Schema (every table is already implicitly filtered to the one "
            "session in question - do not add a session_id filter yourself, "
            "it's handled outside your query):\n"
            f"{schema_description}\n\n"
            "Rules:\n"
            "- Output ONLY the SQL statement: no prose, no markdown fences.\n"
            "- Exactly one statement, ending without a semicolon.\n"
            "- SELECT (or a UNION of SELECTs) only - never write to any table.\n"
            "- Only reference the tables listed above.\n"
            "- Driver identity/name columns live in `results`; join to it "
            "(on \"DriverNumber\") when a query needs a driver's name."
        ),
        messages=[{"role": "user", "content": question}],
    )
    return _strip_code_fence(_extract_text(response).strip())


def _strip_code_fence(text: str) -> str:
    """
    Claude doesn't always follow a "no markdown fences" instruction to the
    letter - strip a leading/trailing ``` or ```sql fence if present rather
    than relying solely on prompt compliance for something the parser needs
    to be exact about.
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"```$", "", text)
    return text.strip()


def validate_select_only(sql: str) -> str:
    """
    Parse `sql` and reject anything that isn't exactly one read-only
    SELECT/UNION statement touching only known tables. This is the first
    line of defense; the Postgres-level role privileges and RLS set up in
    agent_access.sql are what actually can't be bypassed even if this
    check has a gap - but rejecting obviously-wrong SQL here gives a much
    clearer error than a permission-denied buried inside Postgres.
    """
    try:
        statements = [s for s in sqlglot.parse(sql, dialect="postgres") if s is not None]
    except sqlglot.errors.ParseError as exc:
        raise QueryAgentError(f"Generated SQL failed to parse: {exc}") from exc

    if len(statements) != 1:
        raise QueryAgentError(f"Expected exactly one SQL statement, got {len(statements)}.")

    statement = statements[0]
    if not isinstance(statement, (exp.Select, exp.Union)):
        raise QueryAgentError(
            f"Only SELECT statements are allowed, got: {type(statement).__name__}."
        )
    if statement.args.get("into") is not None:
        raise QueryAgentError("SELECT INTO is not allowed.")

    referenced_tables = {t.name for t in statement.find_all(exp.Table)}
    disallowed = referenced_tables - ALLOWED_TABLES
    if disallowed:
        raise QueryAgentError(f"Query references unknown table(s): {disallowed}")

    return statement.sql(dialect="postgres")


def run_query(conn, session_id: int, sql: str) -> list[dict]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SET LOCAL statement_timeout = %s", (QUERY_TIMEOUT_MS,))
        # Scopes every RLS-covered table to this one session, regardless of
        # whether/how `sql` itself filters by session_id - see agent_access.sql.
        cur.execute("SET LOCAL app.session_id = %s", (str(session_id),))
        cur.execute(sql)
        rows = cur.fetchall()
    return [dict(row) for row in rows]


def synthesize_answer(question: str, sql: str, rows: list[dict]) -> str:
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2048,
        system=(
            "You answer a question about F1 session data using ONLY the "
            "query results provided - never invent or assume values not "
            "present in them. If the results are empty, say so plainly. "
            "Choose whatever presentation best suits the question: a "
            "markdown table for a direct data request, prose/analysis for "
            "a comparison or 'what happened' question, or both."
        ),
        messages=[{
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"SQL that was run: {sql}\n\n"
                f"Results ({len(rows)} rows): {rows}"
            ),
        }],
    )
    return _extract_text(response)


def answer_question(year: int, weekend_name: str, session_label: str, question: str) -> str:
    """
    Resolve the already-ingested session the three input fields describe,
    then answer `question` about it. Raises QueryAgentError if that
    session hasn't been ingested yet.
    """
    year = ingestion.validate_year(year)
    weekend_name = ingestion.validate_weekend_name(weekend_name)

    ingestion._ensure_cache_enabled()
    event = ingestion.get_event(year, weekend_name)
    sprint_weekend = ingestion.is_sprint_weekend(event)
    session_label = ingestion.validate_session_label(session_label, sprint_weekend)

    existing = supabase_store.find_session(year, event.RoundNumber, session_label)
    if existing is None:
        raise QueryAgentError(
            f"{year} {event.EventName} {session_label} hasn't been ingested yet. "
            f"Run ingestion/fastF1_ingestion_script.py for it first."
        )
    session_id = existing["id"]

    conn = _db_connection()
    try:
        schema_description = describe_schema(conn)
        raw_sql = generate_sql(question, schema_description)
        safe_sql = validate_select_only(raw_sql)
        rows = run_query(conn, session_id, safe_sql)
    finally:
        conn.close()

    return synthesize_answer(question, safe_sql, rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask a natural-language question about one already-ingested F1 session."
    )
    parser.add_argument("--year", type=int, required=True, help="4-digit season year, e.g. 2024")
    parser.add_argument(
        "--weekend", type=str, required=True, dest="weekend_name",
        help="Weekend/Grand Prix name, e.g. 'Bahrain', 'Silverstone', 'Spa'",
    )
    parser.add_argument(
        "--session", type=str, required=True, dest="session_label",
        help="Session label: P1/P2/P3/Q/R (conventional) or P1/SQ/SR/Q/R (sprint)",
    )
    parser.add_argument("--question", type=str, required=True, help="What to ask about this session")
    args = parser.parse_args()

    # Claude's answers routinely contain non-Latin-1 characters (arrows,
    # en-dashes, etc.). On Windows the console defaults to cp1252, so a
    # bare print() of the answer raises UnicodeEncodeError - force UTF-8
    # output rather than depending on the terminal's code page.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")

    try:
        answer = answer_question(args.year, args.weekend_name, args.session_label, args.question)
    except (ingestion.InvalidInputError, QueryAgentError) as exc:
        parser.error(str(exc))
        return

    print(answer)


if __name__ == "__main__":
    main()
