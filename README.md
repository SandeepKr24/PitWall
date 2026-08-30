# PitWall

**An AI-powered Formula 1 analyst.** Pick a Grand Prix weekend, year, and session,
ingest that session's data, then ask questions about it in plain English:
"What were the results of this race?", "Compare Verstappen and Hamilton lap by
lap." and get answers grounded in the real data.

## How it works

```
FastF1 API ─▶ ingestion/fastF1_ingestion_script.py ─▶ Supabase (PostgreSQL)
                                                            │
                     query_agent/query_agent.py ◀───────────┘
                              │
                              ▼
                     Anthropic Claude  ─▶  answer (table / prose / both)
```

1. **Ingest** — resolve the weekend/year/session, pull nine data categories from
   [FastF1](https://docs.fastf1.dev/) (results, laps, weather, car telemetry, car
   position, track status, session status, race-control messages, session info),
   and store them in Supabase. Already-ingested sessions are detected and skipped.
2. **Query** — Claude writes a single read-only SQL `SELECT` for the question
   against the live schema; it is AST-validated, then executed as a
   `SELECT` only Postgres role scoped to that one session; Claude turns the rows
   into the final answer.

There is **no web UI yet**, the two scripts below stand in for it (see
[Roadmap](#roadmap)).

## Requirements

- Python 3.13
- A Supabase (PostgreSQL) project with:
  - a `sessions` table plus one table per data category (see
    `agent_access.sql` for the table list), and
  - `agent_access.sql` applied once, which creates the read-only
    `query_agent_ro` role and the per-session Row-Level Security policies.
- An Anthropic API key.
- **Run ingestion from a home/residential network.** Formula 1's CDN blocks many
  cloud/datacenter IP ranges, which causes most categories to fail to download
  from a cloud VM.

## Setup

```bash
python -m venv venv
venv/Scripts/python -m pip install -r requirements.txt      # or requirements-dev.txt for tests
```

Create a `.env` in the repo root:

```
SUPABASE_URL=https://<project-ref>.supabase.co
SUPABASE_KEY=<service_role key>          # ingestion only; bypasses RLS
SUPABASE_DB_URL=postgresql://query_agent_ro.<project-ref>:<password>@aws-0-<region>.pooler.supabase.com:6543/postgres
ANTHROPIC_API_KEY=<your key>
```

`SUPABASE_DB_URL` must use Supabase's **Transaction pooler** (port 6543) with the
`query_agent_ro` role and its own password.

## Usage

### Ingest a session

```bash
python ingestion/fastF1_ingestion_script.py --year 2024 --weekend Bahrain --session R
```

- `--session` — `P1 P2 P3 Q R` (conventional weekend) or `P1 SQ SR Q R` (sprint).
- `--weekend` fuzzy-matches country / circuit / official name ("Spa", "Belgium",
  "Silverstone" all resolve).
- `--session all` ingests every session of that weekend; `--weekend all` ingests
  the whole season. Batch runs skip what's already stored, so an interrupted run
  just resumes.
- `--force` re-ingests a session that's already stored (deletes the old copy
  first).

### See what's ingested

```bash
python ingestion/list_sessions.py            # all sessions + per-category row counts
python ingestion/list_sessions.py --year 2024
```

### Ask a question

```bash
python query_agent/query_agent.py --year 2024 --weekend Bahrain --session R \
    --question "What were the results of this race?"
```

The answer is printed to stdout; progress and errors go to stderr.

## Query-agent safety

The SQL Claude generates is treated as untrusted input. Three independent layers:

1. **AST validation** (`sqlglot`): exactly one statement, `SELECT`/`UNION` only,
   only known tables.
2. **`query_agent_ro` role**: `SELECT`-only grants, not a table owner, so it
   structurally cannot write or run DDL.
3. **Row-Level Security**: every per-session table is filtered to the one
   `session_id` the user selected, enforced by Postgres, plus a 10-second
   statement timeout.

## Tests

```bash
venv/Scripts/python -m pip install -r requirements-dev.txt
venv/Scripts/python -m pytest
```

80 tests, all offline (no network, no live database, the Supabase client and
Anthropic SDK are mocked).

## Project layout

```
ingestion/
  fastF1_ingestion_script.py   # fetch from FastF1, dedup-check, store to Supabase
  supabase_store.py            # Supabase persistence layer
  list_sessions.py             # read-only: list ingested sessions + row counts
query_agent/
  query_agent.py               # natural-language question -> SQL -> Supabase -> answer
tests/                         # pytest suite
agent_access.sql               # query_agent_ro role + Row-Level Security (run once)
requirements.txt               # runtime dependencies (pinned)
requirements-dev.txt           # + pytest
```

## Roadmap

The end goal is a deployed web app:

- **Input screen**: dropdowns for weekend and session, a text field for the
  year, session options adapting to the weekend format (conventional vs sprint).
- **Chat interface**: after submitting, land on a chat screen and ask anything
  about that session; the current `query_agent` becomes the backend.
- **Deployment**: host the web frontend and an HTTP API in front of the existing
  ingestion and query modules.

Stack for the web layer is not yet decided.
