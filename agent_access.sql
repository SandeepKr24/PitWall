-- ============================================================================
-- Read-only, session-scoped database access for the natural-language query
-- agent (query_agent/query_agent.py).
--
-- The agent asks an LLM to write a SQL SELECT statement for whatever the
-- user asks in the chat interface, then runs it. An LLM-generated query is
-- untrusted input, same as anything else a user's request could influence,
-- so it must never run with the same privileges the ingestion pipeline
-- uses (the service_role key can write anywhere, no restrictions).
--
-- This sets up two independent layers of defense:
--   1. A dedicated `query_agent_ro` role that can only SELECT - it has no
--      INSERT/UPDATE/DELETE grants and isn't a table owner, so it can't
--      write, DROP, or ALTER no matter what SQL text it's asked to run.
--   2. Row Level Security on every per-session table, restricting every
--      query - regardless of what WHERE clause the LLM did or didn't
--      write - to the one session_id the user actually selected on the
--      input screen. This is enforced by Postgres itself, not by trusting
--      that the LLM's SQL remembered to filter correctly.
--
-- Run this once in the Supabase SQL editor, after schema.sql. Replace
-- 'CHANGE_ME' below with a real password before running, and put the
-- resulting connection string in .env as SUPABASE_DB_URL (see
-- query_agent/query_agent.py's module docstring for the exact format).
-- ============================================================================

CREATE ROLE query_agent_ro LOGIN PASSWORD 'CHANGE_ME';

GRANT USAGE ON SCHEMA public TO query_agent_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO query_agent_ro;

-- sessions itself isn't per-session data (it's the index of sessions), so
-- it stays fully readable rather than being scoped by session_id.
ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;
CREATE POLICY sessions_read_all ON sessions FOR SELECT USING (true);

-- Every other table is restricted to the session set for the current
-- query via `SET LOCAL app.session_id = '<id>'` (done by query_agent.py
-- at the start of each request's transaction, before running the LLM's
-- SQL). FORCE means this applies even to the table owner, not just
-- query_agent_ro - belt and suspenders alongside the SELECT-only grant.
DO $$
DECLARE
    scoped_table TEXT;
BEGIN
    FOR scoped_table IN
        SELECT unnest(ARRAY[
            'results', 'laps', 'weather', 'car_data', 'pos_data',
            'track_status', 'session_status', 'race_control_messages',
            'session_info'
        ])
    LOOP
        EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', scoped_table);
        EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', scoped_table);
        EXECUTE format(
            'CREATE POLICY %I ON %I FOR SELECT '
            'USING (session_id = current_setting(''app.session_id'', true)::bigint)',
            scoped_table || '_session_scoped', scoped_table
        );
    END LOOP;
END $$;
