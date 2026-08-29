"""
FastF1 ingestion script.

The (future) webpage will collect three inputs from the user:
    1) Weekend  -> dropdown, the Grand Prix name (e.g. "Bahrain",
                   "Silverstone", "Spa"), ~30 options per season 
                   (30 because there are rotating calendars and also pre-season testing)
    2) Year     -> text input, 4-digit integer
    3) Session  -> dropdown, options depend on the weekend's format:
                     - conventional weekends: P1, P2, P3, Q, R
                     - sprint weekends:       P1, SQ, SR, Q, R

This script takes those three raw values, validates them, resolves them
to the identifiers the FastF1 API actually expects, and checks Supabase
for a `sessions` row matching that (year, round, session) first - see
supabase_store.py. If the session is already stored there, it's served
from Supabase and FastF1 is never hit for it again. Otherwise, the data
is fetched from FastF1, written out as CSV files locally (results, laps,
weather, car telemetry, car position, track status, session status, race
control messages, session info) and uploaded to Supabase.

Because a single session's data (especially telemetry) can be sizeable,
the local output directory for a given request is automatically deleted
DATA_TTL_SECONDS after ingestion - Supabase is the durable copy, the
local CSVs are just a staging/debug artifact of a given run.

There is no webpage yet, so this is runnable directly from the CLI:
    python fastF1_ingestion_script.py --year 2024 --weekend Bahrain --session R

--weekend all / --session all fan out over a whole season / whole weekend,
ingesting each session in turn (the dedup check skips what's already
stored, so an interrupted batch just resumes on re-run):
    python fastF1_ingestion_script.py --year 2024 --weekend Bahrain --session all
    python fastF1_ingestion_script.py --year 2024 --weekend all --session R

Add --force to re-ingest sessions that are already stored (each existing
Supabase copy is deleted first, then re-fetched). To see what's already
ingested, run list_sessions.py.
"""

import argparse
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

import fastf1
import pandas as pd

import supabase_store

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Constants mirroring the future webpage's dropdown options
# --------------------------------------------------------------------------

# The UI-facing session labels for each weekend format.
CONVENTIONAL_SESSIONS = ["P1", "P2", "P3", "Q", "R"]
SPRINT_SESSIONS = ["P1", "SQ", "SR", "Q", "R"]

# Passed as --weekend or --session to fan out over a whole season / whole
# weekend instead of one session. Not a real UI value - a CLI convenience.
BATCH_TOKEN = "all"

# FastF1 does not use the same short codes as the UI (e.g. it expects
# "FP1" instead of "P1", and calls the sprint race "S" instead of "SR"),
# so this maps each UI label to the identifier FastF1's API expects.
SESSION_ALIASES = {
    "P1": "FP1",
    "P2": "FP2",
    "P3": "FP3",
    "SQ": "SQ",  # Sprint Qualifying
    "SR": "S",   # Sprint (the sprint race itself)
    "Q": "Q",
    "R": "R",
}

# Earliest season FastF1 has reasonably complete session data for.
MIN_SUPPORTED_YEAR = 1950

# Where FastF1 caches raw API responses between runs (speeds up re-runs
# and is required by the library for session.load() to work reliably).
CACHE_DIR = Path(__file__).resolve().parent / ".fastf1_cache"

# Where the ingested CSVs are written, organised by year/round/session.
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"

# How long an ingested session's output directory is kept before it is
# automatically deleted.
DATA_TTL_SECONDS = 15 * 60


class InvalidInputError(ValueError):
    """Raised when one of the three UI inputs fails validation."""


# --------------------------------------------------------------------------
# Input validation - mirrors the constraints the webpage's dropdowns and
# text input will enforce, done here again since the backend can never
# trust the frontend's own validation.
# --------------------------------------------------------------------------

def validate_year(year: int) -> int:
    if not isinstance(year, int) or isinstance(year, bool):
        raise InvalidInputError(f"Year must be an integer, got {year!r}")
    if not (1000 <= year <= 9999):
        raise InvalidInputError(f"Year must be a 4-digit integer, got {year}")
    if year < MIN_SUPPORTED_YEAR:
        raise InvalidInputError(
            f"FastF1 has no data before {MIN_SUPPORTED_YEAR}, got {year}"
        )
    return year


def validate_weekend_name(weekend_name: str) -> str:
    if not isinstance(weekend_name, str) or not weekend_name.strip():
        raise InvalidInputError(f"Weekend must be a non-empty string, got {weekend_name!r}")
    return weekend_name.strip()


def validate_session_label(session_label: str, is_sprint_weekend: bool) -> str:
    session_label = session_label.strip().upper()
    allowed = SPRINT_SESSIONS if is_sprint_weekend else CONVENTIONAL_SESSIONS
    if session_label not in allowed:
        weekend_type = "sprint" if is_sprint_weekend else "conventional"
        raise InvalidInputError(
            f"'{session_label}' is not a valid session for a {weekend_type} "
            f"weekend. Valid options are: {allowed}"
        )
    return session_label


# --------------------------------------------------------------------------
# FastF1 lookups
# --------------------------------------------------------------------------

def _ensure_cache_enabled() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))


def get_event(year: int, weekend_name: str):
    """
    Resolve the weekend dropdown's value to a FastF1 event.

    FastF1 fuzzy-matches `weekend_name` against each event's country,
    location, name and official name, so values like "Bahrain",
    "Silverstone" or "Spa" all resolve correctly without needing an
    exact match against the official Grand Prix name.
    """
    try:
        return fastf1.get_event(year, weekend_name)
    except Exception as exc:
        raise InvalidInputError(
            f"Could not find a {year} event matching '{weekend_name}': {exc}"
        ) from exc


def is_sprint_weekend(event) -> bool:
    return event.EventFormat != "conventional"


def _season_weekend_names(year: int) -> list[str]:
    """Every non-testing event name for a season, in calendar order."""
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
    except Exception as exc:
        raise InvalidInputError(
            f"Could not load the {year} season schedule: {exc}"
        ) from exc
    names = [str(name).strip() for name in schedule["EventName"] if str(name).strip()]
    if not names:
        raise InvalidInputError(f"No non-testing events found for {year}.")
    return names


def _expand_targets(year: int, weekend_name: str, session_label: str) -> list[tuple[str, str]]:
    """
    Turn the (possibly 'all') weekend/session inputs into a concrete list
    of (weekend_name, session_label) pairs to ingest.

    - weekend == 'all'  -> every non-testing event that season
    - session == 'all'  -> every session label valid for that weekend's
                           format (conventional vs sprint)

    Session labels aren't validated here - ingest_session() does that per
    pair, so in a batch one bad combination just fails that pair.
    """
    weekend_all = weekend_name.strip().lower() == BATCH_TOKEN
    session_all = session_label.strip().lower() == BATCH_TOKEN

    weekend_names = _season_weekend_names(year) if weekend_all else [weekend_name.strip()]

    targets: list[tuple[str, str]] = []
    for name in weekend_names:
        if session_all:
            event = get_event(year, name)
            labels = SPRINT_SESSIONS if is_sprint_weekend(event) else CONVENTIONAL_SESSIONS
        else:
            labels = [session_label.strip()]
        targets.extend((name, label) for label in labels)
    return targets


def _slugify(name: str) -> str:
    """Turn an event name like 'Bahrain Grand Prix' into 'bahrain_grand_prix'."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _telemetry_dict_to_frame(telemetry_by_driver: dict) -> pd.DataFrame:
    """
    car_data / pos_data are {driver_number: Telemetry DataFrame} dicts, one
    frame per driver. Flatten that into a single long DataFrame (tagged with
    a DriverNumber column) so it can be written out as one CSV.

    FastF1's Telemetry frames always include a `SessionTime` column
    alongside `Time`, but the car_data/pos_data tables only define `Time`
    (SessionTime is redundant with it) - dropped here so the frame matches
    the destination table's columns exactly, the same way every other
    export category does.
    """
    if not telemetry_by_driver:
        raise fastf1.exceptions.DataNotLoadedError("no telemetry data available")
    frames = []
    for driver_number, telemetry in telemetry_by_driver.items():
        frame = telemetry.copy()
        frame.insert(0, "DriverNumber", driver_number)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).drop(columns=["SessionTime"], errors="ignore")


def _session_info_to_frame(session_info: dict) -> pd.DataFrame:
    """session_info is a single nested dict; flatten it into a one-row frame."""
    if not session_info:
        raise fastf1.exceptions.DataNotLoadedError("no session info available")
    return pd.json_normalize(session_info)


def _schedule_deletion(path: Path, delay_seconds: int = DATA_TTL_SECONDS) -> None:
    """
    Delete `path` after `delay_seconds`.

    This spawns a short-lived, detached background process (rather than a
    thread or timer) specifically so the cleanup still fires even after
    this script's own process has exited - the script is a one-shot CLI
    command today, not a long-running service.
    """
    cleanup_command = (
        f"import shutil, time; "
        f"time.sleep({delay_seconds}); "
        f"shutil.rmtree({str(path)!r}, ignore_errors=True)"
    )
    subprocess.Popen(
        [sys.executable, "-c", cleanup_command],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


# --------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------

# Every data category FastF1 exposes on a loaded Session, and how to turn
# each into a DataFrame. Evaluated lazily (inside the export loop's try
# block) so a category that failed to load doesn't take down the others.
SESSION_DATA_EXPORTS = (
    ("results", lambda s: s.results),
    ("laps", lambda s: s.laps),
    ("weather", lambda s: s.weather_data),
    ("car_data", lambda s: _telemetry_dict_to_frame(s.car_data)),
    ("pos_data", lambda s: _telemetry_dict_to_frame(s.pos_data)),
    ("track_status", lambda s: s.track_status),
    ("session_status", lambda s: s.session_status),
    ("race_control_messages", lambda s: s.race_control_messages),
    ("session_info", lambda s: _session_info_to_frame(s.session_info)),
)


def ingest_session(
    year: int,
    weekend_name: str,
    session_label: str,
    force: bool = False,
) -> Optional[Path]:
    """
    Validate the three UI inputs and resolve them to a FastF1 event/session.

    If that session is already stored in Supabase (checked via the
    `sessions` table - see supabase_store.py), FastF1 is not queried for
    its data at all and this returns None - unless `force` is set, in which
    case the existing copy is deleted and the session is re-fetched from
    scratch. Otherwise, the session is fetched from FastF1, every data
    category it loaded is written to CSV locally and uploaded to Supabase,
    and the local output directory is scheduled for deletion after
    DATA_TTL_SECONDS (Supabase is the durable copy from this point on).

    Returns the local output directory, or None if the session was already
    in Supabase and `force` was not set.
    """
    year = validate_year(year)
    weekend_name = validate_weekend_name(weekend_name)

    _ensure_cache_enabled()

    # Resolving the event only needs the (lightweight, cached) season
    # schedule, not a full session load, so the dedup check below is cheap
    # even when the session turns out to already be in Supabase.
    event = get_event(year, weekend_name)
    sprint_weekend = is_sprint_weekend(event)
    session_label = validate_session_label(session_label, sprint_weekend)

    existing = supabase_store.find_session(year, event.RoundNumber, session_label)
    if existing is not None:
        if not force:
            logger.info(
                "%s %s %s is already in Supabase (session_id=%s); skipping the "
                "FastF1 fetch. Pass --force to re-ingest it.",
                year, event.EventName, session_label, existing["id"],
            )
            return None
        logger.info(
            "%s %s %s is already in Supabase (session_id=%s); --force set, "
            "deleting it and re-ingesting.",
            year, event.EventName, session_label, existing["id"],
        )
        supabase_store.delete_session(existing["id"])

    fastf1_identifier = SESSION_ALIASES[session_label]
    session = event.get_session(fastf1_identifier)
    session.load()

    # Named after the resolved event (not the raw, possibly-fuzzy input)
    # so that e.g. "Spa" and "Belgium" land in the same output folder.
    event_dir = f"{event.RoundNumber:02d}_{_slugify(event.EventName)}"
    out_dir = OUTPUT_DIR / str(year) / event_dir / session_label
    out_dir.mkdir(parents=True, exist_ok=True)

    # FastF1 frequently has partial data available (e.g. weather data is
    # missing for some sessions, older seasons lack telemetry), so each
    # export is best-effort rather than failing the whole ingestion. Each
    # getter is only called *inside* the try block so a category that
    # failed to load doesn't blow up before we even get there.
    data_frames = {}
    for name, get_frame in SESSION_DATA_EXPORTS:
        try:
            frame = get_frame(session)
            frame.to_csv(out_dir / f"{name}.csv", index=False)
            data_frames[name] = frame
        except (fastf1.exceptions.DataNotLoadedError, AttributeError):
            logger.warning("'%s' data was not available for this session, skipping.", name)

    # session_info is stored in Supabase as the raw nested dict (JSONB),
    # not the flattened one-row frame written to session_info.csv above.
    try:
        raw_session_info = session.session_info
    except fastf1.exceptions.DataNotLoadedError:
        raw_session_info = None

    # The generic per-category CSV export loop above already wrote
    # session_info.csv from the flattened frame; drop it from what gets
    # uploaded to Supabase since it's handled separately via raw_session_info.
    data_frames.pop("session_info", None)

    session_id = supabase_store.store_session(
        year=year,
        round_number=event.RoundNumber,
        event_name=event.EventName,
        session_label=session_label,
        data_frames=data_frames,
        session_info=raw_session_info,
    )
    logger.info("Stored as session_id=%s in Supabase.", session_id)

    _schedule_deletion(out_dir)

    return out_dir


# --------------------------------------------------------------------------
# Orchestration - one session, or a whole weekend / season via 'all'
# --------------------------------------------------------------------------

def run_ingestion(
    year: int,
    weekend_name: str,
    session_label: str,
    force: bool = False,
) -> int:
    """
    Ingest one session, or - when weekend and/or session is 'all' - fan
    out over a whole weekend / whole season, ingesting each in turn.

    A single explicit session keeps the original chatty output and lets
    InvalidInputError / RuntimeError propagate (so the CLI can map them to
    exit 2 / exit 1). A batch run instead reports one status line per
    session and keeps going when one fails - the dedup check makes a
    re-run skip whatever already landed, so an interrupted batch is just
    resumed. A configuration/storage failure (RuntimeError) still aborts
    the batch, since it's not a per-session problem.

    Returns a process exit code: 0 if nothing failed, 1 otherwise.
    """
    year = validate_year(year)
    _ensure_cache_enabled()

    is_batch = BATCH_TOKEN in (weekend_name.strip().lower(), session_label.strip().lower())
    targets = _expand_targets(year, weekend_name, session_label)

    if not is_batch:
        (only_weekend, only_session) = targets[0]
        out_dir = ingest_session(year, only_weekend, only_session, force=force)
        if out_dir is not None:
            logger.info(
                "Ingested %s %s session %s -> %s",
                year, only_weekend, only_session.upper(), out_dir,
            )
            logger.info(
                "Local copy will be automatically deleted in %s minutes; the "
                "data itself now lives in Supabase.", DATA_TTL_SECONDS // 60,
            )
        return 0

    logger.info(
        "Batch ingest: %s session(s) for %s%s.",
        len(targets), year, " (--force)" if force else "",
    )
    stored = skipped = failed = 0
    for weekend, session in targets:
        logger.info("=== %s %s %s ===", year, weekend, session)
        try:
            out_dir = ingest_session(year, weekend, session, force=force)
        except RuntimeError as exc:
            logger.error("[FAIL] %s %s %s: %s", year, weekend, session, exc)
            logger.error(
                "Aborting batch: this looks like a configuration or storage "
                "problem, not a per-session one."
            )
            return 1
        except Exception as exc:  # noqa: BLE001 - one bad session must not stop the batch
            failed += 1
            logger.error("[FAIL] %s %s %s: %s", year, weekend, session, exc)
            continue
        if out_dir is None:
            skipped += 1
        else:
            stored += 1

    logger.info(
        "Batch done: %s stored, %s already present, %s failed.",
        stored, skipped, failed,
    )
    return 1 if failed else 0


# --------------------------------------------------------------------------
# CLI entry point - stands in for the webpage until it exists.
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch and ingest F1 session data via FastF1 - one session, "
                    "or a whole weekend / season with --weekend all / --session all."
    )
    parser.add_argument("--year", type=int, required=True, help="4-digit season year, e.g. 2024")
    parser.add_argument(
        "--weekend",
        type=str,
        required=True,
        dest="weekend_name",
        help="Weekend/Grand Prix name (e.g. 'Bahrain', 'Silverstone', 'Spa'), "
             "or 'all' for every event that season",
    )
    parser.add_argument(
        "--session",
        type=str,
        required=True,
        help="Session label: P1/P2/P3/Q/R (conventional) or P1/SQ/SR/Q/R (sprint), "
             "or 'all' for every session that weekend",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest even if a session is already in Supabase "
             "(deletes the existing copy first, then re-fetches from FastF1).",
    )
    args = parser.parse_args()

    # This module's own progress/warnings (and supabase_store's) go through
    # logging; message-only format so it reads like plain output. Root stays
    # at WARNING so third-party INFO chatter (httpx logging every Supabase
    # request URL, etc.) is suppressed - only our two loggers are boosted to
    # INFO. FastF1 configures its own `fastf1` logger separately.
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logger.setLevel(logging.INFO)
    logging.getLogger("supabase_store").setLevel(logging.INFO)

    try:
        exit_code = run_ingestion(
            args.year, args.weekend_name, args.session, force=args.force
        )
    except InvalidInputError as exc:
        # A bad CLI value - show usage alongside the message.
        parser.error(str(exc))
        return
    except RuntimeError as exc:
        # A runtime/config/storage failure (missing env vars, a rolled-back
        # partial upload, etc.) - the args were fine, so usage text would
        # just be noise. supabase_store.SupabaseStoreError lands here too.
        logger.error("Error: %s", exc)
        sys.exit(1)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
