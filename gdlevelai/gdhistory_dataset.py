from __future__ import annotations

import hashlib
import json
import random
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


BASE_URL = "https://history.geometrydash.eu"

APPROX_VERSION_THRESHOLDS_DESC: list[tuple[int, str]] = [
    (1697673600, "2.2"),  # 2023-10-19
    (1484784000, "2.1"),  # 2017-01-19
    (1439942400, "2.0"),  # 2015-08-19
    (1414886400, "1.9"),  # 2014-11-02
    (1408492800, "1.8"),  # 2014-08-20
    (1400112000, "1.7"),  # 2014-05-15
    (1394496000, "1.6"),  # 2014-03-11
    (1389916800, "1.5"),  # 2014-01-17
    (1385856000, "1.4"),  # 2013-12-01
    (1382659200, "1.3"),  # 2013-10-25
    (1379808000, "1.2"),  # 2013-09-22
    (1378684800, "1.1"),  # 2013-09-09
]

APPROX_VERSION_ONLINE_ID_THRESHOLDS_DESC: list[tuple[int, str]] = [
    (8_000_000, "2.2"),
    (500_000, "2.1"),
    (150_000, "2.0"),
    (80_000, "1.9"),
    (45_000, "1.8"),
    (25_000, "1.7"),
    (13_000, "1.6"),
    (8_000, "1.5"),
    (4_000, "1.4"),
    (2_000, "1.3"),
    (900, "1.2"),
    (300, "1.1"),
]


def _log(message: str) -> None:
    print(f"[gdhistory_dataset] {message}", flush=True)


def _to_unix_seconds(value: object) -> int | None:
    if value is None:
        return None
    try:
        ts = int(float(value))
    except (TypeError, ValueError):
        return None
    if ts <= 0:
        return None
    if ts > 10**12:
        ts //= 1000
    return ts


def _sleep_with_cancel(
    seconds: float,
    stop_event: threading.Event | None,
) -> None:
    if seconds <= 0:
        return
    if stop_event is None:
        time.sleep(seconds)
        return

    deadline = time.monotonic() + seconds
    while True:
        if stop_event.is_set():
            raise InterruptedError("Fetch interrupted by user")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        stop_event.wait(timeout=min(0.25, remaining))


def _parse_version_value(value: object) -> float | None:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if num <= 0:
        return None
    if num >= 10 and num <= 99 and abs(num - round(num)) < 1e-9:
        num = num / 10.0
    return num


def infer_approx_version(
    *,
    online_id: int,
    submitted_timestamp: object,
    hit: dict[str, Any] | None = None,
) -> str:
    if hit is not None:
        for key in ("cache_game_version", "cache_version", "game_version", "version"):
            if key not in hit:
                continue
            v = _parse_version_value(hit[key])
            if v is None:
                continue
            if v >= 2.2:
                return "2.2"
            if v >= 2.1:
                return "2.1"
            if v >= 2.0:
                return "2.0"
            if v >= 1.9:
                return "1.9"
            if v >= 1.8:
                return "1.8"
            if v >= 1.7:
                return "1.7"
            if v >= 1.6:
                return "1.6"
            if v >= 1.5:
                return "1.5"
            if v >= 1.4:
                return "1.4"
            if v >= 1.3:
                return "1.3"
            if v >= 1.2:
                return "1.2"
            if v >= 1.1:
                return "1.1"
            return "1.0"

    for threshold, version in APPROX_VERSION_ONLINE_ID_THRESHOLDS_DESC:
        if online_id >= threshold:
            return version

    ts = _to_unix_seconds(submitted_timestamp)
    if ts is not None:
        for threshold, version in APPROX_VERSION_THRESHOLDS_DESC:
            if ts >= threshold:
                return version
        return "1.0"

    return "1.0"


def extract_reported_version(hit: dict[str, Any] | None) -> str | None:
    if hit is None:
        return None
    for key in (
        "cache_game_version",
        "cache_version",
        "game_version",
        "version",
    ):
        if key not in hit:
            continue
        value = hit.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        try:
            num = float(text)
            if num <= 0:
                continue
            if num >= 10 and num <= 99 and abs(num - round(num)) < 1e-9:
                decoded = num / 10.0
                return f"{decoded:.1f}"
            return f"{num:.2f}".rstrip("0").rstrip(".")
        except (TypeError, ValueError):
            return text
    return None


POLITENESS_PROFILES: dict[str, dict[str, float | int]] = {
    "normal": {
        "delay_seconds": 1.5,
        "jitter_seconds": 0.4,
        "batch_pause_every": 35,
        "batch_pause_seconds": 6.0,
        "max_retries": 4,
        "max_requests_per_hour": 1200,
        "cooldown_seconds_on_budget": 90,
    },
    "careful": {
        "delay_seconds": 2.5,
        "jitter_seconds": 0.75,
        "batch_pause_every": 25,
        "batch_pause_seconds": 12.0,
        "max_retries": 5,
        "max_requests_per_hour": 900,
        "cooldown_seconds_on_budget": 180,
    },
    "very_careful": {
        "delay_seconds": 4.0,
        "jitter_seconds": 1.2,
        "batch_pause_every": 20,
        "batch_pause_seconds": 20.0,
        "max_retries": 6,
        "max_requests_per_hour": 600,
        "cooldown_seconds_on_budget": 300,
    },
}


@dataclass
class FetchConfig:
    output_dir: Path
    state_db: Path
    metadata_jsonl: Path
    limit_per_page: int = 200
    max_levels: int = 200
    min_featured_score: int = 1
    min_epic_tier: int = 0
    delay_seconds: float = 2.5
    jitter_seconds: float = 0.75
    max_retries: int = 5
    batch_pause_every: int = 25
    batch_pause_seconds: float = 12.0
    request_timeout: int = 30
    politeness_profile: str = "careful"
    max_requests_per_hour: int = 900
    cooldown_seconds_on_budget: float = 180.0
    global_limiter_db: Path | None = None
    shard_index: int = 0
    shard_count: int = 1
    start_online_id: int | None = None
    resume_from_last_downloaded: bool = True
    workers: int = 3
    level_record_page_size: int = 40
    level_record_max_pages: int = 6
    auto_tune: bool = False
    auto_min_workers: int = 1
    auto_max_workers: int = 6
    auto_target_seconds_per_level: float = 8.0

    def apply_profile(self) -> None:
        _log(f"Applying politeness profile '{self.politeness_profile}'")
        if self.politeness_profile not in POLITENESS_PROFILES:
            valid = ", ".join(sorted(POLITENESS_PROFILES))
            raise ValueError(
                f"Invalid politeness profile '{self.politeness_profile}'. Use one of: {valid}"
            )
        profile = POLITENESS_PROFILES[self.politeness_profile]

        if self.delay_seconds <= 0:
            self.delay_seconds = float(profile["delay_seconds"])
        if self.jitter_seconds <= 0:
            self.jitter_seconds = float(profile["jitter_seconds"])
        if self.batch_pause_every <= 0:
            self.batch_pause_every = int(profile["batch_pause_every"])
        if self.batch_pause_seconds <= 0:
            self.batch_pause_seconds = float(profile["batch_pause_seconds"])
        if self.max_retries <= 0:
            self.max_retries = int(profile["max_retries"])

        if self.max_requests_per_hour <= 0:
            self.max_requests_per_hour = int(profile["max_requests_per_hour"])
        if self.cooldown_seconds_on_budget <= 0:
            self.cooldown_seconds_on_budget = float(
                profile["cooldown_seconds_on_budget"]
            )

    def validate(self) -> None:
        _log(
            "Validating fetch config "
            f"(shard_index={self.shard_index}, shard_count={self.shard_count})"
        )
        if self.shard_count < 1:
            raise ValueError("shard_count must be >= 1")
        if self.shard_index < 0 or self.shard_index >= self.shard_count:
            raise ValueError("shard_index must be in range [0, shard_count)")
        if self.workers < 1:
            raise ValueError("workers must be >= 1")
        if self.level_record_page_size < 1:
            raise ValueError("level_record_page_size must be >= 1")
        if self.level_record_max_pages < 1:
            raise ValueError("level_record_max_pages must be >= 1")
        if self.auto_min_workers < 1:
            raise ValueError("auto_min_workers must be >= 1")
        if self.auto_max_workers < self.auto_min_workers:
            raise ValueError("auto_max_workers must be >= auto_min_workers")
        if self.auto_target_seconds_per_level <= 0:
            raise ValueError("auto_target_seconds_per_level must be > 0")


class GlobalRateLimiter:
    def __init__(
        self,
        db_path: Path,
        max_requests_per_hour: int,
        cooldown_seconds: float,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.db_path = db_path
        self.max_requests_per_hour = max_requests_per_hour
        self.cooldown_seconds = cooldown_seconds
        self.stop_event = stop_event

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            isolation_level=None,
            check_same_thread=False,
        )
        self._lock = threading.Lock()
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS request_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at REAL NOT NULL
            )
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS request_log_created_at_idx ON request_log(created_at)"
        )
        _log(
            "Initialized global limiter "
            f"(db={self.db_path}, max_requests_per_hour={self.max_requests_per_hour}, cooldown={self.cooldown_seconds}s)"
        )

    def close(self) -> None:
        _log("Closing global limiter")
        self.conn.close()

    def acquire(self) -> None:
        if self.max_requests_per_hour <= 0:
            return

        window_seconds = 3600.0
        while True:
            if self.stop_event is not None and self.stop_event.is_set():
                raise InterruptedError("Global limiter acquire interrupted")
            with self._lock:
                now = time.time()
                self.conn.execute("BEGIN IMMEDIATE")
                try:
                    cutoff = now - window_seconds
                    self.conn.execute(
                        "DELETE FROM request_log WHERE created_at < ?", (cutoff,)
                    )

                    row = self.conn.execute(
                        "SELECT COUNT(*), MIN(created_at) FROM request_log"
                    ).fetchone()
                    count = int(row[0] or 0)
                    oldest = float(row[1]) if row[1] is not None else None

                    if count < self.max_requests_per_hour:
                        self.conn.execute(
                            "INSERT INTO request_log(created_at) VALUES (?)", (now,)
                        )
                        self.conn.execute("COMMIT")
                        return

                    self.conn.execute("COMMIT")
                    wait_for_budget = 0.0
                    if oldest is not None:
                        wait_for_budget = max(0.0, (oldest + window_seconds) - now)
                    sleep_for = max(self.cooldown_seconds, wait_for_budget)
                except Exception:
                    self.conn.execute("ROLLBACK")
                    raise

                _log(
                    "Global request budget reached; "
                    f"sleeping {sleep_for:.2f}s before retrying"
                )

            _sleep_with_cancel(sleep_for, self.stop_event)


class GDHistoryClient:
    def __init__(
        self,
        config: FetchConfig,
        limiter: GlobalRateLimiter | None = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.config = config
        self.limiter = limiter
        self.stop_event = stop_event
        self._last_request_ts = 0.0
        self._user_agent = "gdlevelai-dataset-fetcher/0.1 (+careful-rolloff)"

    def _throttle(self) -> None:
        now = time.monotonic()
        target_gap = self.config.delay_seconds + random.uniform(
            0.0, self.config.jitter_seconds
        )
        elapsed = now - self._last_request_ts
        if elapsed < target_gap:
            sleep_for = target_gap - elapsed
            _log(f"Throttling request for {sleep_for:.2f}s")
            _sleep_with_cancel(sleep_for, self.stop_event)

    def _request(
        self, path: str, *, params: dict[str, Any] | None = None, as_json: bool = True
    ) -> Any:
        if params:
            query = urlencode(params)
            url = f"{BASE_URL}{path}?{query}"
        else:
            url = f"{BASE_URL}{path}"

        attempt = 0
        while True:
            if self.stop_event is not None and self.stop_event.is_set():
                raise InterruptedError("Request loop interrupted")
            if self.limiter is not None:
                self.limiter.acquire()
            self._throttle()
            _log(f"Requesting {url} (attempt {attempt + 1})")
            req = Request(url, headers={"User-Agent": self._user_agent})
            try:
                with urlopen(req, timeout=self.config.request_timeout) as response:
                    data = response.read()
                    self._last_request_ts = time.monotonic()
                    _log(f"Received response from {url} ({len(data)} bytes)")
                    if not as_json:
                        return data.decode("utf-8", errors="replace")
                    return json.loads(data.decode("utf-8"))
            except HTTPError as exc:
                code = exc.code
                if attempt >= self.config.max_retries or code not in (
                    429,
                    500,
                    502,
                    503,
                    504,
                ):
                    raise
            except URLError:
                if attempt >= self.config.max_retries:
                    raise

            attempt += 1
            backoff = min(30.0, (2**attempt) + random.uniform(0.0, 1.5))
            _log(f"Request failed; retrying after {backoff:.2f}s")
            _sleep_with_cancel(backoff, self.stop_event)

    def search_levels(
        self,
        *,
        limit: int,
        offset: int,
        min_featured_score: int,
        min_epic_tier: int,
        min_online_id: int | None,
    ) -> dict[str, Any]:
        filters = [
            f"cache_featured >= {min_featured_score}",
            "cache_level_string_available = true",
            "cache_search_available = true",
        ]
        if min_online_id is not None:
            filters.append(f"online_id > {min_online_id}")
        if min_epic_tier > 0:
            filters.append(f"cache_epic >= {min_epic_tier}")

        _log(
            "Searching levels "
            f"(limit={limit}, offset={offset}, min_online_id={min_online_id}, min_featured={min_featured_score}, min_epic={min_epic_tier})"
        )
        return self._request(
            "/api/v1/search/level/advanced/",
            params={
                "limit": str(limit),
                "offset": str(offset),
                "sort": "online_id:asc",
                "filter": " AND ".join(filters),
            },
        )

    def level_info(self, online_id: int) -> dict[str, Any]:
        _log(f"Fetching level info for online_id={online_id}")
        return self._request(f"/api/v1/level/{online_id}/")

    def level_records_page(
        self, online_id: int, *, start_from: int | None, count: int
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"count": str(count)}
        if start_from is not None:
            params["start_from"] = str(start_from)
        _log(
            f"Fetching level records page online_id={online_id}, "
            f"start_from={start_from}, count={count}"
        )
        return self._request(f"/api/v1/level/{online_id}/", params=params)

    def download_record_gmd(self, online_id: int, record_id: int) -> str:
        _log(f"Downloading record .gmd online_id={online_id}, record_id={record_id}")
        return self._request(f"/level/{online_id}/{record_id}/download/", as_json=False)


class FetchState:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fetched_levels (
                online_id INTEGER PRIMARY KEY,
                record_id INTEGER,
                status TEXT NOT NULL,
                file_path TEXT,
                sha256 TEXT,
                file_bytes INTEGER,
                message TEXT,
                fetched_at TEXT NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS level_index (
                online_id INTEGER PRIMARY KEY,
                level_name TEXT,
                creator TEXT,
                featured INTEGER,
                epic_tier INTEGER,
                submitted_timestamp INTEGER,
                level_string_available INTEGER,
                is_public INTEGER,
                is_deleted INTEGER,
                reported_version TEXT,
                approx_version TEXT,
                download_status TEXT,
                last_seen_at TEXT NOT NULL
            )
            """
        )
        self._ensure_level_index_schema()
        self.conn.commit()
        _log(f"Initialized fetch state DB at {self.db_path}")

    def _ensure_level_index_schema(self) -> None:
        rows = self.conn.execute("PRAGMA table_info(level_index)").fetchall()
        columns = {str(row[1]) for row in rows}
        if "reported_version" not in columns:
            _log("Applying schema migration: add level_index.reported_version")
            self.conn.execute(
                "ALTER TABLE level_index ADD COLUMN reported_version TEXT"
            )
        if "approx_version" not in columns:
            _log("Applying schema migration: add level_index.approx_version")
            self.conn.execute("ALTER TABLE level_index ADD COLUMN approx_version TEXT")

        _log("Backfilling level_index.approx_version for missing rows")
        missing_rows = self.conn.execute(
            """
            SELECT online_id, submitted_timestamp, reported_version
            FROM level_index
            WHERE approx_version IS NULL OR TRIM(approx_version) = ''
            """
        ).fetchall()
        if missing_rows:
            updates = [
                (
                    infer_approx_version(
                        online_id=int(row[0]),
                        submitted_timestamp=row[1],
                        hit=(
                            {"cache_game_version": row[2]}
                            if row[2] is not None and str(row[2]).strip()
                            else None
                        ),
                    ),
                    int(row[0]),
                )
                for row in missing_rows
            ]
            self.conn.executemany(
                "UPDATE level_index SET approx_version = ? WHERE online_id = ?",
                updates,
            )

    def close(self) -> None:
        _log("Closing fetch state DB")
        self.conn.close()

    def has_level(self, online_id: int) -> bool:
        row = self.conn.execute(
            "SELECT online_id FROM fetched_levels WHERE online_id = ?", (online_id,)
        ).fetchone()
        return row is not None

    def is_downloaded(self, online_id: int) -> bool:
        row = self.conn.execute(
            "SELECT online_id FROM fetched_levels WHERE online_id = ? AND status = 'ok'",
            (online_id,),
        ).fetchone()
        return row is not None

    def get_last_downloaded_online_id(self) -> int | None:
        row = self.conn.execute(
            "SELECT MAX(online_id) FROM fetched_levels WHERE status = 'ok'"
        ).fetchone()
        if row is None or row[0] is None:
            _log("No previously downloaded levels found in state DB")
            return None
        _log(f"Last downloaded online_id in DB is {int(row[0])}")
        return int(row[0])

    def upsert_level_hit(
        self, hit: dict[str, Any], download_status: str = "seen"
    ) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        online_id = int(hit["online_id"])
        submitted_ts = hit.get("cache_submitted_timestamp")
        reported_version = extract_reported_version(hit)
        approx_version = infer_approx_version(
            online_id=online_id,
            submitted_timestamp=submitted_ts,
            hit=hit,
        )
        self.conn.execute(
            """
            INSERT INTO level_index (
                online_id, level_name, creator, featured, epic_tier,
                submitted_timestamp, level_string_available, is_public,
                is_deleted, reported_version, approx_version, download_status, last_seen_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(online_id) DO UPDATE SET
                level_name=excluded.level_name,
                creator=excluded.creator,
                featured=excluded.featured,
                epic_tier=excluded.epic_tier,
                submitted_timestamp=excluded.submitted_timestamp,
                level_string_available=excluded.level_string_available,
                is_public=excluded.is_public,
                is_deleted=excluded.is_deleted,
                reported_version=excluded.reported_version,
                approx_version=excluded.approx_version,
                download_status=excluded.download_status,
                last_seen_at=excluded.last_seen_at
            """,
            (
                online_id,
                hit.get("cache_level_name"),
                hit.get("cache_username"),
                int(hit.get("cache_featured") or 0),
                int(hit.get("cache_epic") or 0),
                submitted_ts,
                1 if hit.get("cache_level_string_available") else 0,
                1 if hit.get("is_public") else 0,
                1 if hit.get("is_deleted") else 0,
                reported_version,
                approx_version,
                download_status,
                now_iso,
            ),
        )
        self.conn.commit()

    def mark(
        self,
        *,
        online_id: int,
        record_id: int | None,
        status: str,
        file_path: Path | None,
        sha256_hex: str | None,
        file_bytes: int | None,
        message: str,
    ) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        approx_version = infer_approx_version(
            online_id=online_id,
            submitted_timestamp=None,
            hit=None,
        )
        self.conn.execute(
            """
            INSERT INTO fetched_levels (online_id, record_id, status, file_path, sha256, file_bytes, message, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(online_id) DO UPDATE SET
                record_id=excluded.record_id,
                status=excluded.status,
                file_path=excluded.file_path,
                sha256=excluded.sha256,
                file_bytes=excluded.file_bytes,
                message=excluded.message,
                fetched_at=excluded.fetched_at
            """,
            (
                online_id,
                record_id,
                status,
                str(file_path) if file_path else None,
                sha256_hex,
                file_bytes,
                message,
                now_iso,
            ),
        )
        self.conn.execute(
            """
            INSERT INTO level_index (
                online_id, reported_version, approx_version, download_status, last_seen_at
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(online_id) DO UPDATE SET
                reported_version=COALESCE(level_index.reported_version, excluded.reported_version),
                approx_version=COALESCE(level_index.approx_version, excluded.approx_version),
                download_status=excluded.download_status,
                last_seen_at=excluded.last_seen_at
            """,
            (online_id, None, approx_version, status, now_iso),
        )
        self.conn.commit()

    def update_version_info(
        self,
        *,
        online_id: int,
        reported_version: str | None,
        approx_version: str | None,
    ) -> None:
        self.conn.execute(
            """
            UPDATE level_index
            SET
                reported_version = COALESCE(?, reported_version),
                approx_version = COALESCE(?, approx_version)
            WHERE online_id = ?
            """,
            (reported_version, approx_version, online_id),
        )
        self.conn.commit()


def _pick_record(level_response: dict[str, Any]) -> dict[str, Any] | None:
    records = level_response.get("records", [])
    candidates = [r for r in records if r.get("level_string_available")]
    if not candidates:
        _log("No record with level string available for this level")
        return None
    candidates.sort(
        key=lambda r: (r.get("cache_real_date") or "9999", r.get("id") or 10**18)
    )
    return candidates[0]


def _find_record_with_level_string(
    client: GDHistoryClient, online_id: int, config: FetchConfig
) -> tuple[int | None, str, int, str | None]:
    start_from: int | None = None
    pages_checked = 0
    best_reported_version: str | None = None

    while pages_checked < config.level_record_max_pages:
        pages_checked += 1
        response = client.level_records_page(
            online_id,
            start_from=start_from,
            count=config.level_record_page_size,
        )
        records = response.get("records", [])
        if not records:
            return None, "No records returned", pages_checked

        for record in records:
            if best_reported_version is None and isinstance(record, dict):
                best_reported_version = extract_reported_version(record)
            if record.get("level_string_available"):
                record_id = record.get("id")
                if record_id is not None:
                    record_reported_version = (
                        extract_reported_version(record)
                        if isinstance(record, dict)
                        else None
                    )
                    return (
                        int(record_id),
                        "ok",
                        pages_checked,
                        record_reported_version or best_reported_version,
                    )

        last_id = records[-1].get("id")
        if last_id is None or len(records) < config.level_record_page_size:
            return (
                None,
                "No record with level string",
                pages_checked,
                best_reported_version,
            )

        start_from = int(last_id)

    return None, "Reached record page scan limit", pages_checked, best_reported_version


def _apply_auto_tuning(
    config: FetchConfig, batch_stats: dict[str, float | int]
) -> None:
    if not config.auto_tune:
        return

    attempted = int(batch_stats.get("attempted", 0))
    if attempted == 0:
        return

    failed = int(batch_stats.get("failed", 0))
    avg_elapsed = float(batch_stats.get("avg_elapsed_seconds", 0.0))
    avg_pages = float(batch_stats.get("avg_pages_scanned", 1.0))

    old_workers = config.workers
    old_page_size = config.level_record_page_size
    old_limit = config.limit_per_page

    if failed > 0:
        config.workers = max(config.auto_min_workers, config.workers - 1)
        config.limit_per_page = max(50, config.limit_per_page - 50)
    else:
        if (
            avg_elapsed > 0
            and avg_elapsed < config.auto_target_seconds_per_level * 0.75
        ):
            config.workers = min(config.auto_max_workers, config.workers + 1)
        elif avg_elapsed > config.auto_target_seconds_per_level * 1.5:
            config.workers = max(config.auto_min_workers, config.workers - 1)

    if avg_pages >= 3.0:
        config.level_record_page_size = min(100, config.level_record_page_size + 10)
    elif avg_pages <= 1.2:
        config.level_record_page_size = max(20, config.level_record_page_size - 5)

    if failed == 0 and config.workers >= 3:
        config.limit_per_page = min(500, config.limit_per_page + 50)

    if (
        old_workers != config.workers
        or old_page_size != config.level_record_page_size
        or old_limit != config.limit_per_page
    ):
        _log(
            "Auto-tune adjusted config: "
            f"workers {old_workers}->{config.workers}, "
            f"level_record_page_size {old_page_size}->{config.level_record_page_size}, "
            f"limit_per_page {old_limit}->{config.limit_per_page}, "
            f"batch avg_elapsed={avg_elapsed:.2f}s, avg_pages={avg_pages:.2f}, failed={failed}"
        )


def _download_single_level(
    *,
    online_id: int,
    hit: dict[str, Any],
    config: FetchConfig,
    limiter: GlobalRateLimiter,
    stop_event: threading.Event | None = None,
) -> dict[str, Any]:
    started_at = time.monotonic()
    client = GDHistoryClient(config, limiter=limiter, stop_event=stop_event)
    record_id, reason, pages_scanned, record_reported_version = (
        _find_record_with_level_string(client, online_id, config)
    )

    hit_reported_version = extract_reported_version(hit)
    reported_version = record_reported_version or hit_reported_version
    if reported_version is None:
        try:
            info = client.level_info(online_id)
            reported_version = extract_reported_version(info)
        except Exception:
            reported_version = None

    approx_version = infer_approx_version(
        online_id=online_id,
        submitted_timestamp=hit.get("cache_submitted_timestamp"),
        hit=(
            {"cache_game_version": reported_version}
            if reported_version is not None
            else hit
        ),
    )

    if record_id is None:
        return {
            "online_id": online_id,
            "status": "skipped",
            "message": reason,
            "elapsed_seconds": time.monotonic() - started_at,
            "pages_scanned": pages_scanned,
            "reported_version": reported_version,
            "approx_version": approx_version,
        }

    gmd_text = client.download_record_gmd(online_id, record_id)
    gmd_bytes = gmd_text.encode("utf-8")
    sha256_hex = hashlib.sha256(gmd_bytes).hexdigest()

    out_name = f"{online_id}_{record_id}.gmd"
    out_path = config.output_dir / out_name
    out_path.write_bytes(gmd_bytes)

    metadata = {
        "online_id": online_id,
        "record_id": record_id,
        "level_name": hit.get("cache_level_name"),
        "creator": hit.get("cache_username"),
        "reported_version": reported_version,
        "featured": hit.get("cache_featured"),
        "epic_tier": hit.get("cache_epic"),
        "submitted_timestamp": hit.get("cache_submitted_timestamp"),
        "approx_version": approx_version,
        "path": str(out_path),
        "sha256": sha256_hex,
        "bytes": len(gmd_bytes),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }

    return {
        "online_id": online_id,
        "record_id": record_id,
        "status": "ok",
        "path": out_path,
        "sha256": sha256_hex,
        "bytes": len(gmd_bytes),
        "metadata": metadata,
        "reported_version": reported_version,
        "approx_version": approx_version,
        "elapsed_seconds": time.monotonic() - started_at,
        "pages_scanned": pages_scanned,
    }


def _write_metadata_line(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    _log(f"Appended metadata for online_id={payload.get('online_id')}")


def migrate_fetch_state_db(
    state_db: Path,
    metadata_jsonl: Path | None = None,
) -> dict[str, Any]:
    _log(f"Running fetch-state migration for {state_db}")
    state = FetchState(state_db)
    try:
        metadata_reported_updates = 0
        if metadata_jsonl is not None and metadata_jsonl.exists():
            _log(f"Backfilling reported_version from metadata file {metadata_jsonl}")
            updates: list[tuple[str, int]] = []
            with metadata_jsonl.open("r", encoding="utf-8", errors="ignore") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    online_id_val = obj.get("online_id")
                    if online_id_val is None:
                        continue
                    try:
                        online_id = int(online_id_val)
                    except (TypeError, ValueError):
                        continue
                    reported = obj.get("reported_version")
                    if reported is None:
                        continue
                    reported_text = str(reported).strip()
                    if not reported_text:
                        continue
                    updates.append((reported_text, online_id))

            if updates:
                state.conn.executemany(
                    """
                    UPDATE level_index
                    SET reported_version = ?
                    WHERE online_id = ?
                    """,
                    updates,
                )
                state.conn.commit()
                metadata_reported_updates = len(updates)

        rows_to_recompute = state.conn.execute(
            "SELECT online_id, submitted_timestamp, reported_version, approx_version FROM level_index"
        ).fetchall()
        updates: list[tuple[str, int]] = []
        changed = 0
        for row in rows_to_recompute:
            online_id = int(row[0])
            submitted_timestamp = row[1]
            reported_version = row[2]
            old_version = str(row[3]) if row[3] is not None else ""
            hit_hint: dict[str, Any] | None = None
            if reported_version is not None and str(reported_version).strip():
                hit_hint = {"cache_game_version": reported_version}
            new_version = infer_approx_version(
                online_id=online_id,
                submitted_timestamp=submitted_timestamp,
                hit=hit_hint,
            )
            updates.append((new_version, online_id))
            if old_version != new_version:
                changed += 1

        if updates:
            state.conn.executemany(
                "UPDATE level_index SET approx_version = ? WHERE online_id = ?",
                updates,
            )
            state.conn.commit()

        rows = state.conn.execute(
            """
            SELECT COALESCE(approx_version, '[null]') AS approx_version, COUNT(*)
            FROM level_index
            GROUP BY COALESCE(approx_version, '[null]')
            ORDER BY COUNT(*) DESC, approx_version ASC
            """
        ).fetchall()
        distribution = {str(r[0]): int(r[1]) for r in rows}
        total = int(sum(distribution.values()))
        reported_rows = int(
            state.conn.execute(
                "SELECT COUNT(*) FROM level_index WHERE reported_version IS NOT NULL AND TRIM(reported_version) <> ''"
            ).fetchone()[0]
        )
        return {
            "state_db": str(state_db),
            "ok": True,
            "level_index_rows": total,
            "rows_recomputed": len(rows_to_recompute),
            "rows_changed": changed,
            "reported_version_updates_from_metadata": metadata_reported_updates,
            "reported_version_rows": reported_rows,
            "approx_version_distribution": distribution,
        }
    finally:
        state.close()


def download_featured_dataset(config: FetchConfig) -> dict[str, int]:
    _log("Starting featured dataset download")
    config.apply_profile()
    config.validate()

    config.output_dir.mkdir(parents=True, exist_ok=True)

    stop_event = threading.Event()

    limiter_db = config.global_limiter_db or config.state_db
    limiter = GlobalRateLimiter(
        db_path=limiter_db,
        max_requests_per_hour=config.max_requests_per_hour,
        cooldown_seconds=config.cooldown_seconds_on_budget,
        stop_event=stop_event,
    )

    client = GDHistoryClient(config, limiter=limiter, stop_event=stop_event)
    state = FetchState(config.state_db)

    min_online_id = config.start_online_id
    if min_online_id is None and config.resume_from_last_downloaded:
        min_online_id = state.get_last_downloaded_online_id()
    _log(
        "Downloader configuration ready "
        f"(output_dir={config.output_dir}, max_levels={config.max_levels}, min_online_id={min_online_id}, workers={config.workers})"
    )

    offset = 0
    processed = 0
    saved = 0
    skipped = 0
    failed = 0
    not_in_shard = 0
    aborted = False

    try:
        while processed < config.max_levels:
            if stop_event.is_set():
                aborted = True
                break
            page = client.search_levels(
                limit=config.limit_per_page,
                offset=offset,
                min_featured_score=config.min_featured_score,
                min_epic_tier=config.min_epic_tier,
                min_online_id=min_online_id,
            )
            hits = page.get("hits", [])
            _log(f"Search returned {len(hits)} hits at offset {offset}")
            if not hits:
                _log("No more hits returned; ending download loop")
                break

            eligible_hits: list[dict[str, Any]] = []

            for hit in hits:
                if processed >= config.max_levels:
                    break

                online_id = int(hit["online_id"])
                state.upsert_level_hit(hit, download_status="seen")

                if (online_id % config.shard_count) != config.shard_index:
                    not_in_shard += 1
                    continue

                processed += 1
                _log(
                    f"Processing level online_id={online_id} "
                    f"({processed}/{config.max_levels})"
                )

                if state.is_downloaded(online_id):
                    skipped += 1
                    _log(f"Skipping online_id={online_id}; already downloaded in DB")
                    continue

                eligible_hits.append(hit)

            if eligible_hits:
                _log(
                    f"Downloading {len(eligible_hits)} levels with worker pool size {config.workers}"
                )
                batch_attempted = len(eligible_hits)
                batch_ok = 0
                batch_failed = 0
                batch_elapsed_total = 0.0
                batch_pages_total = 0.0
                executor = ThreadPoolExecutor(max_workers=config.workers)
                interrupted = False
                try:
                    futures = [
                        executor.submit(
                            _download_single_level,
                            online_id=int(hit["online_id"]),
                            hit=hit,
                            config=config,
                            limiter=limiter,
                            stop_event=stop_event,
                        )
                        for hit in eligible_hits
                    ]

                    for future in as_completed(futures):
                        if stop_event.is_set():
                            interrupted = True
                            break
                        try:
                            result = future.result()
                            online_id = int(result["online_id"])
                            status = str(result.get("status"))

                            if status == "ok":
                                batch_ok += 1
                                out_path = Path(result["path"])
                                _log(
                                    f"Saved .gmd for online_id={online_id} to {out_path} "
                                    f"({int(result['bytes'])} bytes)"
                                )
                                _write_metadata_line(
                                    config.metadata_jsonl,
                                    dict(result["metadata"]),
                                )
                                state.mark(
                                    online_id=online_id,
                                    record_id=int(result["record_id"]),
                                    status="ok",
                                    file_path=out_path,
                                    sha256_hex=str(result["sha256"]),
                                    file_bytes=int(result["bytes"]),
                                    message="",
                                )
                                state.update_version_info(
                                    online_id=online_id,
                                    reported_version=(
                                        str(result.get("reported_version"))
                                        if result.get("reported_version") is not None
                                        else None
                                    ),
                                    approx_version=(
                                        str(result.get("approx_version"))
                                        if result.get("approx_version") is not None
                                        else None
                                    ),
                                )
                                saved += 1
                                batch_elapsed_total += float(
                                    result.get("elapsed_seconds") or 0.0
                                )
                                batch_pages_total += float(
                                    result.get("pages_scanned") or 1.0
                                )
                                _log(
                                    f"Completed online_id={online_id} successfully "
                                    f"(saved={saved}, skipped={skipped}, failed={failed})"
                                )
                            elif status == "skipped":
                                skipped += 1
                                batch_elapsed_total += float(
                                    result.get("elapsed_seconds") or 0.0
                                )
                                batch_pages_total += float(
                                    result.get("pages_scanned") or 1.0
                                )
                                message = str(result.get("message") or "Skipped")
                                state.mark(
                                    online_id=online_id,
                                    record_id=None,
                                    status="skipped",
                                    file_path=None,
                                    sha256_hex=None,
                                    file_bytes=None,
                                    message=message,
                                )
                                state.update_version_info(
                                    online_id=online_id,
                                    reported_version=(
                                        str(result.get("reported_version"))
                                        if result.get("reported_version") is not None
                                        else None
                                    ),
                                    approx_version=(
                                        str(result.get("approx_version"))
                                        if result.get("approx_version") is not None
                                        else None
                                    ),
                                )
                                _log(f"Skipping online_id={online_id}; {message}")
                            else:
                                failed += 1
                                batch_failed += 1
                                batch_elapsed_total += float(
                                    result.get("elapsed_seconds") or 0.0
                                )
                                batch_pages_total += float(
                                    result.get("pages_scanned") or 1.0
                                )
                                message = str(
                                    result.get("message") or "Unknown worker status"
                                )
                                state.mark(
                                    online_id=online_id,
                                    record_id=None,
                                    status="failed",
                                    file_path=None,
                                    sha256_hex=None,
                                    file_bytes=None,
                                    message=message,
                                )
                                _log(f"Failed online_id={online_id}: {message}")
                        except InterruptedError:
                            interrupted = True
                            stop_event.set()
                            _log("Worker interrupted; stopping fetch loop")
                            break
                        except KeyboardInterrupt:
                            interrupted = True
                            stop_event.set()
                            _log("Interrupt received; stopping fetch loop")
                            break
                        except Exception as exc:
                            failed += 1
                            batch_failed += 1
                            _log(f"Worker future failed: {exc}")

                        if (
                            config.batch_pause_every > 0
                            and processed % config.batch_pause_every == 0
                        ):
                            _log(
                                f"Batch pause reached at processed={processed}; "
                                f"sleeping {config.batch_pause_seconds:.2f}s"
                            )
                            _sleep_with_cancel(config.batch_pause_seconds, stop_event)

                    if interrupted:
                        aborted = True
                finally:
                    executor.shutdown(wait=False, cancel_futures=True)

                if aborted:
                    break

                _apply_auto_tuning(
                    config,
                    {
                        "attempted": batch_attempted,
                        "ok": batch_ok,
                        "failed": batch_failed,
                        "avg_elapsed_seconds": batch_elapsed_total
                        / max(1, batch_attempted),
                        "avg_pages_scanned": batch_pages_total
                        / max(1, batch_attempted),
                    },
                )

            offset += config.limit_per_page
    except (KeyboardInterrupt, InterruptedError):
        aborted = True
        stop_event.set()
        _log("Download interrupted by user; shutting down worker pool")
    finally:
        stop_event.set()
        state.close()
        limiter.close()

    _log(
        "Download finished "
        f"(processed={processed}, saved={saved}, skipped={skipped}, failed={failed}, not_in_shard={not_in_shard})"
    )
    return {
        "processed": processed,
        "saved": saved,
        "skipped": skipped,
        "failed": failed,
        "not_in_shard": not_in_shard,
        "shard_index": config.shard_index,
        "shard_count": config.shard_count,
        "aborted": aborted,
    }
