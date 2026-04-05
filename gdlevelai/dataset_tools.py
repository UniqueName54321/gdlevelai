from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any


def _log(message: str) -> None:
    print(f"[dataset_tools] {message}", flush=True)


def _format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    idx = 0
    while value >= 1024.0 and idx < len(units) - 1:
        value /= 1024.0
        idx += 1
    if idx == 0:
        return f"{int(value)} {units[idx]}"
    return f"{value:.2f} {units[idx]}"


@dataclass
class DatasetSummary:
    downloaded_count: int
    failed_count: int
    skipped_count: int
    indexed_count: int
    indexed_ok_count: int
    indexed_seen_count: int
    gmd_file_count: int
    gmd_total_bytes: int
    last_downloaded_online_id: int | None
    metadata_lines: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "downloaded_count": self.downloaded_count,
            "failed_count": self.failed_count,
            "skipped_count": self.skipped_count,
            "indexed_count": self.indexed_count,
            "indexed_ok_count": self.indexed_ok_count,
            "indexed_seen_count": self.indexed_seen_count,
            "gmd_file_count": self.gmd_file_count,
            "gmd_total_bytes": self.gmd_total_bytes,
            "gmd_total_bytes_human": _format_bytes(self.gmd_total_bytes),
            "last_downloaded_online_id": self.last_downloaded_online_id,
            "metadata_lines": self.metadata_lines,
        }


def _connect(db_path: Path) -> sqlite3.Connection:
    _log(f"Opening state database at {db_path}")
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _safe_count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    _log(f"Counting metadata lines in {path}")
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return sum(1 for _ in f)


def _gmd_disk_usage(dataset_dir: Path) -> tuple[int, int]:
    if not dataset_dir.exists():
        return 0, 0
    _log(f"Scanning .gmd files in {dataset_dir}")
    count = 0
    total_bytes = 0
    for gmd in dataset_dir.glob("*.gmd"):
        count += 1
        try:
            total_bytes += gmd.stat().st_size
        except OSError:
            continue
    return count, total_bytes


def dataset_summary(
    state_db: Path, dataset_dir: Path, metadata_jsonl: Path
) -> DatasetSummary:
    conn = _connect(state_db)
    try:
        _log("Computing dataset summary metrics")
        downloaded_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM fetched_levels WHERE status = 'ok'"
            ).fetchone()[0]
        )
        failed_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM fetched_levels WHERE status = 'failed'"
            ).fetchone()[0]
        )
        skipped_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM fetched_levels WHERE status LIKE 'skipped%'"
            ).fetchone()[0]
        )
        indexed_count = int(
            conn.execute("SELECT COUNT(*) FROM level_index").fetchone()[0]
        )
        indexed_ok_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM level_index WHERE download_status = 'ok'"
            ).fetchone()[0]
        )
        indexed_seen_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM level_index WHERE download_status = 'seen'"
            ).fetchone()[0]
        )

        row = conn.execute(
            "SELECT MAX(online_id) FROM fetched_levels WHERE status = 'ok'"
        ).fetchone()
        last_downloaded_online_id = int(row[0]) if row and row[0] is not None else None
    finally:
        conn.close()
        _log("Closed state database")

    gmd_file_count, gmd_total_bytes = _gmd_disk_usage(dataset_dir)
    metadata_lines = _safe_count_lines(metadata_jsonl)

    return DatasetSummary(
        downloaded_count=downloaded_count,
        failed_count=failed_count,
        skipped_count=skipped_count,
        indexed_count=indexed_count,
        indexed_ok_count=indexed_ok_count,
        indexed_seen_count=indexed_seen_count,
        gmd_file_count=gmd_file_count,
        gmd_total_bytes=gmd_total_bytes,
        last_downloaded_online_id=last_downloaded_online_id,
        metadata_lines=metadata_lines,
    )


def recent_downloads(state_db: Path, limit: int = 20) -> list[dict[str, Any]]:
    conn = _connect(state_db)
    try:
        _log(f"Querying {limit} recent downloads")
        rows = conn.execute(
            """
            SELECT online_id, record_id, file_path, file_bytes, fetched_at
            FROM fetched_levels
            WHERE status = 'ok'
            ORDER BY fetched_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
        _log("Closed state database")


def top_creators(state_db: Path, limit: int = 20) -> list[dict[str, Any]]:
    conn = _connect(state_db)
    try:
        _log(f"Querying top {limit} creators")
        rows = conn.execute(
            """
            SELECT
                COALESCE(creator, '[unknown]') AS creator,
                COUNT(*) AS levels
            FROM level_index
            WHERE download_status = 'ok'
            GROUP BY COALESCE(creator, '[unknown]')
            ORDER BY levels DESC, creator ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
        _log("Closed state database")


def find_levels(state_db: Path, query: str, limit: int = 50) -> list[dict[str, Any]]:
    conn = _connect(state_db)
    try:
        _log(f"Searching indexed levels for query='{query}'")
        like_query = f"%{query}%"
        rows = conn.execute(
            """
            SELECT
                online_id,
                level_name,
                creator,
                featured,
                epic_tier,
                download_status,
                submitted_timestamp
            FROM level_index
            WHERE level_name LIKE ? OR creator LIKE ?
            ORDER BY online_id ASC
            LIMIT ?
            """,
            (like_query, like_query, limit),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
        _log("Closed state database")


def verify_downloads(
    state_db: Path, dataset_dir: Path, limit: int = 200
) -> dict[str, Any]:
    conn = _connect(state_db)
    missing: list[dict[str, Any]] = []
    checked = 0
    try:
        _log(f"Verifying up to {limit} downloaded files exist on disk")
        rows = conn.execute(
            """
            SELECT online_id, record_id, file_path
            FROM fetched_levels
            WHERE status = 'ok'
            ORDER BY fetched_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        for row in rows:
            checked += 1
            stored_path = row["file_path"]
            if stored_path:
                p = Path(stored_path)
                if not p.is_absolute():
                    p = dataset_dir / p
            else:
                p = dataset_dir / f"{row['online_id']}_{row['record_id']}.gmd"

            if not p.exists():
                missing.append(
                    {
                        "online_id": row["online_id"],
                        "record_id": row["record_id"],
                        "expected_path": str(p),
                    }
                )
    finally:
        conn.close()
        _log("Closed state database")

    return {
        "checked": checked,
        "missing_count": len(missing),
        "missing": missing,
    }


def _extract_online_id_from_gmd_name(filename: str) -> int | None:
    if not filename.endswith(".gmd"):
        return None
    stem = filename[:-4]
    if "_" not in stem:
        return None
    candidate = stem.split("_", 1)[0]
    if not candidate.isdigit():
        return None
    return int(candidate)


def scan_gmd_dataset(dataset_dir: Path) -> dict[str, Any]:
    _log(f"Scanning dataset directory {dataset_dir}")
    files: list[dict[str, Any]] = []
    invalid_filenames: list[dict[str, Any]] = []
    online_id_to_files: dict[int, list[str]] = {}

    if not dataset_dir.exists():
        return {
            "files": files,
            "online_ids": set(),
            "invalid_filenames": [
                {
                    "filename": "[dataset_dir_missing]",
                    "path": str(dataset_dir),
                    "reason": "dataset directory does not exist",
                }
            ],
            "duplicate_online_ids": [],
        }

    for path in sorted(dataset_dir.glob("*.gmd")):
        online_id = _extract_online_id_from_gmd_name(path.name)
        size = 0
        try:
            size = path.stat().st_size
        except OSError:
            pass

        if online_id is None:
            invalid_filenames.append(
                {
                    "filename": path.name,
                    "path": str(path),
                    "reason": "expected '<online_id>_<record_id>.gmd'",
                }
            )
            continue

        files.append(
            {
                "path": str(path),
                "filename": path.name,
                "online_id": online_id,
                "file_size": size,
            }
        )
        online_id_to_files.setdefault(online_id, []).append(path.name)

    duplicate_online_ids: list[dict[str, Any]] = []
    for online_id in sorted(online_id_to_files):
        names = online_id_to_files[online_id]
        if len(names) > 1:
            duplicate_online_ids.append(
                {
                    "online_id": online_id,
                    "filenames": sorted(names),
                }
            )

    return {
        "files": files,
        "online_ids": set(online_id_to_files.keys()),
        "invalid_filenames": invalid_filenames,
        "duplicate_online_ids": duplicate_online_ids,
    }


def read_metadata_jsonl(metadata_jsonl: Path) -> dict[str, Any]:
    _log(f"Reading metadata JSONL from {metadata_jsonl}")
    line_count = 0
    valid_entries = 0
    invalid_json_lines: list[int] = []
    missing_online_id_lines: list[int] = []
    online_id_to_lines: dict[int, list[int]] = {}

    if not metadata_jsonl.exists():
        return {
            "line_count": 0,
            "valid_entries": 0,
            "online_ids": set(),
            "invalid_json_lines": [0],
            "missing_online_id_lines": [],
            "duplicate_online_ids": [],
        }

    with metadata_jsonl.open("r", encoding="utf-8", errors="ignore") as f:
        for idx, raw_line in enumerate(f, start=1):
            line_count += 1
            line = raw_line.strip()
            if not line:
                invalid_json_lines.append(idx)
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                invalid_json_lines.append(idx)
                continue

            online_id_value = obj.get("online_id") if isinstance(obj, dict) else None
            if online_id_value is None:
                missing_online_id_lines.append(idx)
                continue

            try:
                online_id = int(online_id_value)
            except (TypeError, ValueError):
                missing_online_id_lines.append(idx)
                continue

            valid_entries += 1
            online_id_to_lines.setdefault(online_id, []).append(idx)

    duplicate_online_ids: list[dict[str, Any]] = []
    for online_id in sorted(online_id_to_lines):
        lines = online_id_to_lines[online_id]
        if len(lines) > 1:
            duplicate_online_ids.append(
                {
                    "online_id": online_id,
                    "lines": lines,
                }
            )

    return {
        "line_count": line_count,
        "valid_entries": valid_entries,
        "online_ids": set(online_id_to_lines.keys()),
        "invalid_json_lines": invalid_json_lines,
        "missing_online_id_lines": missing_online_id_lines,
        "duplicate_online_ids": duplicate_online_ids,
    }


def state_downloaded_online_ids(state_db: Path) -> set[int]:
    conn = _connect(state_db)
    try:
        _log("Reading downloaded online_ids from fetched_levels")
        rows = conn.execute(
            "SELECT online_id FROM fetched_levels WHERE status = 'ok'"
        ).fetchall()
        return {int(r["online_id"]) for r in rows}
    finally:
        conn.close()
        _log("Closed state database")


def reset_global_request_limiter(db_path: Path) -> dict[str, Any]:
    _log(f"Resetting global request limiter in {db_path}")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS request_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at REAL NOT NULL
            )
            """
        )
        before_count = int(
            conn.execute("SELECT COUNT(*) FROM request_log").fetchone()[0]
        )
        conn.execute("DELETE FROM request_log")
        conn.commit()
        after_count = int(
            conn.execute("SELECT COUNT(*) FROM request_log").fetchone()[0]
        )
        return {
            "limiter_db": str(db_path),
            "deleted_rows": before_count,
            "remaining_rows": after_count,
            "ok": True,
        }
    finally:
        conn.close()
        _log("Closed limiter database")


def _normalize_version_tag(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"(\d+)(?:\.(\d+))?", text)
    if not match:
        return None
    major = int(match.group(1))
    minor = int(match.group(2) or 0)
    if major >= 10 and major <= 99 and minor == 0:
        return f"{major // 10}.{major % 10}"
    return f"{major}.{minor}"


def version_report(state_db: Path, *, limit: int = 50) -> dict[str, Any]:
    conn = _connect(state_db)
    try:
        _log("Computing version report from level_index")
        rows = conn.execute(
            "SELECT online_id, reported_version, approx_version FROM level_index"
        ).fetchall()

        reported_dist: dict[str, int] = {}
        approx_dist: dict[str, int] = {}
        mismatches: list[dict[str, Any]] = []
        with_reported = 0

        for row in rows:
            online_id = int(row[0])
            reported_raw = row[1]
            approx_raw = row[2]

            reported_norm = _normalize_version_tag(reported_raw)
            approx_norm = _normalize_version_tag(approx_raw)

            if reported_norm is not None:
                with_reported += 1
                reported_dist[reported_norm] = reported_dist.get(reported_norm, 0) + 1
            if approx_norm is not None:
                approx_dist[approx_norm] = approx_dist.get(approx_norm, 0) + 1

            if reported_norm and approx_norm and reported_norm != approx_norm:
                mismatches.append(
                    {
                        "online_id": online_id,
                        "reported_version": reported_raw,
                        "approx_version": approx_raw,
                    }
                )

        mismatches.sort(key=lambda x: int(x["online_id"]))
        shown, truncated = _truncate_list(mismatches, limit)
        return {
            "counts": {
                "level_index_rows": len(rows),
                "reported_version_rows": with_reported,
                "approx_version_rows": sum(approx_dist.values()),
                "reported_vs_approx_mismatch_count": len(mismatches),
            },
            "reported_version_distribution": dict(
                sorted(reported_dist.items(), key=lambda kv: (-kv[1], kv[0]))
            ),
            "approx_version_distribution": dict(
                sorted(approx_dist.items(), key=lambda kv: (-kv[1], kv[0]))
            ),
            "reported_vs_approx_mismatches": shown,
            "truncated": {
                "reported_vs_approx_mismatches": truncated,
            }
            if truncated > 0
            else {},
        }
    finally:
        conn.close()
        _log("Closed state database")


def online_id_set_report(
    state_db: Path,
    dataset_dir: Path,
    metadata_jsonl: Path,
    *,
    limit: int | None = 50,
) -> dict[str, Any]:
    gmd = scan_gmd_dataset(dataset_dir)
    metadata = read_metadata_jsonl(metadata_jsonl)
    state_ids = state_downloaded_online_ids(state_db)

    gmd_ids = set(gmd["online_ids"])
    metadata_ids = set(metadata["online_ids"])

    all_ids = gmd_ids | metadata_ids | state_ids
    in_all_three = gmd_ids & metadata_ids & state_ids
    in_any_two = (
        (gmd_ids & metadata_ids) | (gmd_ids & state_ids) | (metadata_ids & state_ids)
    ) - in_all_three

    metadata_only = sorted(metadata_ids - gmd_ids - state_ids)
    files_only = sorted(gmd_ids - metadata_ids - state_ids)
    db_only = sorted(state_ids - metadata_ids - gmd_ids)
    metadata_vs_files_only = sorted((metadata_ids ^ gmd_ids) - state_ids)
    metadata_vs_db_only = sorted((metadata_ids ^ state_ids) - gmd_ids)
    files_vs_db_only = sorted((gmd_ids ^ state_ids) - metadata_ids)

    metadata_only_shown, metadata_only_truncated = _truncate_list(metadata_only, limit)
    files_only_shown, files_only_truncated = _truncate_list(files_only, limit)
    db_only_shown, db_only_truncated = _truncate_list(db_only, limit)
    metadata_vs_files_shown, metadata_vs_files_truncated = _truncate_list(
        metadata_vs_files_only, limit
    )
    metadata_vs_db_shown, metadata_vs_db_truncated = _truncate_list(
        metadata_vs_db_only, limit
    )
    files_vs_db_shown, files_vs_db_truncated = _truncate_list(files_vs_db_only, limit)

    truncated: dict[str, int] = {}
    if metadata_only_truncated > 0:
        truncated["only_in_metadata_ids"] = metadata_only_truncated
    if files_only_truncated > 0:
        truncated["only_in_file_ids"] = files_only_truncated
    if db_only_truncated > 0:
        truncated["only_in_db_ids"] = db_only_truncated
    if metadata_vs_files_truncated > 0:
        truncated["metadata_vs_files_symmetric_diff_ids"] = metadata_vs_files_truncated
    if metadata_vs_db_truncated > 0:
        truncated["metadata_vs_db_symmetric_diff_ids"] = metadata_vs_db_truncated
    if files_vs_db_truncated > 0:
        truncated["files_vs_db_symmetric_diff_ids"] = files_vs_db_truncated

    return {
        "counts": {
            "metadata_online_id_count": len(metadata_ids),
            "file_online_id_count": len(gmd_ids),
            "db_online_id_count": len(state_ids),
            "union_online_id_count": len(all_ids),
            "intersection_all_three_count": len(in_all_three),
            "intersection_exactly_two_count": len(in_any_two),
            "only_in_metadata_count": len(metadata_only),
            "only_in_files_count": len(files_only),
            "only_in_db_count": len(db_only),
        },
        "only_in_metadata_ids": metadata_only_shown,
        "only_in_file_ids": files_only_shown,
        "only_in_db_ids": db_only_shown,
        "metadata_vs_files_symmetric_diff_ids": metadata_vs_files_shown,
        "metadata_vs_db_symmetric_diff_ids": metadata_vs_db_shown,
        "files_vs_db_symmetric_diff_ids": files_vs_db_shown,
        "truncated": truncated,
    }


def _truncate_list(values: list[Any], limit: int | None) -> tuple[list[Any], int]:
    if limit is None or limit < 0:
        return values, 0
    if len(values) <= limit:
        return values, 0
    return values[:limit], len(values) - limit


def verify_dataset_consistency(
    state_db: Path,
    dataset_dir: Path,
    metadata_jsonl: Path,
    *,
    limit: int | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    gmd = scan_gmd_dataset(dataset_dir)
    metadata = read_metadata_jsonl(metadata_jsonl)
    state_ids = state_downloaded_online_ids(state_db)

    gmd_ids = set(gmd["online_ids"])
    metadata_ids = set(metadata["online_ids"])

    gmd_without_metadata = sorted(gmd_ids - metadata_ids)
    metadata_without_gmd = sorted(metadata_ids - gmd_ids)
    gmd_without_state = sorted(gmd_ids - state_ids)
    state_without_gmd = sorted(state_ids - gmd_ids)
    metadata_without_state = sorted(metadata_ids - state_ids)

    issues = {
        "invalid_gmd_filenames": list(gmd["invalid_filenames"]),
        "duplicate_gmd_online_ids": list(gmd["duplicate_online_ids"]),
        "duplicate_metadata_online_ids": list(metadata["duplicate_online_ids"]),
        "metadata_invalid_json_lines": list(metadata["invalid_json_lines"]),
        "metadata_missing_online_id_lines": list(metadata["missing_online_id_lines"]),
    }

    mismatches = {
        "gmd_without_metadata": gmd_without_metadata,
        "metadata_without_gmd": metadata_without_gmd,
        "gmd_without_state": gmd_without_state,
        "state_without_gmd": state_without_gmd,
        "metadata_without_state": metadata_without_state,
    }

    issue_count = sum(len(v) for v in issues.values())
    mismatch_count = sum(len(v) for v in mismatches.values())

    if strict:
        ok = issue_count == 0 and mismatch_count == 0
    else:
        ok = mismatch_count == 0

    output_issues: dict[str, Any] = {}
    output_mismatches: dict[str, Any] = {}
    full_issue_counts: dict[str, int] = {}
    full_mismatch_counts: dict[str, int] = {}
    shown_issue_counts: dict[str, int] = {}
    shown_mismatch_counts: dict[str, int] = {}
    truncation: dict[str, int] = {}

    for key, vals in issues.items():
        full_issue_counts[key] = len(vals)
        sliced, removed = _truncate_list(vals, limit)
        output_issues[key] = sliced
        shown_issue_counts[key] = len(sliced)
        if removed > 0:
            truncation[f"issues.{key}"] = removed

    for key, vals in mismatches.items():
        full_mismatch_counts[key] = len(vals)
        sliced, removed = _truncate_list(vals, limit)
        output_mismatches[key] = sliced
        shown_mismatch_counts[key] = len(sliced)
        if removed > 0:
            truncation[f"mismatches.{key}"] = removed

    return {
        "ok": ok,
        "strict": strict,
        "counts": {
            "gmd_file_count": len(gmd["files"]),
            "gmd_online_id_count": len(gmd_ids),
            "metadata_line_count": int(metadata["line_count"]),
            "metadata_valid_entries": int(metadata["valid_entries"]),
            "metadata_online_id_count": len(metadata_ids),
            "state_downloaded_online_id_count": len(state_ids),
        },
        "totals": {
            "issue_count": issue_count,
            "mismatch_count": mismatch_count,
        },
        "full_counts": {
            "issues": full_issue_counts,
            "mismatches": full_mismatch_counts,
        },
        "shown_counts": {
            "issues": shown_issue_counts,
            "mismatches": shown_mismatch_counts,
        },
        "issues": output_issues,
        "mismatches": output_mismatches,
        "truncated": truncation,
    }


def print_verify_human_report(report: dict[str, Any]) -> None:
    counts = report["counts"]
    issues = report["issues"]
    mismatches = report["mismatches"]
    full_counts = report.get("full_counts", {})
    shown_counts = report.get("shown_counts", {})

    full_issue_counts = full_counts.get("issues", {})
    full_mismatch_counts = full_counts.get("mismatches", {})
    shown_issue_counts = shown_counts.get("issues", {})
    shown_mismatch_counts = shown_counts.get("mismatches", {})

    def _count_display(section: str, key: str) -> str:
        if section == "issues":
            total = int(full_issue_counts.get(key, len(issues.get(key, []))))
            shown = int(shown_issue_counts.get(key, len(issues.get(key, []))))
        else:
            total = int(full_mismatch_counts.get(key, len(mismatches.get(key, []))))
            shown = int(shown_mismatch_counts.get(key, len(mismatches.get(key, []))))

        if shown < total:
            return f"{total} (showing first {shown})"
        return str(total)

    print("Dataset verify summary:", flush=True)
    print(
        "- counts: "
        f"gmd_files={counts['gmd_file_count']}, "
        f"gmd_ids={counts['gmd_online_id_count']}, "
        f"metadata_lines={counts['metadata_line_count']}, "
        f"metadata_ids={counts['metadata_online_id_count']}, "
        f"state_ids={counts['state_downloaded_online_id_count']}",
        flush=True,
    )

    print("- issues:", flush=True)
    print(
        f"  invalid_gmd_filenames={_count_display('issues', 'invalid_gmd_filenames')}",
        flush=True,
    )
    print(
        "  "
        f"duplicate_gmd_online_ids={_count_display('issues', 'duplicate_gmd_online_ids')}",
        flush=True,
    )
    print(
        "  "
        f"duplicate_metadata_online_ids={_count_display('issues', 'duplicate_metadata_online_ids')}",
        flush=True,
    )
    print(
        "  "
        f"metadata_invalid_json_lines={_count_display('issues', 'metadata_invalid_json_lines')}",
        flush=True,
    )
    print(
        "  "
        f"metadata_missing_online_id_lines={_count_display('issues', 'metadata_missing_online_id_lines')}",
        flush=True,
    )

    print("- mismatches:", flush=True)
    print(
        "  "
        f"gmd_without_metadata={_count_display('mismatches', 'gmd_without_metadata')}",
        flush=True,
    )
    print(
        "  "
        f"metadata_without_gmd={_count_display('mismatches', 'metadata_without_gmd')}",
        flush=True,
    )
    print(
        f"  gmd_without_state={_count_display('mismatches', 'gmd_without_state')}",
        flush=True,
    )
    print(
        f"  state_without_gmd={_count_display('mismatches', 'state_without_gmd')}",
        flush=True,
    )
    print(
        "  "
        f"metadata_without_state={_count_display('mismatches', 'metadata_without_state')}",
        flush=True,
    )

    if report.get("truncated"):
        print(f"- note: output truncated entries: {report['truncated']}", flush=True)

    print(f"- result: {'OK' if report['ok'] else 'NOT OK'}", flush=True)


def archive_untracked_gmd_files(
    state_db: Path,
    dataset_dir: Path,
    metadata_jsonl: Path,
    *,
    archive_dir: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    report = verify_dataset_consistency(
        state_db,
        dataset_dir,
        metadata_jsonl,
        limit=None,
        strict=True,
    )

    untracked_ids = set(int(x) for x in report["mismatches"]["gmd_without_state"])
    if archive_dir is None:
        archive_dir = dataset_dir.parent / (
            "gmd_untracked_archive_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        )

    files_to_move: list[Path] = []
    for path in sorted(dataset_dir.glob("*.gmd")):
        online_id = _extract_online_id_from_gmd_name(path.name)
        if online_id is not None and online_id in untracked_ids:
            files_to_move.append(path)

    moved_files: list[str] = []
    errors: list[dict[str, str]] = []

    if not dry_run and files_to_move:
        archive_dir.mkdir(parents=True, exist_ok=True)

    for src in files_to_move:
        if dry_run:
            moved_files.append(src.name)
            continue
        try:
            dest = archive_dir / src.name
            if dest.exists():
                stem = src.stem
                suffix = src.suffix
                idx = 1
                while True:
                    candidate = archive_dir / f"{stem}__dup{idx}{suffix}"
                    if not candidate.exists():
                        dest = candidate
                        break
                    idx += 1
            shutil.move(str(src), str(dest))
            moved_files.append(dest.name)
        except Exception as exc:
            errors.append({"file": str(src), "error": str(exc)})

    return {
        "dry_run": dry_run,
        "archive_dir": str(archive_dir),
        "target_online_ids": len(untracked_ids),
        "candidate_files": len(files_to_move),
        "moved_files": len(moved_files),
        "move_errors": len(errors),
        "errors": errors,
    }


def print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=True), flush=True)
