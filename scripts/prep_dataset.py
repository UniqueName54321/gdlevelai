from __future__ import annotations

import argparse
import subprocess
import sys


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and verify dataset")
    parser.add_argument("--max-levels", type=int, default=200)
    parser.add_argument("--output-dir", default="dataset/gmd")
    parser.add_argument("--state-db", default="dataset/state/fetch_state.sqlite3")
    parser.add_argument("--metadata-jsonl", default="dataset/metadata/levels.jsonl")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("extra_args", nargs="*")
    args = parser.parse_args()

    if not args.skip_fetch:
        fetch_command = [
            sys.executable,
            "-m",
            "gdlevelai",
            "fetch",
            "--max-levels",
            str(args.max_levels),
            "--output-dir",
            args.output_dir,
            "--state-db",
            args.state_db,
            "--metadata-jsonl",
            args.metadata_jsonl,
            "--workers",
            str(args.workers),
        ]
        fetch_command.extend(args.extra_args)
        _run(fetch_command)

    verify_command = [
        sys.executable,
        "-m",
        "gdlevelai",
        "dataset",
        "verify",
        "--dataset-dir",
        args.output_dir,
        "--state-db",
        args.state_db,
        "--metadata-jsonl",
        args.metadata_jsonl,
    ]
    _run(verify_command)


if __name__ == "__main__":
    main()
