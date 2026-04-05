from __future__ import annotations

import argparse
from pathlib import Path

from .dataset_tools import (
    archive_untracked_gmd_files,
    dataset_summary,
    find_levels,
    online_id_set_report,
    reset_global_request_limiter,
    version_report,
    print_verify_human_report,
    print_json,
    recent_downloads,
    top_creators,
    verify_dataset_consistency,
)
from .gdhistory_dataset import (
    FetchConfig,
    download_featured_dataset,
    migrate_fetch_state_db,
)


def _log(message: str) -> None:
    print(f"[cli] {message}", flush=True)


def _format_song_choice(song_id: int, is_custom_song: bool = False) -> str:
    if is_custom_song:
        return f"custom_song_id={song_id} (Newgrounds)"

    official_song_names = {
        0: "Stereo Madness",
        1: "Back on Track",
        2: "Polargeist",
        3: "Dry Out",
        4: "Base After Base",
        5: "Cant Let Go",
        6: "Jumper",
        7: "Time Machine",
        8: "Cycles",
        9: "xStep",
        10: "Clutterfunk",
        11: "Theory of Everything",
        12: "Electroman Adventures",
        13: "Clubstep",
        14: "Electrodynamix",
        15: "Hexagon Force",
        16: "Blast Processing",
        17: "Theory of Everything 2",
        18: "Geometrical Dominator",
        19: "Deadlocked",
        20: "Fingerdash",
        21: "Dash",
    }
    name = official_song_names.get(song_id)
    if name is None:
        return f"song_id={song_id}"
    return f"song_id={song_id} ({name})"


def _fetch_command(args: argparse.Namespace) -> None:
    _log("Starting fetch command")
    out_dir = Path(args.output_dir)
    cfg = FetchConfig(
        output_dir=out_dir,
        state_db=Path(args.state_db),
        metadata_jsonl=Path(args.metadata_jsonl),
        limit_per_page=args.limit_per_page,
        max_levels=args.max_levels,
        min_featured_score=args.min_featured_score,
        min_epic_tier=args.min_epic_tier,
        delay_seconds=args.delay if args.delay is not None else 0.0,
        jitter_seconds=args.jitter if args.jitter is not None else 0.0,
        max_retries=args.max_retries if args.max_retries is not None else 0,
        batch_pause_every=args.batch_pause_every
        if args.batch_pause_every is not None
        else 0,
        batch_pause_seconds=args.batch_pause_seconds
        if args.batch_pause_seconds is not None
        else 0.0,
        request_timeout=args.timeout,
        politeness_profile=args.politeness_profile,
        max_requests_per_hour=args.max_requests_per_hour,
        cooldown_seconds_on_budget=args.cooldown_seconds_on_budget,
        global_limiter_db=Path(args.global_limiter_db)
        if args.global_limiter_db
        else None,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        start_online_id=args.start_online_id,
        resume_from_last_downloaded=args.resume_from_last_downloaded,
        workers=args.workers,
        level_record_page_size=args.level_record_page_size,
        level_record_max_pages=args.level_record_max_pages,
        auto_tune=args.auto_tune,
        auto_min_workers=args.auto_min_workers,
        auto_max_workers=args.auto_max_workers,
        auto_target_seconds_per_level=args.auto_target_seconds_per_level,
    )
    if args.delay is not None:
        cfg.delay_seconds = args.delay
    if args.jitter is not None:
        cfg.jitter_seconds = args.jitter
    if args.max_retries is not None:
        cfg.max_retries = args.max_retries
    if args.batch_pause_every is not None:
        cfg.batch_pause_every = args.batch_pause_every
    if args.batch_pause_seconds is not None:
        cfg.batch_pause_seconds = args.batch_pause_seconds

    result = download_featured_dataset(cfg)
    _log("Fetch command completed")
    print(result)


def _train_command(args: argparse.Namespace) -> None:
    _log("Starting train command")
    from .autoregressive_generator import AutoregressiveConfig, train_autoregressive
    from .diffusion_generator import DiffusionConfig, train_diffusion

    if args.model_type == "autoregressive":
        cfg = AutoregressiveConfig(
            seq_len=args.seq_len,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            max_objects_per_file=args.max_objects_per_file,
            max_vocab_size=args.max_vocab_size,
            min_token_freq=args.min_token_freq,
            position_quant=args.position_quant,
            rotation_quant=args.rotation_quant,
            sample_stride=args.sample_stride,
            max_steps_per_epoch=args.max_steps_per_epoch,
            log_every_steps=args.log_every_steps,
            num_threads=args.num_threads,
            torch_compile=args.torch_compile,
            name_max_words=args.name_max_words,
            desc_max_words=args.desc_max_words,
            name_min_words=args.name_min_words,
            desc_min_words=args.desc_min_words,
            max_song_id=args.max_song_id,
            max_custom_song_id=args.max_custom_song_id,
            artifacts_subdir=args.artifacts_subdir,
            checkpoints_subdir=args.checkpoints_subdir,
            samples_subdir=args.samples_subdir,
            save_preprocessed_artifacts=args.save_preprocessed_artifacts,
            save_checkpoint_every_epochs=args.save_checkpoint_every_epochs,
            save_samples_every_epochs=args.save_samples_every_epochs,
            samples_per_epoch=args.samples_per_epoch,
            sample_preview_max_new_tokens=args.sample_preview_max_new_tokens,
            resume_checkpoint=args.resume_checkpoint,
        )
        train_autoregressive(Path(args.dataset_dir), Path(args.model_out), cfg)
    else:
        cfg = DiffusionConfig(
            max_objects=args.max_objects,
            feature_dim=args.feature_dim,
            timesteps=args.timesteps,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            id_scale=args.id_scale,
            x_scale=args.x_scale,
            y_scale=args.y_scale,
            rotation_scale=args.rotation_scale,
            log_every_steps=args.diffusion_log_every_steps,
            dataloader_shuffle=args.dataloader_shuffle,
            dataloader_drop_last=args.dataloader_drop_last,
            dataloader_workers=args.dataloader_workers,
            num_threads=args.num_threads,
        )
        train_diffusion(Path(args.dataset_dir), Path(args.model_out), cfg)
    _log("Train command completed")
    print(f"Saved model to {args.model_out}")


def _sample_command(args: argparse.Namespace) -> None:
    _log("Starting sample command")
    from .autoregressive_generator import sample_autoregressive
    from .diffusion_generator import sample_level

    if args.model_type == "autoregressive":
        sample_result = sample_autoregressive(
            model_path=Path(args.model_path),
            out_path=Path(args.out_path),
            level_name=args.level_name,
            device_override=args.device,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            min_tokens_before_eos=args.min_tokens_before_eos,
            sample_log_every_tokens=args.sample_log_every_tokens,
            ban_special_tokens=args.ban_special_tokens,
            song_id=args.song_id,
            custom_song_id=args.custom_song_id,
            level_description=args.level_description,
            min_objects_before_layout_end=args.min_objects_before_layout_end,
            min_layout_tokens_before_layout_end=args.min_layout_tokens_before_layout_end,
        )
        level_name = str(sample_result.get("level_name", "Untitled"))
        level_description = str(
            sample_result.get(
                "level_description", "Generated by gdlevelai autoregressive baseline"
            )
        )
        song_id = int(sample_result.get("song_id", 1))
        is_custom_song = bool(sample_result.get("is_custom_song", False))
        object_count = int(sample_result.get("object_count", 0))
        valid_objects = int(sample_result.get("valid_objects", object_count))
        attempted_objects = int(sample_result.get("attempted_objects", object_count))
        custom_song_id = sample_result.get("custom_song_id")
        stop_reason = str(sample_result.get("stop_reason", "unknown"))
    else:
        level_name, level_description, song_id, is_custom_song = sample_level(
            model_path=Path(args.model_path),
            out_path=Path(args.out_path),
            level_name=args.level_name,
            seed=args.seed,
            device_override=args.device,
            timesteps_override=args.timesteps_override,
            sample_log_every_steps=args.sample_log_every_steps,
            song_id=args.song_id,
            custom_song_id=args.custom_song_id,
            level_description=args.level_description,
        )
        object_count = -1
        valid_objects = -1
        attempted_objects = -1
        custom_song_id = song_id if is_custom_song else None
        stop_reason = "diffusion_schedule_complete"
    _log("Sample command completed")
    print(
        f"Generated {args.out_path} "
        f"({_format_song_choice(song_id, is_custom_song)}, name='{level_name}', "
        f"description='{level_description}', objects={object_count}, "
        f"valid_objects={valid_objects}, attempted_objects={attempted_objects}, "
        f"custom_song_id={custom_song_id}, stop_reason={stop_reason})"
    )


def _dataset_command(args: argparse.Namespace) -> None:
    _log("Starting dataset command")
    state_db = Path(args.state_db)
    dataset_dir = Path(args.dataset_dir)
    metadata_jsonl = Path(args.metadata_jsonl)

    if args.dataset_action == "summary":
        result = dataset_summary(state_db, dataset_dir, metadata_jsonl).to_dict()
    elif args.dataset_action == "recent":
        result = recent_downloads(state_db, limit=args.limit)
    elif args.dataset_action == "top-creators":
        result = top_creators(state_db, limit=args.limit)
    elif args.dataset_action == "find":
        result = find_levels(state_db, query=args.query, limit=args.limit)
    elif args.dataset_action == "verify":
        result = verify_dataset_consistency(
            state_db,
            dataset_dir,
            metadata_jsonl,
            limit=args.limit,
            strict=args.strict,
        )
    elif args.dataset_action == "archive-untracked":
        result = archive_untracked_gmd_files(
            state_db,
            dataset_dir,
            metadata_jsonl,
            archive_dir=Path(args.archive_dir) if args.archive_dir else None,
            dry_run=args.dry_run,
        )
    elif args.dataset_action == "id-report":
        result = online_id_set_report(
            state_db,
            dataset_dir,
            metadata_jsonl,
            limit=args.limit,
        )
    elif args.dataset_action == "version-report":
        result = version_report(state_db, limit=args.limit)
    elif args.dataset_action == "reset-limiter":
        limiter_db = (
            Path(args.global_limiter_db) if args.global_limiter_db else state_db
        )
        result = reset_global_request_limiter(limiter_db)
    elif args.dataset_action == "migrate-state-db":
        result = migrate_fetch_state_db(state_db, metadata_jsonl=metadata_jsonl)
    else:
        raise RuntimeError(f"Unknown dataset action: {args.dataset_action}")

    _log(f"Dataset command '{args.dataset_action}' completed")
    if args.dataset_action == "verify" and not args.json_only:
        print_verify_human_report(result)
    print_json(result)
    if args.dataset_action == "verify" and not result.get("ok", False):
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gdlevelai")
    sub = parser.add_subparsers(required=True)

    fetch = sub.add_parser("fetch", help="Fetch featured .gmd dataset from GDHistory")
    fetch.add_argument("--output-dir", default="dataset/gmd")
    fetch.add_argument("--state-db", default="dataset/state/fetch_state.sqlite3")
    fetch.add_argument("--metadata-jsonl", default="dataset/metadata/levels.jsonl")
    fetch.add_argument("--limit-per-page", type=int, default=200)
    fetch.add_argument("--max-levels", type=int, default=200)
    fetch.add_argument("--min-featured-score", type=int, default=1)
    fetch.add_argument("--min-epic-tier", type=int, default=0)
    fetch.add_argument(
        "--politeness-profile",
        choices=("normal", "careful", "very_careful"),
        default="careful",
    )
    fetch.add_argument("--max-requests-per-hour", type=int, default=0)
    fetch.add_argument("--cooldown-seconds-on-budget", type=float, default=0.0)
    fetch.add_argument("--global-limiter-db")
    fetch.add_argument("--delay", type=float)
    fetch.add_argument("--jitter", type=float)
    fetch.add_argument("--max-retries", type=int)
    fetch.add_argument("--batch-pause-every", type=int)
    fetch.add_argument("--batch-pause-seconds", type=float)
    fetch.add_argument("--shard-index", type=int, default=0)
    fetch.add_argument("--shard-count", type=int, default=1)
    fetch.add_argument("--start-online-id", type=int)
    fetch.add_argument(
        "--resume-from-last-downloaded",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    fetch.add_argument("--timeout", type=int, default=30)
    fetch.add_argument("--workers", type=int, default=3)
    fetch.add_argument("--level-record-page-size", type=int, default=40)
    fetch.add_argument("--level-record-max-pages", type=int, default=6)
    fetch.add_argument(
        "--auto-tune",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    fetch.add_argument("--auto-min-workers", type=int, default=1)
    fetch.add_argument("--auto-max-workers", type=int, default=6)
    fetch.add_argument("--auto-target-seconds-per-level", type=float, default=8.0)
    fetch.set_defaults(func=_fetch_command)

    train = sub.add_parser("train", help="Train model on .gmd files")
    train.add_argument("--dataset-dir", default="dataset/gmd")
    train.add_argument("--model-out", default="models/autoregressive.pt")
    train.add_argument(
        "--model-type",
        choices=("autoregressive", "diffusion"),
        default="autoregressive",
    )
    train.add_argument("--seq-len", type=int, default=192)
    train.add_argument("--embed-dim", type=int, default=192)
    train.add_argument("--max-objects", type=int, default=256)
    train.add_argument("--feature-dim", type=int, default=4)
    train.add_argument("--timesteps", type=int, default=500)
    train.add_argument("--beta-start", type=float, default=1e-4)
    train.add_argument("--beta-end", type=float, default=0.02)
    train.add_argument("--hidden-dim", type=int, default=512)
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--lr", type=float, default=2e-4)
    train.add_argument("--device", default="auto")
    train.add_argument("--id-scale", type=float, default=2000.0)
    train.add_argument("--x-scale", type=float, default=50000.0)
    train.add_argument("--y-scale", type=float, default=50000.0)
    train.add_argument("--rotation-scale", type=float, default=360.0)
    train.add_argument("--max-objects-per-file", type=int, default=0)
    train.add_argument("--max-vocab-size", type=int, default=50000)
    train.add_argument("--min-token-freq", type=int, default=2)
    train.add_argument("--position-quant", type=int, default=10)
    train.add_argument("--rotation-quant", type=int, default=15)
    train.add_argument("--sample-stride", "--stride", type=int, default=2)
    train.add_argument("--max-steps-per-epoch", type=int, default=2500)
    train.add_argument("--log-every-steps", type=int, default=100)
    train.add_argument("--name-max-words", type=int, default=8)
    train.add_argument("--desc-max-words", type=int, default=20)
    train.add_argument("--name-min-words", type=int, default=1)
    train.add_argument("--desc-min-words", type=int, default=1)
    train.add_argument("--max-song-id", type=int, default=1000)
    train.add_argument("--max-custom-song-id", type=int, default=100000000)
    train.add_argument("--artifacts-subdir", default="artifacts")
    train.add_argument("--checkpoints-subdir", default="checkpoints")
    train.add_argument("--samples-subdir", default="samples")
    train.add_argument(
        "--save-preprocessed-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    train.add_argument("--save-checkpoint-every-epochs", type=int, default=1)
    train.add_argument("--save-samples-every-epochs", type=int, default=1)
    train.add_argument("--samples-per-epoch", type=int, default=1)
    train.add_argument("--sample-preview-max-new-tokens", type=int, default=800)
    train.add_argument("--resume-checkpoint", default="")
    train.add_argument("--diffusion-log-every-steps", type=int, default=20)
    train.add_argument("--num-threads", type=int, default=0)
    train.add_argument(
        "--dataloader-shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    train.add_argument(
        "--dataloader-drop-last",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    train.add_argument("--dataloader-workers", type=int, default=0)
    train.add_argument(
        "--torch-compile",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    train.set_defaults(func=_train_command)

    sample = sub.add_parser("sample", help="Sample a generated .gmd level")
    sample.add_argument("--model-path", default="models/autoregressive.pt")
    sample.add_argument(
        "--model-type",
        choices=("autoregressive", "diffusion"),
        default="autoregressive",
    )
    sample.add_argument("--out-path", default="generated/generated_level.gmd")
    sample.add_argument("--level-name")
    sample.add_argument("--level-description")
    sample.add_argument("--seed", type=int)
    sample.add_argument("--device", default="auto")
    sample.add_argument("--max-new-tokens", type=int, default=4000)
    sample.add_argument("--temperature", type=float, default=1.0)
    sample.add_argument("--top-k", type=int, default=64)
    sample.add_argument("--min-tokens-before-eos", type=int, default=10)
    sample.add_argument("--sample-log-every-tokens", type=int, default=250)
    sample.add_argument(
        "--ban-special-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    sample.add_argument("--timesteps-override", type=int)
    sample.add_argument("--sample-log-every-steps", type=int, default=0)
    sample.add_argument("--song-id", type=int)
    sample.add_argument("--custom-song-id", type=int)
    sample.add_argument("--min-objects-before-layout-end", type=int, default=50)
    sample.add_argument("--min-layout-tokens-before-layout-end", type=int, default=0)
    sample.set_defaults(func=_sample_command)

    dataset = sub.add_parser("dataset", help="Inspect dataset and state DB")
    dataset.add_argument(
        "dataset_action",
        choices=(
            "summary",
            "recent",
            "top-creators",
            "find",
            "verify",
            "id-report",
            "version-report",
            "reset-limiter",
            "migrate-state-db",
            "archive-untracked",
        ),
    )
    dataset.add_argument("--state-db", default="dataset/state/fetch_state.sqlite3")
    dataset.add_argument("--dataset-dir", default="dataset/gmd")
    dataset.add_argument("--metadata-jsonl", default="dataset/metadata/levels.jsonl")
    dataset.add_argument("--limit", type=int, default=20)
    dataset.add_argument("--query", default="")
    dataset.add_argument("--global-limiter-db")
    dataset.add_argument(
        "--json-only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    dataset.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    dataset.add_argument("--archive-dir")
    dataset.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    dataset.set_defaults(func=_dataset_command)

    return parser


def main() -> None:
    _log("Parsing CLI arguments")
    parser = build_parser()
    args = parser.parse_args()
    _log(f"Executing subcommand '{args.func.__name__}'")
    args.func(args)


if __name__ == "__main__":
    main()
