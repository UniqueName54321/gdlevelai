from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def _run(command: list[str]) -> None:
    print(f"[colab_bootstrap] Running: {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)


def _expand_extra_args(values: list[str]) -> list[str]:
    expanded: list[str] = []
    for value in values:
        expanded.extend(shlex.split(value))
    return expanded


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap a Colab-friendly smoke train + sample run"
    )
    parser.add_argument(
        "--install-deps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Install dependencies from requirements.txt",
    )
    parser.add_argument("--requirements", default="requirements.txt")
    parser.add_argument("--dataset-dir", default="dataset/gmd_overfit")
    parser.add_argument("--model-out", default="models/colab_smoke_autoregressive.pt")
    parser.add_argument("--sample-out", default="generated/colab_smoke_level.gmd")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps-per-epoch", type=int, default=20)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-sample", action="store_true")
    parser.add_argument(
        "--train-extra",
        action="append",
        default=[],
        help="Extra args to append to train command (can be repeated)",
    )
    parser.add_argument(
        "--sample-extra",
        action="append",
        default=[],
        help="Extra args to append to sample command (can be repeated)",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    model_out = Path(args.model_out)
    sample_out = Path(args.sample_out)

    model_out.parent.mkdir(parents=True, exist_ok=True)
    sample_out.parent.mkdir(parents=True, exist_ok=True)

    if args.install_deps:
        requirements = Path(args.requirements)
        if not requirements.exists():
            raise FileNotFoundError(f"requirements file not found: {requirements}")
        _run([sys.executable, "-m", "pip", "install", "-r", str(requirements)])

    if not args.skip_train:
        if not dataset_dir.exists():
            raise FileNotFoundError(
                f"dataset directory not found: {dataset_dir}. "
                "Run scripts/prep_dataset.py first or pass --dataset-dir to an existing folder."
            )

        train_command = [
            sys.executable,
            "-m",
            "gdlevelai",
            "train",
            "--dataset-dir",
            str(dataset_dir),
            "--model-out",
            str(model_out),
            "--epochs",
            str(args.epochs),
            "--device",
            args.device,
            "--num-threads",
            str(args.num_threads),
            "--stride",
            str(args.stride),
            "--max-steps-per-epoch",
            str(args.max_steps_per_epoch),
        ]
        train_command.extend(_expand_extra_args(args.train_extra))
        _run(train_command)

    if not args.skip_sample:
        if not model_out.exists():
            raise FileNotFoundError(
                f"model checkpoint not found: {model_out}. "
                "Run without --skip-train or point --model-out to an existing checkpoint."
            )

        sample_command = [
            sys.executable,
            "-m",
            "gdlevelai",
            "sample",
            "--model-path",
            str(model_out),
            "--out-path",
            str(sample_out),
            "--device",
            args.device,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--seed",
            str(args.seed),
        ]
        sample_command.extend(_expand_extra_args(args.sample_extra))
        _run(sample_command)

    print("[colab_bootstrap] Done", flush=True)


if __name__ == "__main__":
    main()
