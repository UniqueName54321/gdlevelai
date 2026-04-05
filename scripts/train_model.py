from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run gdlevelai training with safe defaults"
    )
    parser.add_argument("--dataset-dir", default="dataset/gmd")
    parser.add_argument("--model-out", default="models/autoregressive.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("extra_args", nargs="*")
    args = parser.parse_args()

    command = [
        sys.executable,
        "-m",
        "gdlevelai",
        "train",
        "--dataset-dir",
        args.dataset_dir,
        "--model-out",
        args.model_out,
        "--epochs",
        str(args.epochs),
        "--device",
        args.device,
        "--num-threads",
        str(args.num_threads),
        "--stride",
        str(args.stride),
    ]
    if args.resume_checkpoint:
        command.extend(["--resume-checkpoint", args.resume_checkpoint])
    command.extend(args.extra_args)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
