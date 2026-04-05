from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Run gdlevelai sampling")
    parser.add_argument("--model-path", default="models/autoregressive.pt")
    parser.add_argument("--out-path", default="generated/generated_level.gmd")
    parser.add_argument(
        "--model-type",
        choices=("autoregressive", "diffusion"),
        default="autoregressive",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("extra_args", nargs="*")
    args = parser.parse_args()

    command = [
        sys.executable,
        "-m",
        "gdlevelai",
        "sample",
        "--model-path",
        args.model_path,
        "--out-path",
        args.out_path,
        "--model-type",
        args.model_type,
        "--device",
        args.device,
    ]
    command.extend(args.extra_args)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
