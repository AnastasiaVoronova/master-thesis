from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_MODELS = [
    "deepvk/USER2-base",
    "bert-base-multilingual-uncased",
    "deepvk/RuModernBERT-base",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run two-stage training (binary + multiclass) for multiple backbone models."
    )
    parser.add_argument("--data-path", type=str, default="data/enps_full_2025.xlsx")
    parser.add_argument("--models-root", type=str, default="models")
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--train-indices-path", type=str, default="data/train_indices.npy")
    parser.add_argument("--test-indices-path", type=str, default="data/test_indices.npy")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="HF model names for both stages (backward-compatible fallback).",
    )
    parser.add_argument(
        "--binary-models",
        nargs="+",
        default=None,
        help="HF model names for binary stage only.",
    )
    parser.add_argument(
        "--multiclass-models",
        nargs="+",
        default=None,
        help="HF model names for multiclass stage only.",
    )
    parser.add_argument("--stages", choices=["binary", "multiclass", "both"], default="both")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable to run train.py")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_train_cmd(args: argparse.Namespace, stage: str, model_name: str) -> list[str]:
    cmd = [
        args.python,
        "train.py",
        "--train-stage",
        stage,
        "--data-path",
        args.data_path,
        "--models-root",
        args.models_root,
        "--results-root",
        args.results_root,
        "--train-indices-path",
        args.train_indices_path,
        "--test-indices-path",
        args.test_indices_path,
        "--max-length",
        str(args.max_length),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--device",
        args.device,
    ]
    if stage == "binary":
        cmd.extend(["--binary-model-name", model_name])
    else:
        cmd.extend(["--multiclass-model-name", model_name])
    return cmd


def run_cmd(cmd: list[str], dry_run: bool) -> int:
    print("\n$ " + " ".join(cmd))
    if dry_run:
        return 0
    proc = subprocess.run(cmd)
    return proc.returncode


def main() -> None:
    args = parse_args()
    stages = ["binary", "multiclass"] if args.stages == "both" else [args.stages]

    common_models = args.models if args.models is not None else DEFAULT_MODELS
    binary_models = args.binary_models if args.binary_models is not None else common_models
    multiclass_models = args.multiclass_models if args.multiclass_models is not None else common_models

    train_py = Path("train.py")
    if not train_py.exists():
        raise FileNotFoundError("train.py not found in current directory")

    failures: list[tuple[str, str, int]] = []

    for stage in stages:
        stage_models = binary_models if stage == "binary" else multiclass_models
        for model_name in stage_models:
            code = run_cmd(build_train_cmd(args, stage, model_name), dry_run=args.dry_run)
            if code != 0:
                failures.append((model_name, stage, code))
                if not args.continue_on_error:
                    print(f"\nStopped on first failure: model={model_name}, stage={stage}, code={code}")
                    raise SystemExit(code)

    if failures:
        print("\nCompleted with failures:")
        for model_name, stage, code in failures:
            print(f"- model={model_name}, stage={stage}, exit_code={code}")
        raise SystemExit(1)

    print("\nAll runs completed successfully.")


if __name__ == "__main__":
    main()
