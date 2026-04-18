from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from utils import model_key, resolve_device, resolve_indices_path, set_seed
from model import ENPSClassifier, ModelConfig
from preprocessing import (
    TextPreprocessConfig,
    build_multiclass_from_binary_split,
    load_training_pairs,
    save_split_indices,
    split_binary,
)
from train_model import (
    ENPSDataset,
    UnfreezeStage,
    load_plan_from_json,
    train_with_plan,
)


def make_encodings(tokenizer, texts: list[str], max_length: int) -> dict[str, np.ndarray]:
    encoded = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="np",
    )
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }


def build_loader(
    tokenizer,
    texts: list[str],
    scores: np.ndarray,
    labels: np.ndarray,
    max_length: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    enc = make_encodings(tokenizer, texts, max_length=max_length)
    ds = ENPSDataset(encodings=enc, scores=scores, labels=labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-stage ENPS text classifier training")
    parser.add_argument("--data-path", type=str, required=True, help="Path to XLSX/CSV with Score/A1/C1/A2/C2")
    parser.add_argument("--models-root", type=str, default="models")
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument(
        "--train-stage",
        type=str,
        choices=["binary", "multiclass"],
        default="binary",
        help="Which stage to train",
    )

    parser.add_argument("--binary-model-name", type=str, default="bert-base-multilingual-uncased")
    parser.add_argument("--multiclass-model-name", type=str, default="bert-base-multilingual-uncased")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--binary-plan-json", type=str, default=None)
    parser.add_argument("--multiclass-plan-json", type=str, default=None)

    parser.add_argument("--train-indices-path", type=str, default="data/train_indices.npy")
    parser.add_argument("--test-indices-path", type=str, default="data/test_indices.npy")

    parser.add_argument("--syntax-correction", action="store_true")
    parser.add_argument("--lemmatization", action="store_true")
    parser.add_argument("--stopwords-removal", action="store_true")
    parser.add_argument("--russian-words-path", type=str, default=None)
    parser.add_argument("--symspell-dict-path", type=str, default=None)

    # Binary gradual unfreeze plan (3 stages)
    parser.add_argument("--binary-stage1-epochs", type=int, default=18)
    parser.add_argument("--binary-stage1-lr", type=float, default=1e-4)
    parser.add_argument("--binary-stage1-warmup-ratio", type=float, default=0.10)
    parser.add_argument("--binary-stage1-early-stopping", type=int, default=7)

    parser.add_argument("--binary-stage2-epochs", type=int, default=18)
    parser.add_argument("--binary-stage2-lr", type=float, default=5e-5)
    parser.add_argument("--binary-stage2-warmup-ratio", type=float, default=0.10)
    parser.add_argument("--binary-stage2-early-stopping", type=int, default=7)

    parser.add_argument("--binary-stage3-epochs", type=int, default=18)
    parser.add_argument("--binary-stage3-lr", type=float, default=5e-5)
    parser.add_argument("--binary-stage3-warmup-ratio", type=float, default=0.10)
    parser.add_argument("--binary-stage3-early-stopping", type=int, default=7)

    # Multiclass gradual unfreeze plan (3 stages)
    parser.add_argument("--multi-head-epochs", type=int, default=18)
    parser.add_argument("--multi-head-lr", type=float, default=3e-5)
    parser.add_argument("--multi-head-warmup-ratio", type=float, default=0.06)
    parser.add_argument("--multi-head-early-stopping", type=int, default=10)

    parser.add_argument("--multi-stage1-epochs", type=int, default=50)
    parser.add_argument("--multi-stage1-lr", type=float, default=3e-5)
    parser.add_argument("--multi-stage1-warmup-ratio", type=float, default=0.06)
    parser.add_argument("--multi-stage1-early-stopping", type=int, default=10)

    parser.add_argument("--multi-stage2-epochs", type=int, default=50)
    parser.add_argument("--multi-stage2-lr", type=float, default=1e-5)
    parser.add_argument("--multi-stage2-warmup-ratio", type=float, default=0.06)
    parser.add_argument("--multi-stage2-early-stopping", type=int, default=10)

    return parser.parse_args()


def build_binary_plan_from_args(args: argparse.Namespace) -> list[UnfreezeStage]:
    return [
        UnfreezeStage(
            name="head_only",
            epochs=args.binary_stage1_epochs,
            lr=args.binary_stage1_lr,
            warmup_ratio=args.binary_stage1_warmup_ratio,
            early_stopping=args.binary_stage1_early_stopping,
            unfreeze_pooler=False,
            unfreeze_last_n_layers=0,
            unfreeze_all_encoder_outputs=False,
        ),
        UnfreezeStage(
            name="pooler",
            epochs=args.binary_stage2_epochs,
            lr=args.binary_stage2_lr,
            warmup_ratio=args.binary_stage2_warmup_ratio,
            early_stopping=args.binary_stage2_early_stopping,
            unfreeze_pooler=True,
            unfreeze_last_n_layers=0,
            unfreeze_all_encoder_outputs=False,
        ),
        UnfreezeStage(
            name="encoder_outputs",
            epochs=args.binary_stage3_epochs,
            lr=args.binary_stage3_lr,
            warmup_ratio=args.binary_stage3_warmup_ratio,
            early_stopping=args.binary_stage3_early_stopping,
            unfreeze_pooler=True,
            unfreeze_last_n_layers=0,
            unfreeze_all_encoder_outputs=True,
        ),
    ]


def build_multiclass_plan_from_args(args: argparse.Namespace) -> list[UnfreezeStage]:
    # Matches the staged unfreezing schedule used in the original notebook:
    # head-only -> pooler -> pooler+last{1,2,3,4,5,8,11} layers.
    return [
        UnfreezeStage(
            name="head_only",
            epochs=100,
            lr=1e-4,
            warmup_ratio=0.03,
            early_stopping=10,
            unfreeze_pooler=False,
            unfreeze_last_n_layers=0,
        ),
        UnfreezeStage(
            name="pooler",
            epochs=100,
            lr=5e-5,
            warmup_ratio=0.03,
            early_stopping=10,
            unfreeze_pooler=True,
            unfreeze_last_n_layers=0,
        ),
        UnfreezeStage(
            name="pooler_plus_last1",
            epochs=50,
            lr=5e-5,
            warmup_ratio=0.06,
            early_stopping=10,
            unfreeze_pooler=True,
            unfreeze_last_n_layers=1,
        ),
        UnfreezeStage(
            name="pooler_plus_last2",
            epochs=50,
            lr=5e-5,
            warmup_ratio=0.06,
            early_stopping=10,
            unfreeze_pooler=True,
            unfreeze_last_n_layers=2,
        ),
        UnfreezeStage(
            name="pooler_plus_last3",
            epochs=50,
            lr=5e-5,
            warmup_ratio=0.06,
            early_stopping=10,
            unfreeze_pooler=True,
            unfreeze_last_n_layers=3,
        ),
        UnfreezeStage(
            name="pooler_plus_last4",
            epochs=50,
            lr=5e-5,
            warmup_ratio=0.06,
            early_stopping=10,
            unfreeze_pooler=True,
            unfreeze_last_n_layers=4,
        ),
        UnfreezeStage(
            name="pooler_plus_last5",
            epochs=50,
            lr=5e-5,
            warmup_ratio=0.06,
            early_stopping=10,
            unfreeze_pooler=True,
            unfreeze_last_n_layers=5,
        ),
        UnfreezeStage(
            name="pooler_plus_last8",
            epochs=50,
            lr=3e-5,
            warmup_ratio=0.06,
            early_stopping=10,
            unfreeze_pooler=True,
            unfreeze_last_n_layers=8,
        ),
        UnfreezeStage(
            name="pooler_plus_last11",
            epochs=50,
            lr=1e-5,
            warmup_ratio=0.06,
            early_stopping=10,
            unfreeze_pooler=True,
            unfreeze_last_n_layers=11,
        ),
    ]


def main() -> None:
    args = parse_args()
    set_seed(args.random_seed)
    active_model_name = args.binary_model_name if args.train_stage == "binary" else args.multiclass_model_name
    active_model_key = model_key(active_model_name)
    stage_results_dir = Path(args.results_root) / active_model_key / args.train_stage

    preprocess_cfg = TextPreprocessConfig(
        syntax_correction=args.syntax_correction,
        lemmatization=args.lemmatization,
        stopwords_removal=args.stopwords_removal,
        russian_words_path=args.russian_words_path,
        symspell_dict_path=args.symspell_dict_path,
    )

    print("[1/4] Loading and preprocessing training pairs...")
    full_df = load_training_pairs(args.data_path, preprocess_cfg)

    print("[2/4] Preparing split...")
    train_indices_path = resolve_indices_path(args.train_indices_path)
    test_indices_path = resolve_indices_path(args.test_indices_path)
    if args.train_indices_path and not train_indices_path:
        print(f"[warn] train indices not found at {args.train_indices_path}; fallback to random split.")
    if args.test_indices_path and not test_indices_path:
        print(f"[warn] test indices not found at {args.test_indices_path}; fallback to random split.")

    b_train, b_valid, train_idx, test_idx = split_binary(
        full_df,
        test_size=args.test_size,
        random_state=args.random_seed,
        train_indices_path=train_indices_path,
        test_indices_path=test_indices_path,
    )
    split_dir = stage_results_dir / "splits"
    save_split_indices(train_idx, test_idx, split_dir.as_posix())

    device = resolve_device(args.device)

    run_summary = {
        "data_path": args.data_path,
        "train_stage": args.train_stage,
        "binary_model_name": args.binary_model_name,
        "multiclass_model_name": args.multiclass_model_name,
        "models_root": args.models_root,
        "results_root": args.results_root,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "test_size": args.test_size,
        "seed": args.random_seed,
    }

    if args.train_stage == "binary":
        print("[3/4] Stage-1 binary training...")
        binary_tokenizer = AutoTokenizer.from_pretrained(args.binary_model_name)
        binary_train_loader = build_loader(
            tokenizer=binary_tokenizer,
            texts=b_train["text"].astype(str).tolist(),
            scores=b_train["score"].to_numpy(dtype=np.float32),
            labels=b_train["binary_label"].to_numpy(dtype=np.float32),
            max_length=args.max_length,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
        )
        binary_valid_loader = build_loader(
            tokenizer=binary_tokenizer,
            texts=b_valid["text"].astype(str).tolist(),
            scores=b_valid["score"].to_numpy(dtype=np.float32),
            labels=b_valid["binary_label"].to_numpy(dtype=np.float32),
            max_length=args.max_length,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )
        binary_model = ENPSClassifier(
            ModelConfig(model_name=args.binary_model_name, stage="binary", num_labels=1)
        ).to(device)
        binary_plan = load_plan_from_json(args.binary_plan_json, build_binary_plan_from_args(args))
        binary_summary = train_with_plan(
            model=binary_model,
            train_loader=binary_train_loader,
            valid_loader=binary_valid_loader,
            stage_name="binary",
            model_output_dir=(Path(args.models_root) / model_key(args.binary_model_name) / "binary").as_posix(),
            results_output_dir=(Path(args.results_root) / model_key(args.binary_model_name) / "binary").as_posix(),
            plan=binary_plan,
            device=device,
        )
        run_summary["binary_summary"] = binary_summary
        run_summary_path = stage_results_dir / "run_summary.json"
    else:
        print("[3/4] Multiclass split from binary indices...")
        m_train, m_valid = build_multiclass_from_binary_split(full_df, train_idx=train_idx, test_idx=test_idx)

        print("[4/4] Stage-2 multiclass training...")
        multiclass_tokenizer = AutoTokenizer.from_pretrained(args.multiclass_model_name)
        multiclass_train_loader = build_loader(
            tokenizer=multiclass_tokenizer,
            texts=m_train["text"].astype(str).tolist(),
            scores=m_train["score"].to_numpy(dtype=np.float32),
            labels=m_train["multi_label"].to_numpy(dtype=np.int64),
            max_length=args.max_length,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
        )
        multiclass_valid_loader = build_loader(
            tokenizer=multiclass_tokenizer,
            texts=m_valid["text"].astype(str).tolist(),
            scores=m_valid["score"].to_numpy(dtype=np.float32),
            labels=m_valid["multi_label"].to_numpy(dtype=np.int64),
            max_length=args.max_length,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )
        multiclass_model = ENPSClassifier(
            ModelConfig(model_name=args.multiclass_model_name, stage="multiclass", num_labels=28)
        ).to(device)
        multiclass_plan = load_plan_from_json(
            args.multiclass_plan_json, build_multiclass_plan_from_args(args)
        )
        multiclass_summary = train_with_plan(
            model=multiclass_model,
            train_loader=multiclass_train_loader,
            valid_loader=multiclass_valid_loader,
            stage_name="multiclass",
            model_output_dir=(Path(args.models_root) / model_key(args.multiclass_model_name) / "multiclass").as_posix(),
            results_output_dir=(Path(args.results_root) / model_key(args.multiclass_model_name) / "multiclass").as_posix(),
            plan=multiclass_plan,
            device=device,
        )
        run_summary["multiclass_summary"] = multiclass_summary
        run_summary_path = stage_results_dir / "run_summary.json"

    print("Saving run summary...")
    run_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(run_summary_path, "w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)

    print("Training completed.")


if __name__ == "__main__":
    main()
