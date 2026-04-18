from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from utils import model_key, resolve_device, resolve_indices_path
from model import ENPSClassifier, ModelConfig
from preprocessing import TextPreprocessConfig, load_training_pairs, split_binary
from train_model import ENPSDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate two-stage ENPS classifier")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--binary-checkpoint", type=str, default=None)
    parser.add_argument("--multiclass-checkpoint", type=str, default=None)
    parser.add_argument("--models-root", type=str, default="models")
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument(
        "--binary-models",
        nargs="+",
        default=None,
        help="HF model names for binary stage in grid mode. If omitted, use all available binary checkpoints.",
    )
    parser.add_argument(
        "--multiclass-models",
        nargs="+",
        default=None,
        help="HF model names for multiclass stage in grid mode. If omitted, use all available multiclass checkpoints.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "grid"],
        default="single",
        help="single: validate one binary+multiclass pair; grid: validate all binary x multiclass best checkpoints",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)

    parser.add_argument("--eval-split", type=str, choices=["train", "test", "all"], default="test")
    parser.add_argument("--train-indices-path", type=str, default="data/train_indices.npy")
    parser.add_argument("--test-indices-path", type=str, default="data/test_indices.npy")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-seed", type=int, default=42)

    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--syntax-correction", action="store_true")
    parser.add_argument("--lemmatization", action="store_true")
    parser.add_argument("--stopwords-removal", action="store_true")
    parser.add_argument("--russian-words-path", type=str, default=None)
    parser.add_argument("--symspell-dict-path", type=str, default=None)
    return parser.parse_args()


def _load_model(checkpoint_path: str, device: torch.device) -> ENPSClassifier:
    payload = torch.load(checkpoint_path, map_location=device)
    model_cfg = ModelConfig(**payload["model_config"])
    model = ENPSClassifier(model_cfg)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _make_loader(
    tokenizer,
    texts: list[str],
    scores: np.ndarray,
    labels: np.ndarray | None,
    max_length: int,
    batch_size: int,
) -> DataLoader:
    encoded = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="np",
    )
    ds = ENPSDataset(
        encodings={"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]},
        scores=scores,
        labels=labels,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


@torch.no_grad()
def _predict_binary_logits(model: ENPSClassifier, loader: DataLoader, device: torch.device) -> np.ndarray:
    logits_all = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["attention_mask"], batch["scores"])
        logits_all.append(logits.cpu().numpy())
    return np.concatenate(logits_all, axis=0)


@torch.no_grad()
def _predict_multiclass_logits(model: ENPSClassifier, loader: DataLoader, device: torch.device) -> np.ndarray:
    logits_all = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["attention_mask"], batch["scores"])
        logits_all.append(logits.cpu().numpy())
    return np.concatenate(logits_all, axis=0)


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    roc_auc = roc_auc_score(y_true, y_prob) if np.unique(y_true).size > 1 else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
    }


def _macro_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def _discover_checkpoints(models_root: Path, stage: str) -> list[Path]:
    found: list[Path] = []
    if not models_root.exists():
        return found
    for model_dir in models_root.iterdir():
        if not model_dir.is_dir():
            continue
        ckpt = model_dir / stage / "best_ckpt.pt"
        if ckpt.exists():
            found.append(ckpt)
    return sorted(found)


def _resolve_checkpoints(
    models_root: Path,
    stage: str,
    requested_models: list[str] | None,
) -> list[Path]:
    if not requested_models:
        return _discover_checkpoints(models_root, stage)

    resolved: list[Path] = []
    missing: list[str] = []
    for model_name in requested_models:
        ckpt = models_root / model_key(model_name) / stage / "best_ckpt.pt"
        if ckpt.exists():
            resolved.append(ckpt)
        else:
            missing.append(model_name)
    if missing:
        print(f"[warn] Missing {stage} checkpoints for: {', '.join(missing)}")
    return sorted(resolved)


def _build_binary_cache_entry(
    eval_df: pd.DataFrame,
    binary_checkpoint: str,
    binary_model: ENPSClassifier,
    binary_tokenizer,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    binary_loader = _make_loader(
        tokenizer=binary_tokenizer,
        texts=eval_df["text"].astype(str).tolist(),
        scores=eval_df["score"].to_numpy(dtype=np.float32),
        labels=eval_df["binary_label"].to_numpy(dtype=np.float32),
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    binary_logits = _predict_binary_logits(binary_model, binary_loader, device)
    binary_probs = 1.0 / (1.0 + np.exp(-binary_logits))
    binary_pred = (binary_probs >= 0.5).astype(int)
    y_true_binary = eval_df["binary_label"].to_numpy(dtype=int)
    stage1_metrics = _binary_metrics(y_true_binary, binary_pred, binary_probs)

    return {
        "binary_checkpoint": binary_checkpoint,
        "binary_model_name": binary_model.cfg.model_name,
        "binary_probs": binary_probs,
        "binary_pred": binary_pred,
        "stage1_metrics": stage1_metrics,
    }


def _evaluate_pair(
    eval_df: pd.DataFrame,
    binary_cached: dict[str, object],
    multiclass_checkpoint: str,
    multiclass_model: ENPSClassifier,
    multiclass_tokenizer,
    args: argparse.Namespace,
    device: torch.device,
    out_dir: Path,
) -> None:
    eval_df = eval_df.copy()
    eval_df["stage1_prob"] = np.asarray(binary_cached["binary_probs"])
    eval_df["stage1_pred"] = np.asarray(binary_cached["binary_pred"])

    oracle_pos = eval_df[eval_df["label"] > 1].copy()
    if not oracle_pos.empty:
        oracle_loader = _make_loader(
            tokenizer=multiclass_tokenizer,
            texts=oracle_pos["text"].astype(str).tolist(),
            scores=oracle_pos["score"].to_numpy(dtype=np.float32),
            labels=None,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )
        oracle_logits = _predict_multiclass_logits(multiclass_model, oracle_loader, device)
        oracle_pred = np.argmax(oracle_logits, axis=1)
        oracle_true = (oracle_pos["label"].to_numpy(dtype=int) - 2).astype(int)
        stage2_oracle_metrics = _macro_metrics(oracle_true, oracle_pred)
    else:
        stage2_oracle_metrics = {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0}

    eval_df["final_pred_merged"] = 0
    pred_pos = eval_df[eval_df["stage1_pred"] == 1].copy()
    if not pred_pos.empty:
        pred_pos_loader = _make_loader(
            tokenizer=multiclass_tokenizer,
            texts=pred_pos["text"].astype(str).tolist(),
            scores=pred_pos["score"].to_numpy(dtype=np.float32),
            labels=None,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )
        pred_pos_logits = _predict_multiclass_logits(multiclass_model, pred_pos_loader, device)
        pred_pos_cls = np.argmax(pred_pos_logits, axis=1)
        eval_df.loc[pred_pos.index, "final_pred_merged"] = pred_pos_cls + 1

    y_true_merged = np.where(eval_df["label"].to_numpy(dtype=int) <= 1, 0, eval_df["label"].to_numpy(dtype=int) - 1)
    y_pred_merged = eval_df["final_pred_merged"].to_numpy(dtype=int)
    end_to_end_metrics = _macro_metrics(y_true_merged, y_pred_merged)

    report = {
        "eval_split": args.eval_split,
        "num_samples": int(len(eval_df)),
        "stage1_binary_metrics": binary_cached["stage1_metrics"],
        "stage2_oracle_metrics": stage2_oracle_metrics,
        "end_to_end_merged_metrics": end_to_end_metrics,
        "binary_checkpoint": binary_cached["binary_checkpoint"],
        "multiclass_checkpoint": multiclass_checkpoint,
        "binary_model_name": binary_cached["binary_model_name"],
        "multiclass_model_name": multiclass_model.cfg.model_name,
    }

    with open(out_dir / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    pred_cols = ["text", "score", "label", "binary_label", "stage1_prob", "stage1_pred", "final_pred_merged"]
    eval_df.loc[:, pred_cols].to_csv(out_dir / "validation_predictions.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    preprocess_cfg = TextPreprocessConfig(
        syntax_correction=args.syntax_correction,
        lemmatization=args.lemmatization,
        stopwords_removal=args.stopwords_removal,
        russian_words_path=args.russian_words_path,
        symspell_dict_path=args.symspell_dict_path,
    )

    print("[1/5] Loading data...")
    full_df = load_training_pairs(args.data_path, preprocess_cfg)

    train_idx_path = resolve_indices_path(args.train_indices_path)
    test_idx_path = resolve_indices_path(args.test_indices_path)
    train_df, test_df, _, _ = split_binary(
        full_df,
        test_size=args.test_size,
        random_state=args.random_seed,
        train_indices_path=train_idx_path,
        test_indices_path=test_idx_path,
    )
    if args.eval_split == "train":
        eval_df = train_df.copy()
    elif args.eval_split == "test":
        eval_df = test_df.copy()
    else:
        eval_df = full_df.copy()
        eval_df["binary_label"] = (eval_df["label"] > 1).astype(np.float32)

    if args.mode == "single":
        if not args.binary_checkpoint or not args.multiclass_checkpoint:
            raise ValueError("For --mode single provide both --binary-checkpoint and --multiclass-checkpoint.")

        print("[2/5] Loading models...")
        binary_model = _load_model(args.binary_checkpoint, device)
        multiclass_model = _load_model(args.multiclass_checkpoint, device)
        if args.output_dir:
            out_dir = Path(args.output_dir)
        else:
            b_key = model_key(binary_model.cfg.model_name)
            m_key = model_key(multiclass_model.cfg.model_name)
            two_stage_key = b_key if b_key == m_key else f"{b_key}__{m_key}"
            out_dir = Path(args.results_root) / two_stage_key / "two-stage"
        out_dir.mkdir(parents=True, exist_ok=True)

        binary_tokenizer = AutoTokenizer.from_pretrained(binary_model.cfg.model_name)
        multiclass_tokenizer = AutoTokenizer.from_pretrained(multiclass_model.cfg.model_name)

        print("[3/5] Evaluating single pair...")
        binary_cached = _build_binary_cache_entry(
            eval_df=eval_df,
            binary_checkpoint=args.binary_checkpoint,
            binary_model=binary_model,
            binary_tokenizer=binary_tokenizer,
            args=args,
            device=device,
        )
        _evaluate_pair(
            eval_df=eval_df,
            binary_cached=binary_cached,
            multiclass_checkpoint=args.multiclass_checkpoint,
            multiclass_model=multiclass_model,
            multiclass_tokenizer=multiclass_tokenizer,
            args=args,
            device=device,
            out_dir=out_dir,
        )
        print("[4/5] Done.")
        print(f"Saved to: {out_dir}")
        return

    print("[2/5] Discovering checkpoints...")
    models_root = Path(args.models_root)
    binary_ckpts = _resolve_checkpoints(models_root, "binary", args.binary_models)
    multiclass_ckpts = _resolve_checkpoints(models_root, "multiclass", args.multiclass_models)
    if not binary_ckpts:
        raise FileNotFoundError(f"No binary checkpoints found in {models_root}")
    if not multiclass_ckpts:
        raise FileNotFoundError(f"No multiclass checkpoints found in {models_root}")

    print("[3/5] Preparing binary stage (one binary model at a time)...")
    binary_failures: list[tuple[str, str]] = []
    any_binary_ok = False
    tokenizer_cache: dict[str, object] = {}
    out_root = Path(args.output_dir) if args.output_dir else Path(args.results_root)
    print("[4/5] Running all binary x multiclass pairs...")
    failures: list[tuple[str, str, str]] = []
    for b_ckpt in binary_ckpts:
        try:
            b_model = _load_model(b_ckpt.as_posix(), device)
            b_tok = AutoTokenizer.from_pretrained(b_model.cfg.model_name)
            b_cached = _build_binary_cache_entry(
                eval_df=eval_df,
                binary_checkpoint=b_ckpt.as_posix(),
                binary_model=b_model,
                binary_tokenizer=b_tok,
                args=args,
                device=device,
            )
            any_binary_ok = True
            print(f"  binary: {b_model.cfg.model_name}")
        except Exception as exc:
            if not args.continue_on_error:
                raise
            binary_failures.append((b_ckpt.as_posix(), str(exc)))
            print(f"[warn] skip binary model {b_ckpt.as_posix()}: {exc}")
            continue

        for m_ckpt in multiclass_ckpts:
            try:
                m_model = _load_model(m_ckpt.as_posix(), device)
                m_model_name = m_model.cfg.model_name
                m_tok = tokenizer_cache.get(m_model_name)
                if m_tok is None:
                    m_tok = AutoTokenizer.from_pretrained(m_model_name)
                    tokenizer_cache[m_model_name] = m_tok
            except Exception as exc:
                msg = f"multiclass model load failed: {exc}"
                if args.continue_on_error:
                    failures.append((b_ckpt.as_posix(), m_ckpt.as_posix(), msg))
                    print(f"[warn] {msg}")
                    continue
                raise

            print(f"    multiclass: {m_model.cfg.model_name}")
            try:
                b_key = model_key(str(b_cached["binary_model_name"]))
                m_key = model_key(m_model.cfg.model_name)
                two_stage_key = b_key if b_key == m_key else f"{b_key}__{m_key}"
                out_dir = out_root / two_stage_key / "two-stage"
                out_dir.mkdir(parents=True, exist_ok=True)

                _evaluate_pair(
                    eval_df=eval_df,
                    binary_cached=b_cached,
                    multiclass_checkpoint=m_ckpt.as_posix(),
                    multiclass_model=m_model,
                    multiclass_tokenizer=m_tok,
                    args=args,
                    device=device,
                    out_dir=out_dir,
                )
                print(f"      saved: {two_stage_key}")
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                failures.append((b_ckpt.as_posix(), m_ckpt.as_posix(), str(exc)))
                print(f"[warn] failed pair b={b_ckpt.as_posix()} m={m_ckpt.as_posix()}: {exc}")
                print(traceback.format_exc(limit=1).strip())

    if not any_binary_ok:
        print("[5/5] No valid binary models after stage-1.")
        for b_ckpt, err in binary_failures:
            print(f"- binary={b_ckpt}, error={err}")
        raise SystemExit(1)

    if failures:
        print("[5/5] Done with failures:")
        for b_ckpt, err in binary_failures:
            print(f"- binary={b_ckpt}, error={err}")
        for b_ckpt, m_ckpt, err in failures:
            print(f"- binary={b_ckpt}, multiclass={m_ckpt}, error={err}")
        if args.continue_on_error:
            print("[5/5] Completed with partial failures (continue-on-error).")
            return
        raise SystemExit(1)

    if binary_failures:
        print("[5/5] Done with skipped binary models:")
        for b_ckpt, err in binary_failures:
            print(f"- binary={b_ckpt}, error={err}")
        if args.continue_on_error:
            print("[5/5] Completed with skipped binary models (continue-on-error).")
            return
        raise SystemExit(1)

    print("[5/5] Done.")


if __name__ == "__main__":
    main()
