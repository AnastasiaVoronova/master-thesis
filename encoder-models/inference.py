from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from utils import resolve_device
from model import ENPSClassifier, ModelConfig
from preprocessing import TextPreprocessConfig, index_to_category, load_data, preprocess_text_series
from train_model import ENPSDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-stage ENPS inference")
    parser.add_argument("--input-path", type=str, required=True, help="Input XLSX/CSV with Score, A1, A2")
    parser.add_argument("--output-path", type=str, required=True, help="Path for resulting CSV")

    parser.add_argument("--binary-checkpoint", type=str, required=True, help="Checkpoint for stage-1 binary model")
    parser.add_argument(
        "--multiclass-checkpoint", type=str, required=True, help="Checkpoint for stage-2 multiclass model"
    )

    parser.add_argument("--binary-model-name", type=str, default=None, help="Optional tokenizer/model override for stage-1")
    parser.add_argument(
        "--multiclass-model-name", type=str, default=None, help="Optional tokenizer/model override for stage-2"
    )

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--binary-max-length", type=int, default=64)
    parser.add_argument("--multiclass-max-length", type=int, default=64)
    parser.add_argument("--binary-threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--syntax-correction", action="store_true")
    parser.add_argument("--lemmatization", action="store_true")
    parser.add_argument("--stopwords-removal", action="store_true")
    parser.add_argument("--russian-words-path", type=str, default=None)
    parser.add_argument("--symspell-dict-path", type=str, default=None)
    return parser.parse_args()


def _load_model_from_checkpoint(path: str, device: torch.device, expected_stage: str) -> ENPSClassifier:
    payload = torch.load(path, map_location=device)
    cfg = ModelConfig(**payload["model_config"])
    if cfg.stage != expected_stage:
        raise ValueError(f"Checkpoint {path} has stage='{cfg.stage}', expected '{expected_stage}'.")
    model = ENPSClassifier(cfg)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _tokenize(tokenizer, texts: list[str], max_length: int) -> dict[str, np.ndarray]:
    encoded = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="np",
    )
    return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}


def _make_inference_loader(
    tokenizer,
    texts: list[str],
    scores: np.ndarray,
    max_length: int,
    batch_size: int,
) -> DataLoader:
    enc = _tokenize(tokenizer, texts, max_length=max_length)
    ds = ENPSDataset(encodings=enc, scores=scores, labels=None)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


@torch.no_grad()
def _predict_binary_probs(
    model: ENPSClassifier,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    probs_all: list[np.ndarray] = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["attention_mask"], batch["scores"]).cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))
        probs_all.append(probs)
    return np.concatenate(probs_all, axis=0)


@torch.no_grad()
def _predict_multiclass(
    model: ENPSClassifier,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    logits_all: list[np.ndarray] = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["attention_mask"], batch["scores"]).cpu().numpy()
        logits_all.append(logits)
    logits = np.concatenate(logits_all, axis=0)
    return np.argmax(logits, axis=1)


def _run_column_inference(
    df: pd.DataFrame,
    col: str,
    binary_model: ENPSClassifier,
    multi_model: ENPSClassifier,
    binary_tokenizer,
    multi_tokenizer,
    preprocess_cfg: TextPreprocessConfig,
    binary_max_length: int,
    multiclass_max_length: int,
    binary_threshold: float,
    batch_size: int,
    device: torch.device,
) -> pd.DataFrame:
    pairs = df.loc[:, ["Score", col]].copy().rename(columns={"Score": "score", col: "text"})
    pairs["text"] = preprocess_text_series(pairs["text"], preprocess_cfg)
    pairs = pairs.dropna(subset=["text"]).copy()
    if pairs.empty:
        return pairs.assign(problem_stated=False, class_idx=np.nan, category=np.nan)

    binary_loader = _make_inference_loader(
        tokenizer=binary_tokenizer,
        texts=pairs["text"].astype(str).tolist(),
        scores=pairs["score"].to_numpy(dtype=np.float32),
        max_length=binary_max_length,
        batch_size=batch_size,
    )
    problem_prob = _predict_binary_probs(binary_model, binary_loader, device)
    pairs["problem_probability"] = problem_prob
    pairs["problem_stated"] = (problem_prob >= binary_threshold).astype(bool)

    pairs["class_idx"] = np.nan
    pairs["category"] = np.nan

    positives = pairs[pairs["problem_stated"]].copy()
    if not positives.empty:
        multi_loader = _make_inference_loader(
            tokenizer=multi_tokenizer,
            texts=positives["text"].astype(str).tolist(),
            scores=positives["score"].to_numpy(dtype=np.float32),
            max_length=multiclass_max_length,
            batch_size=batch_size,
        )
        pred_multi = _predict_multiclass(multi_model, multi_loader, device)
        class_idx = pred_multi + 2
        positives["class_idx"] = class_idx
        positives["category"] = [index_to_category(int(v)) for v in class_idx]
        pairs.loc[positives.index, "class_idx"] = positives["class_idx"]
        pairs.loc[positives.index, "category"] = positives["category"]

    return pairs


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

    binary_model = _load_model_from_checkpoint(args.binary_checkpoint, device, expected_stage="binary")
    multiclass_model = _load_model_from_checkpoint(
        args.multiclass_checkpoint, device, expected_stage="multiclass"
    )

    binary_tokenizer = AutoTokenizer.from_pretrained(args.binary_model_name or binary_model.cfg.model_name)
    multi_tokenizer = AutoTokenizer.from_pretrained(args.multiclass_model_name or multiclass_model.cfg.model_name)

    raw = load_data(args.input_path)
    result = raw.copy()

    col1 = _run_column_inference(
        df=raw,
        col="A1",
        binary_model=binary_model,
        multi_model=multiclass_model,
        binary_tokenizer=binary_tokenizer,
        multi_tokenizer=multi_tokenizer,
        preprocess_cfg=preprocess_cfg,
        binary_max_length=args.binary_max_length,
        multiclass_max_length=args.multiclass_max_length,
        binary_threshold=args.binary_threshold,
        batch_size=args.batch_size,
        device=device,
    )
    col2 = _run_column_inference(
        df=raw,
        col="A2",
        binary_model=binary_model,
        multi_model=multiclass_model,
        binary_tokenizer=binary_tokenizer,
        multi_tokenizer=multi_tokenizer,
        preprocess_cfg=preprocess_cfg,
        binary_max_length=args.binary_max_length,
        multiclass_max_length=args.multiclass_max_length,
        binary_threshold=args.binary_threshold,
        batch_size=args.batch_size,
        device=device,
    )

    result["cat1"] = "РЅРµС‚ РѕС‚РІРµС‚Р°"
    result["cat2"] = "РЅРµС‚ РѕС‚РІРµС‚Р°"
    if not col1.empty:
        result.loc[col1.index, "cat1"] = col1["category"].fillna("РЅРµС‚ РѕС‚РІРµС‚Р°")
    if not col2.empty:
        result.loc[col2.index, "cat2"] = col2["category"].fillna("РЅРµС‚ РѕС‚РІРµС‚Р°")

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Inference completed. Saved to: {out_path}")


if __name__ == "__main__":
    main()
