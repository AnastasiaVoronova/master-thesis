from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

from model import ENPSClassifier


class ENPSDataset(Dataset):
    def __init__(self, encodings: dict[str, np.ndarray], scores: np.ndarray, labels: Optional[np.ndarray] = None):
        self.input_ids = torch.tensor(encodings["input_ids"], dtype=torch.long)
        self.attention_mask = torch.tensor(encodings["attention_mask"], dtype=torch.long)
        self.scores = torch.tensor(scores, dtype=torch.float32)
        self.labels = None if labels is None else torch.tensor(labels)

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "scores": self.scores[idx],
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


@dataclass
class UnfreezeStage:
    name: str
    epochs: int
    lr: float
    warmup_ratio: float
    early_stopping: int
    unfreeze_pooler: bool = False
    unfreeze_last_n_layers: int = 0
    unfreeze_all_encoder_outputs: bool = False
    unfreeze_all_encoder: bool = False


def default_binary_plan() -> list[UnfreezeStage]:
    return [
        UnfreezeStage(
            name="head_only",
            epochs=18,
            lr=1e-4,
            warmup_ratio=0.10,
            early_stopping=7,
            unfreeze_pooler=False,
            unfreeze_last_n_layers=0,
            unfreeze_all_encoder_outputs=False,
        ),
        UnfreezeStage(
            name="pooler",
            epochs=18,
            lr=5e-5,
            warmup_ratio=0.10,
            early_stopping=7,
            unfreeze_pooler=True,
            unfreeze_last_n_layers=0,
            unfreeze_all_encoder_outputs=False,
        ),
        UnfreezeStage(
            name="encoder_outputs",
            epochs=18,
            lr=5e-5,
            warmup_ratio=0.10,
            early_stopping=7,
            unfreeze_pooler=True,
            unfreeze_last_n_layers=0,
            unfreeze_all_encoder_outputs=True,
        ),
    ]


def default_multiclass_plan() -> list[UnfreezeStage]:
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


def load_plan_from_json(path: Optional[str], default_plan: list[UnfreezeStage]) -> list[UnfreezeStage]:
    if not path:
        return default_plan
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [UnfreezeStage(**item) for item in raw]


def _set_all_encoder_frozen(model: ENPSClassifier) -> None:
    for p in model.encoder.parameters():
        p.requires_grad = False


def _resolve_module_by_path(root: nn.Module, path: str) -> Optional[nn.Module]:
    cur: nn.Module = root
    for part in path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
        if not isinstance(cur, nn.Module):
            return None
    return cur


def _as_layer_list(module: Optional[nn.Module]) -> list[nn.Module]:
    if module is None:
        return []
    if isinstance(module, nn.ModuleList):
        return list(module)
    if isinstance(module, nn.Sequential):
        return list(module)
    return []


def _get_encoder_layers(model: ENPSClassifier) -> list[nn.Module]:
    encoder = model.encoder

    # Fast-path for common known layouts.
    candidate_paths = (
        "encoder.layer",  # BERT/RoBERTa/Electra style
        "layers",  # ModernBERT and similar
        "layer",  # Some wrappers expose this directly
        "block",  # T5-style encoder stack
        "transformer.layer",  # ALBERT-like layouts
        "transformer.h",  # GPT-like block stack
        "model.layers",  # LLaMA-like wrappers
        "text_model.encoder.layers",  # CLIP-like wrappers
        "backbone.encoder.layer",
        "backbone.layers",
    )
    for path in candidate_paths:
        layers = _as_layer_list(_resolve_module_by_path(encoder, path))
        if layers:
            return layers

    # Generic fallback: choose the most likely transformer block stack.
    best_name = ""
    best_layers: list[nn.Module] = []
    best_score = -10**9
    for name, submodule in encoder.named_modules():
        if not isinstance(submodule, nn.ModuleList) or len(submodule) == 0:
            continue
        if not all(isinstance(m, nn.Module) for m in submodule):
            continue

        lowered = name.lower()
        score = len(submodule)
        if "encoder" in lowered:
            score += 1000
        if "decoder" in lowered:
            score -= 1000
        if any(token in lowered for token in ("layer", "layers", "block", "blocks", "transformer")):
            score += 200

        if score > best_score:
            best_score = score
            best_name = name
            best_layers = list(submodule)

    if best_layers:
        print(f"[unfreeze] detected encoder block stack: {best_name} ({len(best_layers)} layers)")
    else:
        print("[unfreeze] warning: no encoder block stack detected; only head/pooler will be trainable")
    return best_layers


def _find_pooler_module(encoder: nn.Module) -> Optional[nn.Module]:
    candidate_paths = (
        "pooler",
        "encoder.pooler",
        "bert.pooler",
        "roberta.pooler",
        "deberta.pooler",
        "electra.pooler",
        "text_model.pooler",
        "backbone.pooler",
    )
    for path in candidate_paths:
        module = _resolve_module_by_path(encoder, path)
        if module is not None and any(True for _ in module.parameters(recurse=True)):
            return module

    for name, module in encoder.named_modules():
        if name.lower().endswith("pooler") and any(True for _ in module.parameters(recurse=True)):
            return module
    return None


def _unfreeze_pooler_or_fallback(model: ENPSClassifier) -> None:
    encoder = model.encoder
    pooler = _find_pooler_module(encoder)
    if pooler is not None:
        for p in pooler.parameters():
            p.requires_grad = True
        return

    # Models without pooler (for example ModernBERT): unfreeze top-1 encoder block.
    layers = _get_encoder_layers(model)
    if layers:
        for p in layers[-1].parameters():
            p.requires_grad = True


def _unfreeze_encoder_outputs_compat(model: ENPSClassifier) -> None:
    for layer in _get_encoder_layers(model):
        # BERT/RoBERTa-like block.
        if hasattr(layer, "output") and layer.output is not None:
            for p in layer.output.parameters():
                p.requires_grad = True
            continue

        # ModernBERT-like block.
        matched = False
        for attr in ("mlp_norm", "mlp"):
            part = getattr(layer, attr, None)
            if part is not None:
                for p in part.parameters():
                    p.requires_grad = True
                matched = True
        if matched:
            continue

        # Safe fallback for unknown blocks.
        for p in layer.parameters():
            p.requires_grad = True


def apply_unfreeze_stage(model: ENPSClassifier, stage: UnfreezeStage) -> None:
    _set_all_encoder_frozen(model)

    if stage.unfreeze_all_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = True
        return

    if stage.unfreeze_pooler:
        _unfreeze_pooler_or_fallback(model)

    if stage.unfreeze_all_encoder_outputs:
        _unfreeze_encoder_outputs_compat(model)

    if stage.unfreeze_last_n_layers > 0:
        layers = _get_encoder_layers(model)
        for layer in layers[-stage.unfreeze_last_n_layers :]:
            for p in layer.parameters():
                p.requires_grad = True


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


def _macro_roc_auc_ovr(y_true: np.ndarray, logits: np.ndarray) -> float:
    # Macro ROC-AUC for multiclass in one-vs-rest setup.
    # Skip classes that do not have both positive and negative examples in y_true.
    if logits.ndim != 2 or logits.shape[0] == 0:
        return 0.0

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    probs = exp_logits / np.clip(np.sum(exp_logits, axis=1, keepdims=True), a_min=1e-12, a_max=None)

    aucs: list[float] = []
    for cls in range(probs.shape[1]):
        y_bin = (y_true == cls).astype(int)
        if np.unique(y_bin).size < 2:
            continue
        aucs.append(float(roc_auc_score(y_bin, probs[:, cls])))
    if not aucs:
        return 0.0
    return float(np.mean(aucs))


def _multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    roc_auc = _macro_roc_auc_ovr(y_true, logits)

    # Macro specificity через one-vs-rest.
    classes = np.unique(np.concatenate([y_true, y_pred]))
    specs = []
    for cls in classes:
        y_true_bin = (y_true == cls).astype(int)
        y_pred_bin = (y_pred == cls).astype(int)
        tn, fp, _, _ = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    specificity = float(np.mean(specs)) if specs else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
    }


def _compute_metrics(stage_name: str, y_true: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    if stage_name == "binary":
        probs = 1.0 / (1.0 + np.exp(-logits))
        pred = (probs >= 0.5).astype(int)
        return _binary_metrics(y_true.astype(int), pred, probs)
    pred = np.argmax(logits, axis=1)
    return _multiclass_metrics(y_true.astype(int), pred, logits)


def _batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def evaluate_model(
    model: ENPSClassifier,
    dataloader: DataLoader,
    stage_name: str,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    model.eval()
    losses: list[float] = []
    logits_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            batch = _batch_to_device(batch, device)
            labels = batch["labels"]
            logits = model(batch["input_ids"], batch["attention_mask"], batch["scores"])
            if stage_name == "binary":
                loss = loss_fn(logits, labels.float())
            else:
                loss = loss_fn(logits, labels.long())
            losses.append(loss.item())
            logits_all.append(logits.detach().cpu().numpy())
            labels_all.append(labels.detach().cpu().numpy())

    stacked_logits = np.concatenate(logits_all, axis=0)
    stacked_labels = np.concatenate(labels_all, axis=0)
    metrics = _compute_metrics(stage_name, stacked_labels, stacked_logits)
    return float(np.mean(losses)), metrics


def _append_csv_row(path: Path, row: dict[str, object]) -> None:
    write_header = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _make_optimizer_and_scheduler(
    model: ENPSClassifier,
    train_loader: DataLoader,
    stage: UnfreezeStage,
) -> tuple[torch.optim.Optimizer, object]:
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=stage.lr, eps=1e-8)
    total_steps = len(train_loader) * stage.epochs
    warmup_steps = math.ceil(stage.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler


def save_checkpoint(
    model: ENPSClassifier,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_metric: float,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "best_metric": best_metric,
        "model_config": asdict(model.cfg),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    torch.save(payload, path)


def load_checkpoint(model: ENPSClassifier, path: str, map_location: str = "cpu") -> dict:
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model_state_dict"])
    return payload


def train_with_plan(
    model: ENPSClassifier,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    stage_name: str,
    model_output_dir: str,
    results_output_dir: str,
    plan: list[UnfreezeStage],
    device: torch.device,
) -> dict[str, object]:
    model_dir = Path(model_output_dir)
    results_dir = Path(results_output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = results_dir / "metrics.csv"

    if stage_name == "binary":
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()

    best_metric = -1.0
    global_epoch = 0
    best_ckpt_path = model_dir / "best_ckpt.pt"
    last_ckpt_path = model_dir / "last_ckpt.pt"

    for schedule in plan:
        apply_unfreeze_stage(model, schedule)
        optimizer, scheduler = _make_optimizer_and_scheduler(model, train_loader, schedule)

        epochs_without_improve = 0
        for local_epoch in range(schedule.epochs):
            global_epoch += 1
            started = time.time()
            model.train()
            train_losses: list[float] = []

            for batch in train_loader:
                batch = _batch_to_device(batch, device)
                optimizer.zero_grad()
                labels = batch["labels"]
                logits = model(batch["input_ids"], batch["attention_mask"], batch["scores"])
                if stage_name == "binary":
                    loss = loss_fn(logits, labels.float())
                else:
                    loss = loss_fn(logits, labels.long())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                train_losses.append(loss.item())

            avg_train_loss = float(np.mean(train_losses))
            valid_loss, valid_metrics = evaluate_model(model, valid_loader, stage_name, loss_fn, device)
            elapsed = time.time() - started

            row = {
                "global_epoch": global_epoch,
                "stage_name": schedule.name,
                "stage_epoch": local_epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": valid_loss,
                "precision": valid_metrics["precision"],
                "recall": valid_metrics["recall"],
                "specificity": valid_metrics["specificity"],
                "f1": valid_metrics["f1"],
                "roc_auc": valid_metrics.get("roc_auc", 0.0),
                "lr": optimizer.param_groups[0]["lr"],
                "elapsed_sec": elapsed,
            }
            _append_csv_row(metrics_csv, row)
            print(
                f"[{stage_name}] {schedule.name} epoch {local_epoch + 1}/{schedule.epochs} | "
                f"train_loss={avg_train_loss:.5f} val_loss={valid_loss:.5f} "
                f"f1={valid_metrics['f1']:.5f}"
            )

            if valid_metrics["f1"] > best_metric:
                best_metric = valid_metrics["f1"]
                epochs_without_improve = 0
                save_checkpoint(model, optimizer, scheduler, global_epoch, best_metric, best_ckpt_path)
            else:
                epochs_without_improve += 1

            if epochs_without_improve >= schedule.early_stopping:
                print(
                    f"[{stage_name}] early stopping on stage={schedule.name}, "
                    f"patience={schedule.early_stopping}"
                )
                break

        save_checkpoint(model, optimizer, scheduler, global_epoch, best_metric, last_ckpt_path)

    _save_training_plots(metrics_csv, results_dir)

    summary = {
        "stage": stage_name,
        "best_f1": best_metric,
        "metrics_csv": str(metrics_csv),
        "best_checkpoint": str(best_ckpt_path),
        "last_checkpoint": str(last_ckpt_path),
        "results_dir": str(results_dir),
        "model_dir": str(model_dir),
        "plan": [asdict(s) for s in plan],
    }
    with open(results_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def _save_training_plots(metrics_csv: Path, results_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        return

    if not metrics_csv.exists():
        return

    df = pd.read_csv(metrics_csv)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["global_epoch"], df["train_loss"], label="train_loss")
    ax.plot(df["global_epoch"], df["val_loss"], label="val_loss")
    ax.set_xlabel("global_epoch")
    ax.set_ylabel("loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "loss_curve.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["global_epoch"], df["precision"], label="precision")
    ax.plot(df["global_epoch"], df["recall"], label="recall")
    ax.plot(df["global_epoch"], df["specificity"], label="specificity")
    ax.plot(df["global_epoch"], df["f1"], label="f1")
    if "roc_auc" in df.columns:
        ax.plot(df["global_epoch"], df["roc_auc"], label="roc_auc")
    ax.set_xlabel("global_epoch")
    ax.set_ylabel("score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "metrics_curve.png", dpi=140)
    plt.close(fig)
