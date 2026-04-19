import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from preprocess import get_category_name

SENTINEL = -1


def load_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        df = pd.read_excel(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")
    missing_cols = {"Y", "pred_idx"} - set(df.columns)
    if missing_cols:
        raise ValueError(f"File missing columns: {missing_cols}")
    return df


def apply_01_merge(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    out[out == 1] = 0
    return out


def compute_multiclass_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, valid_labels: list
) -> dict:
    extra = [p for p in np.unique(y_pred) if p not in set(valid_labels)]
    all_labels = valid_labels + extra
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    n = len(valid_labels)
    total = len(y_true)

    TP = np.array([cm[i, i] for i in range(n)])
    FP = np.array([cm[:, i].sum() - cm[i, i] for i in range(n)])
    FN = np.array([cm[i, :].sum() - cm[i, i] for i in range(n)])
    TN = total - TP - FP - FN

    def safe_div(a, b):
        return float(a) / float(b) if b > 0 else 0.0

    prec = [safe_div(TP[i], TP[i] + FP[i]) for i in range(n)]
    rec  = [safe_div(TP[i], TP[i] + FN[i]) for i in range(n)]
    f1   = [safe_div(2 * prec[i] * rec[i], prec[i] + rec[i]) for i in range(n)]
    spec = [safe_div(TN[i], TN[i] + FP[i]) for i in range(n)]
    support = [int(cm[i, :].sum()) for i in range(n)]

    sum_TP, sum_FP, sum_FN, sum_TN = TP.sum(), FP.sum(), FN.sum(), TN.sum()
    prec_micro = safe_div(sum_TP, sum_TP + sum_FP)
    rec_micro  = safe_div(sum_TP, sum_TP + sum_FN)
    f1_micro   = safe_div(2 * prec_micro * rec_micro, prec_micro + rec_micro)
    spec_micro = safe_div(sum_TN, sum_TN + sum_FP)
    accuracy   = safe_div(sum_TP, total)

    per_class = [
        {
            "label": valid_labels[i],
            "precision": prec[i],
            "recall": rec[i],
            "f1": f1[i],
            "specificity": spec[i],
            "support": support[i],
        }
        for i in range(n)
    ]

    return {
        "accuracy": accuracy,
        "precision_micro": prec_micro,
        "precision_macro": float(np.mean(prec)),
        "recall_micro": rec_micro,
        "recall_macro": float(np.mean(rec)),
        "f1_micro": f1_micro,
        "f1_macro": float(np.mean(f1)),
        "specificity_micro": spec_micro,
        "specificity_macro": float(np.mean(spec)),
        "per_class": per_class,
    }


def compute_binary_metrics(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> dict:
    accuracy = accuracy_score(y_true_bin, y_pred_bin)
    prec_micro = precision_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)
    prec_macro = precision_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)
    rec_micro  = recall_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)
    rec_macro  = recall_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)
    f1_micro   = f1_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)
    f1_macro   = f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)

    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    tn0, fp0 = int(cm[1, 1]), int(cm[1, 0])
    tn1, fp1 = int(cm[0, 0]), int(cm[0, 1])
    spec0 = tn0 / (tn0 + fp0) if (tn0 + fp0) > 0 else 0.0
    spec1 = tn1 / (tn1 + fp1) if (tn1 + fp1) > 0 else 0.0
    spec_micro = (tn0 + tn1) / (tn0 + fp0 + tn1 + fp1) if (tn0 + fp0 + tn1 + fp1) > 0 else 0.0
    spec_macro = (spec0 + spec1) / 2

    try:
        roc_auc = roc_auc_score(y_true_bin, y_pred_bin)
    except ValueError:
        roc_auc = float("nan")

    return {
        "accuracy": accuracy,
        "precision_micro": prec_micro,
        "precision_macro": prec_macro,
        "recall_micro": rec_micro,
        "recall_macro": rec_macro,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "specificity_micro": spec_micro,
        "specificity_macro": spec_macro,
        "roc_auc": roc_auc,
    }


def _fmt(x: float) -> str:
    if x != x:  # nan
        return "nan   "
    return f"{x:.4f}"


def _category_name(label: int) -> str:
    if label == 0:
        return "ок / нет конкретного ответа"
    return get_category_name(label)


def format_multiclass_block(m: dict, include_per_class: bool = False) -> str:
    lines = [
        "--- Multiclass (0/1 interchangeable) ---",
        f"Accuracy:                  {_fmt(m['accuracy'])}",
        f"Precision micro/macro:     {_fmt(m['precision_micro'])} / {_fmt(m['precision_macro'])}",
        f"Recall    micro/macro:     {_fmt(m['recall_micro'])} / {_fmt(m['recall_macro'])}",
        f"F1        micro/macro:     {_fmt(m['f1_micro'])} / {_fmt(m['f1_macro'])}",
        f"Specificity micro/macro:   {_fmt(m['specificity_micro'])} / {_fmt(m['specificity_macro'])}",
    ]
    if include_per_class and m.get("per_class"):
        lines.append("")
        lines.append("Per-class metrics:")
        header = f"  {'Idx':>3}  {'Category':<40}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'Spec':>6}  {'Support':>7}"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for pc in m["per_class"]:
            name = _category_name(pc["label"])
            lines.append(
                f"  {pc['label']:>3}  {name:<40}  "
                f"{_fmt(pc['precision']):>6}  {_fmt(pc['recall']):>6}  "
                f"{_fmt(pc['f1']):>6}  {_fmt(pc['specificity']):>6}  {pc['support']:>7}"
            )
    return "\n".join(lines)


def format_binary_block(m: dict) -> str:
    lines = [
        "--- Binary (negative: 0+1, positive: 2-29) ---",
        f"Accuracy:                  {_fmt(m['accuracy'])}",
        f"Precision micro/macro:     {_fmt(m['precision_micro'])} / {_fmt(m['precision_macro'])}",
        f"Recall    micro/macro:     {_fmt(m['recall_micro'])} / {_fmt(m['recall_macro'])}",
        f"F1        micro/macro:     {_fmt(m['f1_micro'])} / {_fmt(m['f1_macro'])}",
        f"Specificity micro/macro:   {_fmt(m['specificity_micro'])} / {_fmt(m['specificity_macro'])}",
        f"ROC-AUC:                   {_fmt(m['roc_auc'])}",
    ]
    return "\n".join(lines)


def run_evaluation(df: pd.DataFrame, input_path: Path) -> str:
    missing_mask = df["pred_idx"].isna()
    missing_count = int(missing_mask.sum())
    missing_pct = 100.0 * missing_count / len(df) if len(df) > 0 else 0.0

    # --- Scenario A: without missing ---
    df_valid = df[~missing_mask].copy()
    y_true_a = df_valid["Y"].values.astype(int)
    y_pred_a = df_valid["pred_idx"].values.astype(int)

    y_true_mc_a = apply_01_merge(y_true_a)
    y_pred_mc_a = apply_01_merge(y_pred_a)
    valid_labels_a = sorted(set(y_true_mc_a))
    mc_a = compute_multiclass_metrics(y_true_mc_a, y_pred_mc_a, valid_labels_a)

    y_true_bin_a = (y_true_a >= 2).astype(int)
    y_pred_bin_a = (y_pred_a >= 2).astype(int)
    bin_a = compute_binary_metrics(y_true_bin_a, y_pred_bin_a)

    # --- Scenario B: with missing (treated as incorrect) ---
    y_true_b = df["Y"].values.astype(int)

    y_pred_mc_b = df["pred_idx"].fillna(SENTINEL).values.astype(int)
    non_sentinel = y_pred_mc_b != SENTINEL
    y_pred_mc_b[non_sentinel] = apply_01_merge(y_pred_mc_b[non_sentinel])
    y_true_mc_b = apply_01_merge(y_true_b)
    valid_labels_b = sorted(set(y_true_mc_b))
    mc_b = compute_multiclass_metrics(y_true_mc_b, y_pred_mc_b, valid_labels_b)

    y_true_bin_b = (y_true_b >= 2).astype(int)
    y_pred_bin_b = (df["pred_idx"].fillna(0).values.astype(int) >= 2).astype(int)
    y_pred_bin_b[missing_mask.values] = 1 - y_true_bin_b[missing_mask.values]
    bin_b = compute_binary_metrics(y_true_bin_b, y_pred_bin_b)

    lines = [
        f"Input file: {input_path}",
        f"Missing predictions: {missing_count} ({missing_pct:.1f}%)",
        "",
        "=" * 60,
        "WITHOUT MISSING PREDICTIONS",
        "=" * 60,
        format_multiclass_block(mc_a, include_per_class=True),
        "",
        format_binary_block(bin_a),
        "",
        "=" * 60,
        "WITH MISSING PREDICTIONS (treated as incorrect)",
        "=" * 60,
        format_multiclass_block(mc_b, include_per_class=False),
        "",
        format_binary_block(bin_b),
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate one-label classification predictions."
    )
    parser.add_argument("prediction_file", help="Path to predictions file (.xlsx or .csv)")
    args = parser.parse_args()

    input_path = Path(args.prediction_file)
    try:
        df = load_file(input_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    report = run_evaluation(df, input_path)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / f"result_{input_path.stem}.txt"
    output_path.write_text(report, encoding="utf-8")

    print(report)
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
