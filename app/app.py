from __future__ import annotations

import asyncio
import json
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.requests import Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "encoder_models"))

from encoder_models.inference import _load_model_from_checkpoint, _make_inference_loader
from encoder_models.preprocessing import TextPreprocessConfig, index_to_category, preprocess_text_series

BASE_DIR = Path(__file__).parent
BINARY_CHECKPOINT = BASE_DIR / "final_model" / "binary.pt"
MULTICLASS_CHECKPOINT = BASE_DIR / "final_model" / "multiclass.pt"
RUNS_DIR = BASE_DIR / "runs"
UPLOADS_DIR = BASE_DIR / "uploads"
METADATA_FILE = RUNS_DIR / "metadata.json"

CAT1_COL = "Категория вопроса 2"
CAT2_COL = "Категория вопроса 5"
BATCH_SIZE = 64
MAX_LENGTH = 64

app_state: dict = {}
progress_store: dict[str, dict] = {}
cancel_events: dict[str, threading.Event] = {}
run_futures: dict[str, Future] = {}
run_params: dict[str, dict] = {}
executor = ThreadPoolExecutor(max_workers=1)


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@asynccontextmanager
async def lifespan(app: FastAPI):
    RUNS_DIR.mkdir(exist_ok=True)
    UPLOADS_DIR.mkdir(exist_ok=True)
    if not METADATA_FILE.exists():
        METADATA_FILE.write_text("[]")

    device = _resolve_device()
    print(f"[startup] Using device: {device}")

    binary_model = _load_model_from_checkpoint(str(BINARY_CHECKPOINT), device, "binary")
    multi_model = _load_model_from_checkpoint(str(MULTICLASS_CHECKPOINT), device, "multiclass")
    binary_tok = AutoTokenizer.from_pretrained(binary_model.cfg.model_name)
    multi_tok = AutoTokenizer.from_pretrained(multi_model.cfg.model_name)

    app_state.update({
        "binary_model": binary_model,
        "multi_model": multi_model,
        "binary_tok": binary_tok,
        "multi_tok": multi_tok,
        "device": device,
    })
    print("[startup] Models loaded and ready.")
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _set_progress(run_id: str, percent: int, message: str, status: str = "running") -> None:
    progress_store[run_id] = {"percent": percent, "message": message, "status": status}


def _append_metadata(record: dict) -> None:
    data = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    data.append(record)
    METADATA_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _check_cancel(run_id: str) -> None:
    if cancel_events.get(run_id, threading.Event()).is_set():
        raise RuntimeError("__cancelled__")


# ── Batch loops (own the iteration to report per-batch progress) ──────────────

@torch.no_grad()
def _binary_loop(
    run_id: str,
    loader,
    pct_start: int,
    pct_end: int,
    col_label: str,
) -> np.ndarray:
    probs_all: list[np.ndarray] = []
    total = len(loader)
    model = app_state["binary_model"]
    device = app_state["device"]
    for i, batch in enumerate(loader):
        _check_cancel(run_id)
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["attention_mask"], batch["scores"]).cpu().numpy()
        probs_all.append(1.0 / (1.0 + np.exp(-logits)))
        pct = pct_start + (pct_end - pct_start) * (i + 1) / total
        _set_progress(run_id, int(pct), f"Бинарная классификация «{col_label}», батч {i + 1}/{total}")
    return np.concatenate(probs_all)


@torch.no_grad()
def _multiclass_loop(
    run_id: str,
    loader,
    pct_start: int,
    pct_end: int,
    col_label: str,
) -> np.ndarray:
    logits_all: list[np.ndarray] = []
    total = len(loader)
    model = app_state["multi_model"]
    device = app_state["device"]
    for i, batch in enumerate(loader):
        _check_cancel(run_id)
        batch = {k: v.to(device) for k, v in batch.items()}
        logits_all.append(
            model(batch["input_ids"], batch["attention_mask"], batch["scores"]).cpu().numpy()
        )
        pct = pct_start + (pct_end - pct_start) * (i + 1) / total
        _set_progress(run_id, int(pct), f"Мультиклассовая классификация «{col_label}», батч {i + 1}/{total}")
    return np.argmax(np.concatenate(logits_all), axis=1)


def _infer_col(
    run_id: str,
    df: pd.DataFrame,
    score_col: str,
    text_col: str,
    threshold: float,
    binary_pct: tuple[int, int],
    multi_pct: tuple[int, int],
) -> pd.DataFrame:
    preprocess_cfg = TextPreprocessConfig()
    pairs = df[[score_col, text_col]].copy()
    pairs.columns = pd.Index(["score", "text"])
    pairs["text"] = preprocess_text_series(pairs["text"], preprocess_cfg)
    pairs = pairs.dropna(subset=["text"]).copy()

    if pairs.empty:
        _set_progress(run_id, multi_pct[1], f"«{text_col}»: нет данных для классификации")
        return pairs.assign(category=pd.NA)

    _check_cancel(run_id)
    binary_loader = _make_inference_loader(
        tokenizer=app_state["binary_tok"],
        texts=pairs["text"].astype(str).tolist(),
        scores=pairs["score"].to_numpy(dtype=np.float32),
        max_length=MAX_LENGTH,
        batch_size=BATCH_SIZE,
    )
    _check_cancel(run_id)
    problem_prob = _binary_loop(run_id, binary_loader, *binary_pct, col_label=text_col)
    pairs["problem_stated"] = (problem_prob >= threshold).astype(bool)
    pairs["category"] = pd.Series([pd.NA] * len(pairs), dtype=object, index=pairs.index)

    positives = pairs[pairs["problem_stated"]].copy()
    if not positives.empty:
        _check_cancel(run_id)
        multi_loader = _make_inference_loader(
            tokenizer=app_state["multi_tok"],
            texts=positives["text"].astype(str).tolist(),
            scores=positives["score"].to_numpy(dtype=np.float32),
            max_length=MAX_LENGTH,
            batch_size=BATCH_SIZE,
        )
        _check_cancel(run_id)
        pred_multi = _multiclass_loop(run_id, multi_loader, *multi_pct, col_label=text_col)
        class_idx = pred_multi + 2
        positives["category"] = [index_to_category(int(v)) for v in class_idx]
        pairs.loc[positives.index, "category"] = positives["category"]
    else:
        _set_progress(run_id, multi_pct[1], f"«{text_col}»: позитивных нет, пропуск мультиклассификации")

    return pairs


# ── Background task ───────────────────────────────────────────────────────────

def _make_cancelled_record(run_id: str) -> dict:
    params = run_params.get(run_id, {})
    return {
        "id": run_id,
        "timestamp": datetime.now().isoformat(),
        "input_filename": params.get("input_filename", "unknown"),
        "row_count": 0,
        "threshold": params.get("threshold", 0.5),
        "score_col": params.get("score_col", ""),
        "a1_col": params.get("a1_col", ""),
        "a2_col": params.get("a2_col", ""),
        "status": "cancelled",
        "error": None,
    }


def _inference_task(
    run_id: str,
    input_path: Path,
    output_path: Path,
    threshold: float,
    original_filename: str,
    score_col: str,
    a1_col: str,
    a2_col: str,
) -> None:
    try:
        _set_progress(run_id, 0, "Загрузка файла...")
        raw = pd.read_excel(input_path)
        row_count = len(raw)

        # Progress weights: binary A1=40%, multi A1=10%, binary A2=40%, multi A2=10%
        col1 = _infer_col(run_id, raw, score_col, a1_col, threshold,
                          binary_pct=(0, 40), multi_pct=(40, 50))
        col2 = _infer_col(run_id, raw, score_col, a2_col, threshold,
                          binary_pct=(50, 90), multi_pct=(90, 100))

        _set_progress(run_id, 100, "Сохранение результатов...")
        result = raw.copy()
        result[CAT1_COL] = "нет ответа"
        result[CAT2_COL] = "нет ответа"
        if not col1.empty:
            result.loc[col1.index, CAT1_COL] = col1["category"].fillna("нет ответа")
        if not col2.empty:
            result.loc[col2.index, CAT2_COL] = col2["category"].fillna("нет ответа")

        # Insert result cols right after their source columns
        cols = [c for c in result.columns if c not in (CAT1_COL, CAT2_COL)]
        cols.insert(cols.index(a1_col) + 1, CAT1_COL)
        cols.insert(cols.index(a2_col) + 1, CAT2_COL)
        result[cols].to_excel(output_path, index=False)

        _append_metadata({
            "id": run_id,
            "timestamp": datetime.now().isoformat(),
            "input_filename": original_filename,
            "row_count": row_count,
            "threshold": threshold,
            "score_col": score_col,
            "a1_col": a1_col,
            "a2_col": a2_col,
            "status": "completed",
            "error": None,
        })
        _set_progress(run_id, 100, "Готово!", status="completed")

    except RuntimeError as exc:
        if "__cancelled__" in str(exc):
            _append_metadata(_make_cancelled_record(run_id))
            _set_progress(run_id, 0, "Отменено пользователем", status="cancelled")
        else:
            _append_metadata({
                "id": run_id,
                "timestamp": datetime.now().isoformat(),
                "input_filename": original_filename,
                "row_count": 0,
                "threshold": threshold,
                "score_col": score_col,
                "a1_col": a1_col,
                "a2_col": a2_col,
                "status": "failed",
                "error": str(exc),
            })
            _set_progress(run_id, 0, f"Ошибка: {exc}", status="failed")

    except Exception as exc:
        _append_metadata({
            "id": run_id,
            "timestamp": datetime.now().isoformat(),
            "input_filename": original_filename,
            "row_count": 0,
            "threshold": threshold,
            "score_col": score_col,
            "a1_col": a1_col,
            "a2_col": a2_col,
            "status": "failed",
            "error": str(exc),
        })
        _set_progress(run_id, 0, f"Ошибка: {exc}", status="failed")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/api/upload")
async def upload_file(file: UploadFile):
    upload_id = str(uuid.uuid4())
    path = UPLOADS_DIR / f"{upload_id}.xlsx"
    path.write_bytes(await file.read())

    df_headers = pd.read_excel(path, nrows=0)
    columns = [str(c) for c in df_headers.columns.tolist()]

    def guess(patterns: list[str]) -> str:
        for col in columns:
            if any(p.lower() in col.lower() for p in patterns):
                return col
        return columns[0] if columns else ""

    return {
        "upload_id": upload_id,
        "filename": file.filename or "upload.xlsx",
        "columns": columns,
        "guessed": {
            "score_col": guess(["1.", "score"]),
            "a1_col": guess(["2.", "A1"]),
            "a2_col": guess(["5.", "A2"]),
        },
    }


@app.post("/api/run")
async def start_run(
    upload_id: str = Form(...),
    score_col: str = Form(...),
    a1_col: str = Form(...),
    a2_col: str = Form(...),
    threshold: float = Form(0.5),
    original_filename: str = Form("upload.xlsx"),
):
    upload_path = UPLOADS_DIR / f"{upload_id}.xlsx"
    if not upload_path.exists():
        raise HTTPException(status_code=400, detail="Файл не найден. Загрузите файл заново.")

    run_id = str(uuid.uuid4())
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True)
    input_path = run_dir / "input.xlsx"
    output_path = run_dir / "output.xlsx"
    upload_path.rename(input_path)

    cancel_events[run_id] = threading.Event()
    run_params[run_id] = {
        "input_filename": original_filename,
        "threshold": threshold,
        "score_col": score_col,
        "a1_col": a1_col,
        "a2_col": a2_col,
    }
    _set_progress(run_id, 0, "В очереди...", status="pending")
    fut = executor.submit(
        _inference_task, run_id, input_path, output_path,
        threshold, original_filename, score_col, a1_col, a2_col,
    )
    run_futures[run_id] = fut
    return {"run_id": run_id}


@app.post("/api/runs/{run_id}/cancel")
async def cancel_run(run_id: str):
    ev = cancel_events.get(run_id)
    if ev:
        ev.set()
    fut = run_futures.get(run_id)
    if fut and fut.cancel():
        # Task was queued but not yet started — write metadata here
        _append_metadata(_make_cancelled_record(run_id))
        _set_progress(run_id, 0, "Отменено пользователем", status="cancelled")
    return {"ok": True}


@app.get("/api/runs")
async def list_runs():
    if not METADATA_FILE.exists():
        return []
    data = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    return list(reversed(data))


@app.get("/api/runs/{run_id}/download")
async def download_run(run_id: str):
    output_path = RUNS_DIR / run_id / "output.xlsx"
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Файл не найден")

    runs_data = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    run_record = next((r for r in runs_data if r["id"] == run_id), None)
    if run_record:
        ts = datetime.fromisoformat(run_record["timestamp"])
        base = Path(run_record["input_filename"]).stem
        filename = f"{base}_processed_{ts.strftime('%Y%m%d_%H%M%S')}.xlsx"
    else:
        filename = f"results_{run_id[:8]}.xlsx"

    return FileResponse(
        path=str(output_path),
        filename=filename,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.get("/api/progress/{run_id}")
async def progress_stream(run_id: str):
    async def event_generator():
        while True:
            state = progress_store.get(
                run_id, {"percent": 0, "message": "Ожидание...", "status": "pending"}
            )
            yield f"data: {json.dumps(state, ensure_ascii=False)}\n\n"
            if state["status"] in ("completed", "failed", "cancelled"):
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
