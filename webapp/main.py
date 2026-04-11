"""
FastAPI web demo for ViHateT5 — Vietnamese Hate Speech Detection.

Run with:
    uvicorn webapp.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import io
import time
import unicodedata
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch
import pandas as pd
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Load .env for HF_TOKEN
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

T5_MODELS = {
    "ViHateT5 Fine-tune Balanced (Best)": "NCPhat2005/vit5_finetune_balanced",
    "ViHateT5 Fine-tune Hate-only": "NCPhat2005/vit5_finetune_hate_only",
    "ViHateT5 Fine-tune Multi": "NCPhat2005/vit5_finetune_multi",
    "ViHateT5 Reimplementation": "NCPhat2005/vihatet5_reimpl",
}

LOCAL_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

# Map HuggingFace model IDs → local folder names
_HF_TO_LOCAL = {
    "NCPhat2005/vit5_finetune_balanced": "vit5_finetune_balanced",
    "NCPhat2005/vit5_finetune_hate_only": "vit5_finetune_hate_only",
    "NCPhat2005/vit5_finetune_multi": "vit5_finetune_multi",
    "NCPhat2005/vihatet5_reimpl": "vihatet5_reimpl",
}

TASK_PREFIXES = {
    "vihsd": "hate-speech-detection",
    "victsd": "toxic-speech-detection",
    "vihos": "hate-spans-detection",
}

TASK_LABELS = {
    "vihsd": "ViHSD — Hate Speech Detection",
    "victsd": "ViCTSD — Toxic Speech Detection",
    "vihos": "ViHOS — Hate Spans Detection",
}

# ---------------------------------------------------------------------------
# Model cache  (loaded once, kept in memory)
# ---------------------------------------------------------------------------
_model_cache: dict[str, tuple] = {}


def _resolve_model_path(model_id: str) -> str:
    """Return a local path if the model exists on disk, else the HF id."""
    local_name = _HF_TO_LOCAL.get(model_id)
    if local_name:
        local_path = LOCAL_MODEL_DIR / local_name
        if local_path.exists() and (local_path / "config.json").exists():
            return str(local_path)
    return model_id


def load_model(model_id: str):
    """Load and optionally quantize a T5 model. Results are cached."""
    if model_id in _model_cache:
        return _model_cache[model_id]

    from transformers import T5ForConditionalGeneration, PreTrainedTokenizerFast

    source = _resolve_model_path(model_id)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(source)
    model = T5ForConditionalGeneration.from_pretrained(source)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        # Try dynamic INT8 quantization for faster CPU inference
        try:
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
        except Exception:
            pass  # Quantization not supported (e.g. ARM/MPS) — use FP32

    model.to(device)
    model.eval()
    _model_cache[model_id] = (model, tokenizer, device)
    return model, tokenizer, device


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def run_inference(text: str, task_key: str, model, tokenizer, device) -> str:
    prefix = TASK_PREFIXES[task_key]
    input_text = f"{prefix}: {text}"
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    ).to(device)
    with torch.no_grad():
        ids = model.generate(**inputs, max_length=256, num_beams=1, do_sample=False)
    return tokenizer.decode(ids[0], skip_special_tokens=True).strip()


def extract_hate_spans(original: str, tagged: str) -> list[tuple[int, int]]:
    tag = "[hate]"
    tagged_l = unicodedata.normalize("NFC", tagged.lower())
    orig_l = unicodedata.normalize("NFC", original.lower())

    subs: list[str] = []
    pos = tagged_l.find(tag)
    while pos != -1:
        end = tagged_l.find(tag, pos + len(tag))
        if end == -1:
            break
        subs.append(tagged_l[pos + len(tag) : end])
        pos = tagged_l.find(tag, end + len(tag))

    spans: list[tuple[int, int]] = []
    for sub in subs:
        idx = orig_l.find(sub)
        while idx != -1:
            spans.append((idx, idx + len(sub)))
            idx = orig_l.find(sub, idx + 1)
    return sorted(set(spans))


# ---------------------------------------------------------------------------
# App lifespan — preload default model
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load the default (best) model at startup
    default_id = T5_MODELS["ViHateT5 Fine-tune Balanced (Best)"]
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, load_model, default_id)
    yield
    _model_cache.clear()


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ViHateT5 Demo",
    description="Vietnamese Hate Speech Detection — DS200.Q21 Group 02",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    device = "GPU 🟢" if torch.cuda.is_available() else "CPU 🔵"
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "models": T5_MODELS,
            "tasks": TASK_LABELS,
            "device": device,
        },
    )


@app.post("/api/predict")
async def predict(request: Request):
    body = await request.json()
    text: str = body.get("text", "").strip()
    task: str = body.get("task", "vihsd")
    model_label: str = body.get("model", "ViHateT5 Fine-tune Balanced (Best)")

    if not text:
        return JSONResponse({"error": "Text is required."}, status_code=400)
    if task not in TASK_PREFIXES:
        return JSONResponse({"error": f"Invalid task: {task}"}, status_code=400)

    model_id = T5_MODELS.get(model_label)
    if not model_id:
        return JSONResponse({"error": f"Unknown model: {model_label}"}, status_code=400)

    loop = asyncio.get_running_loop()

    # Load model (cached after first call)
    model, tokenizer, device = await loop.run_in_executor(None, load_model, model_id)

    t0 = time.perf_counter()
    output = await loop.run_in_executor(
        None, run_inference, text, task, model, tokenizer, device
    )
    elapsed = time.perf_counter() - t0

    result: dict = {
        "task": task,
        "task_label": TASK_LABELS[task],
        "raw_output": output,
        "inference_time": round(elapsed, 3),
        "device": device,
    }

    if task == "vihos":
        spans = extract_hate_spans(text, output)
        result["spans"] = [{"start": s, "end": e, "text": text[s:e]} for s, e in spans]
    else:
        result["label"] = output.upper()

    # Run all 3 tasks for comprehensive view
    all_results = {}
    for tk in TASK_PREFIXES:
        if tk == task:
            all_results[tk] = result.get("label", output)
            continue
        out = await loop.run_in_executor(
            None, run_inference, text, tk, model, tokenizer, device
        )
        if tk == "vihos":
            spans = extract_hate_spans(text, out)
            all_results[tk] = f"{len(spans)} span(s)" if spans else "None"
        else:
            all_results[tk] = out.upper()
    result["all_tasks"] = all_results

    return JSONResponse(result)


@app.post("/api/batch")
async def batch_predict(
    file: UploadFile = File(...),
    task: str = Form("vihsd"),
    model_label: str = Form("ViHateT5 Fine-tune Balanced (Best)"),
    text_column: str = Form("text"),
):
    if task not in TASK_PREFIXES:
        return JSONResponse({"error": f"Invalid task: {task}"}, status_code=400)
    model_id = T5_MODELS.get(model_label)
    if not model_id:
        return JSONResponse({"error": f"Unknown model: {model_label}"}, status_code=400)

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception:
        return JSONResponse({"error": "Could not parse CSV file."}, status_code=400)

    if text_column not in df.columns:
        return JSONResponse(
            {"error": f"Column '{text_column}' not found. Available: {list(df.columns)}"},
            status_code=400,
        )

    loop = asyncio.get_running_loop()
    model, tokenizer, device = await loop.run_in_executor(None, load_model, model_id)

    predictions = []
    for text in df[text_column].astype(str):
        out = await loop.run_in_executor(
            None, run_inference, text, task, model, tokenizer, device
        )
        if task == "vihos":
            spans = extract_hate_spans(text, out)
            predictions.append({"text": text, "raw_output": out, "num_spans": len(spans)})
        else:
            predictions.append({"text": text, "prediction": out.upper()})

    result_df = pd.DataFrame(predictions)
    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"},
    )


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": list(_model_cache.keys()),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
