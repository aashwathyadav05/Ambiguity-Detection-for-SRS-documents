"""
Ambiguity Detection for SRS Documents — FastAPI Backend
Powered by fine-tuned RoBERTa
"""

import sys
import re
from pathlib import Path
from collections import Counter
from contextlib import asynccontextmanager

import torch
import pdfplumber
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ─────────────────────────────────────────────
#  Resolve project root so we can import src.*
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import rule_based_ambiguity

# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────
LABEL_NAMES = {
    0: "Lexical",
    1: "Syntactic",
    2: "Semantic",
    3: "Syntax",
    4: "Pragmatic",
    5: "Clean",
}

LABEL_DESC = {
    "Lexical":   "A word has multiple meanings (e.g., 'process', 'handle', 'light').",
    "Syntactic": "Sentence structure allows multiple parse trees.",
    "Semantic":  "Meaning is unclear even with a fixed parse tree.",
    "Syntax":    "Structural/grammatical issue causing misinterpretation.",
    "Pragmatic": "Context-dependent meaning; intent unclear without extra knowledge.",
    "Clean":     "Requirement is clear and unambiguous.",
}

BAR_COLORS = {
    "Lexical":   "#eab308",
    "Syntactic": "#f97316",
    "Semantic":  "#a855f7",
    "Syntax":    "#fb923c",
    "Pragmatic": "#f43f5e",
    "Clean":     "#22c55e",
}

MODEL_PATH = PROJECT_ROOT / "models" / "roberta-ambiguity-final"

# ─────────────────────────────────────────────
#  Global model state
# ─────────────────────────────────────────────
_tokenizer = None
_model = None
_model_loaded = False


def _load_model():
    """Load model once at startup."""
    global _tokenizer, _model, _model_loaded
    if MODEL_PATH.exists():
        try:
            _tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
            _model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH))
            _model.eval()
            _model_loaded = True
            print(f"[OK] Model loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"[FAIL] Failed to load model: {e}")
            _model_loaded = False
    else:
        print(f"[WARN] Model not found at {MODEL_PATH}. Rule-based heuristics only.")
        _model_loaded = False


# ─────────────────────────────────────────────
#  Lifespan (startup / shutdown)
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


# ─────────────────────────────────────────────
#  FastAPI app
# ─────────────────────────────────────────────
app = FastAPI(
    title="SRS Ambiguity Detector API",
    description="Classify linguistic ambiguities in Software Requirements Specifications",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
def split_sentences(text: str) -> list[str]:
    """Split text into sentences (same logic as original app.py)."""
    raw = re.split(r'(?<=[.!?])\s+|\n+', text.strip())
    return [s.strip() for s in raw if len(s.strip()) > 8]


def predict_sentence(sentence: str) -> dict:
    """Run model inference on a single sentence."""
    if not _model_loaded:
        return {
            "model_label": "N/A",
            "model_confidence": 0.0,
            "model_probabilities": {v: 0.0 for v in LABEL_NAMES.values()},
        }

    inputs = _tokenizer(
        sentence, return_tensors="pt",
        truncation=True, max_length=128, padding=True,
    )
    with torch.no_grad():
        logits = _model(**inputs).logits

    probs = torch.softmax(logits, dim=-1).squeeze().tolist()
    pred_id = int(torch.argmax(logits))
    label = LABEL_NAMES[pred_id]
    conf = probs[pred_id]

    return {
        "model_label": label,
        "model_confidence": round(conf, 4),
        "model_probabilities": {
            LABEL_NAMES[i]: round(p, 4) for i, p in enumerate(probs)
        },
    }


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF bytes using pdfplumber."""
    import io
    text = ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


# ─────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────
@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": _model_loaded,
        "model_path": str(MODEL_PATH),
    }


@app.get("/api/labels")
async def get_labels():
    """Return label metadata for the frontend."""
    labels = []
    for idx, name in LABEL_NAMES.items():
        labels.append({
            "id": idx,
            "name": name,
            "description": LABEL_DESC[name],
            "color": BAR_COLORS[name],
        })
    return {"labels": labels}


@app.post("/api/analyze")
async def analyze_document(
    file: UploadFile = File(...),
    max_sentences: int = Query(default=50, ge=1, le=500),
):
    """
    Analyze an uploaded SRS document (.txt or .pdf).
    Returns sentence-level ambiguity classifications.
    """
    # Validate file type
    filename = file.filename or "unknown"
    content_type = file.content_type or ""
    ext = Path(filename).suffix.lower()

    if ext not in (".txt", ".pdf") and "pdf" not in content_type and "text" not in content_type:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: '{ext}'. Upload a .txt or .pdf file.",
        )

    # Read file bytes
    file_bytes = await file.read()

    # Extract text
    if ext == ".pdf" or "pdf" in content_type:
        raw_text = extract_text_from_pdf(file_bytes)
    else:
        raw_text = file_bytes.decode("utf-8", errors="ignore")

    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract any text from the uploaded file.")

    # Split into sentences
    sentences = split_sentences(raw_text)
    if not sentences:
        raise HTTPException(status_code=400, detail="No valid sentences found in the document.")

    # Cap at max_sentences
    to_analyze = sentences[:max_sentences]

    # Analyze each sentence
    results = []
    for sent in to_analyze:
        # Model prediction
        model_result = predict_sentence(sent)

        # Rule-based prediction
        rule_cls, rule_flags = rule_based_ambiguity(sent)

        results.append({
            "sentence": sent,
            "model_label": model_result["model_label"],
            "model_confidence": model_result["model_confidence"],
            "model_probabilities": model_result["model_probabilities"],
            "rule_label": rule_cls,
            "rule_flags": rule_flags,
        })

    # Build summary counts
    if _model_loaded:
        counts = Counter(r["model_label"] for r in results)
    else:
        counts = Counter(r["rule_label"] for r in results)

    summary = {name: counts.get(name, 0) for name in LABEL_NAMES.values()}

    return {
        "filename": filename,
        "total_sentences": len(sentences),
        "analyzed_sentences": len(to_analyze),
        "model_loaded": _model_loaded,
        "summary": summary,
        "results": results,
    }
