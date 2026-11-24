"""HTTP service. Startup runs the skew canaries; `make serve` fails fast if they are red.

Endpoints
  GET  /v1/health            versions, canaries, which models are loaded
  POST /v1/classify          one text -> topic
  POST /v1/classify:batch    many texts -> topics, length-bucketed (extension)
  POST /v1/analyze           topic + dialect + entities + similar resolved cases
  GET  /demo                 the demo page
"""
import os
import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field, StringConstraints

from . import config as C
from . import dialect as D
from . import preprocess as P
from .engine import load_ner, load_topic

STATE: dict = {}


@asynccontextmanager
async def lifespan(app):
    P.assert_canaries()                                   # skew canaries, asserted at startup
    topic, topic_name = load_topic()
    STATE.update(topic=topic, topic_name=topic_name, ner=load_ner(), dialect=D.load(), search=None)
    try:
        from .search import Searcher
        if (C.INDEX_DIR / "manifest.json").exists():
            STATE["search"] = Searcher()
    except Exception as e:                                # keep serving; /health says why
        STATE["search_error"] = str(e)
    # end-to-end canary: PII in the canary string must never come back out
    probe = STATE["topic"].predict_proba([P.CANARY_STRING])
    assert probe.shape[0] == 1
    assert "000000" not in P.preprocess(P.CANARY_STRING)
    yield
    STATE.clear()


app = FastAPI(title="Shifa", version="1.1.0", lifespan=lifespan)

# Input limits. A patient message is a few hundred characters; these caps keep one request from tying up a worker.
MAX_TEXT_CHARS = 2000
MAX_BATCH_ITEMS = 256
MAX_BODY_BYTES = int(os.environ.get("SHIFA_MAX_BODY_BYTES", 1_000_000))

# Stripped, non-blank, length-capped. Applies to every text, including each item of a batch.
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_TEXT_CHARS)]


@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    """Reject oversized bodies before they are read or parsed (413). Put a proxy limit in front in production too."""
    if request.method == "POST":
        size = request.headers.get("content-length")
        if size is None:
            if request.headers.get("transfer-encoding", "").lower() == "chunked":
                return JSONResponse({"detail": "Content-Length is required"}, status_code=411)
        elif not size.isdigit() or int(size) > MAX_BODY_BYTES:
            return JSONResponse({"detail": f"request body over {MAX_BODY_BYTES} bytes"}, status_code=413)
    return await call_next(request)


class One(BaseModel):
    text: Text


class Many(BaseModel):
    texts: list[Text] = Field(min_length=1, max_length=MAX_BATCH_ITEMS)
    bucket: bool = True
    max_items: int = Field(32, ge=1, le=128)
    max_tokens: int = Field(2048, ge=16, le=16384)


class Analyze(One):
    k: int = Field(5, ge=1, le=20)


def _topic(texts, bucket=False, **kw):
    m = STATE["topic"]
    probs = m.predict_proba_bucketed(texts, **kw) if bucket else m.predict_proba(texts)
    return [{"label": m.labels[int(p.argmax())], "confidence": round(float(p.max()), 4),
             "probs": {l: round(float(v), 4) for l, v in zip(m.labels, p)}} for p in probs]


@app.get("/v1/health")
def health():
    return {"status": "ok", "preprocess_version": P.VERSION, "canaries_green": not P.run_canaries(),
            "topic_model": STATE["topic_name"], "ner": STATE["ner"] is not None,
            "dialect_id": STATE["dialect"] is not None, "search": STATE["search"] is not None,
            "search_error": STATE.get("search_error")}


@app.post("/v1/classify")
def classify(req: One):
    t0 = time.perf_counter()
    out = _topic([req.text])[0]
    return {**out, "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}


@app.post("/v1/classify:batch")
def classify_batch(req: Many):
    t0 = time.perf_counter()
    res = _topic(req.texts, bucket=req.bucket, max_items=req.max_items, max_tokens=req.max_tokens)
    dt = time.perf_counter() - t0
    return {"results": res, "n": len(res), "bucketed": req.bucket,
            "latency_ms": round(dt * 1000, 2), "items_per_sec": round(len(res) / dt, 1)}


@app.post("/v1/analyze")
def analyze(req: Analyze):
    t0 = time.perf_counter()
    masked = P.preprocess(req.text, "multilingual")
    out = {"preprocess_version": P.VERSION, "masked": masked, "topic": _topic([req.text])[0],
           "dialect": None, "entities": [], "similar": []}
    if STATE["dialect"] is not None:
        out["dialect"] = D.predict(STATE["dialect"], [req.text])[0]
    if STATE["ner"] is not None:
        out["entities"] = STATE["ner"].entities(req.text)
    if STATE["search"] is not None:
        out["similar"] = STATE["search"].search(req.text, k=req.k)
    out["models"] = {"topic": STATE["topic_name"], "ner": "onnx-fp32" if STATE["ner"] else None}
    out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return out


@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
def root():
    return RedirectResponse("/demo")


@app.get("/demo", response_class=HTMLResponse)
def demo():
    page = C.ROOT / "demo" / "index.html"
    if not page.exists():
        raise HTTPException(404, "demo/index.html missing")
    return page.read_text(encoding="utf-8")
