"""Jev-compatible decision server for the MiniCPM5-2B router.

    ROUTER_ADAPTER=<hf-repo-or-local-dir> uvicorn router.server:app --port 8009

POST /v1/systemone   TypeSafe/Jev request shape (state + choice / noul / score questions), so Kev's benchmark
                     (`python -m kev.benchmark --remote http://127.0.0.1:8009 ...`) and the typesafe-sdk work against it.
POST /v1/action      tools + user message -> tool_call / request_for_info / cannot_answer / direct
POST /v1/route       tools + conversation -> which tool to call, or NONE
"""
import json
import math
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Literal, Union

import torch
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from . import prompts as P

BASE_MODEL = os.environ.get("ROUTER_BASE_MODEL", "openbmb/MiniCPM5-2B")
ADAPTER = os.environ.get("ROUTER_ADAPTER")              # Hub repo id or local folder with adapter_config.json
MODEL_NAME = os.environ.get("ROUTER_MODEL_NAME", "minicpm5-2b-jev-router")
API_KEY = os.environ.get("ROUTER_API_KEY")               # optional bearer token
MAX_TOKENS = int(os.environ.get("ROUTER_MAX_TOKENS", "8192"))

JSONContent = Union[str, dict, list, int, float, bool, None]
S = {}


class Question(BaseModel):
    type: Literal["choice", "noul", "score"]
    instructions: JSONContent = None
    criteria: JSONContent = None


class SystemOneRequest(BaseModel):
    state: JSONContent
    model: str = MODEL_NAME
    questions: dict[str, Question] = Field(min_length=1)


class ActionRequest(BaseModel):
    user_message: str
    tools: list[Union[dict, str]] = []


class RouteRequest(BaseModel):
    tools: list[dict] = Field(min_length=1, max_length=len(P.LETTERS) - 1)
    messages: list[dict[str, Any]] | None = None
    user_message: str | None = None


def load():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    if not ADAPTER:
        raise RuntimeError("set ROUTER_ADAPTER to the adapter's Hub repo id or local folder")
    cuda = torch.cuda.is_available()
    dtype = (torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16) if cuda else torch.float32
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=dtype, device_map="auto" if cuda else None)
    model = PeftModel.from_pretrained(model, ADAPTER).merge_and_unload().eval()
    calib = {}
    path = os.path.join(ADAPTER, "calibration.json")
    if not os.path.exists(path):
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(ADAPTER, "calibration.json")
        except Exception:
            path = None
    if path:
        calib = json.load(open(path))
    sample = P.chat_prompt(tok, P.DECISION_SYSTEM, "x")
    base = tok(sample, add_special_tokens=False)["input_ids"]
    joiner = calib.get("joiner", " ")
    letter_ids = []
    for L in P.LETTERS:
        ids = tok(sample + joiner + L, add_special_tokens=False)["input_ids"]
        if len(ids) != len(base) + 1 or ids[:len(base)] != base:
            raise RuntimeError("option letters are not single tokens after the prefill")
        letter_ids.append(ids[-1])
    S.update(tok=tok, model=model, letter_ids=letter_ids, temperature=float(calib.get("temperature", 1.0)),
             device=model.get_input_embeddings().weight.device)


@asynccontextmanager
async def lifespan(app):
    load()
    yield


app = FastAPI(title="MiniCPM5-2B Jev-style router", lifespan=lifespan)


def check_key(authorization):
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(401, "invalid or missing bearer token")


@torch.inference_mode()
def probs(system, user, n):
    if not 1 <= n <= len(P.LETTERS):
        raise HTTPException(422, f"between 1 and {len(P.LETTERS)} options are supported, got {n}")
    tok, model = S["tok"], S["model"]
    ids = tok(P.chat_prompt(tok, system, user), add_special_tokens=False, return_tensors="pt")["input_ids"]
    if ids.shape[1] > MAX_TOKENS:
        raise HTTPException(422, f"prompt is {ids.shape[1]} tokens; the limit is {MAX_TOKENS}")
    try:
        out = model(input_ids=ids.to(S["device"]), logits_to_keep=1)
    except TypeError:
        out = model(input_ids=ids.to(S["device"]))
    logits = out.logits[0, -1].float()[S["letter_ids"][:n]] / S["temperature"]
    return torch.softmax(logits, 0).tolist(), ids.shape[1]


def choice_confidence(p):
    k = len(p)
    return 1.0 if k == 1 else (max(p) - 1 / k) / (1 - 1 / k)


def score_confidence(p):
    n = len(p)
    if n == 1:
        return 1.0
    mode = max(range(n), key=p.__getitem__)
    spread = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
    return max(0.0, 1.0 - sum(pi * abs(i - mode) for i, pi in enumerate(p)) / spread)


def answer(state, q):
    qd = q.model_dump()
    opts = P.decision_options(qd)
    p, n_tokens = probs(P.DECISION_SYSTEM, P.decision_user(state, qd, opts), len(opts))
    keys = [k for k, _ in opts]
    dist = dict(zip(keys, p))
    if q.type == "noul":
        a = {"type": "noul", "noul": round(dist["true"], 4)}
    elif q.type == "choice":
        a = {"type": "choice", "choice": max(dist, key=dist.get), "confidence": round(choice_confidence(p), 4),
             "probabilities": {k: round(v, 4) for k, v in dist.items()}}
    else:
        a = {"type": "score", "score": round(sum(i * v for i, v in enumerate(p)), 4),
             "legend": {k: text for k, text in opts}, "probabilities": {k: round(v, 4) for k, v in dist.items()},
             "confidence": round(score_confidence(p), 4)}
    return a, n_tokens


@app.post("/v1/systemone")
def systemone(req: SystemOneRequest, authorization: str | None = Header(None)):
    check_key(authorization)
    t0, answers, tokens = time.perf_counter(), {}, 0
    for qid, q in req.questions.items():
        answers[qid], n = answer(req.state, q)
        tokens += n
    return {"model": req.model, "answers": answers, "usage": {"input_tokens": tokens, "output_tokens": 0},
            "latency_ms": round(1000 * (time.perf_counter() - t0), 1)}


@app.post("/v1/action")
def action(req: ActionRequest, authorization: str | None = Header(None)):
    check_key(authorization)
    q = Question(**P.action_question())
    a, _ = answer(P.action_state(req.tools, req.user_message), q)
    return {"action": a["choice"], "confidence": a["confidence"], "probabilities": a["probabilities"]}


@app.post("/v1/route")
def route(req: RouteRequest, authorization: str | None = Header(None)):
    check_key(authorization)
    messages = req.messages or [{"role": "user", "content": req.user_message or ""}]
    p, _ = probs(P.ROUTE_SYSTEM, P.route_user(messages, req.tools), len(req.tools) + 1)
    names = [t["name"] for t in req.tools] + ["NONE"]
    best = max(range(len(p)), key=p.__getitem__)
    return {"tool": None if names[best] == "NONE" else names[best], "confidence": round(choice_confidence(p), 4),
            "probabilities": {k: round(v, 4) for k, v in zip(names, p)}}


@app.get("/v1/models")
def models():
    return {"data": [{"id": MODEL_NAME, "base_model": BASE_MODEL, "adapter": ADAPTER,
                      "temperature": S.get("temperature")}]}


@app.get("/health")
def health():
    return {"ok": "model" in S}
