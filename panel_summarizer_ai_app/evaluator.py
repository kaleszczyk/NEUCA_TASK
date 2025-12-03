from typing import List, Dict, Any, Tuple
import os, re
import numpy as np
_cross_encoders_cache: Dict[str, Any] = {}

def _get_ce(model_name: str):
    try:
        if model_name in _cross_encoders_cache:
            return _cross_encoders_cache[model_name]
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder(model_name)
        _cross_encoders_cache[model_name] = ce
        print(f"[EVAL] CrossEncoder loaded & cached: {model_name}")
        return ce
    except Exception as e:
        print(f"[EVAL] CrossEncoder load failed for {model_name}: {e}")
        return None

RELEVANCY_MODEL = os.getenv("RELEVANCY_CE_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
NLI_MODEL = os.getenv("NLI_CE_MODEL", "cross-encoder/nli-deberta-v3-base")

def score_relevancy(question: str, answer_text: str) -> float:
    ce = _get_ce(RELEVANCY_MODEL)
    try:
        pairs = [(question, answer_text)]
        scores = ce.predict(pairs)
        # nie używaj "if scores" – sprawdzaj długość
        val = float(scores[0]) if len(scores) > 0 else 0.0
        print(f"[EVAL][relevancy] model={RELEVANCY_MODEL} pairs={len(pairs)} score={val}")
        return val
    except Exception as e:
        print(f"[EVAL][relevancy] ERROR: {e}")
        return 0.0

def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r'(?<=[\.\?\!])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]

def score_faithfulness(answer_text: str, contexts: List[Dict[str, Any]]) -> Tuple[float, List[Dict[str, Any]]]:
    ce = _get_ce(NLI_MODEL)
    sentences = _split_sentences(answer_text)
    ctx_texts = [(c.get("text") or "") for c in (contexts or [])]
    results = []
    if len(sentences) == 0 or len(ctx_texts) == 0:
        print(f"[EVAL][faithfulness] skip: sentences={len(sentences)} ctx_texts={len(ctx_texts)}")
        return 0.0, []
    try:
        for i, s in enumerate(sentences):
            pairs = [(s, ct) for ct in ctx_texts]
            scores = ce.predict(pairs)
            # użyj np.max, a nie "if scores"
            best = float(np.max(scores)) if len(scores) > 0 else 0.0
            results.append({"sentence": s, "best_ctx_score": best})
            print(f"[EVAL][faithfulness] model={NLI_MODEL} sent#{i} pairs={len(pairs)} best={best}")
        avg = sum(r["best_ctx_score"] for r in results) / max(1, len(results))
        print(f"[EVAL][faithfulness] avg={avg} sentences={len(sentences)}")
        return float(avg), results
    except Exception as e:
        print(f"[EVAL][faithfulness] ERROR: {e}")
        return 0.0, []

def evaluate_answer_crossencoder(question: str, answer_text: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
    print(f"[EVAL] inputs: q_len={len(question or '')} ans_len={len(answer_text or '')} ctx_count={len(contexts or [])}")
    rel = score_relevancy(question, answer_text)
    faith_avg, per_sentence = score_faithfulness(answer_text, contexts)
    def normalize_0_1(x: float) -> float:
        return round(max(0.0, min(1.0, x)), 4)
    out = {
        "relevancy": normalize_0_1(rel),            
        "faithfulness": normalize_0_1(faith_avg),   
        "details": {
            "raw_relevancy": rel,
            "raw_faithfulness_avg": faith_avg,
            "per_sentence": per_sentence[:20],
            "models": {"relevancy": RELEVANCY_MODEL, "faithfulness": NLI_MODEL},
        },
    }
    print(f"[EVAL] output: {out}")
    return out