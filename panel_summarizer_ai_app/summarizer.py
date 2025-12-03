from typing import Optional, List, Dict, Any
from openai import OpenAI
from config import OPENAI_API_KEY
import os
import json as _json

client = OpenAI(api_key=OPENAI_API_KEY)

# Prompty do podsumowania
SUMMARY_PROMPT_SYSTEM = (
    "Jesteś asystentem przygotowującym zwięzłe, rzeczowe podsumowanie debaty/panelu. "
    "Zachowaj neutralność, bez ocen, bez dodawania informacji spoza transkryptu."
)
SUMMARY_PROMPT_USER_TEMPLATE = (
    "Przygotuj strukturalne podsumowanie poniższej treści.\n\n"
    "Wymagany format:\n"
    "1. Główne tematy (lista)\n"
    "2. Uczestnicy / mówcy (lista z krótką charakterystyką roli jeśli wynika z tekstu)\n"
    "3. Stanowiska i argumenty (podziel według mówców)\n"
    "4. Punkty sporne / kontrasty\n"
    "5. Wnioski / otwarte kwestie\n"
    "6. Jeśli są dane liczbowe lub zobowiązania – wypisz je osobno.\n\n"
    "Treść do podsumowania:\n---\n{panel_text}\n---\n"
    "Zwróć wynik w języku polskim."
)

# Prompty do QA
ANSWER_PROMPT_SYSTEM = (
    "Jesteś asystentem QA. Odpowiadasz wyłącznie na podstawie podanych kontekstów. "
    "Jeśli odpowiedź nie wynika z kontekstu, zwróć dokładnie: 'Brak danych w dostarczonym kontekście.' "
    "Nie zgaduj. Używaj zwięzłego stylu."
)
ANSWER_PROMPT_USER_TEMPLATE = (
    "KONTEKSTY (fragmenty transkryptu):\n{contexts}\n\n"
    "PYTANIE: {question}\n\n"
    "Instrukcje:\n"
    "- Odpowiadaj tylko jeśli informacja występuje w kontekstach.\n"
    "- Jeśli brak, zwróć: Brak danych w dostarczonym kontekście.\n"
    "- Nie dodawaj źródeł ani numerów fragmentów.\n"
    "ODPOWIEDŹ:"
)


SUMMARIZE_MODEL =  "gpt-4.1"
SUMMARIZE_INPUT_TOKEN_BUDGET = 16000
PARTIAL_MAX_TOKENS = 8000

def _estimate_tokens(text: str) -> int:
    try:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(SUMMARIZE_MODEL)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)

def _split_text_by_tokens_optimal(text: str, token_budget: int) -> List[str]:
    total = _estimate_tokens(text)
    if total <= token_budget:
        return [text]

    import math
    n_parts = max(2, math.ceil(total / token_budget))
    paragraphs = text.split("\n\n")
    if len(paragraphs) <= n_parts:
        step = math.ceil(len(text) / n_parts)
        return [text[i:i+step] for i in range(0, len(text), step)]

    target_tokens = math.ceil(total / n_parts)
    parts: List[str] = []
    acc: List[str] = []
    acc_tokens = 0
    for para in paragraphs:
        para_tokens = _estimate_tokens(para)
        if acc and (acc_tokens + para_tokens) > target_tokens and len(parts) < (n_parts - 1):
            parts.append("\n\n".join(acc))
            acc, acc_tokens = [], 0
        acc.append(para)
        acc_tokens += para_tokens
    if acc:
        parts.append("\n\n".join(acc))

    while len(parts) > n_parts:
        parts[-2] = parts[-2] + "\n\n" + parts[-1]
        parts.pop()

    fixed: List[str] = []
    for p in parts:
        if _estimate_tokens(p) <= token_budget:
            fixed.append(p)
        else:
            t = _estimate_tokens(p)
            sub_n = math.ceil(t / token_budget)
            step = math.ceil(len(p) / sub_n)
            fixed.extend([p[i:i+step] for i in range(0, len(p), step)])
    return fixed

def _summarize_fragment(fragment: str) -> str:
    system = "Jesteś asystentem, który wiernie i zwięźle streszcza dany fragment, zachowując kluczowe fakty."
    user = f"Streć możliwie dokładnie (bez dodawania nowych informacji) poniższy fragment:\n\n---\n{fragment}\n---"
    r = client.chat.completions.create(
        model=SUMMARIZE_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
        max_tokens=PARTIAL_MAX_TOKENS 
    )
    return r.choices[0].message.content or ""

def _format_context_blocks(context_list: List[Dict[str, Any]]) -> str:
    blocks = []
    for i, ctx in enumerate(context_list, start=1):
        text = ctx.get("text", "")
        speaker = ctx.get("speaker", "UNKNOWN")
        start = ctx.get("start")
        end = ctx.get("end")
        header = f"[CTX {i}] speaker={speaker} time={start}-{end}"
        blocks.append(f"{header}\n{text}")
    return "\n\n".join(blocks)

def _normalize_contexts(contexts_or_results: Any) -> List[Dict[str, Any]]:
    if isinstance(contexts_or_results, dict) and "documents" in contexts_or_results and "metadatas" in contexts_or_results:
        docs = contexts_or_results.get("documents", [[]])[0] or []
        metas = contexts_or_results.get("metadatas", [[]])[0] or []
        out: List[Dict[str, Any]] = []
        for d, m in zip(docs, metas):
            m = m or {}
            out.append({
                "text": d,
                "speaker": m.get("speaker"),
                "start": m.get("start"),
                "end": m.get("end"),
            })
        return out
    if isinstance(contexts_or_results, list):
        return contexts_or_results
    return []

def summarize(text: str, max_tokens: int = 800):
    # krótkie wejście – pojedyncze zapytanie
    if _estimate_tokens(text) <= SUMMARIZE_INPUT_TOKEN_BUDGET:
        user_content = SUMMARY_PROMPT_USER_TEMPLATE.format(panel_text=text)

        def _gen_short():
            stream_resp = client.chat.completions.create(
                model=SUMMARIZE_MODEL,
                messages=[
                    {"role": "system", "content": SUMMARY_PROMPT_SYSTEM},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.2,
                max_tokens=max_tokens,  # <-- zamiast max_tokens
                stream=True
            )
            for chunk in stream_resp:
                try:
                    delta = chunk.choices[0].delta.content  # type: ignore[attr-defined]
                except Exception:
                    delta = None
                if delta:
                    yield delta
        return _gen_short()

    # długie wejście – map-reduce i finalne podsumowanie
    parts = _split_text_by_tokens_optimal(text, SUMMARIZE_INPUT_TOKEN_BUDGET)
    partial_summaries: List[str] = []
    for idx, frag in enumerate(parts, start=1):
        ps = _summarize_fragment(frag)
        partial_summaries.append(f"[CZĘŚĆ {idx}] {ps}")

    combined = "\n\n".join(partial_summaries)
    user_content = SUMMARY_PROMPT_USER_TEMPLATE.format(panel_text=combined)

    def _gen_long():
        stream_resp = client.chat.completions.create(
            model=SUMMARIZE_MODEL,
            messages=[
                {"role": "system", "content": SUMMARY_PROMPT_SYSTEM},
                {"role": "user", "content": user_content}
            ],
            temperature=0.2,
            max_tokens=max_tokens,  # <-- zamiast max_tokens
            stream=True
        )
        for chunk in stream_resp:
            try:
                delta = chunk.choices[0].delta.content  # type: ignore[attr-defined]
            except Exception:
                delta = None
            if delta:
                yield delta
    return _gen_long()

def _parse_questions_llm(raw: str, model: str = "gpt-4.1") -> List[str]:
    """
    Używa LLM do wyodrębnienia listy pytań z wejścia użytkownika.
    Zwraca listę pytań; w razie problemów – [raw] jako fallback.
    """
    raw = (raw or "").strip()
    if not raw:
        return []
    sys_msg = (
        "Jesteś asystentem, który wyodrębnia jedno lub więcej pytań z podanego tekstu. "
        "Zwróć TYLKO poprawny JSON: {\"questions\": [\"pytanie 1?\", \"pytanie 2?\", ...]} bez dodatkowego tekstu."
    )
    user_msg = (
        "Tekst użytkownika (może zawierać kilka pytań, zdania, wypunktowania):\n"
        f"{raw}\n\n"
        "Upewnij się, że każde pytanie kończy się znakiem '?'."
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_completion_tokens=256
        )
        content = resp.choices[0].message.content or ""
        data = _json.loads(content)
        qs = data.get("questions", [])
        qs = [q.strip() for q in qs if isinstance(q, str) and q.strip()]
        if qs:
            return qs
    except Exception:
        pass
    # Fallback: traktuj cały tekst jako jedno pytanie
    return [raw if raw.endswith("?") else f"{raw}?"]

def answer(contexts, question: str, model: str = "gpt-4.1", top_k: int = None):
    ctx_list = _normalize_contexts(contexts)
    if top_k is not None and isinstance(ctx_list, list):
        ctx_list = ctx_list[:top_k]

    # separacja wielu pytań
    sub_questions = _parse_questions_llm(question, model=model)

    if len(sub_questions) == 1:
        context_block = _format_context_blocks(ctx_list)
        user_content = ANSWER_PROMPT_USER_TEMPLATE.format(contexts=context_block, question=sub_questions[0])

        def _gen_single():
            stream_resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": ANSWER_PROMPT_SYSTEM},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.0,
                stream=True
            )
            for chunk in stream_resp:
                try:
                    delta = chunk.choices[0].delta.content  
                except Exception:
                    delta = None
                if delta:
                    yield delta
        return _gen_single()

    # wiele pytań – każde osobno, w jednym strumieniu
    def _gen_multi():
        context_block = _format_context_blocks(ctx_list)
        for i, q in enumerate(sub_questions, start=1):
            yield f"\n\n[Pytanie {i}] {q}\n"
            user_content = ANSWER_PROMPT_USER_TEMPLATE.format(contexts=context_block, question=q)
            stream_resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": ANSWER_PROMPT_SYSTEM},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.0,
                stream=True
            )
            for chunk in stream_resp:
                try:
                    delta = chunk.choices[0].delta.content 
                except Exception:
                    delta = None
                if delta:
                    yield delta
        yield "\n"
    return _gen_multi()