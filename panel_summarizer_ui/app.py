import re
from typing import Optional
import requests
import gradio as gr

API = "http://127.0.0.1:8000"

def _extract_video_id(url: str) -> Optional[str]:
    m = re.search(r"[?&]v=([A-Za-z0-9_\-]{6,})", url)
    if m:
        return m.group(1)
    m = re.search(r"youtu\.be/([A-Za-z0-9_\-]{6,})", url)
    if m:
        return m.group(1)
    return None

def process_video(url):
    # generator: pokaż spinner od razu i aktualizuj status w trakcie
    yield "Inicjalizacja...", None, gr.update(visible=True)
    if not url:
        yield "Brak URL", None, gr.update(visible=False)
        return
    try:
        yield "Wywołuję /process_youtube (to może potrwać)...", None, gr.update(visible=True)
        r = requests.post(f"{API}/process_youtube", json={"url": url}, timeout=600)
        if r.status_code != 200:
            yield f"ERROR {r.status_code}: {r.text}", None, gr.update(visible=False)
            return
        data = r.json()
        vid = _extract_video_id(url)
        mode = data.get("mode")
        idx = data.get("indexed_chunks")
        msg = f"OK. mode={mode}, zindeksowano={idx}"
        yield msg, vid, gr.update(visible=False)
    except Exception as e:
        yield f"Exception: {e}", None, gr.update(visible=False)

def get_summary_stream(video_id, max_tokens):
    # stream z /summarize
    payload = {}
    if video_id:
        payload["video_id"] = video_id
    payload["max_tokens"] = int(max_tokens or 2000)
    buf = ""
    try:
        with requests.post(f"{API}/summarize_stream", json=payload) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=1024):
                if not chunk:
                    continue
                buf += chunk.decode("utf-8", errors="ignore")
                yield buf
    except Exception as e:
        yield f"Exception: {e}"

def ask_question_stream(question, top_k):
    # stream z /ask; po zakończeniu: dociąg kontekstów oraz wywołanie /ask_eval_ce i pokazanie metryk w "Uwagi"
    if not question:
        yield "Brak pytania", "", "Brak pytania do ewaluacji."
        return
    buf = ""
    try:
        with requests.post(
            f"{API}/ask_stream",
            json={"question": question, "top_k": int(top_k or 5), "collection_name": "panel"}
        ) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=1024):
                if not chunk:
                    continue
                buf += chunk.decode("utf-8", errors="ignore")
                # w trakcie streamu nie mamy metryk – placeholder
                yield buf, "", "Ewaluacja będzie dostępna po zakończeniu odpowiedzi..."
        # Po zakończeniu streamu – pobierz użyte konteksty
        ctx_preview = ""
        try:
            rr = requests.get(f"{API}/used_contexts", timeout=30)
            if rr.status_code == 200:
                data = rr.json()
                used = data.get("contexts", [])
                chunks_preview = []
                for c in used:
                    speaker = c.get("speaker")
                    start = c.get("start")
                    end = c.get("end")
                    text = c.get("text") or ""
                    chunks_preview.append(f"{speaker} [{start}-{end}]: {text}")
                ctx_preview = "\n\n".join(chunks_preview)
            else:
                ctx_preview = f"ERROR {rr.status_code}: {rr.text}"
        except Exception as e2:
            ctx_preview = f"Exception: {e2}"
        # Wywołaj ewaluację CE (relevancy i faithfulness)
        eval_md = ""
        try:
            er = requests.post(f"{API}/ask_eval_ce", json={"question": question, "top_k": int(top_k or 5)}, timeout=60)
            if er.status_code == 200:
                ed = er.json()
                m = ed.get("metrics_ce", {})
                fai = m.get("faithfulness", 0)
                rel = m.get("relevancy", 0)
                eval_md = f"Metryki ewaluacji (CE):\n- Faithfulness: {fai}\n- Relevancy: {rel}"
            else:
                eval_md = f"[CE] error {er.status_code}: {er.text}"
        except Exception as e3:
            eval_md = f"[CE] exception: {e3}"
        # Zwróć: odpowiedź, konteksty, metryki do sekcji Uwagi
        yield buf, ctx_preview, eval_md
    except Exception as e:
        yield f"Exception: {e}", "", f"Exception: {e}"


with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Panel Summarizer")

    with gr.Row():
        url = gr.Textbox(label="YouTube URL", placeholder="https://youtu.be/...")
        process_btn = gr.Button("Przetwórz / Załaduj")
        spinner = gr.HTML(
            """
            <div style="display:flex;align-items:center;height:100%">
              <div style="
                border: 6px solid #f3f3f3;
                border-top: 6px solid #3498db;
                border-radius: 50%;
                width: 28px; height: 28px;
                animation: spin 1s linear infinite;
              "></div>
            </div>
            <style>
              @keyframes spin { 0% { transform: rotate(0deg);} 100% { transform: rotate(360deg);} }
            </style>
            """,
            visible=False,
        )

    status = gr.Textbox(label="Status", interactive=False)
    video_id_box = gr.Textbox(label="video_id (auto)", interactive=False)
    process_btn.click(process_video, inputs=url, outputs=[status, video_id_box, spinner])

    gr.Markdown("## 📌 Podsumowanie")
    with gr.Row():
        max_tokens = gr.Number(label="Limit tokenów ", value=2000)
        summary_btn = gr.Button("Podsumuj")
    summary_out = gr.Markdown()
    summary_btn.click(get_summary_stream, inputs=[video_id_box, max_tokens], outputs=summary_out)

    gr.Markdown("## 💬 Zapytania")
    question = gr.Textbox(label="Pytanie")
    top_k = gr.Number(label="Ile kontekstów (top_k)", value=5)
    ask_btn = gr.Button("Zapytaj")
    answer_out = gr.Textbox(lines=8, label="Odpowiedź")
    contexts_out = gr.Textbox(lines=10, label="Użyte chunki")
    # Dodaj miejsce na metryki w sekcji Uwagi (Markdown aktualizowany po ask)
    eval_out = gr.Markdown()  # specjalne miejsce na wynik ewaluacji
    ask_btn.click(ask_question_stream, inputs=[question, top_k], outputs=[answer_out, contexts_out, eval_out])

    gr.Markdown("## ℹ️ Uwagi")
    gr.Markdown(
        "- Podsumowanie i odpowiedzi mogą trwać, zależnie od długości materiału i modelu.\n"
        "- Konteksty są dociągane po zakończeniu strumienia odpowiedzi.\n"
        "- Metryki ewaluacji (CE) pojawiają się po zakończeniu odpowiedzi w polu powyżej."
    )

demo.queue().launch(server_name="127.0.0.1", server_port=7860)


