import os
import json
import chromadb
from typing import List, Dict, Any
from langchain_openai import OpenAIEmbeddings

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
_client = chromadb.Client()
embedding = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")

#zwraca kolekcje lub tworzy nową
def get_collection(name: str):
    return _client.get_or_create_collection(name)

#zapisuje chunki do bazy wektorowej z metadanymi
def store_chunks(collection, chunks):
    documents = [c["text"] for c in chunks]
    metadatas = [{
        "id": c.get("id", i),
        "speaker": c.get("speaker", "UNKNOWN"),
        "start": c.get("start"),
        "end": c.get("end"),
    } for i, c in enumerate(chunks)]
    ids = [str(m["id"]) for m in metadatas]
    vectors = embedding.embed_documents(documents)
    collection.upsert(documents=documents, metadatas=metadatas, ids=ids, embeddings=vectors)

# zapytanie do bazy wektorowej
def query_db(collection, question: str, n_results: int = 5):
    q_vec = embedding.embed_query(question)
    return collection.query(query_embeddings=[q_vec], n_results=n_results)

# initializacja cross-encodera do rerankingu 
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    print(f"[RERANK] CrossEncoder loaded: {CROSS_ENCODER_MODEL}")
except Exception as e:
    _cross_encoder = None
    print(f"[RERANK] CrossEncoder not available: {e}")

def query_and_rerank_crossencoder(collection, question: str, n_candidates: int = 20, top_k: int = 5) -> List[Dict[str, Any]]:
    top_k = max(1, min(int(top_k), 20)) #ograniczenie top_k do [1,20]
    res = query_db(collection, question, n_results=n_candidates)
    ctxs = _build_contexts_from_query(res)
    return _cross_encode_rerank(question, ctxs, top_k)

# reranking z cross-encoderem
def _build_contexts_from_query(res: Dict[str, Any]) -> List[Dict[str, Any]]:
    docs = res.get("documents", [[]])[0] or []
    metas = res.get("metadatas", [[]])[0] or []
    contexts: List[Dict[str, Any]] = []
    for d, m in zip(docs, metas):
        m = m or {}
        contexts.append({
            "text": d,
            "speaker": m.get("speaker"),
            "start": m.get("start"),
            "end": m.get("end"),
        })
    return contexts

def _cross_encode_rerank(question: str, contexts: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    if not contexts or _cross_encoder is None:
        return contexts[:top_k]
    pairs = [(question, c.get("text") or "") for c in contexts]
    scores = _cross_encoder.predict(pairs)  
    items = [{**c, "rerank_score": float(s)} for c, s in zip(contexts, scores)]
    items.sort(key=lambda x: x["rerank_score"], reverse=True)
    return items[:top_k]



