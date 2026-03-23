import asyncio
import os
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from queue import Queue

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from llama_cpp import Llama
from myllm import ChatGPTClient
from utils import MyUtils

# ── Config ─────────────────────────────────────────────────────────────────────

EMBEDDING_MODEL_PATH = "../embed_model/bge-m3-FP16.gguf"
VECTOR_DB_DIR        = "../rag_vector_db"
TOP_K                = 3    # hadiths per book fed to the LLM prompt
TOP_K_UI             = 2    # hadiths per book shown in the UI


N_EMBED_WORKERS = 2

# ── GPU semaphore — only 1 Llama.embed() at a time ─────────────────────────────
_gpu_sem = threading.Semaphore(1)

# ── Pre-load all vector DBs into RAM once ──────────────────────────────────────
# Avoids re-reading pickle files on every request (major latency win).
print("Pre-loading vector databases into RAM…")
VECTOR_DBS: dict[str, list] = {}
for _fname in sorted(os.listdir(VECTOR_DB_DIR)):
    _book = _fname.split(".")[0]
    with open(os.path.join(VECTOR_DB_DIR, _fname), "rb") as _f:
        VECTOR_DBS[_book] = pickle.load(_f)
    print(f"  ✓ {_book}: {len(VECTOR_DBS[_book])} vectors")
print(f"All {len(VECTOR_DBS)} books loaded into RAM.\n")

print("Loading embedding model instance(s)…")
_model_pool: Queue = Queue()
for _i in range(N_EMBED_WORKERS):
    _model_pool.put(
        Llama(model_path=EMBEDDING_MODEL_PATH, embedding=True, verbose=False)
    )
    print(f"  ✓ Embedding model instance {_i + 1}/{N_EMBED_WORKERS} ready")
print()

util = MyUtils()

_executor = ThreadPoolExecutor(max_workers=N_EMBED_WORKERS + 4)



@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Server is ready — accepting requests.\n")
    yield
    _executor.shutdown(wait=True)
    print("Thread pool shut down cleanly.")


app = FastAPI(title="Hadith RAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ─────────────────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    query: str

class AskResponse(BaseModel):
    answer: str
    books: dict[str, list[str]]


# ── Blocking worker (runs in thread pool) ───────────────────────────────────────

def _embed_and_search(query: str) -> dict[str, list[str]]:
    """
    1. Borrow a Llama instance from the pool.
    2. Embed the query — serialised via _gpu_sem so only 1 thread uses GPU at once.
    3. Cosine-search all books in RAM — pure CPU, runs concurrently across threads.
    4. Return pool item and results.
    """
    model = _model_pool.get()   # waits only if all worker slots are busy
    try:
        with _gpu_sem:
            query_vec = model.embed(query)

        results: dict[str, list[str]] = {}
        for book_name, vectors in VECTOR_DBS.items():
            sims = [
                (text, util.cosine_similarity(query_vec, emb))
                for text, emb in vectors
            ]
            sims.sort(key=lambda x: x[1], reverse=True)
            results[book_name] = [t for t, _ in sims[:TOP_K]]

        return results
    finally:
        _model_pool.put(model)  # always return — even on exception


def _call_llm(prompt: str) -> str:
    """Instantiate a fresh ChatGPTClient and fire the request (sync wrapper)."""
    llm = ChatGPTClient()
    return llm.ask_text_question(prompt)


def build_prompt(query: str, similar: dict) -> str:
    return f"""
Below is a dictionary of hadith.

{similar}

From the above dictionary, select the 2 hadith that are most relevant to the following question.

Question:
{query}

Instructions:
1. The response must be entirely in Bangla.
2. Write the hadith as it is in the dictionary witout the source.
3. Do not include any explanation, introduction, or additional text.
4. Do not use any special characters (such as **, ``, --- etc.).
5. Follow the format below:

Full hadith text - Book name(Hadith number from the source.)
"""


# ── Route ────────────────────────────────────────────────────────────────────────

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    query = req.query.strip()
    print("*"*100)
    print(query)
    print("*"*100)
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    loop = asyncio.get_event_loop()

    # Step 1 — embed + vector search (blocking GPU+CPU work → thread pool)
    similar = await loop.run_in_executor(_executor, _embed_and_search, query)

    # Step 2 — LLM call (blocking network call → thread pool)
    # All 20 users' LLM calls run concurrently here because OpenAI/Groq
    # is an external service — no GPU contention on your machine.
    prompt = build_prompt(query, similar)
    answer = await loop.run_in_executor(_executor, _call_llm, prompt)

    top2 = {book: texts[:TOP_K_UI] for book, texts in similar.items()}
    return AskResponse(answer=answer, books=top2)


@app.get("/health")
async def health():
    pool_available = _model_pool.qsize()
    return {
        "status": "ok",
        "books_loaded": list(VECTOR_DBS.keys()),
        "embed_workers": N_EMBED_WORKERS,
        "embed_pool_free": pool_available,
    }


# ── Entry point ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=5123,
        workers=1,      # ← MUST stay 1: the GPU model pool lives in this process
        loop="asyncio",
        log_level="info",
    )