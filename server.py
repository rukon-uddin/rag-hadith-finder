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
HADITH_BOOK_SQUENCE = ['Top 3',"bukhari","muslim","abudawud","tirmidhi","nasai","ibnmajah","malik","nawawi"]
TOP_K                = 5    # hadiths per book fed to the LLM prompt
TOP_K_UI             = 1    # hadiths per book shown in the UI


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
    model = _model_pool.get()
    try:
        with _gpu_sem:
            query_vec = model.embed(query)

        results: dict[str, list[str]] = {}
        final_results: dict[str, list[str]] = {}
        all_sims: list[tuple[str, float]] = []
        
        for book_name, vectors in VECTOR_DBS.items():
            sims = [
                (text, util.cosine_similarity(query_vec, emb))
                for text, emb in vectors
            ]
            sims.sort(key=lambda x: x[1], reverse=True)
            results[book_name] = [t for t, _ in sims[:TOP_K]]
            all_sims.extend(sims)  # collect for global top 3

        all_sims.sort(key=lambda x: x[1], reverse=True)
        results["Top 3"] = [t for t, _ in all_sims[:4]]

        for i in HADITH_BOOK_SQUENCE:
            final_results[i] = results[i]

        return final_results
    finally:
        _model_pool.put(model)


def _call_llm(prompt: str) -> str:
    """Instantiate a fresh ChatGPTClient and fire the request (sync wrapper)."""
    llm = ChatGPTClient()
    return llm.ask_text_question(prompt)


def build_prompt(query: str, similar: dict) -> str:
    return f"""
Below is a dictionary of hadith.

{similar}

A user has asked the following question:

Question:
{query}

Instructions:
1. Read the question carefully and check if the hadith in the dictionary can answer it.
2. If the hadith can answer the question, write a single, direct answer in Bangla based strictly on the hadith content. Do not add any information beyond what the hadith states.
3. If no hadith in the dictionary clearly answers the question, respond only with: "এই প্রশ্নের উত্তর প্রদত্ত হাদিসে পাওয়া যায়নি।"
4. The response must be entirely in Bangla.
5. After the answer, cite the hadith(s) you based your answer on using the format below.
6. Do not include any explanation, introduction, or additional text beyond the answer and citation.
7. Do not use any special characters (such as **, ``, --- etc.).

Format:
Answer in Bangla.

Source: Book name (Hadith number)
"""


def build_query_prompt(query):
    return f"""
আপনি একজন সহায়ক ইসলামিক অনুসন্ধান সহকারী।

ব্যবহারকারী একটি প্রশ্ন বা বাক্য লিখেছে যা অগোছালো বা অসম্পূর্ণ হতে পারে। 
আপনার কাজ হলো মূল প্রশ্নের অর্থ ঠিক রেখে আরও ৩টি ভিন্ন বাক্য তৈরি করা, 
যাতে একই বিষয়ে হাদিস খোঁজা সহজ হয়।

নিয়ম:
১. সব বাক্য সম্পূর্ণ বাংলায় হবে।
২. মূল প্রশ্নের অর্থ পরিবর্তন করা যাবে না।
৩. প্রতিটি বাক্য আলাদা ভাবে লিখতে হবে।
৪. অতিরিক্ত ব্যাখ্যা বা কোনো মন্তব্য লিখবেন না।
৫. শুধু ৩টি নতুন বাক্য লিখবেন।

ব্যবহারকারীর প্রশ্ন:
{query}

উত্তর:
"""


# ── Route ────────────────────────────────────────────────────────────────────────

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    query = req.query.strip()
    query_prompt = build_query_prompt(query)
    
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    loop = asyncio.get_event_loop()

    generate_query = await loop.run_in_executor(_executor, _call_llm, query_prompt)
    print("*"*100)
    print(query)
    print('-'*100)
    print(generate_query)
    print("*"*100)
    # Step 1 — embed + vector search (blocking GPU+CPU work → thread pool)
    similar = await loop.run_in_executor(_executor, _embed_and_search, generate_query)

    # Step 2 — LLM call (blocking network call → thread pool)
    # All 20 users' LLM calls run concurrently here because OpenAI/Groq
    # is an external service — no GPU contention on your machine.
    # prompt = build_prompt(query, similar)
    # answer = await loop.run_in_executor(_executor, _call_llm, prompt)

    # top2 = {book: texts[:TOP_K_UI] for book, texts in similar.items()}
    top2 = {}
    for book, texts in similar.items():
        top2[book] = texts[:TOP_K_UI]
        if book == 'Top 3':
            top2[book] = texts[:4]

    return AskResponse(answer='', books=top2)


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