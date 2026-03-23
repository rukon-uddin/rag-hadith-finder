# rag-hadith-finder — হাদিস অনুসন্ধান

Search Islamic hadith collections **in Bangla**. Ask your question in Bangla, get your answer in Bangla.

## How it works

1. Type a question in Bangla
2. The system searches across 8 major hadith books using semantic vector search
3. An LLM reads the matching hadiths and writes a grounded answer — in Bangla

## Supported Books

Bukhari · Muslim · Abu Dawud · Tirmidhi · Nasa'i · Ibn Majah · Malik · Nawawi

## Stack

- FastAPI + uvicorn backend
- BGE-M3 (GGUF) for local embeddings
- OpenAI `gpt-4.1-mini` for answer generation
- Vanilla HTML/CSS/JS frontend — no build step

## Setup

```bash
# Set your OpenAI key
export OPENAI_API_KEY=sk-...

# Place model and vector DBs at:
# ../embed_model/bge-m3-FP16.gguf
# ../rag_vector_db/<book>.pkl

# Run
python server.py
```

Then open `index.html` in your browser.
