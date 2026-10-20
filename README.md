# Digital Twin — agentic portfolio assistant (v1.1)

An AI agent that answers questions about you (CV, skills, experience, projects, live GitHub activity).
**LangChain** `create_agent` (LangChain 1.x / LangGraph runtime) · **NVIDIA-hosted models** · **Qdrant hybrid RAG** ·
streaming **FastAPI** service. A pop-up website widget can be bolted on later.

## Architecture

```
Widget ──SSE──▶ FastAPI /v1/chat
                 │ rate-limit · input guard · session lock · timeout · <think> stripping
                 ▼
          create_agent (LangGraph)
          middleware: dynamic prompt → retry → model fallback → tool-call cap → summarization
                 │ tool calling
   ┌─────────────┼──────────────────────────┬────────────────────┐
   ▼             ▼                          ▼                    ▼
search_knowledge  github_* (4 read-only,   get_contact_links   optional MCP
 Qdrant HYBRID:    TTL-cached, locked to    (deterministic)     (allow-listed,
 dense (NVIDIA     ONE username)                                fail-closed)
 embed) + BM25 → RRF fusion → NVIDIA rerank
```



## What changed in v1.1

- **Qdrant replaces Chroma**, with *hybrid* search: dense NVIDIA embeddings + BM25 keywords fused by RRF, then reranked. Keywords matter for a CV (`YOLO26`, `TensorRT`, `SAHI`, `RF-DETR`), which pure vector search blurs.
- **Three Qdrant modes** (no code change, just env): embedded local · Docker service · Qdrant Cloud free tier.
- **CV is** `knowledge/cv.md`: structured Markdown made from your Word resume (phone number deliberately excluded; ingest also redacts phone numbers as a safety net).
- **GitHub READMEs are indexed** (selectable) *and* live GitHub tools remain (see "READMEs vs tools").
- Your `.env` is honoured: `NVIDIA_BASE_URL`, `PERSONA=first_person`, your models, LangSmith (the app now exports `.env` to the process so LangSmith tracing actually switches on).
- Reasoning-model safe: `<think>…</think>` is stripped from streams; token budgets raised.
- Expired `GITHUB_TOKEN` now degrades to unauthenticated requests instead of breaking GitHub tools.



## Quick start

```bash
uv venv .venv && .\.venv\Scripts\Activate.ps1 
uv sync --extra dev or 
cp .env.example .env    # fill NVIDIA_API_KEY (+ GITHUB_TOKEN, LANGSMITH_API_KEY)
uv run python scripts/check_models.py # verifies models, tool-calling, Qdrant, sparse model
uv run python -m twin.ingest --github  # index knowledge/ + GitHub READMEs
uv run uvicorn twin.api:app --port 8000
curl -N localhost:8000/v1/chat -H 'content-type: application/json' -d '{"message":"Where do you work?"}'
```



### Choosing a Qdrant mode


| Mode              | Set                                       | Best for                 | Watch out                                                                                                                                                                                                  |
| ----------------- | ----------------------------------------- | ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Embedded          | leave `QDRANT_URL` empty                  | local dev, tiny deploys  | file lock: stop the API before running ingest                                                                                                                                                              |
| Docker (compose)  | automatic via `docker compose up --build` | any VPS/VM you control   | needs a persistent volume (provided)                                                                                                                                                                       |
| Qdrant Cloud free | `QDRANT_URL` + `QDRANT_API_KEY`           | PaaS hosts without disks | free clusters are **paused after 1 idle week** and deleted after 4; the API pings Qdrant every `QDRANT_KEEPALIVE_HOURS` (12) to keep it awake, but if your API host also sleeps, add an external cron ping |


For a ~50-chunk corpus, Qdrant vs Chroma speed is irrelevant (the LLM dominates latency). The reasons to use Qdrant here are hybrid search and a production-grade server you can scale or move later.

## Knowledge base

- `knowledge/cv.md` — primary source. Edit it, then re-run `python -m twin.ingest --github`.
- Add `knowledge/faq.md` from `_templates/faq.md` (roles sought, availability, location, working style).
- Keep your original `.docx`/PDF outside this folder.

**Which CV format is best?** Structured **Markdown** > text > PDF ≈ Word. Markdown keeps headings, so each chunk is
stored with its job/project/section path and retrieval can answer "what did you do at X?" precisely. PDF/Word
extraction flattens layout (header tables, tabs, columns) and mixes sections. Keep Word as your editing master; treat
`cv.md` as the version the twin reads.

## READMEs vs tools (GitHub)

MCP/tools do **not** index anything — they are on-demand lookups the model must decide to call. Use both:

- **Indexed READMEs** (`--github`): fast, semantic, answers broad questions ("computer-vision projects?") in one retrieval.
Control it with `GITHUB_README_REPOS` (force-include) and `GITHUB_README_LIMIT`. Re-run ingest after READMEs change
(e.g. a weekly GitHub Action, or on each deploy).
- **Live tools**: always-fresh facts — recent activity, newest repos, stars, languages.



## API contract (for the widget)

`GET /v1/config` · `GET /healthz` · `GET /readyz` (503 until indexed) · `POST /v1/chat {message, session_id?}` → SSE:


| event            | data                    | client action                        |
| ---------------- | ----------------------- | ------------------------------------ |
| `session`        | `{session_id}`          | store; send with follow-ups          |
| `tool`           | `{name,label,status}`   | show "Searching CV & notes…"         |
| `reset`          | `{}`                    | clear text streamed so far this turn |
| `token`          | `{text}`                | append                               |
| `done` / `error` | `{tools}` / `{message}` | finish / show error                  |


Errors: `429` (+`Retry-After`), `409`, `413`, `422`, `503`.

## Production notes

- Run **one** uvicorn worker (in-memory sessions + rate limiter). Scale up, not out, or move both to Redis/Postgres.
- Set `CORS_ORIGINS` to your real site; `TRUST_PROXY=true` only behind your own proxy.
- `.env` is git-ignored. Never commit or upload it; rotate any key that has been shared.
- Tests: `pytest` (guardrails, text filters) · `python evals/run_evals.py` (live regression against your real CV facts).

