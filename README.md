# Digital Twin — agentic portfolio assistant

A production-oriented AI agent that answers questions about you (CV, skills, experience, projects, live GitHub activity).
Built on **LangChain `create_agent`** (LangChain 1.x / LangGraph runtime) with **NVIDIA-hosted models** (free API catalog).
Ships as a streaming FastAPI service; a pop-up website widget can be bolted on later.

## Architecture

```
Widget ──SSE──▶ FastAPI /v1/chat
                 │ rate-limit · input guard · session lock · timeout
                 ▼
          create_agent (LangGraph)
          middleware: dynamic prompt → retry → model fallback → tool-call cap → summarization
                 │ tool calling
   ┌─────────────┼──────────────────────────┬────────────────────┐
   ▼             ▼                          ▼                    ▼
search_knowledge  github_* (4 read-only,   get_contact_links   optional MCP
(RAG: Chroma +    TTL-cached, bound to      (deterministic,     (allow-listed,
 NVIDIA embed →   ONE username)             from env)           fail-closed)
 NVIDIA rerank)
```

### Design decisions
| Decision | Why |
|---|---|
| `create_agent`, not Deep Agents | A Q&A twin needs a short tool loop, not planning/sub-agents/filesystem. Deep Agents adds latency and tokens for no gain here. |
| Agentic RAG (retrieve as a tool) | The model decides *when* to look things up, so chit-chat/refusals cost zero retrieval calls. |
| Retrieve 12 → rerank to 4 | Dense recall + cross-encoder precision; far fewer irrelevant chunks, shorter prompts, faster answers. |
| Contextual chunk headers + Markdown-aware splitting | Each chunk carries `file — section`, which markedly helps retrieval on short CV-style facts. |
| Native GitHub tools instead of GitHub MCP | Faster (no extra hop), cached, read-only, locked to your username. A public chatbot should not hold a token-bearing MCP server with write tools. MCP loader exists for other sources but is **fail-closed** (explicit allowlist). |
| Llama-3.3-70B primary, Nemotron-3-Super fallback, Llama-3.1-8B summarizer | 70B = reliable tool calling without reasoning-token latency; fallback keeps you up if one model is throttled/removed; the 8B only compresses long chats. |
| In-memory sessions with TTL + cap, single worker | Zero infra, no memory leak. Swap the checkpointer for Postgres/Redis if you ever scale to multiple replicas. |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env                     # add NVIDIA_API_KEY, identity, GitHub username
cp knowledge/_templates/*.md knowledge/  # fill in with real facts; add knowledge/cv.pdf
python scripts/check_models.py           # verify model ids + latency (catalog rotates!)
python -m twin.ingest --github           # build the vector index (+ top repo READMEs)
uvicorn twin.api:app --port 8000
```

Test it:
```bash
curl -N localhost:8000/v1/chat -H 'content-type: application/json' \
     -d '{"message":"What are your strongest skills?"}'
```

Docker: `docker compose up --build` (auto-ingests on first start).
Re-index after editing knowledge: `docker compose exec twin python -m twin.ingest --github`.

## API contract (for the widget)

`GET /v1/config` → `{owner, starter_questions, max_message_chars}`
`GET /healthz` (liveness) · `GET /readyz` (503 until the index is populated)
`POST /v1/chat` `{message, session_id?}` → `text/event-stream`

| event | data | client action |
|---|---|---|
| `session` | `{session_id}` | store it; send it with every follow-up |
| `tool` | `{name,label,status:start\|end}` | show "Searching CV & notes…" |
| `reset` | `{}` | clear text streamed so far this turn (preamble before a tool call) |
| `token` | `{text}` | append to the bubble |
| `done` | `{tools}` | finish |
| `error` | `{message}` | show friendly error |

Errors: `429` (+`Retry-After`), `409` (turn in progress), `413`, `422`, `503`.

## Production hardening (built in)
- Per-IP sliding-window limit **and** a global daily cap to protect the free NVIDIA quota
- Deterministic prompt-injection / length / control-char guard before any LLM call
- Prompt rules: ground every claim in tools, no invention, no commitments (salary/dates), scope-limited, retrieved text treated as untrusted data
- Read-only tools; GitHub owner is not model-controllable; repo names regex-validated
- Retry → fallback model → graceful rerank/GitHub degradation; request timeout; bounded tool loop
- CORS allow-list, non-root container, no secrets in prompts or logs, message content not logged

## Before going live
1. `pytest` (guardrails/rate-limit) and `python evals/run_evals.py` after editing `evals/golden.yaml` with your real facts.
2. Set `CORS_ORIGINS` to your site; set `TRUST_PROXY=true` only behind your own proxy (Nginx/Caddy/Cloudflare).
3. Put Cloudflare Turnstile (or similar) in front of the widget if you expect bots.
4. Optional tracing: set `LANGSMITH_TRACING=true` + key.

## Extending
- Hybrid (BM25 + vector) search: add a keyword retriever and fuse before reranking.
- New data source: add a `@tool` in `src/twin/tools/` and register it in `agent.py`.
- Blog/Notion/etc. via MCP: set `MCP_SERVERS_JSON` and `MCP_TOOL_ALLOWLIST` (read-only tools only).
