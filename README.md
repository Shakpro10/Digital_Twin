# Digital Twin — Agentic AI Portfolio Assistant

**A retrieval-grounded AI digital twin that answers questions about a professional's background, skills, work experience, projects, and public GitHub activity.** Built with FastAPI, LangChain agents, NVIDIA-hosted language models, hybrid retrieval with Qdrant, and locally run embedding and reranking models.

![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB)
![API](https://img.shields.io/badge/API-FastAPI-009688)
![Agents](https://img.shields.io/badge/Agents-LangChain%20%2B%20LangGraph-1f2937)
![Retrieval](https://img.shields.io/badge/RAG-Qdrant-DC244C)
![LLM](https://img.shields.io/badge/LLM-NVIDIA%20API%20Catalog-76B900)
![Deployment](https://img.shields.io/badge/Deployment-Docker-2496ED)

## Overview

Digital Twin is an API-first, agentic portfolio assistant designed to represent a professional's documented profile on a portfolio website or in a chat widget. It combines a curated knowledge base with indexed GitHub README files and read-only tools that can retrieve current public GitHub information.

Rather than relying on the language model's general knowledge to answer personal or career questions, the agent is instructed to retrieve supporting information from its tools. If the information is not documented, it should say so instead of inventing employers, dates, metrics, skills, salary expectations, or commitments.

The service exposes a streaming chat endpoint using Server-Sent Events (SSE), making it suitable for integration with a custom website widget or other frontend.

> **Project status:** Portfolio-grade application under active development. Model availability, API quotas, and end-to-end behavior depend on the configured NVIDIA endpoint, GitHub API access, and Qdrant deployment. Run the checks in this README before deploying.

## Key features

- **Agentic question answering:** Uses LangChain's `create_agent` and LangGraph runtime to decide when to search knowledge, retrieve GitHub information, or return configured contact links.
- **Hybrid RAG:** Combines dense semantic retrieval and BM25 sparse keyword retrieval in Qdrant, fuses rankings using Reciprocal Rank Fusion (RRF), and reranks candidates.
- **Local embedding and reranking:** Runs FastEmbed/ONNX models locally, avoiding an embedding API call for each query.
- **Professional knowledge base:** Supports Markdown, text, and PDF files in `knowledge/`; Markdown headings are retained as contextual section metadata.
- **GitHub README indexing:** Optionally indexes selected public repository READMEs for broad semantic questions about projects.
- **Live, read-only GitHub tools:** Can list public repositories, inspect repository details, summarize recent activity, and retrieve profile statistics for one configured GitHub username.
- **Streaming FastAPI endpoint:** Streams session, tool status, text token, completion, and error events for responsive UI integration.
- **Model resilience:** Configurable primary, fallback, and summarization models, with retry and fallback middleware.
- **Conversation sessions:** Maintains in-memory conversation checkpoints and session expiry for follow-up questions.
- **Safety controls:** Input validation, basic prompt-injection filtering, request rate limits, daily request cap, bounded tool calls, and removal of inline `<think>...</think>` content from streamed output.
- **Optional MCP integration:** MCP tools are disabled unless server configuration and an explicit tool allowlist are supplied.
- **Operational endpoints:** Includes health and readiness checks, with readiness reflecting whether the knowledge collection is populated.
- **Docker support:** Includes a Dockerfile and Docker Compose setup with Qdrant persistence and a project-local FastEmbed cache.

## How it works

```text
                      ┌─────────────────────────┐
                      │ Website widget / client││ 
                      └────────────┬────────────┘
                                   │ POST /v1/chat
                                   │ Server-Sent Events
                                   ▼
                      ┌─────────────────────────┐
                      │       FastAPI API       │
                      │ validation · limits     │
                      │ session · stream        │
                      └────────────┬────────────┘
                                   ▼
                      ┌─────────────────────────┐
                      │ LangChain agent         │
                      │ LangGraph runtime       │
                      │ retry · fallback        │
                      │ tool-call limit         │
                      │ conversation summary    │
                      └───────┬─────────┬───────┘
                              │         │
                   ┌──────────┘         └──────────────┐
                   ▼                                   ▼
       ┌────────────────────────┐         ┌────────────────────────┐
       │ search_knowledge       │         │ Live GitHub tools      │
       │ Qdrant hybrid retrieval│         │ read-only, one owner   │
       └────────────┬───────────┘         └────────────────────────┘
                    │
                    ▼
       ┌────────────────────────────────────────────────┐
       │ Dense vectors + BM25 sparse vectors            │
       │ Reciprocal Rank Fusion → local cross-encoder   │
       └─────────────────────────┬──────────────────────┘
                                 ▼
                    ┌────────────────────────┐
                    │ Grounded response      │
                    │ streamed to the client │
                    └────────────────────────┘

Knowledge files ──┐
                  ├── ingestion ──► Qdrant collection
GitHub READMEs ───┘
```

### Retrieval pipeline

1. Read supported files from `knowledge/` and, optionally, selected public GitHub repository READMEs.
2. Redact phone-number patterns when `REDACT_PHONES` is enabled.
3. Split Markdown by heading and then split long sections into overlapping text chunks.
4. Generate dense embeddings with the configured local FastEmbed model.
5. Generate BM25 sparse representations for keyword matching.
6. Store dense and sparse vectors, source metadata, section names, and PDF page metadata in Qdrant.
7. At query time, retrieve candidates using hybrid search and fuse rankings with Reciprocal Rank Fusion.
8. Rerank candidates with the configured local cross-encoder and return the highest-ranked excerpts to the `search_knowledge` tool.
9. The agent uses those excerpts as evidence to formulate a concise answer.

GitHub README indexing and live GitHub tools serve different purposes. Indexed READMEs help answer broad questions about projects; live tools provide up-to-date repository and activity information.

## Architecture and technology stack

| Layer | Technology | Role |
|---|---|---|
| API | FastAPI, Uvicorn | HTTP endpoints and streaming responses |
| Agent orchestration | LangChain `create_agent`, LangGraph | Tool selection, conversation flow, middleware |
| Chat models | NVIDIA API Catalog via `ChatNVIDIA` | Primary responses, fallback, summarization |
| Dense embeddings | FastEmbed; default `BAAI/bge-small-en-v1.5` | Local semantic vectors (384 dimensions by default) |
| Sparse retrieval | FastEmbed `Qdrant/bm25` | Keyword-oriented matching |
| Vector database | Qdrant | Dense and sparse vector storage and retrieval |
| Reranker | FastEmbed; default `Xenova/ms-marco-MiniLM-L-6-v2` | Reranks retrieved passages locally |
| Knowledge ingestion | LangChain text splitters, `pypdf` | Markdown, TXT, and PDF ingestion |
| GitHub integration | GitHub REST API via `httpx` | Read-only public repository and activity tools |
| Optional tool integration | MCP adapters | Explicitly allow-listed external tools |
| Configuration | Pydantic Settings, `python-dotenv` | Environment-based configuration |
| Local model cache | FastEmbed cache directory | Reuses downloaded model artifacts |
| Deployment | Docker, Docker Compose | Containerized API and Qdrant service |
| Testing | pytest, YAML-based evaluation script | Unit tests and LLM-in-the-loop regression checks |

## Project structure

```text
digitaltwin/
├── knowledge/
│   ├── cv.md                     # Main professional profile
│   ├── README.md                 # Knowledge-base authoring guidance
│   └── _templates/               # Example profile, projects, FAQ templates
├── scripts/
│   ├── check_models.py           # Model, retrieval, and service smoke checks
│   └── entrypoint.sh             # Container startup and initial ingestion
├── evals/
│   ├── golden.yaml               # Curated regression questions and expectations
│   └── run_evals.py              # Agent behavior regression runner
├── src/twin/
│   ├── api.py                    # FastAPI routes and SSE streaming
│   ├── agent.py                  # Agent construction and middleware
│   ├── config.py                 # Settings and environment configuration
│   ├── guardrails.py             # Input checks and in-memory rate limiter
│   ├── ingest.py                 # Knowledge ingestion and chunking
│   ├── llm.py                    # Chat model and local model factories
│   ├── local_models.py           # Dense embedding and reranker wrappers
│   ├── prompts.py                # Digital-twin persona and grounding rules
│   ├── retrieval.py              # Qdrant client, indexing and hybrid retrieval
│   ├── sessions.py               # In-memory session lifecycle
│   ├── textfilter.py             # Phone redaction and reasoning-tag filtering
│   └── tools/
│       ├── knowledge.py          # Knowledge search and contact-link tools
│       ├── github.py             # Read-only GitHub tools and README loader
│       └── mcp_tools.py          # Optional allow-listed MCP tools
├── tests/
│   ├── test_guardrails.py
│   └── test_textfilter.py
├── .cache/fastembed/             # Local model cache (generated; not committed)
├── .env.example                  # Safe configuration template
├── .gitignore
├── .dockerignore
├── Dockerfile
├── compose.yaml
├── pyproject.toml
├── uv.lock
├── REVIEW_CHANGES.md
└── README.md
```

## Requirements

- Python **3.12 or later**
- [`uv`](https://docs.astral.sh/uv/) for dependency and environment management
- An NVIDIA API Catalog key for hosted chat-model inference
- A Qdrant instance, or the embedded local Qdrant mode
- GitHub username for live GitHub tools and README ingestion (optional but recommended)
- Docker Desktop / Docker Engine with the Compose plugin for container deployment

The first local embedding, reranking, or sparse-model use may download model files. Ensure the environment can reach the relevant model repositories and has sufficient disk space.

## Quick start — Windows PowerShell

### 1. Enter the project directory

Open PowerShell at the extracted project root (the directory containing `pyproject.toml`).

### 2. Create and configure `.env`

```powershell
Copy-Item .env.example .env
```

Open `.env` and set at minimum:

```dotenv
NVIDIA_API_KEY=your_nvidia_api_key
OWNER_NAME=Your Name
```

Replace the placeholders with your own values. Do not commit or share `.env`. Add optional settings such as `GITHUB_USERNAME`, `GITHUB_TOKEN`, public contact links, and Qdrant Cloud credentials only when needed.

### 3. Install dependencies

```powershell
uv sync --extra dev
```

### 4. Test configured models and services

```powershell
uv run python scripts/check_models.py
```

This checks tool calling for the configured primary, fallback, and summarizer chat models, the dense embedding model, reranker, Qdrant connection, and sparse model. A successful tool-calling check does not by itself prove structured-output support or complete end-to-end RAG behavior.

### 5. Index the knowledge base

Index the local `knowledge/` files:

```powershell
uv run python -m twin.ingest
```

Index local knowledge plus selected GitHub repository READMEs:

```powershell
uv run python -m twin.ingest --github
```

The GitHub README selection is controlled by `GITHUB_README_REPOS` and `GITHUB_README_LIMIT`. A full ingestion rebuilds the configured collection. Back up any data you need before rebuilding a populated collection.

### 6. Start the API

```powershell
uv run uvicorn twin.api:app --host 127.0.0.1 --port 8000
```

The API listens on `http://127.0.0.1:8000`. The project does not expose a homepage at `/`; use the documented endpoints below.

### 7. Send a streaming chat request

In a second PowerShell window:

```powershell
$body = @{ message = "Where do you work?" } | ConvertTo-Json -Compress
curl.exe -N http://127.0.0.1:8000/v1/chat `
  -H "Content-Type: application/json" `
  -d $body
```

or this if the one above doesn't work as curl.exe -d $body can mishandle how that string is passed to the executable

```powershell
$body = @{ message = "Where do you work?" } | ConvertTo-Json -Compress
$body | Set-Content -Path request.json -Encoding utf8

curl.exe -N http://127.0.0.1:8000/v1/chat `
  -H "Content-Type: application/json" `
  --data-binary "@request.json"
```

or this if you're using powershell 7

```powershell
$body = @{ message = "Where do you work?" } | ConvertTo-Json -Compress

curl.exe -N http://127.0.0.1:8000/v1/chat `
  -H "Content-Type: application/json" `
  --data-raw "$body"
```

Also, verify the endpoint; first, check that the server is healthy:

```powershell
curl.exe http://127.0.0.1:8000/healthz
```

The response is an SSE stream. You should see events for the session, any tools used, generated text, and completion or error.

## Docker deployment

Docker Compose starts the FastAPI service and a Qdrant server with a persistent named volume.

1. Create `.env` from `.env.example`, then set `NVIDIA_API_KEY` and `OWNER_NAME`.
2. Review `CORS_ORIGINS` and optional GitHub settings.
3. From the project root, run:

```bash
docker compose config
docker compose up --build -d
docker compose logs -f twin
```

The container entrypoint runs ingestion if the collection is empty and includes GitHub README ingestion when `INGEST_GITHUB` is enabled in Compose. The Qdrant service's data is stored in the named `qdrant-data` volume. The FastEmbed cache is persisted in the project-local `.cache/fastembed/` directory.

To re-index after changing files under `knowledge/` or after changing the selected GitHub repositories, run:

```bash
docker compose exec twin python -m twin.ingest --github
```

**Note:** this is a full collection rebuild, not a partial update. The ingestion process recreates the configured collection before adding the new chunks.

## Configuration

Settings are loaded from `.env` and environment variables. Environment variables can override values in `.env`. See `.env.example` for the complete commented template.

### Required settings

| Variable | Purpose |
|---|---|
| `NVIDIA_API_KEY` | Authenticates NVIDIA-hosted chat-model requests |
| `OWNER_NAME` | Name used to configure the digital-twin persona |

### Chat models

| Variable | Default | Purpose |
|---|---|---|
| `NVIDIA_BASE_URL` | Provider integration default | Optional NVIDIA-compatible API base URL |
| `LLM_MODEL` | `nvidia/nemotron-3.5-super-vl-120b-a12b` | Primary chat model |
| `FALLBACK_LLM_MODEL` | `openai/gpt-oss-20b` | Fallback model if primary inference fails persistently |
| `SUMMARIZER_MODEL` | `openai/gpt-oss-20b` | Model used to summarize longer conversations |
| `LLM_TEMPERATURE` | `0.2` | Sampling temperature |
| `LLM_MAX_TOKENS` | `2000` | Main chat model token budget |
| `SUMMARIZER_MAX_TOKENS` | `1000` | Summarization token budget |

Model identifiers and free-endpoint availability can change. Use `scripts/check_models.py` to check current connectivity and tool-call behavior before deployment.

### Retrieval and model cache

| Variable | Default | Purpose |
|---|---|---|
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Local dense embedding model |
| `RERANK_MODEL` | `Xenova/ms-marco-MiniLM-L-6-v2` | Local cross-encoder reranker |
| `SPARSE_MODEL` | `Qdrant/bm25` | Sparse keyword model |
| `FASTEMBED_CACHE_PATH` | `.cache/fastembed` | Project-local cache for downloaded local models |
| `QDRANT_COLLECTION` | `twin_knowledge` | Collection name |
| `CHUNK_SIZE` | `900` | Recursive text chunk size |
| `CHUNK_OVERLAP` | `120` | Overlap between chunks |
| `RETRIEVE_K` | `12` | Number of initial retrieval candidates |
| `RERANK_TOP_N` | `4` | Number of passages retained after reranking |
| `REDACT_PHONES` | `true` | Redacts phone-number patterns during ingestion |

The default embedding model produces 384-dimensional vectors. If you change `EMBED_MODEL`, re-index the collection so its vector dimensions and stored embeddings match the selected model.

### Qdrant modes

| Mode | Configuration | Notes |
|---|---|---|
| Embedded local | Leave `QDRANT_URL` empty | Stores local data under `data/qdrant` by default. Stop the API before ingesting to avoid embedded-storage file-lock conflicts. |
| Docker Compose | Use the supplied `compose.yaml` | Compose config points the API at `http://qdrant:6333` and persists Qdrant data in `qdrant-data`. |
| Qdrant Cloud | Set `QDRANT_URL` and `QDRANT_API_KEY` | Useful for hosted API deployments where local disk may be ephemeral. |

Additional Qdrant settings:

- `QDRANT_URL`: Qdrant server URL; empty selects embedded mode.
- `QDRANT_API_KEY`: optional for local Qdrant, normally required for a secured cloud cluster.
- `QDRANT_PATH`: local embedded database directory, default `data/qdrant`.
- `QDRANT_KEEPALIVE_HOURS`: interval for periodic Qdrant pings when using a remote Qdrant URL; default `12`.

### Identity, GitHub, API, and optional integrations

| Variable | Default | Purpose |
|---|---|---|
| `OWNER_TITLE` | `AI Engineer` | Role/title used in the persona |
| `PERSONA` | `third_person` | `third_person` or `first_person` voice |
| `CONTACT_EMAIL` | empty | Public contact email returned by the contact tool |
| `LINKEDIN_URL` | empty | Public LinkedIn link |
| `WEBSITE_URL` | empty | Portfolio or website URL |
| `CALENDAR_URL` | empty | Scheduling link |
| `GITHUB_USERNAME` | empty | The single GitHub account used by live tools and README ingestion |
| `GITHUB_TOKEN` | empty | Optional GitHub API token; improves authenticated API rate limits |
| `GITHUB_README_REPOS` | empty | Comma-separated repository names to prioritize for README ingestion |
| `GITHUB_README_LIMIT` | `10` | Maximum README documents selected for ingestion |
| `GITHUB_CACHE_TTL_S` | `900` | Live GitHub response cache duration |
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated allowed browser origins |
| `RATE_LIMIT_PER_MIN` | `12` | Per-client request limit per minute |
| `DAILY_REQUEST_CAP` | `3000` | In-memory daily request cap |
| `MAX_MESSAGE_CHARS` | `1000` | Guardrail maximum message length (the request schema also enforces an upper bound) |
| `REQUEST_TIMEOUT_S` | `90` | Maximum duration for an agent request |
| `MAX_TOOL_CALLS` | `4` | Tool-call limit per turn |
| `SESSION_TTL_S` | `1800` | Idle session expiry |
| `MAX_SESSIONS` | `2000` | Maximum tracked sessions |
| `TRUST_PROXY` | `false` | Trust forwarded client IP headers only behind a controlled proxy |
| `MCP_SERVERS_JSON` | empty | Optional MCP server configuration in JSON |
| `MCP_TOOL_ALLOWLIST` | empty | Comma-separated names of MCP tools explicitly allowed |
| `LANGSMITH_TRACING` and related settings | disabled unless configured | Optional LangSmith tracing and project configuration |

MCP is fail-closed: setting server configuration alone does not expose tools; tool names must also be explicitly allow-listed. Only allow trusted servers and tools, especially in a public-facing service.

## API reference

Interactive Swagger and ReDoc pages are disabled in the current FastAPI configuration.

### `GET /healthz`

Returns a basic process health response:

```json
{"status": "ok"}
```

This confirms the API process responds; it does not guarantee the knowledge collection is populated or that all external services are available.

### `GET /readyz`

Returns `{"status":"ready"}` when the configured knowledge collection contains indexed data. Returns HTTP `503` when the knowledge base is empty.

### `GET /v1/config`

Returns public widget bootstrap data, including the configured owner name, starter questions, and maximum message length.

### `POST /v1/chat`

Request body:

```json
{
  "message": "Where do you work?",
  "session_id": null
}
```

- `message` is required.
- `session_id` is optional. Store the returned session ID and send it with follow-up messages to continue the same in-memory conversation.
- The response uses `Content-Type: text/event-stream`.

SSE events:

| Event | Data | Typical client behavior |
|---|---|---|
| `session` | `{"session_id":"..."}` | Store the session ID |
| `tool` | Tool name, status, and sometimes a user-facing label | Show search/activity status |
| `reset` | `{}` | Clear streamed text for the current turn if the model emitted a preamble before calling a tool |
| `token` | `{"text":"..."}` | Append text to the displayed answer |
| `done` | Metadata including tools used | Mark the answer complete |
| `error` | `{"message":"..."}` | Display the error |

The service can return HTTP `409` when a session already has an active request, `413` for an oversized message, `422` for invalid input, `429` when rate-limited, and `503` when at capacity or not ready.

## Knowledge-base authoring

The `knowledge/` directory is the source of truth for curated professional facts.

Supported formats:

- `.md` — recommended for structured profiles and project descriptions
- `.txt`
- `.pdf` — text extraction is page-based; scanned PDFs without an OCR text layer may yield little or no content

Files and directories whose names start with `_` or `.` are skipped by ingestion. The `_templates/` directory contains examples and is intentionally ignored.

Recommended workflow:

1. Edit `knowledge/cv.md` and add supporting Markdown files for experience, projects, FAQs, or other facts.
2. Keep headings meaningful (`#`, `##`, `###`) so chunks retain section context.
3. Avoid adding confidential or unnecessary personal information.
4. Rebuild the index:

   ```powershell
   uv run python -m twin.ingest --github
   ```

5. Ask a few known questions and verify the answer against the source material.

Markdown is recommended because headings and section paths improve retrieval context. Keep an editable Word/PDF CV separately if needed; use `cv.md` as the version indexed by the digital twin.

## GitHub README indexing vs live tools

These capabilities complement each other:

**Indexed README documents**
- Add repository README content to the RAG corpus during ingestion.
- Useful for broad questions such as “What computer-vision projects has this person worked on?”
- Configured with `GITHUB_README_REPOS` and `GITHUB_README_LIMIT`.
- Stale until ingestion is run again.

**Live GitHub tools**
- Query the configured owner's public repositories, repository details, recent activity, and profile statistics at request time.
- Read-only and constrained to the configured `GITHUB_USERNAME`; the model cannot select an arbitrary GitHub owner.
- Responses are cached for `GITHUB_CACHE_TTL_S` seconds.
- A GitHub token is optional but helps with API rate limits. If a configured token is rejected, the integration attempts unauthenticated access.

## Testing and evaluation

Run the unit tests:

```powershell
uv run pytest
```

The current tests cover input guardrails and text filtering. They do not replace end-to-end deployment tests.

Run model and dependency smoke checks:

```powershell
uv run python scripts/check_models.py
```

This checks tool-call behavior for the configured chat models, dense embeddings, reranking, Qdrant access, and the sparse model. It does not independently test structured-output compliance or the full HTTP streaming path.

Run the LLM-in-the-loop evaluation suite:

```powershell
uv run python evals/run_evals.py
```

The cases in `evals/golden.yaml` check expected tool routing, grounded answers, refusal of undocumented information, and basic prompt-injection resistance. These evaluations call the configured model endpoints and may consume API quota or fail if a model endpoint is unavailable.

For an end-to-end check, start the API, confirm `/readyz`, then send a question whose answer is explicitly documented in `knowledge/cv.md`.

## Security, privacy, and operational notes

- **Secrets:** Keep `.env` private. Never put real keys in the README, `.env.example`, ZIP files, source control, or issue reports. Rotate any credential that has been exposed.
- **Knowledge base:** Treat indexed files as content that may be returned to users. Include only information you are comfortable exposing through the assistant.
- **Prompt injection:** Input filtering and system instructions are defense-in-depth, not a guarantee against every attack. Retrieved documents and tool output are treated as untrusted data by the prompt.
- **GitHub access:** Live tools are read-only and restricted to the configured account, but public repository information may still be returned to visitors.
- **MCP:** Keep the tool allowlist narrow and avoid enabling write-capable tools in a public-facing deployment.
- **CORS:** Set `CORS_ORIGINS` to your actual frontend origins. Avoid permissive wildcard settings for a public application.
- **Proxy headers:** Keep `TRUST_PROXY=false` unless the API is behind a proxy you control.
- **Single worker:** The current session store and rate limiter are in memory. Run a single Uvicorn worker and a single application replica unless you move session/checkpoint and rate-limit state to a shared backend.
- **Request limits:** Tune rate limits, daily caps, timeout, and tool-call count to protect hosted model quotas and keep response times predictable.
- **Observability:** Optional LangSmith tracing can help debug agent execution. Review the provider's privacy and retention settings before tracing sensitive data.
- **Model availability:** Model IDs, tool-calling support, free-tier access, latency, and quotas can change. Re-run smoke tests after changing models or dependencies.

## Known limitations

- Conversation checkpoints, session locks, and rate-limit counters are in memory and do not persist across process restarts.
- Horizontal scaling requires a shared session/checkpoint backend and shared rate limiting.
- A successful `/healthz` response is only a process-level health check; `/readyz` checks whether the knowledge collection is populated.
- Local embedding and reranking models must be downloaded and loaded; first-run latency can be higher than subsequent requests.
- The ingestion command rebuilds the configured Qdrant collection rather than applying a partial incremental update.
- GitHub README content is a snapshot until re-indexed.
- `check_models.py` tests tool-call behavior but does not guarantee every model's structured-output behavior or full agent quality.
- PDF ingestion extracts text; OCR is not included.
- The included guardrail tests and golden evaluations are useful regression checks, not a formal security audit or guarantee of factual correctness.

## Troubleshooting

**The API starts, but `/` returns 404**  
This project does not define a root route. Use `/healthz`, `/readyz`, `/v1/config`, or `POST /v1/chat`.

**`/readyz` returns 503**  
The knowledge collection is empty or unavailable. Check the Qdrant connection and run `uv run python -m twin.ingest --github`.

**A model check fails or times out**  
Check the NVIDIA API key, configured model ID, endpoint availability, rate limits, and current provider status. Test primary and fallback models individually and re-run `scripts/check_models.py`. A timeout is not necessarily an invalid key.

**GitHub information is missing**  
Set `GITHUB_USERNAME`; optionally configure `GITHUB_TOKEN`. Re-run ingestion to refresh indexed README documents. Live tools and indexed README content are separate paths.

**Local Qdrant reports a storage lock**  
In embedded mode, stop the API before running ingestion. Docker and Qdrant Cloud use server mode and avoid the embedded local-file lock pattern.

**PowerShell JSON requests fail**  
Build the body with `ConvertTo-Json -Compress` and use `curl.exe`, as shown in the quick-start section. PowerShell's `curl` alias may behave differently from the standalone curl executable.

**The first startup is slow**  
FastEmbed may need to download local model artifacts. Check `.cache/fastembed/` and ensure the runtime user can write to the directory.

## Roadmap ideas

Potential future improvements include:

- Add end-to-end API tests covering SSE event sequences and error paths.
- Add a dedicated structured-output validation check for supported model endpoints.
- Add persistent sessions and shared rate limiting for multi-replica deployment.
- Add authentication or deployment-level access controls if the endpoint should not be public.
- Add automated scheduled README re-ingestion and evaluation.
- Add OCR for scanned PDFs if they become a required source format.
- Add latency and retrieval-quality benchmarks across primary and fallback models.

These are possible next steps, not claims that the features are already implemented.

## License

No license file or explicit license declaration was verified in the reviewed project archive. Add a `LICENSE` file and update this section if you intend to publish the project with a specific open-source license.

## Acknowledgements

Built with FastAPI, LangChain, LangGraph, NVIDIA-hosted language models, Qdrant, FastEmbed, Hugging Face model artifacts, and the GitHub REST API.
