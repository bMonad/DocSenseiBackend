# DocSensei — AI Document Summarizer & Q&A

Upload a PDF, DOCX, PPTX, TXT, Markdown, or HTML file → it is split into topics → click a topic to see its content, an **AI summary**, and a **chat scoped to that topic**. PDFs retain rendered page previews; other formats show extracted text.

```
┌──────────────┬──────────────────────┬───────────────────────────┐
│  Topics      │  Page preview        │  Summary of the topic     │
│  (sidebar)   │  (rendered from PDF) ├───────────────────────────┤
│  ▸ Ch 1      │                      │  Chat about this topic    │
│    1.1 …     │                      │  [ ask anything… ] [Send] │
└──────────────┴──────────────────────┴───────────────────────────┘
```

**Stack:** FastAPI · LangChain · Groq (`qwen/qwen3.8-27b`, Gemini fallback) · local FastEmbed `all-MiniLM-L6-v2` · Chroma · PyMuPDF

## Run it

**Backend** (Python 3.10+)
```bash
cd DcoSenseibackend
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                      # set GROQ_API_KEY; GEMINI_API_KEY is optional
uvicorn app.main:app --reload --port 8000
```

## How it works

| Step | What happens | Where |
|---|---|---|
| 1. Parse | Format adapters extract source content into one section model. PDFs use bookmarks/font heuristics/page windows; DOCX uses heading styles; PPTX follows presentation slide order; Markdown/HTML use headings; text is grouped into bounded parts. | `services/document_parser.py`, `services/pdf_parser.py` |
| 2. Section tree | Each section stores only its *own* text once; a chapter = its subtree. Page ranges cover descendants. | `domain.py` |
| 3. Index | Own-text is chunked (1000/150) and embedded into Chroma with `doc_id` + `section_id` metadata. | `services/vectorstore.py` |
| 4. Preview | PDFs stream through the browser's native viewer with HTTP byte-range support; other formats expose extracted section text. | `api/routes.py` |
| 5. Summary | Generated on click, cached on disk. Fits in context → one call. Too long → map-reduce, capped at 8 map calls. | `services/summarizer.py` |
| 6. Chat | Section small enough → full text is the context (no retrieval misses). Otherwise MMR retrieval filtered to the section's subtree. Answer streams over SSE. | `services/chat.py` |

### Decisions worth knowing
- **Hybrid context** (full text vs. RAG) beats always-RAG: small sections get perfect recall, big ones stay within the free-tier token budget. Tune `DIRECT_CONTEXT_CHARS` in `config.py`.
- **Qwen 3.8 thinking is off** (`reasoning_effort="none"`, `reasoning_format="hidden"`). Summaries and grounded Q&A don't need it, and it saves latency and free-tier tokens. Set `GROQ_REASONING_EFFORT=default` to turn it on.
- **Document id = SHA-256 of the file** (with source type included for non-PDFs): re-uploading the same file is instant and reuses embeddings and cached summaries.
- **Prompt-injection hygiene**: PDF text is delimited and declared untrusted in every prompt; the UI renders Markdown without raw HTML.
- **Stateless chat API**: the browser sends the last turns; the server keeps no session state.
- **Free-tier friendliness**: 350 ms debounce before summarizing (skimming the sidebar costs nothing), client-side summary cache, no automatic Groq retries, sequential bounded map-reduce, a per-process 24 RPM Groq request pacer, 1,024-token response caps, and a 4,000-character chat-history budget.
- **Provider failover**: if Groq returns a rate-limit error, requests fall back to Gemini when `GEMINI_API_KEY` is configured.
- **Abuse protection**: uploads, chats, and summary requests are limited to 10/minute and 100/day per client IP, plus a 24/minute and 100/day app-wide cap. Override with `API_RATE_LIMIT_*` and `API_GLOBAL_RATE_LIMIT_*`. The limiter is in-memory and per process; use a shared limiter store before deploying multiple workers or instances.
- Upload hardening: magic-byte check, size/page limits, hex-only document ids (no path traversal).

### API
| Method | Path | Notes |
|---|---|---|
| POST | `/api/documents` | multipart `file` → `{id, title, source_type, unit_count, unit_label, outline[]}` |
| GET | `/api/documents/{id}` | same payload (used to restore the last session) |
| GET | `/api/documents/{id}/file` | original PDF with HTTP byte-range support for native browser viewing |
| GET | `/api/documents/{id}/pages/{n}/image?scale=1.6` | PNG |
| GET | `/api/documents/{id}/sections/{sid}/content` | extracted section/subtree text |
| GET | `/api/documents/{id}/sections/{sid}/summary` | `?refresh=true` to regenerate |
| POST | `/api/documents/{id}/sections/{sid}/chat` | body `{question, history[]}` → SSE `data: {"token": …}` … `[DONE]` |

## Known limits → good next steps
- **Live webpage URLs** are not fetched yet; HTML files can be uploaded. Add URL ingestion as a separate source adapter with strict public-HTTPS-only destination checks, redirect/DNS revalidation, response size/time limits, and content-type validation. Never let a user-controlled URL reach internal services or cloud metadata endpoints.
- **Cloud Run storage is ephemeral**: local document files and Chroma data can disappear when an instance is replaced and are not shared across instances. Before scaling or relying on saved documents, put source/metadata in object storage or a database and use a hosted vector service. Keep Chroma for local development; evaluate Qdrant Cloud's current free allowance for production, with backups/export because free-tier terms can change.
- **Keep ingestion synchronous for small files**, as it is now. If larger files or concurrency make requests slow, move parse/embed into a bounded background job and expose ingestion status before adding more infrastructure.
- **Scanned PDFs** are rejected (no text layer). Add OCR (e.g. `ocrmypdf`) in `parse_pdf`.
- **Multi-column layouts** can interleave columns; PyMuPDF's `get_text("blocks")` + column clustering would fix it.
- Heading detection by font is a heuristic. For messy PDFs, ask the LLM to clean up titles, or add a "merge/split topic" UI.
- Upload runs synchronously. For big files move parse + index to a background job and poll progress.
- Add source citations (page numbers) to answers: store `page` in chunk metadata, return it as an SSE `sources` event.
- Add authentication and per-user document ownership before serving multiple users.

## Troubleshooting
- *"GROQ_API_KEY is not set"* → fill `backend/.env`, restart uvicorn.
- *First upload is slow* → the embedding model (~90 MB) downloads once; the server pre-loads it on startup.
- *Rate-limit message* → free tier limits; wait a few seconds. Lower `summary_max_parts` if long chapters keep hitting it.
- *Model name rejected* → check `https://api.groq.com/openai/v1/models` and set `GROQ_MODEL`.


---
## Credits

<div align="center">

**Designed and developed by -** **[bMonad (Shivansh Singh)](https://github.com/bMonad)**
  
### Let's Build Something Meaningful 🚀

</div>

If you find this project useful or inspiring, consider giving the repo a star. You're welcome to use it as a reference for your own learning, but please don't copy it as-is. Make it your own.

---

<div align="center">

**Shivansh Singh | bMonad** · Software Engineer · AI Engineer · React.js · Python · LangChain · Generative AI 

Thank you for visiting this project!
