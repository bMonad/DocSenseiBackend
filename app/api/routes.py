from __future__ import annotations

import json
import logging
from pathlib import Path
from datetime import datetime, timezone

import pymupdf
from fastapi import APIRouter, HTTPException, File, UploadFile, status, Query, Response
from fastapi.responses import JSONResponse, FileResponse

from app.config import get_settings
from app.domain import Section, subtree, subtree_text, to_outline
from app.schemas import DocumentOut, SectionNode, SummaryOut
from app.services.vectorstore import is_embeddings_loaded
from app.services.repository import get_repository
from app.services.document_parser import SUPPORTED_TYPES, parse_document
from app.services.vectorstore import ensure_indexed
from app.services.llm import describe__llm_error
from app.services.summarizer import summarize_section

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

MIN_SUMMARY_CHARS = 80

# ------ Helper methods ------

def _document_out(doc_id: str) -> DocumentOut:
  parsed = get_repository().load(doc_id)
  
  return DocumentOut(
    id = doc_id,
    title = parsed.title,
    page_count = parsed.page_count if parsed.source_type == "pdf" else None,
    source_type = parsed.source_type,
    unit_count = parsed.unit_count,
    unit_label = parsed.unit_label,
    outline = to_outline(parsed.sections)
  )


def _scope(doc_id: str, section_id: str) -> tuple[list[Section], list[Section]]:
  """All sections, the requested section + descendants. 404 if either is unknown."""
  all_sections = get_repository().load(doc_id).sections
  try:
    return all_sections, subtree(all_sections, section_id)
  except KeyError:
    raise HTTPException(status.HTTP_404_NOT_FOUND, "Section not found!") from None


# ------ Health Check Endpoint ------

@router.get("/health")
def health() -> JSONResponse:
  return JSONResponse(
    content={
      "status": "Ok" if is_embeddings_loaded() else "Loading Embedding",
      "timestamp": datetime.now(timezone.utc).isoformat()
    },
    status_code=status.HTTP_200_OK if is_embeddings_loaded() else status.HTTP_503_SERVICE_UNAVAILABLE
  )


# ------ Document Upload Endpoint ------

@router.post("/documents", response_model=DocumentOut)
def upload_document(file: UploadFile = File(...)) -> DocumentOut:
  settings = get_settings()
  repo = get_repository()

  limit = settings.max_upload_mb * 1024 * 1024
  data = file.file.read(limit + 1)

  if len(data) > limit:
    raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, f"File is larger than {settings.max_upload_mb} MB.")
  filename = file.filename or "Untitled"
  extension = Path(filename).suffix.lower()
  source_type = SUPPORTED_TYPES.get(extension)
  if source_type is None:
    supported = ", ".join(SUPPORTED_TYPES)
    raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"Unsupported file type. Supported extensions: {supported}.")

  doc_id = repo.make_id(data)
  if not repo.exists(doc_id):
    try: 
      parsed = parse_document(data, filename, max_pages = settings.max_pages)
    except ValueError as exc:
      raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    repo.save(doc_id, parsed, data, extension)

  ensure_indexed(doc_id, repo.load(doc_id).sections)
  return _document_out(doc_id)


# ----- Retrieve Document with doc_id ------ 

@router.get("/documents/{doc_id}", response_model=DocumentOut)
def get_document(doc_id: str) -> DocumentOut:
  return _document_out(doc_id)


# ------ Render the file ------ 

@router.get("/document/{doc_id}/file")
def document_file(doc_id: str) -> FileResponse:
  repo = get_repository()
  if repo.load(doc_id).source_type  != "pdf":
    raise HTTPException(status.HTTP_404_NOT_FOUND, "PDF file is not available for this document.")
  return FileResponse(
    repo.pdf_path(doc_id),
    media_type="application/pdf",
    filename="document.pdf",
    content_disposition_type="inline",
    headers={
      "Accept-Ranges": "bytes",
      "Cache-Control": "private, max-age=86400, immutable",
      "X-Content-Type-Options": "nosniff",
    },
  )


# ------ Section Content Endpoint ------

@router.get("/documents/{doc_id}/sections/{section_id}/content")
def section_content(doc_id: str, section_id: str) -> dict[str, str]:
    _, scope = _scope(doc_id, section_id)
    return {"content": subtree_text(scope)}


# ------ Section Summary Endpoint ------

@router.get("/documents/{doc_id}/sections/{section_id}/summary", response_model=SummaryOut)
def section_summary(doc_id: str, section_id: str, refresh: bool = False) -> SummaryOut:
  repo = get_repository()
  _, scope = _scope(doc_id, section_id)

  if not refresh and (cached := repo.get_summary(doc_id, section_id)):
    return SummaryOut(summary = cached, cached = True)

  text = subtree_text(scope)
  if len(text) < MIN_SUMMARY_CHARS:
    return SummaryOut(summary = "_This section has to little text to summarize._", cached = False)

  try:
    summary = summarize_section(scope[0].title, text)
  except Exception as exc:
    log.exception("Summary failed for %s/%s", doc_id, section_id)
    status_code, detail = describe__llm_error(exc)
    raise HTTPException(status_code=status_code, detail=detail) from exc

  repo.save_summary(doc_id, section_id, summary)
  return SummaryOut(summary=summary, cached=False)