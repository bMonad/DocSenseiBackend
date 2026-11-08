from __future__ import annotations

import json
import logging
from pathlib import Path
from datetime import datetime, timezone

import pymupdf
from fastapi import APIRouter, HTTPException, File, UploadFile, status, Query, Response
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.domain import Section, subtree, subtree_text
from app.schemas import DocumentOut, SectionNode, SummaryOut
from app.services.vectorstore import is_embeddings_loaded
from app.services.repository import get_repository
from app.services.pdf_parser import parse_pdf
from app.services.vectorstore import ensure_indexed
from app.services.llm import describe__llm_error
from app.services.summarizer import summarize_section

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

MIN_SUMMARY_CHARS = 80

# ------ Helper methods ------

def _document_out(doc_id: str) -> DocumentOut:
  parsed = get_repository().load(doc_id)
  nodes = {
    section.id: SectionNode(
      id=section.id,
      title=section.title,
      level=section.level,
      page_start=section.page_start,
      page_end=section.page_end,
    )
    for section in parsed.sections
  }
  outline: list[SectionNode] = []
  for section in parsed.sections:
    node = nodes[section.id]
    parent = nodes.get(section.parent_id) if section.parent_id else None
    if parent:
      parent.children.append(node)
    else:
      outline.append(node)

  return DocumentOut(
    id = doc_id,
    title = parsed.title,
    page_count = parsed.page_count,
    outline = outline
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
  if not data.startswith(b"%PDF-"):
    raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Only PDF files are supported.")

  doc_id = repo.make_id(data)
  if not repo.exists(doc_id):
    try: 
      parsed = parse_pdf(data, fallback_title=Path(file.filename or "Untitled").stem, max_pages = settings.max_pages)
    except ValueError as exc:
      raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    repo.save(doc_id, parsed, data)

  ensure_indexed(doc_id, repo.load(doc_id).sections)
  return _document_out(doc_id)


# ----- Retrieve Document with doc_id ------ 

@router.get("/documents/{doc_id}", response_model=DocumentOut)
def get_document(doc_id: str) -> DocumentOut:
  return _document_out(doc_id)


# ------ Retrieve Image of page ------ 

@router.get("/document/{doc_id}/pages/{page}/image")
def page_image(doc_id: str, page: int, scale: float = Query(1.6, ge = 0.5, le = 3.0)) -> Response:
  path = get_repository().pdf_path(doc_id)
  with pymupdf.open(path) as pdf:
    if not 1 <= page <= pdf.page_count:
      raise HTTPException(status.HTTP_404_NOT_FOUND, "Page Not Found!")
    pixmap = pdf[page - 1].get_pixmap(matrix = pymupdf.Matrix(scale, scale), alpha = False)
    png = pixmap.tobytes("png")
  return Response(png, media_type = "image/png", headers = {
    "Cache-Control": "public, max-age=86400, immutable"
  })


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