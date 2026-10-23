from __future__ import annotations

import json
import logging
from pathlib import Path
from datetime import datetime, timezone

import pymupdf
from fastapi import APIRouter, HTTPException, File, UploadFile, status
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.schemas import DocumentOut
from app.services.vectorstore import is_embeddings_loaded
from app.services.repository import get_repository
from app.services.pdf_parser import parse_pdf

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


""" Health Check Endpoint """
@router.get("/health")
def health() -> JSONResponse:
  return JSONResponse(
    content={
      "status": "Ok" if is_embeddings_loaded() else "Loading Embedding",
      "timestamp": datetime.now(timezone.utc).isoformat()
    },
    status_code=status.HTTP_200_OK if is_embeddings_loaded() else status.HTTP_503_SERVICE_UNAVAILABLE
  )


""" Document Upload Endpoint """
@router.post("/documents", response_model=DocumentOut)
def upload_document(file: UploadFile = File(...)) -> DocumentOut:
  settings = get_settings()
  repo = get_repository()

  limit = settings.max_upload_mb * 1024 * 1024
  data = file.file.read(limit+1)

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
