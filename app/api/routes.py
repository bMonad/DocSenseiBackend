from __future__ import annotations

import json
import logging
from pathlib import Path
from datetime import datetime, timezone

import pymupdf
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from app.services.vectorstore import is_embeddings_loaded

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


@router.get("/health")
def health() -> dict:
  return JSONResponse(
    content={
      "status": "Ok" if is_embeddings_loaded() else "Loading Embedding",
      "timestamp": datetime.now(timezone.utc).isoformat()
    },
    status_code=status.HTTP_200_OK if is_embeddings_loaded() else status.HTTP_503_SERVICE_UNAVAILABLE
  )
