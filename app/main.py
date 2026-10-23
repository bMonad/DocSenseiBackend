from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.config import get_settings
from app.services.vectorstore import get_embeddings
from app.services.repository import DocumentNotFound


logging.basicConfig(
  level = logging.INFO, 
  format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger('docsensei')


@asynccontextmanager
async def lifespan(_: FastAPI):
  # Load (and on first run, download) the embedding model
  try:
    await asyncio.to_thread(get_embeddings)
  except Exception:
    log.exception("Could not preload the embedding model; it will load on first use")
  yield


app = FastAPI(
  title = get_settings().app_name,
  version = get_settings().app_version,
  lifespan = lifespan
)


app.add_middleware(
  CORSMiddleware, 
  allow_origins=get_settings().cors_origins,
  allow_methods=["GET", "POST"],
  allow_headers=["Content-Type"]
)


@app.exception_handler(DocumentNotFound)
async def document_not_found(_: Request, __: DocumentNotFound) -> JSONResponse:
  return JSONResponse(
    {
    "detail": "Document not found!"
    },
    status_code=status.HTTP_404_NOT_FOUND
  )


app.include_router(router)