from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.config import get_settings
from app.rate_limit import ApiRateLimiter
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


settings = get_settings()


app = FastAPI(
  title = settings.app_name,
  version = settings.app_version,
  lifespan = lifespan
)

rate_limiter = ApiRateLimiter(
  per_minute=settings.api_rate_limit_per_minute,
  per_day=settings.api_rate_limit_per_day,
  global_per_minute=settings.api_global_rate_limit_per_minute,
  global_per_day=settings.api_global_rate_limit_per_day,
)


app.add_middleware(
  CORSMiddleware, 
  allow_origins=settings.cors_origins,
  allow_methods=["GET", "POST"],
  allow_headers=["Content-Type"]
)


@app.exception_handler(DocumentNotFound)
async def document_not_found(_: Request, _exc: DocumentNotFound) -> JSONResponse:
  del _exc
  return JSONResponse(
    {
    "detail": "Document not found!"
    },
    status_code=status.HTTP_404_NOT_FOUND
  )


@app.middleware("http")
async def limit_expensive_requests(request: Request, call_next):
  path = request.url.path
  method = request.method
  is_upload = method == "POST" and path == "/api/documents"
  is_chat = (
    method == "POST" 
    and path.startswith("/api/documents")
    and path.endswith("/chat")
  )
  is_summary = (
    method == "GET" 
    and path.startswith("/api/documents")
    and path.endswith("/summary")
  )
  if is_upload or is_chat or is_summary:
    client = request.client.host if request.client else "unknown"
    retry_after = rate_limiter.retry_after(client)
    if retry_after is not None:
      return JSONResponse(
        { "detail": "Too many requests. Please wait before trying again."},
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        headers={"Retry-After": str(retry_after)}
      )
  return await call_next(request)


app.include_router(router)