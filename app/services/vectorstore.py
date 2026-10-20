from __future__ import annotations

import logging
import threading
from functools import lru_cache

from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings

from app.config import get_settings

log = logging.getLogger(__name__)


class LocalFastEmbeddings(Embeddings):
  """LangChain-compatible embeddings backed by FastEmbed (ONNX, no PyTorch)."""

  def __init__(self, model_name: str, **kwargs) -> None:
    self._model = TextEmbedding(model_name=model_name, **kwargs)
    self._lock = threading.Lock()

  def embed_documents(self, texts: list[str]) -> list[list[float]]:
    with self._lock:
      return [v.tolist() for v in self._model.embed(texts)]
  
  def embed_query(self, texts: str) -> list[float]:
    with self._lock:
      return next(iter(self._model.query_embed(texts))).tolist()

@lru_cache
def get_embeddings() -> LocalFastEmbeddings:
  settings = get_settings()
  log.info("Loading embedding model %s", settings.embedding_model)
  return LocalFastEmbeddings(model_name = settings.embedding_model)


def is_embeddings_loaded() -> bool:
  return get_embeddings.cache_info().currsize > 0