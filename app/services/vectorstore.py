from __future__ import annotations

import logging
import threading
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.domain import Section

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


@lru_cache
def get_vectorstore() -> Chroma:
  settings = get_settings()
  return Chroma(
    collection_name = "docsensei",
    embedding_function = get_embeddings(),
    persist_directory = str(settings.data_dir / "chroma")
  )


def _is_indexed(doc_id: str) -> bool:
  found = get_vectorstore().get(where = {"doc_id": doc_id}, limit = 1)
  return bool(found["ids"])


def ensure_indexed(doc_id: str, sections: list[Section]) -> None:
  if _is_indexed(doc_id):
    return

  settings = get_settings()
  splitter = RecursiveCharacterTextSplitter(
    chunk_size = settings.chunk_size,
    chunk_overlap = settings.chunk_overlap,
    separators = ["\n\n", "\n", ". ", " ", ""]
  )
  docs: list[Document] = []
  ids: list[str] = []
  for section in sections:
    if not section.own_text:
      continue
    for n, chunk in enumerate(splitter.split_text(section.own_text)):
      docs.append(
        Document(
          page_content = chunk,
          metadata = {"doc_id": doc_id, "section_id": section.id, "title": section.title}
        )
      )
      ids.append(f"{doc_id} : {section.id} : {n}")
  if docs:
    get_vectorstore().add_documents(docs, ids = ids)
  log.info("Indexed %d chunks for document %s", len(docs), doc_id)