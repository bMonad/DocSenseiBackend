from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
  model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

  # App Profile
  app_name: str
  app_description: str
  app_version: str

  # LLM (Groq)
  groq_api_key: str = ""
  groq_model: str = "qwen/qwen3.8-27b"
  groq_reasoning_effort: str = "none"

  # Embeddings / vector store
  embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
  data_dir: Path = Path("data")
  chunk_size: int = 1000
  chunk_overlap: int = 150
  retrieval_k: int = 5

  # Limits
  max_upload_mb: int = 25
  max_pages: int = 400

  # Context budgeting (characters, ~4 chars per token).
  # Sections that fit are sent whole; larger ones use retrieval / map-reduce.
  direct_context_chars: int = 12_000
  summary_direct_chars: int = 20_000
  summary_max_parts: int = 8
  history_turns: int = 6

  cors_origins: list[str] = ["http://localhost:5173"]

@lru_cache
def get_settings() -> Settings:
  return Settings()