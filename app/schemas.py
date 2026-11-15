from __future__ import annotations
from typing import Literal, TypedDict
from pydantic import BaseModel, Field


class SectionNode(TypedDict):
  id: str
  title: str
  level: int
  page_start: int
  page_end: int
  children: list["SectionNode"]


class DocumentOut(BaseModel):
  id: str
  title: str
  page_count: int | None
  source_type: str = "pdf"
  unit_count: int = 0
  unit_label: str = "pages"
  outline: list[SectionNode]


class SummaryOut(BaseModel):
  summary: str
  cached: bool


class ChatTurn(BaseModel):
  role: Literal["user", "assistant"]
  content: str = Field(max_length=8000)


class ChatRequest(BaseModel):
  question: str = Field(min_length=1, max_length=2000)
  history: list[ChatTurn] = Field(default_factory=list, max_length=20)