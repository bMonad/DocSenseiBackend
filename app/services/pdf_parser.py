from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
import pymupdf

from app.domain import ParsedDocument, Section

log = logging.getLogger(__name__)

MAX_LEVEL = 3                   # deeper headings fold into their level-3 parent
BOLD_FLAG = 16                  # PyMuPDF span flag for bold
EDGE_BAND = 0.09                # top/bottom 9% of a page = header/footer zone
MIN_TOTAL_CHARS = 200           # below this we assume a scanned PDF
MIN_PREAMBLE_CHARS = 300        # text before the first heading becomes "Front matter"
FALLBACK_PAGES_PER_SECTION = 5    

NUMBERED = re.compile(r"^\d+(\.\d+)*[.)]?\s+\S")


class ParseError(ValueError):
  """Raised for PDFs we cannot turn into sections."""


@dataclass(slots=True)
class Line:
  page: int
  y: float
  text: str
  size: float
  bold: bool


def extract_lines(doc: pymupdf.Document) -> list[Line]:
  """Return lines in form of list from document."""
  lines: list[Line] = []
  for index in range(doc.page_count):
    page = doc[index]
    height = page.rect.height or 1.0
    for block in page.get_text("dict", sort="True")["blocks"]:
      if block.get("type") != 0:
        continue
      for raw in block["lines"]:
        spans = [s for s in raw["spans"] if s["text"].strip()]
        if not spans:
          continue
        text = re.sub(r"\s+", " ", "".join(s["text"] for s in spans)).strip()
        dominant = max(spans, key=lambda s: len(s["text"]))
        bold = bool(dominant["flags"] & BOLD_FLAG) or "bold" in dominant["font"].lower()
        lines.append(Line(index + 1, raw["bbox"][1] / height, text, float(dominant["size"]), bold))
  return lines

def drop_page_furniture(lines: list[Line], page_count: int) -> list[Line]:
  """Remove running headers/footers and page numbers."""
  if page_count < 4:
    return lines

  def in_edge(line: Line) -> bool:
    return line.y < EDGE_BAND or line.y > 1-EDGE_BAND

  def key(text: str) -> str:
    return re.sub(r"\d+", "#", text.lower()).strip()

  pages_by_key: dict[str, set[int]] = defaultdict(set)
  for line in lines:
    if in_edge(line):
      pages_by_key[key(line.text)].add(line.page)

  threshold = max(3, int(page_count * 0.4))
  furniture = {k for k, pages in pages_by_key.items() if len(pages) >= threshold}
  return [ln for ln in lines if not (in_edge(ln) and key(ln.text) in furniture)]

def parse_pdf(data: bytes, fallback_title: str, max_pages: int = 400) -> ParsedDocument:
  """Return data into ParsedDocument format after parsing PDF."""
  try:
    doc = pymupdf.open(stream=data, filetype="pdf")
  except (RuntimeError, ValueError) as exc:
    raise ParseError("This file could not be read as a PDF.") from exc

  with doc:
    if doc.needs_pass:
      raise ParseError("This PDF is password-protected. Remove the password and try again.")
    if doc.page_count == 0:
      raise ParseError("This PDF has no page.")
    if doc.page_count > max_pages:
      raise ParseError(f"This PDF has {doc.page_count} pages; the limit is {max_pages}.")

    lines = drop_page_furniture(extract_lines(doc), doc.page_count)
    if sum(len(ln.text) for ln in lines) < MIN_TOTAL_CHARS:
      raise ParseError("No selectable text found. Scanned PDFs need OCR, which isn't supported yet.")

