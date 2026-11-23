"""Format-specific extraction into the shared section model."""
from __future__ import annotations

import re
import posixpath
import zipfile
from html.parser import HTMLParser
from io import BytesIO
from pathlib import PurePath
from xml.etree import ElementTree

from app.domain import ParsedDocument, Section
from app.services.pdf_parser import ParseError, parse_pdf

SUPPORTED_TYPES = {
    ".pdf": "pdf",
    ".txt": "text",
    ".md": "markdown",
    ".markdown": "markdown",
    ".html": "html",
    ".htm": "html",
    ".docx": "docx",
    ".pptx": "pptx",
}

_WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_DRAWING_NS = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
_PRESENTATION_NS = "{http://schemas.openxmlformats.org/presentationml/2006/main}"
_RELATIONSHIP_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_PACKAGE_RELATIONSHIP_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"
_MAX_ARCHIVE_CONTENT_BYTES = 100 * 1024 * 1024
_MAX_SECTIONS = 300
_MAX_SECTION_CHARS = 6000


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[tuple[int | None, str]] = []
        self._tag: str | None = None
        self._parts: list[str] = []
        self._ignored = 0

    def handle_starttag(self, tag: str, _attrs: list[tuple[str, str | None]]) -> None:
        del _attrs
        if tag in {"script", "style", "noscript"}:
            self._ignored += 1
        elif self._ignored == 0 and (tag in {"p", "li", "blockquote"} or re.fullmatch(r"h[1-6]", tag)):
            self._flush()
            self._tag = tag

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._ignored:
            self._ignored -= 1
        elif tag == self._tag:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._ignored == 0:
            self._parts.append(data)

    def close(self) -> None:
        super().close()
        self._flush()

    def _flush(self) -> None:
        text = re.sub(r"\s+", " ", " ".join(self._parts)).strip()
        if text:
            level = int(self._tag[1]) if self._tag and self._tag.startswith("h") else None
            self.blocks.append((level, text))
        self._parts.clear()
        self._tag = None


def parse_document(data: bytes, filename: str, max_pages: int = 300) -> ParsedDocument:
    extension = PurePath(filename).suffix.lower()
    source_type = SUPPORTED_TYPES.get(extension)
    if source_type is None:
        supported = ", ".join(SUPPORTED_TYPES)
        raise ParseError(f"Unsupported file type. Supported extensions: {supported}.")

    title = PurePath(filename).stem.strip() or "Untitled"
    if source_type == "pdf":
        if not data.startswith(b"%PDF-"):
            raise ParseError("This file is not a valid PDF.")
        parsed = parse_pdf(data, fallback_title=title, max_pages=max_pages)
        parsed.source_type = "pdf"
        parsed.unit_count = parsed.page_count
        parsed.unit_label = "pages"
        return parsed
    if source_type == "docx":
        return _build_document(title, source_type, _docx_blocks(data), unit_label="sections")
    if source_type == "pptx":
        return _parse_pptx(data, title, max_pages)
    if source_type == "html":
        parser = _TextExtractor()
        try:
            parser.feed(data.decode("utf-8-sig"))
            parser.close()
        except (UnicodeDecodeError, ValueError) as exc:
            raise ParseError("This HTML file could not be decoded as UTF-8.") from exc
        return _build_document(title, source_type, parser.blocks, unit_label="sections")

    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ParseError("Text files must use UTF-8 encoding.") from exc

    if source_type == "markdown":
        blocks: list[tuple[int | None, str]] = []
        for line in text.splitlines():
            match = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$", line)
            if match:
                blocks.append((len(match.group(1)), match.group(2).strip()))
            elif line.strip():
                blocks.append((None, line.strip()))
    else:
        blocks = [(None, part.strip()) for part in re.split(r"\n\s*\n", text) if part.strip()]
    return _build_document(title, source_type, blocks, unit_label="sections")


def _archive(data: bytes, required: str) -> zipfile.ZipFile:
    try:
        archive = zipfile.ZipFile(BytesIO(data))
        infos = archive.infolist()
        if sum(item.file_size for item in infos) > _MAX_ARCHIVE_CONTENT_BYTES:
            archive.close()
            raise ParseError("The Office document expands beyond the supported size limit.")
        if required not in archive.namelist():
            archive.close()
            raise ParseError("This Office file is missing required document data.")
        return archive
    except zipfile.BadZipFile as exc:
        raise ParseError("This Office file is damaged or has an invalid format.") from exc


def _xml(archive: zipfile.ZipFile, path: str) -> ElementTree.Element:
    try:
        return ElementTree.fromstring(archive.read(path))
    except (KeyError, OSError, RuntimeError, zipfile.BadZipFile, ElementTree.ParseError) as exc:
        raise ParseError("This Office file contains invalid document data.") from exc


def _docx_blocks(data: bytes) -> list[tuple[int | None, str]]:
    with _archive(data, "word/document.xml") as archive:
        root = _xml(archive, "word/document.xml")
    body = root.find(f"{_WORD_NS}body")
    if body is None:
        raise ParseError("This Word document has no readable body.")

    blocks: list[tuple[int | None, str]] = []
    for paragraph in body.iter(f"{_WORD_NS}p"):
        text = "".join(
            ("\t" if item.tag == f"{_WORD_NS}tab" else "\n" if item.tag == f"{_WORD_NS}br" else item.text or "")
            for item in paragraph.iter()
            if item.tag in {f"{_WORD_NS}t", f"{_WORD_NS}tab", f"{_WORD_NS}br"}
        ).strip()
        if not text:
            continue
        style = paragraph.find(f"{_WORD_NS}pPr/{_WORD_NS}pStyle")
        style_name = style.get(f"{_WORD_NS}val", "") if style is not None else ""
        heading = re.fullmatch(r"Heading([1-6])", style_name, re.IGNORECASE)
        heading_level = int(heading.group(1)) if heading is not None else None
        blocks.append((heading_level, text))
    return blocks


def _parse_pptx(data: bytes, title: str, max_pages: int) -> ParsedDocument:
    with _archive(data, "ppt/presentation.xml") as archive:
        presentation = _xml(archive, "ppt/presentation.xml")
        slide_ids = presentation.findall(f".//{_PRESENTATION_NS}sldId")
        relationships_path = "ppt/_rels/presentation.xml.rels"
        if slide_ids and relationships_path in archive.namelist():
            relationships = _xml(archive, relationships_path)
            targets = {
                item.get("Id"): item.get("Target")
                for item in relationships.findall(f"{_PACKAGE_RELATIONSHIP_NS}Relationship")
            }
            slide_paths: list[str] = []
            for slide in slide_ids:
                slide_id = slide.get(f"{_RELATIONSHIP_NS}id")
                if slide_id is None:
                    continue
                target = targets.get(slide_id)
                if target is None:
                    continue
                path = posixpath.normpath(posixpath.join("ppt", target))
                if path.startswith("ppt/slides/") and re.fullmatch(r"ppt/slides/slide\d+\.xml", path):
                    slide_paths.append(path)
        else:
            def slide_number(name: str) -> int:
                match = re.search(r"slide(\d+)", name)
                return int(match.group(1)) if match is not None else 0

            slide_paths = sorted(
                (name for name in archive.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)),
                key=slide_number,
            )
        if len(slide_paths) > max_pages:
            raise ParseError(f"This presentation has {len(slide_paths)} slides; the limit is {max_pages}.")
        if not slide_paths or (slide_ids and len(slide_paths) != len(slide_ids)):
            raise ParseError("This PowerPoint file has no readable slides.")
        sections = []
        for number, path in enumerate(slide_paths, 1):
            root = _xml(archive, path)
            paragraphs = [
                "".join(item.text or "" for item in paragraph.iter(f"{_DRAWING_NS}t")).strip()
                for paragraph in root.findall(f".//{_DRAWING_NS}p")
            ]
            paragraphs = [paragraph for paragraph in paragraphs if paragraph]
            title_text = ""
            for shape in root.findall(f".//{_PRESENTATION_NS}sp"):
                shape_text = [
                    "".join(item.text or "" for item in paragraph.iter(f"{_DRAWING_NS}t")).strip()
                    for paragraph in shape.findall(f".//{_DRAWING_NS}p")
                ]
                shape_text = [text for text in shape_text if text]
                if shape_text:
                    placeholder = shape.find(f"{_PRESENTATION_NS}nvSpPr/{_PRESENTATION_NS}nvPr/{_PRESENTATION_NS}ph")
                    if not title_text and placeholder is not None and placeholder.get("type") in {"title", "ctrTitle"}:
                        title_text = shape_text[0]
            sections.append(Section(
                id=f"s{number - 1:04d}",
                title=title_text or f"Slide {number}",
                level=1,
                parent_id=None,
                page_start=number,
                page_end=number,
                own_text="\n\n".join(paragraphs),
            ))
    if not any(section.own_text for section in sections):
        raise ParseError("No readable text was found in this presentation.")
    return ParsedDocument(title, len(sections), sections, "pptx", len(sections), "slides")


def _build_document(
    title: str, source_type: str, blocks: list[tuple[int | None, str]], unit_label: str
) -> ParsedDocument:
    if not any(text.strip() for _, text in blocks):
        raise ParseError("No readable text was found in this document.")

    headings_exist = any(level is not None for level, _ in blocks)
    prepared: list[tuple[int | None, str]] = []
    if headings_exist:
        prepared = blocks
    else:
        paragraphs: list[str] = []
        current_size = 0
        part_number = 1
        for _, text in blocks:
            if paragraphs and current_size + len(text) > _MAX_SECTION_CHARS and part_number < _MAX_SECTIONS:
                prepared.append((1, f"Part {part_number}"))
                prepared.extend((None, paragraph) for paragraph in paragraphs)
                paragraphs, current_size = [], 0
                part_number += 1
            paragraphs.append(text)
            current_size += len(text)
        if paragraphs:
            prepared.append((1, f"Part {part_number}"))
            prepared.extend((None, paragraph) for paragraph in paragraphs)

    sections: list[Section] = []
    stack: list[Section] = []
    current: Section | None = None

    def add_section(level: int, heading: str) -> Section:
        level = min(max(level, 1), stack[-1].level + 1 if stack else 1, 3)
        while stack and stack[-1].level >= level:
            stack.pop()
        section = Section(
            id=f"s{len(sections):04d}", title=heading, level=level,
            parent_id=stack[-1].id if stack else None,
            page_start=len(sections) + 1, page_end=len(sections) + 1,
        )
        sections.append(section)
        stack.append(section)
        return section

    for level, text in prepared:
        if level is not None:
            current = add_section(min(max(level, 1), 3), text)
        else:
            if current is None:
                current = add_section(1, "Introduction")
            current.own_text = f"{current.own_text}\n\n{text}".strip()
    if len(sections) > _MAX_SECTIONS:
        raise ParseError(f"This document has too many sections; the limit is {_MAX_SECTIONS}.")
    if not any(section.own_text for section in sections):
        raise ParseError("No readable text was found in this document.")
    return ParsedDocument(title, 0, sections, source_type, len(sections), unit_label)
