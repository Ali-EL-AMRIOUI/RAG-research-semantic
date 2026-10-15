
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    title_path: list[str] = field(default_factory=list)
    chunk_index: int = 0

    def get_full_title(self) -> str:
        return " > ".join(self.title_path) if self.title_path else "Document"

    def to_document(self) -> Document:
        return Document(
            page_content=self.text,
            metadata={
                "title_path": self.get_full_title(),
                "chunk_index": self.chunk_index,
                **self.metadata,
            },
        )


class IndustrialChunker:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        preserve_tables: bool = True,
        preserve_code_blocks: bool = True,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.preserve_tables = preserve_tables
        self.preserve_code_blocks = preserve_code_blocks

        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap // 2,
            separators=["\n", ". ", " ", ""],
            length_function=len,
        )

        logger.info(
            "Chunker initialized: size=%s, overlap=%s",
            chunk_size,
            chunk_overlap,
        )

    def _extract_titles(
        self,
        text: str,
        current_path: list[str] | None = None,
    ) -> tuple[str, list[str]]:
        title_path = list(current_path) if current_path else []
        cleaned_lines = []

        for line in text.split("\n"):
            if line.startswith("# "):
                title_path = [line[2:].strip()]
            elif line.startswith("## ") and title_path:
                title_path = [title_path[0], line[3:].strip()]
            elif line.startswith("### ") and len(title_path) >= 2:
                title_path = title_path[:2] + [line[4:].strip()]
            elif line.startswith("#### ") and len(title_path) >= 3:
                title_path = title_path[:3] + [line[5:].strip()]
            else:
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines), title_path

    def _protect_blocks(self, text: str) -> tuple[str, dict[str, str]]:
        placeholders = {}
        protected_text = text

        if self.preserve_code_blocks:
            code_pattern = r"```[\s\S]*?```"
            code_blocks = re.findall(code_pattern, protected_text)
            for i, block in enumerate(code_blocks):
                placeholder = f"__CODE_BLOCK_{i}__"
                placeholders[placeholder] = block
                protected_text = protected_text.replace(block, placeholder, 1)

        if self.preserve_tables:
            table_pattern = (
                r"(\|[\s\S]+?\|\n[\|\-\s]+\n(?:\|[\s\S]+?\|\n?)+)"
            )
            tables = re.findall(table_pattern, protected_text)
            for i, table in enumerate(tables):
                placeholder = f"__TABLE_{i}__"
                placeholders[placeholder] = table
                protected_text = protected_text.replace(table, placeholder, 1)

        return protected_text, placeholders

    def _restore_blocks(
        self,
        text: str,
        placeholders: dict[str, str],
    ) -> str:
        for placeholder, original in placeholders.items():
            text = text.replace(placeholder, original)
        return text

    def _get_overlap_prefix(self, finished_chunk_text: str) -> str:
        if self.chunk_overlap <= 0 or not finished_chunk_text:
            return ""

        overlap = finished_chunk_text[-self.chunk_overlap:]
        space_idx = overlap.find(" ")
        if space_idx != -1:
            overlap = overlap[space_idx + 1:]
        return overlap.strip()

    def split_markdown(
        self,
        text: str,
        metadata: dict | None = None,
    ) -> list[Chunk]:
        if not text or not text.strip():
            return []

        metadata = metadata or {}
        protected_text, placeholders = self._protect_blocks(text)
        paragraphs = protected_text.split("\n\n")

        chunks = []
        current_chunk = ""
        current_titles: list[str] = []

        def flush_chunk() -> str:
            nonlocal current_chunk

            finished_text = current_chunk.strip()
            if not finished_text:
                current_chunk = ""
                return ""

            chunks.append(
                Chunk(
                    text=self._restore_blocks(finished_text, placeholders),
                    metadata=metadata.copy(),
                    title_path=current_titles.copy(),
                    chunk_index=len(chunks),
                )
            )
            current_chunk = ""
            return self._get_overlap_prefix(finished_text)

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_to_add, new_titles = self._extract_titles(
                para,
                current_path=current_titles,
            )

            # Flush the previous section before changing its title metadata.
            if new_titles != current_titles:
                if current_chunk.strip():
                    flush_chunk()
                current_titles = new_titles

            if not para_to_add.strip():
                continue

            if len(para_to_add) > self.chunk_size:
                if current_chunk.strip():
                    flush_chunk()

                for sub in self.fallback_splitter.split_text(para_to_add):
                    chunks.append(
                        Chunk(
                            text=self._restore_blocks(sub, placeholders),
                            metadata=metadata.copy(),
                            title_path=current_titles.copy(),
                            chunk_index=len(chunks),
                        )
                    )
                current_chunk = ""

            elif len(current_chunk) + len(para_to_add) <= self.chunk_size:
                current_chunk += para_to_add + "\n\n"

            else:
                overlap_prefix = flush_chunk()
                current_chunk = (
                    (overlap_prefix + "\n\n") if overlap_prefix else ""
                ) + para_to_add + "\n\n"

        if current_chunk.strip():
            chunks.append(
                Chunk(
                    text=self._restore_blocks(
                        current_chunk.strip(),
                        placeholders,
                    ),
                    metadata=metadata.copy(),
                    title_path=current_titles.copy(),
                    chunk_index=len(chunks),
                )
            )

        if not chunks:
            fallback_chunks = self.fallback_splitter.split_text(text)
            for i, chunk_text in enumerate(fallback_chunks):
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        metadata=metadata.copy(),
                        title_path=[],
                        chunk_index=i,
                    )
                )

        logger.info(
            "%s chunks created (overlap=%s)",
            len(chunks),
            self.chunk_overlap,
        )
        return chunks

    def split_text(
        self,
        text: str,
        source: str = "unknown",
    ) -> list[Chunk]:
        chunks = self.fallback_splitter.split_text(text)
        return [
            Chunk(
                text=chunk,
                metadata={"source": source},
                title_path=[],
                chunk_index=i,
            )
            for i, chunk in enumerate(chunks)
        ]

    def get_stats(self, chunks: list[Chunk]) -> dict:
        if not chunks:
            return {
                "total_chunks": 0,
                "avg_size": 0,
                "min_size": 0,
                "max_size": 0,
            }

        sizes = [len(chunk.text) for chunk in chunks]
        return {
            "total_chunks": len(chunks),
            "avg_size": sum(sizes) // len(sizes),
            "min_size": min(sizes),
            "max_size": max(sizes),
            "chunks_with_titles": sum(
                1 for chunk in chunks if chunk.title_path
            ),
        }
