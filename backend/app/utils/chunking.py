import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

@dataclass
class Chunk:
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    title_path: List[str] = field(default_factory=list)
    chunk_index: int = 0
    
    def get_full_title(self) -> str:
        return " > ".join(self.title_path) if self.title_path else "Document"
    
    def to_document(self) -> Document:
        return Document(
            page_content=self.text,
            metadata={
                "title_path": self.get_full_title(),
                "chunk_index": self.chunk_index,
                **self.metadata
            }
        )


class IndustrialChunker:
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        preserve_tables: bool = True,
        preserve_code_blocks: bool = True
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.preserve_tables = preserve_tables
        self.preserve_code_blocks = preserve_code_blocks
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap // 2,
            separators=["\n", ". ", " ", ""],
            length_function=len,
        )
        
        logger.info(f"Chunker initialized: size={chunk_size}, overlap={chunk_overlap}")
    
    def _extract_titles(self, text: str) -> Tuple[str, List[str]]:
        title_path = []
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            if line.startswith('# '):
                title_path = [line[2:].strip()]
            elif line.startswith('## ') and title_path:
                title_path = [title_path[0], line[3:].strip()]
            elif line.startswith('### ') and len(title_path) >= 2:
                title_path = title_path[:2] + [line[4:].strip()]
            elif line.startswith('#### ') and len(title_path) >= 3:
                title_path = title_path[:3] + [line[5:].strip()]
            else:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines), title_path
    
    def _protect_blocks(self, text: str) -> Tuple[str, Dict[str, str]]:
        placeholders = {}
        protected_text = text
        
        if self.preserve_code_blocks:
            code_pattern = r'```[\s\S]*?```'
            code_blocks = re.findall(code_pattern, protected_text)
            for i, block in enumerate(code_blocks):
                placeholder = f"__CODE_BLOCK_{i}__"
                placeholders[placeholder] = block
                protected_text = protected_text.replace(block, placeholder)
        
        if self.preserve_tables:
            table_pattern = r'(\|[\s\S]+?\|\n[\|\-\s]+\n(?:\|[\s\S]+?\|\n?)+)'
            tables = re.findall(table_pattern, protected_text)
            for i, table in enumerate(tables):
                placeholder = f"__TABLE_{i}__"
                placeholders[placeholder] = table
                protected_text = protected_text.replace(table, placeholder)
        
        return protected_text, placeholders
    
    def _restore_blocks(self, text: str, placeholders: Dict[str, str]) -> str:
        for placeholder, original in placeholders.items():
            text = text.replace(placeholder, original)
        return text
    
    def split_markdown(self, text: str, metadata: Dict = None) -> List[Chunk]:
        if not text or not text.strip():
            return []
        
        metadata = metadata or {}
        
        protected_text, placeholders = self._protect_blocks(text)
        
        text_without_titles, global_title_path = self._extract_titles(protected_text)
        
        chunks = []
        
        paragraphs = text_without_titles.split('\n\n')
        
        current_chunk = ""
        current_titles = global_title_path.copy()
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_without_titles, local_titles = self._extract_titles(para)
            if local_titles:
                current_titles = local_titles
            
            para_to_add = para_without_titles if local_titles else para
            
            if len(current_chunk) + len(para_to_add) <= self.chunk_size:
                current_chunk += para_to_add + "\n\n"
            else:
                if current_chunk:
                    chunks.append(Chunk(
                        text=self._restore_blocks(current_chunk.strip(), placeholders),
                        metadata=metadata.copy(),
                        title_path=current_titles.copy(),
                        chunk_index=len(chunks)
                    ))
                current_chunk = para_to_add + "\n\n"
        
        if current_chunk:
            chunks.append(Chunk(
                text=self._restore_blocks(current_chunk.strip(), placeholders),
                metadata=metadata.copy(),
                title_path=current_titles.copy(),
                chunk_index=len(chunks)
            ))
        
        if not chunks:
            fallback_chunks = self.fallback_splitter.split_text(text[:5000])
            for i, chunk_text in enumerate(fallback_chunks):
                chunks.append(Chunk(
                    text=chunk_text,
                    metadata=metadata.copy(),
                    title_path=[],
                    chunk_index=i
                ))
        
        logger.info(f"{len(chunks)} chunks created")
        return chunks
    
    def split_text(self, text: str, source: str = "unknown") -> List[Chunk]:
        chunks = self.fallback_splitter.split_text(text)
        return [
            Chunk(
                text=chunk,
                metadata={"source": source},
                title_path=[],
                chunk_index=i
            )
            for i, chunk in enumerate(chunks)
        ]
    
    def get_stats(self, chunks: List[Chunk]) -> Dict:
        if not chunks:
            return {"total_chunks": 0, "avg_size": 0, "min_size": 0, "max_size": 0}
        
        sizes = [len(chunk.text) for chunk in chunks]
        return {
            "total_chunks": len(chunks),
            "avg_size": sum(sizes) // len(sizes),
            "min_size": min(sizes),
            "max_size": max(sizes),
            "chunks_with_titles": sum(1 for c in chunks if c.title_path),
        }