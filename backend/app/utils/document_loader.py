import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from hashlib import md5
import json
from io import BytesIO

from unstructured.partition.auto import partition
from PIL import Image
import pytesseract

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_index: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "metadata": self.metadata,
            "chunk_index": self.chunk_index
        }


class IndustrialDocumentLoader:
    
    SUPPORTED_EXTENSIONS = {
        '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.rtf',
        '.html', '.htm', '.txt', '.md', '.jpg', '.jpeg', '.png', '.tiff'
    }
    
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        use_cache: bool = True,
        cache_dir: Optional[Path] = None,
        ocr_enabled: bool = True
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_cache = use_cache
        self.ocr_enabled = ocr_enabled
        self.cache_dir = cache_dir or Path("./cache_documents")
        
        if use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DocumentLoader initialized: chunk_size={chunk_size}, ocr={ocr_enabled}")
    
    def _get_file_hash(self, file_path: Path) -> str:
        with open(file_path, "rb") as f:
            return md5(f.read()).hexdigest()
    
    def _get_cache_path(self, file_path: Path) -> Path:
        return self.cache_dir / f"{self._get_file_hash(file_path)}.json"
    
    def _save_to_cache(self, file_path: Path, chunks: List[DocumentChunk]) -> None:
        if not self.use_cache:
            return
        data = {
            "source": str(file_path),
            "chunks": [chunk.to_dict() for chunk in chunks]
        }
        with open(self._get_cache_path(file_path), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _load_from_cache(self, file_path: Path) -> Optional[List[DocumentChunk]]:
        if not self.use_cache:
            return None
        cache_path = self._get_cache_path(file_path)
        if not cache_path.exists():
            return None
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [
                DocumentChunk(
                    text=chunk["text"],
                    metadata=chunk.get("metadata", {}),
                    chunk_index=chunk.get("chunk_index", i)
                )
                for i, chunk in enumerate(data["chunks"])
            ]
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            return None
    
    def _extract_text_from_image(self, image_path: Path) -> str:
        try:
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img, lang='fra+eng')
            logger.info(f"OCR extracted {len(text)} characters from {image_path.name}")
            return text
        except Exception as e:
            logger.error(f"OCR error on {image_path.name}: {e}")
            return ""
    
    def load_document(self, file_path: Path) -> str:
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        ext = file_path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported format: {ext}")
        
        logger.info(f"Loading: {file_path.name}")
        
        if self.use_cache:
            cached = self._load_from_cache(file_path)
            if cached:
                return "\n\n".join([c.text for c in cached])
        
        if ext in self.IMAGE_EXTENSIONS:
            content = self._extract_text_from_image(file_path)
            if not content.strip():
                raise ValueError(f"No text detected in image {file_path.name}")
            
            if self.use_cache:
                temp_chunks = self._chunk_text(content, {"source": file_path.name})
                self._save_to_cache(file_path, temp_chunks)
            
            return content
        
        try:
            elements = partition(
                filename=str(file_path),
                strategy="hi_res" if self.ocr_enabled else "fast",
                languages=["fra", "eng"]
            )
            
            text_parts = []
            for el in elements:
                if hasattr(el, 'text') and el.text:
                    text_parts.append(el.text)
            
            content = "\n\n".join(text_parts)
            
            if not content.strip():
                raise ValueError(f"No text extracted from {file_path.name}")
            
            logger.info(f"{len(content)} characters extracted")
            
            if self.use_cache:
                temp_chunks = self._chunk_text(content, {"source": file_path.name})
                self._save_to_cache(file_path, temp_chunks)
            
            return content
            
        except Exception as e:
            logger.error(f"Extraction error for {file_path.name}: {e}")
            raise e
    
    def _chunk_text(self, text: str, metadata: Dict = None) -> List[DocumentChunk]:
        if not text:
            return []
        
        metadata = metadata or {}
        
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) <= self.chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(DocumentChunk(
                        text=current_chunk.strip(),
                        metadata=metadata.copy(),
                        chunk_index=len(chunks)
                    ))
                current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(DocumentChunk(
                text=current_chunk.strip(),
                metadata=metadata.copy(),
                chunk_index=len(chunks)
            ))
        
        logger.info(f"{len(chunks)} chunks created")
        return chunks
    
    def load_and_chunk(self, file_path: Path, metadata: Dict = None) -> List[DocumentChunk]:
        metadata = metadata or {}
        metadata["source"] = file_path.name
        metadata["extension"] = file_path.suffix.lower()
        
        if self.use_cache:
            cached = self._load_from_cache(file_path)
            if cached:
                for chunk in cached:
                    chunk.metadata.update(metadata)
                return cached
        
        content = self.load_document(file_path)
        chunks = self._chunk_text(content, metadata)
        
        if self.use_cache:
            self._save_to_cache(file_path, chunks)
        
        return chunks
    
    def load_from_bytes(self, file_bytes: BytesIO, filename: str, metadata: Dict = None) -> List[DocumentChunk]:
        import tempfile
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
            tmp.write(file_bytes.getvalue())
            tmp_path = tmp.name
        
        try:
            return self.load_and_chunk(Path(tmp_path), metadata)
        finally:
            Path(tmp_path).unlink(missing_ok=True)