import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from hashlib import md5
import json

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    text: str
    metadata: Dict[str, any] = field(default_factory=dict)
    page_number: Optional[int] = None
    chunk_index: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "metadata": self.metadata,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index
        }


class IndustrialDocumentLoader:
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        use_cache: bool = True,
        cache_dir: Optional[Path] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_cache = use_cache
        self.cache_dir = cache_dir or Path("./cache_documents")
        
        if use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.do_code_enrichment = True
        
        format_options = {
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
        
        self.converter = DocumentConverter(format_options=format_options)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
            length_function=len,
        )
        
        logger.info(f"DocumentLoader initialized: chunk_size={chunk_size}, overlap={chunk_overlap}, OCR=True")