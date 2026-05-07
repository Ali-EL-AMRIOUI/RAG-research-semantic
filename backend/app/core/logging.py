import logging
import sys
from typing import Optional
from datetime import datetime
from pathlib import Path

SIMPLE_FORMAT = "%(levelname)s - %(message)s"
STANDARD_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DETAILED_FORMAT = "%(asctime)s - %(name)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s"
JSON_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: str = "INFO",
    format_type: str = "standard",
    log_file: Optional[str] = None,
    console: bool = True
) -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    formats = {
        "simple": SIMPLE_FORMAT,
        "standard": STANDARD_FORMAT,
        "detailed": DETAILED_FORMAT,
        "json": JSON_FORMAT
    }
    log_format = formats.get(format_type, STANDARD_FORMAT)
    
    formatter = logging.Formatter(log_format, datefmt=DATE_FORMAT)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    root_logger.handlers.clear()
    
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)
    
    logging.info(f"Logging configured: level={level}, format={format_type}")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


class RequestLogger:
    
    def __init__(self, name: str = "api"):
        self.logger = get_logger(name)
    
    def log_request(self, method: str, path: str, client_ip: str = None, **kwargs):
        extra = f" [IP: {client_ip}]" if client_ip else ""
        self.logger.info(f"Request {method} {path}{extra} - {kwargs if kwargs else ''}")
    
    def log_response(self, method: str, path: str, status_code: int, duration_ms: float):
        status_emoji = "OK" if status_code < 400 else "ERROR" if status_code < 500 else "CRITICAL"
        self.logger.info(f"{status_emoji} {method} {path} -> {status_code} ({duration_ms:.2f}ms)")
    
    def log_error(self, method: str, path: str, error: Exception, **kwargs):
        self.logger.error(f"ERROR {method} {path}: {str(error)} - {kwargs}")


class PerformanceLogger:
    
    def __init__(self, name: str = "performance"):
        self.logger = get_logger(name)
    
    def log_embedding_time(self, count: int, duration_ms: float):
        self.logger.info(f"Embedding: {count} texts in {duration_ms:.2f}ms ({duration_ms/count:.2f}ms/text)")
    
    def log_search_time(self, query: str, results: int, duration_ms: float):
        preview = query[:50] + "..." if len(query) > 50 else query
        self.logger.info(f"Search: '{preview}' -> {results} results in {duration_ms:.2f}ms")
    
    def log_rerank_time(self, count: int, duration_ms: float):
        self.logger.info(f"Reranking: {count} documents in {duration_ms:.2f}ms")
    
    def log_llm_time(self, prompt_tokens: int, response_tokens: int, duration_ms: float):
        total_tokens = prompt_tokens + response_tokens
        self.logger.info(f"LLM: {total_tokens} tokens ({prompt_tokens} prompt, {response_tokens} response) in {duration_ms:.2f}ms")


if not logging.getLogger().handlers:
    setup_logging(level="INFO", format_type="standard")