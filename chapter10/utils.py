"""
Utilities Module for GraphRAG System
Provides logging, file handling, and helper functions
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List
import json
import hashlib
from functools import wraps
import time


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        
        return super().format(record)


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    console_output: bool = True
) -> logging.Logger:
    """
    Setup a logger with file and console handlers
    
    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level
        console_output: Whether to output to console
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = ColoredFormatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    return logger


def timing_decorator(func):
    """Decorator to measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger = logging.getLogger(func.__module__)
        logger.info(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        
        return result
    return wrapper


class FileValidator:
    """Validate files before processing"""
    
    def __init__(self, allowed_extensions: tuple, max_size_mb: int):
        """
        Initialize validator
        
        Args:
            allowed_extensions: Tuple of allowed file extensions
            max_size_mb: Maximum file size in MB
        """
        self.allowed_extensions = allowed_extensions
        self.max_size_bytes = max_size_mb * 1024 * 1024
    
    def validate(self, file_path: Path) -> tuple[bool, Optional[str]]:
        """
        Validate a file
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if file exists
        if not file_path.exists():
            return False, f"File does not exist: {file_path}"
        
        # Check if it's a file
        if not file_path.is_file():
            return False, f"Not a file: {file_path}"
        
        # Check extension
        if file_path.suffix.lower() not in self.allowed_extensions:
            return False, f"Invalid extension. Allowed: {self.allowed_extensions}"
        
        # Check size
        file_size = file_path.stat().st_size
        if file_size > self.max_size_bytes:
            return False, f"File too large: {file_size / 1024 / 1024:.2f}MB (max: {self.max_size_bytes / 1024 / 1024}MB)"
        
        return True, None


def compute_file_hash(file_path: Path) -> str:
    """
    Compute SHA256 hash of a file
    
    Args:
        file_path: Path to file
    
    Returns:
        Hex digest of file hash
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


class DocumentMetadata:
    """Manage document metadata"""
    
    def __init__(self, metadata_file: Path):
        """
        Initialize metadata manager
        
        Args:
            metadata_file: Path to metadata JSON file
        """
        self.metadata_file = metadata_file
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> dict:
        """Load metadata from file"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_metadata(self):
        """Save metadata to file"""
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=4)
    
    def add_document(self, file_path: Path, status: str = "uploaded"):
        """Add or update document metadata"""
        file_hash = compute_file_hash(file_path)
        
        self.metadata[file_hash] = {
            "filename": file_path.name,
            "path": str(file_path),
            "size_bytes": file_path.stat().st_size,
            "hash": file_hash,
            "status": status,
            "uploaded_at": datetime.now().isoformat(),
            "processed_at": None,
            "chunks_count": 0,
            "entities_count": 0,
            "relationships_count": 0
        }
        
        self.save_metadata()
        return file_hash
    
    def update_processing_status(
        self,
        file_hash: str,
        status: str,
        chunks_count: int = 0,
        entities_count: int = 0,
        relationships_count: int = 0
    ):
        """Update processing status"""
        if file_hash in self.metadata:
            self.metadata[file_hash].update({
                "status": status,
                "processed_at": datetime.now().isoformat(),
                "chunks_count": chunks_count,
                "entities_count": entities_count,
                "relationships_count": relationships_count
            })
            self.save_metadata()
    
    def get_document(self, file_hash: str) -> Optional[dict]:
        """Get document metadata by hash"""
        return self.metadata.get(file_hash)
    
    def list_documents(self, status: Optional[str] = None) -> List[dict]:
        """List all documents, optionally filtered by status"""
        if status:
            return [
                doc for doc in self.metadata.values()
                if doc["status"] == status
            ]
        return list(self.metadata.values())
    
    def is_processed(self, file_hash: str) -> bool:
        """Check if document has been processed"""
        doc = self.get_document(file_hash)
        return doc and doc["status"] == "processed"


def format_bytes(bytes_size: int) -> str:
    """Format bytes to human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"


def print_banner(title: str, width: int = 60):
    """Print a formatted banner"""
    border = "═" * width
    print(f"\n╔{border}╗")
    print(f"║ {title.center(width - 2)} ║")
    print(f"╚{border}╝\n")


def print_section(title: str):
    """Print a section header"""
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def print_success(message: str):
    """Print success message"""
    print(f"\033[32m✓ {message}\033[0m")


def print_error(message: str):
    """Print error message"""
    print(f"\033[31m✗ {message}\033[0m")


def print_warning(message: str):
    """Print warning message"""
    print(f"\033[33m⚠ {message}\033[0m")


def print_info(message: str):
    """Print info message"""
    print(f"\033[36mℹ {message}\033[0m")


class ProgressTracker:
    """Track progress of operations"""
    
    def __init__(self, total: int, description: str = "Processing"):
        """
        Initialize progress tracker
        
        Args:
            total: Total number of items
            description: Description of operation
        """
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
    
    def update(self, increment: int = 1):
        """Update progress"""
        self.current += increment
        self._print_progress()
    
    def _print_progress(self):
        """Print progress bar"""
        percentage = (self.current / self.total) * 100 if self.total > 0 else 0
        filled = int(50 * self.current / self.total) if self.total > 0 else 0
        bar = '█' * filled + '░' * (50 - filled)
        
        elapsed = time.time() - self.start_time
        eta = (elapsed / self.current * (self.total - self.current)) if self.current > 0 else 0
        
        print(f'\r{self.description}: |{bar}| {percentage:.1f}% ({self.current}/{self.total}) ETA: {eta:.1f}s', end='')
        
        if self.current >= self.total:
            print()  # New line when complete
    
    def complete(self):
        """Mark as complete"""
        self.current = self.total
        self._print_progress()
        print_success(f"{self.description} completed in {time.time() - self.start_time:.2f}s")


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry function on failure"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries} attempts failed")
                        raise
        return wrapper
    return decorator
