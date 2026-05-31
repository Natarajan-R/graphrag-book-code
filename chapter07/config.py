"""
Configuration Module for GraphRAG System
Handles all configuration settings and environment management
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import json


@dataclass
class Neo4jConfig:
    """Neo4j database configuration"""
    uri: str
    username: str
    password: str
    database: str = "neo4j"
    
    def validate(self) -> bool:
        """Validate Neo4j configuration"""
        return all([self.uri, self.username, self.password])


@dataclass
class LLMConfig:
    """LLM configuration for Ollama"""
    model: str
    embedding_model:str
    temperature: float = 0.0
    base_url: str = "http://localhost:11434"
    
    def validate(self) -> bool:
        """Validate LLM configuration"""
        return bool(self.model)


@dataclass
class ProcessingConfig:
    """Document processing configuration"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_file_size_mb: int = 50
    allowed_extensions: tuple = (".pdf", ".txt", ".md")
    batch_size: int = 10
    
    # Graph extraction parameters
    allowed_nodes: Optional[list] = None
    allowed_relationships: Optional[list] = None
    node_properties: bool = True
    relationship_properties: bool = True


@dataclass
class PathConfig:
    """Path configuration for the system"""
    upload_dir: Path
    processed_dir: Path
    logs_dir: Path
    cache_dir: Path
    
    def create_directories(self):
        """Create all necessary directories"""
        for path in [self.upload_dir, self.processed_dir, self.logs_dir, self.cache_dir]:
            path.mkdir(parents=True, exist_ok=True)


class Config:
    """Main configuration class that manages all settings"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration
        
        Args:
            config_file: Optional path to JSON config file
        """
        # Set base directory
        self.base_dir = Path(__file__).parent
        
        # Load from file if provided, otherwise use defaults
        if config_file and Path(config_file).exists():
            self._load_from_file(config_file)
        else:
            self._load_defaults()
    
    def _load_defaults(self):
        """Load default configuration"""
        # Neo4j Configuration
        self.neo4j = Neo4jConfig(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "your_password_here"),
            database=os.getenv("NEO4J_DATABASE", "neo4j")
        )
        
        # LLM Configuration
        self.llm = LLMConfig(
            model=os.getenv("LLM_MODEL", "qwen16k"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )
        
        # Processing Configuration
        self.processing = ProcessingConfig(
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            max_file_size_mb=int(os.getenv("MAX_FILE_SIZE_MB", "10"))
        )
        
        # Path Configuration
        self.paths = PathConfig(
            upload_dir=self.base_dir / "data" / "uploads",
            processed_dir=self.base_dir / "data" / "processed",
            logs_dir=self.base_dir / "logs",
            cache_dir=self.base_dir / "cache"
        )
        
        # Create directories
        self.paths.create_directories()
    
    def _load_from_file(self, config_file: str):
        """Load configuration from JSON file"""
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        # Parse Neo4j config
        neo4j_data = config_data.get("neo4j", {})
        self.neo4j = Neo4jConfig(**neo4j_data)
        
        # Parse LLM config
        llm_data = config_data.get("llm", {})
        self.llm = LLMConfig(**llm_data)
        
        # Parse processing config
        processing_data = config_data.get("processing", {})
        self.processing = ProcessingConfig(**processing_data)
        
        # Parse paths
        paths_data = config_data.get("paths", {})
        self.paths = PathConfig(
            upload_dir=Path(paths_data.get("upload_dir", "data/uploads")),
            processed_dir=Path(paths_data.get("processed_dir", "data/processed")),
            logs_dir=Path(paths_data.get("logs_dir", "logs")),
            cache_dir=Path(paths_data.get("cache_dir", "cache"))
        )
        self.paths.create_directories()
    
    def save_to_file(self, config_file: str):
        """Save current configuration to JSON file"""
        config_data = {
            "neo4j": {
                "uri": self.neo4j.uri,
                "username": self.neo4j.username,
                "password": self.neo4j.password,
                "database": self.neo4j.database
            },
            "llm": {
                "model": self.llm.model,
                "embedding_model":self.llm.embedding_model,
                "temperature": self.llm.temperature,
                "base_url": self.llm.base_url
            },
            "processing": {
                "chunk_size": self.processing.chunk_size,
                "chunk_overlap": self.processing.chunk_overlap,
                "max_file_size_mb": self.processing.max_file_size_mb
            },
            "paths": {
                "upload_dir": str(self.paths.upload_dir),
                "processed_dir": str(self.paths.processed_dir),
                "logs_dir": str(self.paths.logs_dir),
                "cache_dir": str(self.paths.cache_dir)
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=4)
    
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate all configuration settings
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        if not self.neo4j.validate():
            errors.append("Neo4j configuration is incomplete")
        
        if not self.llm.validate():
            errors.append("LLM configuration is incomplete")
        
        return len(errors) == 0, errors
    
    def __repr__(self):
        """String representation of configuration"""
        return f"""
GraphRAG Configuration:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Neo4j:
  URI: {self.neo4j.uri}
  Database: {self.neo4j.database}
  
LLM:
  Model: {self.llm.model}
  Temperature: {self.llm.temperature}
  
Processing:
  Chunk Size: {self.processing.chunk_size}
  Chunk Overlap: {self.processing.chunk_overlap}
  Max File Size: {self.processing.max_file_size_mb}MB
  
Paths:
  Uploads: {self.paths.upload_dir}
  Processed: {self.paths.processed_dir}
  Logs: {self.paths.logs_dir}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


# Global config instance
_config_instance = None


def get_config(config_file: Optional[str] = None) -> Config:
    """
    Get the global configuration instance (Singleton pattern)
    
    Args:
        config_file: Optional path to config file (only used on first call)
    
    Returns:
        Config instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_file)
    return _config_instance


def reload_config(config_file: Optional[str] = None):
    """Force reload of configuration"""
    global _config_instance
    _config_instance = Config(config_file)
    return _config_instance
