"""
Document Upload Manager
Handles file uploads, validation, and organization
"""

import shutil
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from config import get_config
from utils import (
    setup_logger,
    FileValidator,
    DocumentMetadata,
    compute_file_hash,
    format_bytes,
    print_banner,
    print_section,
    print_success,
    print_error,
    print_warning,
    print_info
)


class DocumentUploader:
    """Manages document uploads and organization"""
    
    def __init__(self):
        """Initialize the document uploader"""
        self.config = get_config()
        
        # Setup logger
        log_file = self.config.paths.logs_dir / f"upload_{datetime.now().strftime('%Y%m%d')}.log"
        self.logger = setup_logger("DocumentUploader", log_file)
        
        # Setup validator
        self.validator = FileValidator(
            allowed_extensions=self.config.processing.allowed_extensions,
            max_size_mb=self.config.processing.max_file_size_mb
        )
        
        # Setup metadata manager
        metadata_file = self.config.paths.processed_dir / "metadata.json"
        self.metadata = DocumentMetadata(metadata_file)
        
        self.logger.info("DocumentUploader initialized")
    
    def upload_file(self, source_path: str, copy: bool = True) -> Optional[str]:
        """
        Upload a single file
        
        Args:
            source_path: Path to source file
            copy: Whether to copy file (True) or move it (False)
        
        Returns:
            File hash if successful, None otherwise
        """
        source = Path(source_path)
        
        # Validate file
        is_valid, error_msg = self.validator.validate(source)
        if not is_valid:
            self.logger.error(f"Validation failed for {source.name}: {error_msg}")
            print_error(f"Validation failed: {error_msg}")
            return None
        
        # Check if already uploaded
        file_hash = compute_file_hash(source)
        if self.metadata.is_processed(file_hash):
            self.logger.info(f"File {source.name} already processed (hash: {file_hash[:8]}...)")
            print_warning(f"File already processed: {source.name}")
            return file_hash
        
        # Copy/move file to upload directory
        destination = self.config.paths.upload_dir / source.name
        
        # Handle duplicate filenames
        counter = 1
        while destination.exists():
            stem = source.stem
            suffix = source.suffix
            destination = self.config.paths.upload_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        
        try:
            if copy:
                shutil.copy2(source, destination)
                self.logger.info(f"Copied {source.name} to {destination}")
            else:
                shutil.move(str(source), str(destination))
                self.logger.info(f"Moved {source.name} to {destination}")
            
            # Add to metadata
            self.metadata.add_document(destination, status="uploaded")
            
            file_size = format_bytes(destination.stat().st_size)
            print_success(f"Uploaded: {destination.name} ({file_size}) [Hash: {file_hash[:8]}...]")
            
            return file_hash
            
        except Exception as e:
            self.logger.error(f"Failed to upload {source.name}: {e}")
            print_error(f"Upload failed: {e}")
            return None
    
    def upload_directory(self, directory_path: str, recursive: bool = False) -> List[str]:
        """
        Upload all files from a directory
        
        Args:
            directory_path: Path to directory
            recursive: Whether to process subdirectories
        
        Returns:
            List of successfully uploaded file hashes
        """
        directory = Path(directory_path)
        
        if not directory.exists() or not directory.is_dir():
            self.logger.error(f"Invalid directory: {directory_path}")
            print_error("Invalid directory path")
            return []
        
        # Find all files
        if recursive:
            files = [f for f in directory.rglob("*") if f.is_file()]
        else:
            files = [f for f in directory.glob("*") if f.is_file()]
        
        # Filter by allowed extensions
        allowed_files = [
            f for f in files
            if f.suffix.lower() in self.config.processing.allowed_extensions
        ]
        
        print_section(f"Uploading {len(allowed_files)} files from {directory.name}")
        
        uploaded_hashes = []
        for file_path in allowed_files:
            file_hash = self.upload_file(str(file_path))
            if file_hash:
                uploaded_hashes.append(file_hash)
        
        print_info(f"Successfully uploaded {len(uploaded_hashes)}/{len(allowed_files)} files")
        
        return uploaded_hashes
    
    def list_uploaded_files(self, status: Optional[str] = None) -> List[dict]:
        """
        List uploaded files
        
        Args:
            status: Filter by status (uploaded, processing, processed, failed)
        
        Returns:
            List of document metadata
        """
        return self.metadata.list_documents(status)
    
    def get_file_info(self, file_hash: str) -> Optional[dict]:
        """Get information about a specific file"""
        return self.metadata.get_document(file_hash)
    
    def remove_file(self, file_hash: str) -> bool:
        """
        Remove a file from the upload directory
        
        Args:
            file_hash: Hash of file to remove
        
        Returns:
            True if successful
        """
        doc = self.metadata.get_document(file_hash)
        if not doc:
            self.logger.error(f"File not found: {file_hash}")
            print_error("File not found")
            return False
        
        file_path = Path(doc["path"])
        if file_path.exists():
            try:
                file_path.unlink()
                self.logger.info(f"Removed file: {file_path.name}")
                print_success(f"Removed: {file_path.name}")
                return True
            except Exception as e:
                self.logger.error(f"Failed to remove {file_path.name}: {e}")
                print_error(f"Removal failed: {e}")
                return False
        
        return False
    
    def display_statistics(self):
        """Display upload statistics"""
        all_docs = self.metadata.list_documents()
        
        if not all_docs:
            print_info("No documents uploaded yet")
            return
        
        # Count by status
        status_counts = {}
        total_size = 0
        
        for doc in all_docs:
            status = doc["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
            total_size += doc["size_bytes"]
        
        print_section("Upload Statistics")
        print(f"  Total Documents: {len(all_docs)}")
        print(f"  Total Size: {format_bytes(total_size)}")
        print(f"\n  Status Breakdown:")
        for status, count in status_counts.items():
            print(f"    {status.capitalize()}: {count}")


def main():
    """Main function for standalone usage"""
    import argparse
    
    print_banner("GraphRAG Document Uploader")
    
    parser = argparse.ArgumentParser(description="Upload documents for GraphRAG processing")
    parser.add_argument("path", help="Path to file or directory to upload")
    parser.add_argument("-r", "--recursive", action="store_true", help="Process subdirectories")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying")
    parser.add_argument("--list", action="store_true", help="List uploaded files")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    
    args = parser.parse_args()
    
    uploader = DocumentUploader()
    
    if args.list:
        print_section("Uploaded Documents")
        docs = uploader.list_uploaded_files()
        if docs:
            for i, doc in enumerate(docs, 1):
                print(f"\n{i}. {doc['filename']}")
                print(f"   Status: {doc['status']}")
                print(f"   Size: {format_bytes(doc['size_bytes'])}")
                print(f"   Hash: {doc['hash'][:16]}...")
                print(f"   Uploaded: {doc['uploaded_at']}")
        else:
            print_info("No documents found")
        return
    
    if args.stats:
        uploader.display_statistics()
        return
    
    # Upload file or directory
    path = Path(args.path)
    
    if path.is_file():
        uploader.upload_file(args.path, copy=not args.move)
    elif path.is_dir():
        uploader.upload_directory(args.path, recursive=args.recursive)
    else:
        print_error(f"Invalid path: {args.path}")
    
    # Show statistics
    print()
    uploader.display_statistics()


if __name__ == "__main__":
    main()
