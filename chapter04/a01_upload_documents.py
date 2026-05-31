"""
upload_documents.py
Stage 1: Document upload with validation (No Neo4j - just file handling)
"""

import os
import hashlib
import shutil
from pathlib import Path
from typing import Dict

# Configuration
DATA_DIR = Path(os.getenv('DATA_DIR', './data'))
UPLOAD_DIR = DATA_DIR / 'uploads'

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class DocumentUploader:
    """Handles document upload and validation - No database operations"""

    def calculate_hash(self, filepath: str) -> str:
        """Calculate SHA-256 hash of file"""
        sha256_hash = hashlib.sha256()

        with open(filepath, 'rb') as f:
            # Read in 4KB chunks
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

        return sha256_hash.hexdigest()

    def validate_file(self, filepath: str) -> Dict:
        """Validate file exists and get metadata"""
        path = Path(filepath)

        # Check existence
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # Check readability
        if not os.access(filepath, os.R_OK):
            raise PermissionError(f"Cannot read file: {filepath}")

        # Check file type
        allowed_extensions = {'.pdf', '.txt', '.md', '.docx'}
        if path.suffix.lower() not in allowed_extensions:
            raise ValueError(
                f"Unsupported file type: {path.suffix}. "
                f"Allowed: {allowed_extensions}"
            )

        # Get metadata
        file_stat = path.stat()

        metadata = {
            'filename': path.name,
            'filepath': str(path.absolute()),
            'extension': path.suffix.lower(),
            'size_bytes': file_stat.st_size,
            'size_mb': round(file_stat.st_size / (1024 * 1024), 2),
        }

        return metadata

    def check_duplicate(self, file_hash: str) -> bool:
        """Check if document already processed (by checking for processed.json)"""
        # Look for existing processed file
        for processed_file in Path('.').glob('*_processed.json'):
            import json
            try:
                with open(processed_file, 'r') as f:
                    data = json.load(f)
                    if data.get('file_hash') == file_hash:
                        return True, processed_file.name
            except:
                continue
        
        return False, None

    def copy_to_uploads(self, filepath: str, file_hash: str) -> str:
        """Copy file to uploads directory"""
        path = Path(filepath)
        
        # Create destination filename with hash prefix
        dest_filename = f"{file_hash[:8]}_{path.name}"
        dest_path = UPLOAD_DIR / dest_filename
        
        # Copy file if not already there
        if not dest_path.exists():
            shutil.copy2(filepath, dest_path)
        
        return str(dest_path)

    def upload_document(self, filepath: str, force: bool = False) -> Dict:
        """
        Upload and validate a document (file operations only, no database)

        Args:
            filepath: Path to document
            force: If True, reprocess even if duplicate

        Returns:
            Dict with upload status and file hash
        """
        print(f"\n{'='*70}")
        print(f"Document Upload & Validation")
        print(f"{'='*70}")
        print(f"File: {filepath}\n")

        # Step 1: Validate file
        print("[1/4] Validating file...", end=" ")
        try:
            metadata = self.validate_file(filepath)
            print("✓")
            print(f"      Size: {metadata['size_mb']} MB")
            print(f"      Type: {metadata['extension']}")
        except Exception as e:
            print(f"✗\n✗ Error: {e}")
            return {'status': 'error', 'message': str(e)}

        # Step 2: Calculate hash
        print("[2/4] Calculating hash...", end=" ")
        try:
            file_hash = self.calculate_hash(filepath)
            print("✓")
            print(f"      Hash: {file_hash[:16]}...")
        except Exception as e:
            print(f"✗\n✗ Error: {e}")
            return {'status': 'error', 'message': str(e)}

        # Step 3: Check for duplicates
        print("[3/4] Checking for duplicates...", end=" ")
        is_duplicate, existing_file = self.check_duplicate(file_hash)

        if is_duplicate and not force:
            print(f"✗")
            print(f"      Duplicate found: {existing_file}")
            print(f"\n⚠️  Document already processed. Use --force to reprocess.")
            return {
                'status': 'duplicate',
                'file_hash': file_hash,
                'existing_file': existing_file
            }
        elif is_duplicate and force:
            print("⚠️  (forcing reprocess)")
        else:
            print("✓")

        # Step 4: Copy to uploads directory
        print("[4/4] Copying to uploads directory...", end=" ")
        try:
            uploaded_path = self.copy_to_uploads(filepath, file_hash)
            print("✓")
            print(f"      Location: {uploaded_path}")
        except Exception as e:
            print(f"✗\n✗ Error: {e}")
            return {'status': 'error', 'message': str(e)}

        print(f"\n{'='*70}")
        print("✓ Upload complete! Ready for text extraction.")
        print(f"{'='*70}")
        print(f"\nNext step:")
        print(f"  python extract_text.py {filepath}")

        return {
            'status': 'success',
            'file_hash': file_hash,
            'uploaded_path': uploaded_path,
            'metadata': metadata
        }


# Command-line interface
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python upload_documents.py <filepath> [--force]")
        print("\nExample:")
        print("  python upload_documents.py research_paper.pdf")
        print("  python upload_documents.py research_paper.pdf --force")
        print("\nThis script only validates and copies the file.")
        print("No database operations are performed here.")
        sys.exit(1)

    filepath = sys.argv[1]
    force = '--force' in sys.argv

    uploader = DocumentUploader()
    result = uploader.upload_document(filepath, force=force)
    
    if result['status'] == 'success':
        print(f"\n✓ File uploaded successfully")
        print(f"  Hash: {result['file_hash']}")
        print(f"  Path: {result['uploaded_path']}")
    elif result['status'] == 'duplicate':
        print(f"\n⚠️  Duplicate detected")
        print(f"  Hash: {result['file_hash']}")
        print(f"  Existing: {result['existing_file']}")
    else:
        print(f"\n✗ Upload failed: {result.get('message', 'Unknown error')}")
        sys.exit(1)
