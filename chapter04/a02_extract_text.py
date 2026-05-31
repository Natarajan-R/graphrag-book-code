"""
extract_text.py
Stage 2: Text extraction and chunking (creates processed.json with document + chunks)
"""

import os
import hashlib
import json
from pathlib import Path
from typing import Dict, List

# Try to import PDF library
try:
    from pypdf import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("Warning: pypdf not installed. Install with: pip install pypdf")


class TextExtractor:
    """Extract text from documents and create chunks"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize extractor
        
        Args:
            chunk_size: Number of words per chunk
            chunk_overlap: Number of words overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def calculate_hash(self, filepath: str) -> str:
        """Calculate SHA-256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def extract_text(self, filepath: str) -> str:
        """Extract text from document based on file type"""
        
        file_ext = Path(filepath).suffix.lower()
        
        if file_ext in ['.txt', '.md']:
            return self._extract_from_txt(filepath)
        elif file_ext == '.pdf':
            if not PDF_AVAILABLE:
                raise RuntimeError("pypdf not installed. Cannot process PDF files.")
            return self._extract_from_pdf(filepath)
        elif file_ext == '.docx':
            return self._extract_from_docx(filepath)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
    
    def _extract_from_txt(self, filepath: str) -> str:
        """Extract text from TXT/MD file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            with open(filepath, 'r', encoding='latin-1') as f:
                return f.read()
    
    def _extract_from_pdf(self, filepath: str) -> str:
        """Extract text from PDF file"""
        reader = PdfReader(filepath)
        text_parts = []
        
        for page in reader.pages:
            text = page.extract_text()
            if text.strip():
                text_parts.append(text)
        
        return "\n\n".join(text_parts)
    
    def _extract_from_docx(self, filepath: str) -> str:
        """Extract text from DOCX file"""
        try:
            from docx import Document
        except ImportError:
            raise RuntimeError("python-docx not installed. Install with: pip install python-docx")
        
        doc = Document(filepath)
        text_parts = []
        
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_parts.append(paragraph.text)
        
        return "\n\n".join(text_parts)
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text"""
        import re
        
        # Remove excessive whitespace
        lines = text.split('\n')
        cleaned_lines = [line.strip() for line in lines if line.strip()]
        
        # Join with single newline
        text = '\n'.join(cleaned_lines)
        
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        return text
    
    def create_chunks(self, text: str) -> List[Dict]:
        """Split text into overlapping chunks"""
        
        words = text.split()
        
        if len(words) <= self.chunk_size:
            return [{
                'chunk_id': 0,
                'text': text,
                'word_count': len(words),
                'start_word': 0,
                'end_word': len(words)
            }]
        
        chunks = []
        chunk_id = 0
        start_idx = 0
        
        while start_idx < len(words):
            end_idx = min(start_idx + self.chunk_size, len(words))
            chunk_words = words[start_idx:end_idx]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                'chunk_id': chunk_id,
                'text': chunk_text,
                'word_count': len(chunk_words),
                'start_word': start_idx,
                'end_word': end_idx
            })
            
            chunk_id += 1
            start_idx += self.chunk_size - self.chunk_overlap
            
            if start_idx >= len(words):
                break
        
        return chunks
    
    def process_document(self, filepath: str) -> Dict:
        """
        Complete processing pipeline
        Creates processed.json with document metadata AND chunks
        """
        
        path = Path(filepath)
        
        print(f"\n{'='*70}")
        print(f"Text Extraction & Chunking")
        print(f"{'='*70}")
        print(f"File: {filepath}\n")
        
        # Get file metadata
        file_stat = path.stat()
        
        # Calculate hash
        print("[1/5] Calculating file hash...", end=" ")
        file_hash = self.calculate_hash(filepath)
        print("✓")
        print(f"      Hash: {file_hash[:16]}...")
        
        # Extract text
        print("[2/5] Extracting text...", end=" ")
        raw_text = self.extract_text(filepath)
        print(f"✓ ({len(raw_text)} chars)")
        
        # Clean text
        print("[3/5] Cleaning text...", end=" ")
        cleaned_text = self.clean_text(raw_text)
        word_count = len(cleaned_text.split())
        print(f"✓ ({word_count} words)")
        
        # Create chunks
        print("[4/5] Creating chunks...", end=" ")
        chunks = self.create_chunks(cleaned_text)
        print(f"✓ ({len(chunks)} chunks)")
        
        # Prepare complete data structure
        result = {
            # Document metadata (for Neo4j Document node)
            'filename': path.name,
            'file_hash': file_hash,
            'filepath': str(path.absolute()),
            'extension': path.suffix.lower(),
            'size_bytes': file_stat.st_size,
            'size_mb': round(file_stat.st_size / (1024 * 1024), 2),
            
            # Processing metadata
            'raw_text_length': len(raw_text),
            'cleaned_text_length': len(cleaned_text),
            'word_count': word_count,
            'chunk_count': len(chunks),
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            
            # Chunks data
            'chunks': chunks
        }
        
        # Save to JSON
        print("[5/5] Saving to JSON...", end=" ")
        output_file = f"{path.stem}_processed.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print("✓")
        
        print(f"\n{'='*70}")
        print("✓ Text extraction complete!")
        print(f"{'='*70}")
        print(f"\nOutput: {output_file}")
        print(f"  Document: {result['filename']}")
        print(f"  Hash: {file_hash}")
        print(f"  Words: {word_count}")
        print(f"  Chunks: {len(chunks)}")
        
        print(f"\nNext step:")
        print(f"  python extract_entities.py {output_file}")
        
        return result


# Command-line interface
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python extract_text.py <filepath>")
        print("\nExample:")
        print("  python extract_text.py research_paper.pdf")
        print("\nSupported formats: .txt, .pdf, .md, .docx")
        print("\nCreates: <filename>_processed.json")
        print("  Contains: document metadata + text chunks")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    if not os.path.exists(filepath):
        print(f"✗ Error: File not found: {filepath}")
        sys.exit(1)
    
    extractor = TextExtractor()
    try:
        result = extractor.process_document(filepath)
        print(f"\n✓ Success!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
