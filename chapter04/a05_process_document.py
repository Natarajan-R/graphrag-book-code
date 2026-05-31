"""
process_document.py
Complete document processing pipeline
"""

import sys
import json
from pathlib import Path

from a01_upload_documents import DocumentUploader
from a02_extract_text import TextExtractor
from a03_extract_entities import EntityExtractor
from a04_build_graph import GraphBuilder


def process_document(filepath: str, force: bool = False):
    """
    Complete pipeline: Upload → Extract → Build Graph

    Args:
        filepath: Path to document
        force: Force reprocess if duplicate
    """
    print("\n" + "="*70)
    print(" GRAPHRAG DOCUMENT PROCESSING PIPELINE")
    print("="*70 + "\n")

    # Stage 1: Upload
    print("STAGE 1: Upload & Validation")
    print("-" * 70)
    uploader = DocumentUploader()
    try:
        upload_result = uploader.upload_document(filepath, force=force)

        if upload_result['status'] == 'error':
            print(f"\n✗ Upload failed: {upload_result['message']}")
            return False

        if upload_result['status'] == 'duplicate' and not force:
            print("\n! Document already processed. Use --force to reprocess.")
            return False

        file_hash = upload_result['file_hash']
        print(f"✓ Upload complete. Hash: {file_hash[:16]}...\n")

    except:
        print(f"✓ Upload failed. Hash: {file_hash[:16]}...\n")

    # Stage 2: Text Extraction
    print("STAGE 2: Text Extraction & Preprocessing")
    print("-" * 70)
    extractor = TextExtractor()
    text_result = extractor.process_document(filepath)

    # Save intermediate result
    processed_file = Path(filepath).stem + "_processed.json"
    with open(processed_file, 'w') as f:
        json.dump(text_result, f, indent=2)
    print(f"✓ Saved processed text to: {processed_file}\n")

    # Stage 3: Entity Extraction
    print("STAGE 3: Entity & Relationship Extraction")
    print("-" * 70)
    entity_extractor = EntityExtractor()
    chunk_extractions = entity_extractor.extract_from_chunks(
        text_result['chunks']
    )
    merged_entities = entity_extractor.merge_extractions(chunk_extractions)

    # Save intermediate result
    entities_file = Path(filepath).stem + "_entities.json"
    with open(entities_file, 'w') as f:
        json.dump(merged_entities, f, indent=2)
    print(f"✓ Saved entities to: {entities_file}\n")

    # Stage 4: Graph Construction
    print("STAGE 4: Graph Construction & Embedding")
    print("-" * 70)
    builder = GraphBuilder()
    try:
        file_hash = builder.build_graph(processed_file, entities_file)
        
        # Show statistics
        print("Document Statistics:")
        print("-" * 70)

        doc_stats = builder.get_document_statistics(file_hash)

        if doc_stats:
            print(f"  Filename: {doc_stats['filename']}")
            print(f"  Status: {doc_stats['status']}")
            print(f"  Chunks: {doc_stats['chunks']}")
            print(f"  Entities: {doc_stats['entities']}")
            print(f"  Relationships: {doc_stats['relationships']}")

        print(f"\n✓ Graph updated successfully!")
        print(f"  Total entities in graph: {doc_stats['entities']}")
        print(f"  Total relationships: {doc_stats['relationships']}\n")

    finally:
        builder.close()

    # Summary
    print("="*70)
    print(" ✓ PIPELINE COMPLETE")
    print("="*70)
    print(f"\nDocument: {filepath}")
    print(f"File hash: {file_hash}")
    print(f"\nExtracted:")
    print(f"  • {merged_entities['statistics']['unique_entities']} entities")
    print(f"  • {merged_entities['statistics']['unique_relationships']} relationships")
    print(f"\nIntermediate files:")
    print(f"  • {processed_file}")
    print(f"  • {entities_file}")
    print("\nGraph database updated with new knowledge!")
    print("="*70 + "\n")

    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_document.py <filepath> [--force]")
        sys.exit(1)

    filepath = sys.argv[1]
    force = '--force' in sys.argv

    success = process_document(filepath, force=force)
    sys.exit(0 if success else 1)