"""
Graph Extractor Module
Extracts entities and relationships from documents using LLM
"""

# Add this import at the top of extract_graph.py
from entities import (
    GraphFirstExtractor,
    EntityNode,
    EntityRelationship,
    ENHANCED_EXTRACTION_PROMPT,
    create_extraction_prompt
)

from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import json

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document

from config import get_config
from utils import (
    setup_logger,
    DocumentMetadata,
    timing_decorator,
    ProgressTracker,
    print_banner,
    print_section,
    print_success,
    print_error,
    print_info,
    retry_on_failure
)


class GraphExtractor:
    """Extracts graph structure from documents"""

    def __init__(self):
        """Initialize the graph extractor"""
        self.config = get_config()

        # Setup logger
        log_file = self.config.paths.logs_dir / f"extraction_{datetime.now().strftime('%Y%m%d')}.log"
        self.logger = setup_logger("GraphExtractor", log_file)

        # Setup metadata manager
        metadata_file = self.config.paths.processed_dir / "metadata.json"
        self.metadata = DocumentMetadata(metadata_file)

        # Initialize LLM
        self._initialize_llm()

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.processing.chunk_size,
            chunk_overlap=self.config.processing.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        self.logger.info("GraphExtractor initialized")

    @retry_on_failure(max_retries=3, delay=2.0)
    def _initialize_llm(self):
        """Initialize the LLM with retry logic"""
        try:
            self.llm = ChatOllama(
                model=self.config.llm.model,
                temperature=self.config.llm.temperature,
                base_url=self.config.llm.base_url
            )
            self.logger.info(f"LLM initialized: {self.config.llm.model}")
            print_success(f"Connected to LLM: {self.config.llm.model}")
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM: {e}")
            print_error(f"LLM initialization failed: {e}")
            raise

    def _load_document(self, file_path: Path) -> List[Document]:
        """
        Load document based on file type

        Args:
            file_path: Path to document

        Returns:
            List of loaded documents
        """
        self.logger.info(f"Loading document: {file_path.name}")

        extension = file_path.suffix.lower()

        try:
            if extension == ".pdf":
                loader = PyPDFLoader(str(file_path))
            elif extension in [".txt", ".md"]:
                loader = TextLoader(str(file_path))
            else:
                raise ValueError(f"Unsupported file type: {extension}")

            documents = loader.load()
            self.logger.info(f"Loaded {len(documents)} page(s) from {file_path.name}")
            return documents

        except Exception as e:
            self.logger.error(f"Failed to load {file_path.name}: {e}")
            raise

    def _chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks

        Args:
            documents: List of documents to chunk

        Returns:
            List of chunked documents
        """
        self.logger.info("Chunking documents...")
        chunks = self.text_splitter.split_documents(documents)
        self.logger.info(f"Created {len(chunks)} chunks")
        return chunks

    @timing_decorator

    def _extract_graph_data(self, chunks: List[Document]) -> List:
        """
        Extract graph structure with Graph-First architecture
        Captures ALL versions including contradictions
        """
        self.logger.info("Extracting entities and relationships (Graph-First)...")
        print_section("Extracting Graph Structure (Graph-First)")

        all_extractors = []
        progress = ProgressTracker(len(chunks), "Processing chunks")

        for i, chunk in enumerate(chunks):
            try:
                # Get page number from metadata
                page_num = chunk.metadata.get('page', i)

                # Create enhanced prompt
                prompt = create_extraction_prompt(
                    chunk.page_content,
                    page_number=page_num
                )

                # Call LLM
                response = self.llm.invoke(prompt)

                # Parse response (assuming JSON format)
                try:
                    import json
                    # Extract JSON from response
                    content = response.content if hasattr(response, 'content') else str(response)

                    # Try to find JSON in response
                    start_idx = content.find('{')
                    end_idx = content.rfind('}') + 1
                    if start_idx >= 0 and end_idx > start_idx:
                        json_str = content[start_idx:end_idx]
                        extraction_data = json.loads(json_str)
                    else:
                        self.logger.warning(f"No JSON found in chunk {i}")
                        progress.update()
                        continue

                    # Create extractor for this chunk
                    chunk_extractor = GraphFirstExtractor()

                    # Add entities
                    for entity_data in extraction_data.get("entities", []):
                        node = EntityNode(
                            id=entity_data.get("id", ""),
                            type=entity_data.get("type", "Unknown"),
                            properties=entity_data.get("properties", {}),
                            source_page=entity_data.get("source_page", page_num),
                            source_text=chunk.page_content[:200],
                            confidence=entity_data.get("confidence", "stated"),
                            version=entity_data.get("version")
                        )
                        chunk_extractor.add_node(node)

                    # Add relationships
                    for rel_data in extraction_data.get("relationships", []):
                        relationship = EntityRelationship(
                            source_id=rel_data.get("source", ""),
                            target_id=rel_data.get("target", ""),
                            type=rel_data.get("type", "RELATED"),
                            properties=rel_data.get("properties", {}),
                            source_page=rel_data.get("source_page", page_num),
                            source_text=chunk.page_content[:200],
                            confidence=rel_data.get("confidence", "stated"),
                            version=rel_data.get("version"),
                            nature=rel_data.get("nature"),
                            route=rel_data.get("route")
                        )
                        chunk_extractor.add_relationship(relationship)

                    all_extractors.append(chunk_extractor)

                except json.JSONDecodeError as e:
                    self.logger.warning(f"JSON parse error in chunk {i}: {e}")

                progress.update()

            except Exception as e:
                self.logger.error(f"Error processing chunk {i}: {e}")
                progress.update()
                continue

        progress.complete()

        # Merge all extractors
        from entities import merge_extractors
        merged_extractor = merge_extractors(*all_extractors)

        # Detect contradictions
        contradictions = merged_extractor.detect_contradictions()

        if contradictions:
            print_info(f"⚠️  Detected {len(contradictions)} contradictions")
            for i, contradiction in enumerate(contradictions[:3], 1):  # Show first 3
                print(f"  {i}. {contradiction['reason']}")

        # Convert to format compatible with rest of pipeline
        # We'll store the enhanced data for build_graph.py to use
        return merged_extractor



    def _count_entities_and_relationships(self, graph_documents: List) -> Tuple[int, int]:
        """
        Count total entities and relationships

        Args:
            graph_documents: List of graph documents

        Returns:
            Tuple of (entities_count, relationships_count)
        """
        entities = set()
        relationships = 0

        for graph_doc in graph_documents:
            # Count unique entities
            for node in graph_doc.nodes:
                entities.add((node.type, node.id))

            # Count relationships
            relationships += len(graph_doc.relationships)

        return len(entities), relationships

    def _save_extraction_results(self, file_hash: str, extractor: GraphFirstExtractor, chunks: List[Document]):
        """
        Save extraction results with Graph-First metadata
        """
        output_dir = self.config.paths.processed_dir / file_hash
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save chunks (unchanged)
        chunks_file = output_dir / "chunks.json"
        chunks_data = [
            {
                "page_content": chunk.page_content,
                "metadata": chunk.metadata
            }
            for chunk in chunks
        ]

        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)

        # Save Graph-First extraction data
        graph_file = output_dir / "graph_data.json"
        summary = extractor.get_extraction_summary()

        with open(graph_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Save contradictions separately for easy access
        if summary['contradictions']:
            contradictions_file = output_dir / "contradictions.json"
            with open(contradictions_file, 'w', encoding='utf-8') as f:
                json.dump(summary['contradictions'], f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved extraction results to {output_dir}")
        if summary['contradictions']:
            self.logger.info(f"Found {len(summary['contradictions'])} contradictions")


    @timing_decorator
    def extract_from_file(self, file_hash: str) -> Optional[Dict]:
        """
        Extract graph structure from a single file

        Args:
            file_hash: Hash of file to process

        Returns:
            Dictionary with extraction results
        """
        # Reload metadata to ensure we have the latest data
        self.metadata = DocumentMetadata(self.config.paths.processed_dir / "metadata.json")

        # Get file metadata
        doc_metadata = self.metadata.get_document(file_hash)
        if not doc_metadata:
            self.logger.error(f"File not found: {file_hash}")
            print_error("File not found in metadata")
            return None

        file_path = Path(doc_metadata["path"])
        if not file_path.exists():
            self.logger.error(f"File does not exist: {file_path}")
            print_error("File does not exist")
            return None

        print_banner(f"Extracting: {file_path.name}")

        # Update status
        self.metadata.update_processing_status(file_hash, "processing")

        try:
            # Step 1: Load document
            print_info("Loading document...")
            raw_documents = self._load_document(file_path)

            # Step 2: Chunk document
            print_info("Chunking document...")
            chunks = self._chunk_documents(raw_documents)

            # Step 3: Extract graph structure (Graph-First)
            extractor = self._extract_graph_data(chunks)

            # Step 4: Get counts
            summary = extractor.get_extraction_summary()
            entities_count = summary['nodes_count']
            relationships_count = summary['relationships_count']
            contradictions_count = summary['contradictions_count']

            # Step 5: Save results
            print_info("Saving extraction results...")
            self._save_extraction_results(file_hash, extractor, chunks)

            # Update metadata with contradiction info
            self.metadata.update_processing_status(
                file_hash,
                "extracted",
                chunks_count=len(chunks),
                entities_count=entities_count,
                relationships_count=relationships_count
                #contradictions_count=contradictions_count  # NEW
            )

            results = {
                "file_hash": file_hash,
                "filename": file_path.name,
                "chunks_count": len(chunks),
                "entities_count": entities_count,
                "relationships_count": relationships_count,
                "contradictions_count": contradictions_count,  # NEW
                "extractor": extractor
            }

            print_section("Extraction Summary")
            print(f"  Chunks: {len(chunks)}")
            print(f"  Entities: {entities_count}")
            print(f"  Relationships: {relationships_count}")
            if contradictions_count > 0:
                print(f"  ⚠️  Contradictions: {contradictions_count}")
            print_success("Extraction completed successfully!")

            return results

        except Exception as e:
            self.logger.error(f"Extraction failed for {file_path.name}: {e}")
            self.metadata.update_processing_status(file_hash, "failed")
            print_error(f"Extraction failed: {e}")
            return None

    def extract_from_multiple(self, file_hashes: List[str]) -> List[Dict]:
        """
        Extract graph structure from multiple files

        Args:
            file_hashes: List of file hashes to process

        Returns:
            List of extraction results
        """
        results = []

        print_banner(f"Batch Extraction: {len(file_hashes)} files")

        for i, file_hash in enumerate(file_hashes, 1):
            print(f"\n{'=' * 60}")
            print(f"Processing {i}/{len(file_hashes)}")
            print(f"{'=' * 60}")

            result = self.extract_from_file(file_hash)
            if result:
                results.append(result)

        print_section("Batch Extraction Complete")
        print(f"  Successful: {len(results)}/{len(file_hashes)}")

        return results


def main():
    """Main function for standalone usage"""
    import argparse

    print_banner("GraphRAG Graph Extractor")

    parser = argparse.ArgumentParser(description="Extract graph structure from documents")
    parser.add_argument("--hash", help="File hash to process")
    parser.add_argument("--all", action="store_true", help="Process all uploaded files")
    parser.add_argument("--pending", action="store_true", help="Process pending files only")

    args = parser.parse_args()

    extractor = GraphExtractor()
    metadata = DocumentMetadata(get_config().paths.processed_dir / "metadata.json")

    if args.hash:
        # Process single file
        extractor.extract_from_file(args.hash)
    elif args.all:
        # Process all files
        docs = metadata.list_documents()
        hashes = [doc["hash"] for doc in docs]
        extractor.extract_from_multiple(hashes)
    elif args.pending:
        # Process only uploaded (not yet extracted) files
        docs = metadata.list_documents(status="uploaded")
        hashes = [doc["hash"] for doc in docs]
        if hashes:
            extractor.extract_from_multiple(hashes)
        else:
            print_info("No pending files to process")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
