"""
Main Orchestrator Module
Coordinates the entire GraphRAG pipeline
"""

import sys
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from config import get_config, reload_config
from upload_documents import DocumentUploader
from extract_graph import GraphExtractor
from build_graph import GraphBuilder
from query_graph import QueryEngine
from utils import (
    setup_logger,
    DocumentMetadata,
    print_banner,
    print_section,
    print_success,
    print_error,
    print_info,
    print_warning,
    timing_decorator
)


class GraphRAGPipeline:
    """Main pipeline orchestrator for GraphRAG system"""

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the pipeline

        Args:
            config_file: Optional path to configuration file
        """
        # Load configuration
        if config_file:
            reload_config(config_file)

        self.config = get_config()

        # Setup logger
        log_file = self.config.paths.logs_dir / f"pipeline_{datetime.now().strftime('%Y%m%d')}.log"
        self.logger = setup_logger("GraphRAGPipeline", log_file)

        # Initialize components
        self.uploader = DocumentUploader()
        self.extractor = GraphExtractor()
        self.builder = GraphBuilder()
        self.query_engine = QueryEngine()

        # Metadata manager
        metadata_file = self.config.paths.processed_dir / "metadata.json"
        self.metadata = DocumentMetadata(metadata_file)

        self.logger.info("GraphRAG Pipeline initialized")

    @timing_decorator
    def run_full_pipeline(
        self,
        source_path: str,
        recursive: bool = False,
        skip_upload: bool = False,
        skip_extraction: bool = False,
        skip_building: bool = False
    ) -> bool:
        """
        Run the complete pipeline: upload -> extract -> build

        Args:
            source_path: Path to file or directory
            recursive: Process subdirectories
            skip_upload: Skip upload phase
            skip_extraction: Skip extraction phase
            skip_building: Skip graph building phase

        Returns:
            True if successful
        """
        print_banner("GraphRAG Full Pipeline")

        try:
            # Phase 1: Upload
            if not skip_upload:
                print_section("Phase 1: Document Upload")
                path = Path(source_path)

                if path.is_file():
                    file_hashes = [self.uploader.upload_file(source_path)]
                elif path.is_dir():
                    file_hashes = self.uploader.upload_directory(source_path, recursive)
                else:
                    print_error("Invalid source path")
                    return False

                # Filter out None values
                file_hashes = [h for h in file_hashes if h]

                if not file_hashes:
                    print_error("No files uploaded successfully")
                    return False

                print_success(f"Uploaded {len(file_hashes)} file(s)")
            else:
                # Get pending files
                docs = self.metadata.list_documents(status="uploaded")
                file_hashes = [doc["hash"] for doc in docs]
                print_info(f"Using {len(file_hashes)} pending file(s)")

            # Phase 2: Extract
            if not skip_extraction:
                print_section("Phase 2: Graph Extraction")

                extraction_results = []
                for file_hash in file_hashes:
                    result = self.extractor.extract_from_file(file_hash)
                    if result:
                        extraction_results.append(result)

                if not extraction_results:
                    print_error("No files extracted successfully")
                    return False

                print_success(f"Extracted {len(extraction_results)} file(s)")
                file_hashes = [r["file_hash"] for r in extraction_results]
            else:
                # Get extracted files
                docs = self.metadata.list_documents(status="extracted")
                file_hashes = [doc["hash"] for doc in docs]
                print_info(f"Using {len(file_hashes)} extracted file(s)")

            # Phase 3: Build
            if not skip_building:
                print_section("Phase 3: Graph Building")

                success_count = 0
                for file_hash in file_hashes:
                    if self.builder.build_graph_from_file(file_hash):
                        success_count += 1

                if success_count == 0:
                    print_error("No graphs built successfully")
                    return False

                print_success(f"Built {success_count} graph(s)")

            # Display final statistics
            print_section("Pipeline Complete")
            self.display_system_status()

            return True

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            print_error(f"Pipeline failed: {e}")
            return False

    def process_pending_files(self) -> bool:
        """
        Process all pending files through the pipeline

        Returns:
            True if successful
        """
        print_banner("Processing Pending Files")

        # Get files in each stage
        uploaded = self.metadata.list_documents(status="uploaded")
        extracted = self.metadata.list_documents(status="extracted")

        total_pending = len(uploaded) + len(extracted)

        if total_pending == 0:
            print_info("No pending files to process")
            return True

        print_info(f"Found {len(uploaded)} uploaded, {len(extracted)} extracted files")

        success = True

        # Extract uploaded files
        if uploaded:
            print_section("Extracting Files")
            for doc in uploaded:
                if not self.extractor.extract_from_file(doc["hash"]):
                    success = False

        # Build extracted files
        if extracted or uploaded:
            print_section("Building Graphs")
            extracted = self.metadata.list_documents(status="extracted")
            for doc in extracted:
                if not self.builder.build_graph_from_file(doc["hash"]):
                    success = False

        print_section("Processing Complete")
        self.display_system_status()

        return success

    def display_system_status(self):
        """Display comprehensive system status"""
        print_section("System Status")

        # Document statistics
        all_docs = self.metadata.list_documents()
        status_counts = {}
        for doc in all_docs:
            status = doc["status"]
            status_counts[status] = status_counts.get(status, 0) + 1

        print("\n📄 Documents:")
        print(f"  Total: {len(all_docs)}")
        for status, count in status_counts.items():
            print(f"  {status.capitalize()}: {count}")

        # Database statistics
        try:
            db_stats = self.builder.get_database_stats()
            print("\n🗄️  Neo4j Database:")
            print(f"  Nodes: {db_stats.get('nodes', 0)}")
            print(f"  Relationships: {db_stats.get('relationships', 0)}")
            print(f"  Node Types: {len(db_stats.get('node_labels', []))}")
            print(f"  Relationship Types: {len(db_stats.get('relationship_types', []))}")
        except Exception as e:
            print_warning(f"Could not retrieve database stats: {e}")

        # Query statistics
        query_history = self.query_engine.get_query_history(100)
        print(f"\n🔍 Queries:")
        print(f"  Total: {len(query_history)}")

    def validate_system(self) -> bool:
        """
        Validate system configuration and connectivity

        Returns:
            True if system is ready
        """
        print_banner("System Validation")

        all_ok = True

        # Validate configuration
        print_info("Validating configuration...")
        is_valid, errors = self.config.validate()
        if is_valid:
            print_success("Configuration valid")
        else:
            for error in errors:
                print_error(error)
            all_ok = False

        # Test Neo4j connection
        print_info("Testing Neo4j connection...")
        try:
            if self.builder.test_connection():
                print_success("Neo4j connection OK")
            else:
                print_error("Neo4j connection failed")
                all_ok = False
        except Exception as e:
            print_error(f"Neo4j test failed: {e}")
            all_ok = False

        # Test LLM connection
        print_info("Testing LLM connection...")
        try:
            # Simple test query
            test_result = self.extractor.llm.invoke("Say 'OK'")
            if test_result:
                print_success("LLM connection OK")
            else:
                print_error("LLM connection failed")
                all_ok = False
        except Exception as e:
            print_error(f"LLM test failed: {e}")
            all_ok = False

        # Check directories
        print_info("Checking directories...")
        for path in [
            self.config.paths.upload_dir,
            self.config.paths.processed_dir,
            self.config.paths.logs_dir,
            self.config.paths.cache_dir
        ]:
            if path.exists():
                print_success(f"  {path.name} exists")
            else:
                print_warning(f"  {path.name} missing (will be created)")

        if all_ok:
            print_success("\n✓ System validation passed")
        else:
            print_error("\n✗ System validation failed")

        return all_ok

    def reset_system(self, confirm: bool = False):
        """
        Reset the entire system (clear database and metadata)

        Args:
            confirm: Must be True to execute
        """
        if not confirm:
            print_warning("Reset requires confirmation. Use --confirm flag")
            return

        print_banner("System Reset")
        print_warning("This will delete all data!")

        # Clear Neo4j database
        print_info("Clearing Neo4j database...")
        self.builder.clear_database(confirm=True)

        # Clear metadata
        print_info("Clearing metadata...")
        self.metadata.metadata = {}
        self.metadata.save_metadata()

        # Clear query history
        print_info("Clearing query history...")
        self.query_engine.clear_history()

        print_success("System reset complete")


def main():
    """Main entry point"""
    import argparse

    print_banner("GraphRAG System")

    parser = argparse.ArgumentParser(
        description="GraphRAG System - Advanced Knowledge Graph RAG Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline on a single file
  python main.py --run document.pdf

  # Run full pipeline on a directory
  python main.py --run /path/to/docs --recursive

  # Process pending files
  python main.py --process-pending

  # Start interactive query mode
  python main.py --query

  # System status
  python main.py --status

  # Validate system
  python main.py --validate
        """
    )

    parser.add_argument("--run", metavar="PATH", help="Run full pipeline on file/directory")
    parser.add_argument("--recursive", action="store_true", help="Process directories recursively")
    parser.add_argument("--process-pending", action="store_true", help="Process pending files")
    parser.add_argument("--query", action="store_true", help="Start interactive query mode")
    parser.add_argument("--status", action="store_true", help="Show system status")
    parser.add_argument("--validate", action="store_true", help="Validate system configuration")
    parser.add_argument("--reset", action="store_true", help="Reset system (requires --confirm)")
    parser.add_argument("--confirm", action="store_true", help="Confirm destructive operations")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--skip-upload", action="store_true", help="Skip upload phase")
    parser.add_argument("--skip-extraction", action="store_true", help="Skip extraction phase")
    parser.add_argument("--skip-building", action="store_true", help="Skip building phase")

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = GraphRAGPipeline(config_file=args.config)

    # Execute commands
    if args.validate:
        pipeline.validate_system()
    elif args.status:
        pipeline.display_system_status()
    elif args.reset:
        pipeline.reset_system(confirm=args.confirm)
    elif args.query:
        pipeline.query_engine.interactive_mode()
    elif args.run:
        pipeline.run_full_pipeline(
            args.run,
            recursive=args.recursive,
            skip_upload=args.skip_upload,
            skip_extraction=args.skip_extraction,
            skip_building=args.skip_building
        )
    elif args.process_pending:
        pipeline.process_pending_files()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
