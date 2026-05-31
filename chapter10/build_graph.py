"""
Graph Builder Module
Builds and stores knowledge graph in Neo4j database
AND generates Vector Embeddings during ingestion.
WITH GRAPH-FIRST ARCHITECTURE SUPPORT
"""

from typing import List, Dict, Optional
from datetime import datetime
import json

from langchain_neo4j import Neo4jGraph
from langchain_ollama import OllamaEmbeddings
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError

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


class GraphBuilder:
    """Builds and manages knowledge graph in Neo4j"""

    def __init__(self):
        """Initialize the graph builder"""
        self.config = get_config()

        # Setup logger
        log_file = self.config.paths.logs_dir / f"graph_builder_{datetime.now().strftime('%Y%m%d')}.log"
        self.logger = setup_logger("GraphBuilder", log_file)

        # Setup metadata manager
        metadata_file = self.config.paths.processed_dir / "metadata.json"
        self.metadata = DocumentMetadata(metadata_file)

        # Initialize Embeddings Model (Nomic)
        self.embedder = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url=self.config.llm.base_url
        )

        # Initialize Neo4j connection
        self._initialize_neo4j()

        self.logger.info("GraphBuilder initialized")

    @retry_on_failure(max_retries=3, delay=2.0)
    def _initialize_neo4j(self):
        """Initialize Neo4j connection"""
        try:
            self.graph = Neo4jGraph(
                url=self.config.neo4j.uri,
                username=self.config.neo4j.username,
                password=self.config.neo4j.password,
                database=self.config.neo4j.database
            )
            self.logger.info("Connected to Neo4j database")
        except Exception as e:
            self.logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def test_connection(self) -> bool:
        try:
            result = self.graph.query("RETURN 1 as test")
            return len(result) > 0
        except Exception:
            return False

    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        try:
            stats = self.graph.query("""
                MATCH (n)
                OPTIONAL MATCH ()-[r]->()
                RETURN count(DISTINCT n) as nodes, count(DISTINCT r) as rels
            """)
            return {
                "nodes": stats[0]['nodes'],
                "relationships": stats[0]['rels']
            }
        except Exception:
            return {}

    def create_indexes(self):
        """Create ALL indexes (Graph + Vector)"""
        print_info("Creating database indexes...")

        # 1. Standard Graph Indexes
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (n:Document) ON (n.id)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.id)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Chunk) ON (n.id)",
            # Fulltext index for keyword search on Entities
            "CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS FOR (e:Entity) ON EACH [e.id]"
        ]

        for index_query in indexes:
            try:
                self.graph.query(index_query)
            except Exception as e:
                self.logger.warning(f"Index creation warning: {e}")

        # 2. Vector Index (768 Dimensions for Nomic)
        vector_query = """
        CREATE VECTOR INDEX chunk_vector_index IF NOT EXISTS
        FOR (c:Chunk)
        ON (c.embedding)
        OPTIONS {indexConfig: {
            `vector.dimensions`: 768,
            `vector.similarity_function`: 'cosine'
        }}
        """
        try:
            self.graph.query(vector_query)
            self.logger.info("Vector index created/verified")
        except Exception as e:
            self.logger.warning(f"Vector index warning: {e}")

        print_success("Indexes created")

    def create_constraints(self):
        """Create constraints for data integrity"""
        print_info("Creating database constraints...")
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Document) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE"
        ]
        for q in constraints:
            try:
                self.graph.query(q)
            except Exception:
                pass
        print_success("Constraints created")

    def _load_extraction_data(self, file_hash: str) -> Optional[Dict]:
        extraction_dir = self.config.paths.processed_dir / file_hash
        if not extraction_dir.exists():
            return None

        with open(extraction_dir / "chunks.json", 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        with open(extraction_dir / "graph_data.json", 'r', encoding='utf-8') as f:
            graph = json.load(f)

        return {"chunks": chunks, "graph_data": graph}

    @timing_decorator
    def _build_document_node(self, file_hash: str, doc_metadata: Dict):
        query = """
        MERGE (d:Document {id: $file_hash})
        SET d.filename = $filename,
            d.uploaded_at = $uploaded_at,
            d.processed_at = $processed_at
        """
        self.graph.query(query, {
            "file_hash": file_hash,
            "filename": doc_metadata["filename"],
            "uploaded_at": doc_metadata["uploaded_at"],
            "processed_at": doc_metadata["processed_at"]
        })

    @timing_decorator
    def _build_chunk_nodes(self, file_hash: str, chunks_data: List[Dict]):
        """Create chunk nodes AND generate embeddings immediately"""
        progress = ProgressTracker(len(chunks_data), "Creating chunks & embeddings")

        # Prepare text list for batch embedding
        texts = [chunk["page_content"] for chunk in chunks_data]

        # Generate Embeddings
        try:
            embeddings = self.embedder.embed_documents(texts)
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            raise e

        # Ingest into Neo4j
        for i, (chunk, embedding) in enumerate(zip(chunks_data, embeddings)):
            chunk_id = f"{file_hash}_chunk_{i}"

            query = """
            MATCH (d:Document {id: $file_hash})
            MERGE (c:Chunk {id: $chunk_id})
            SET c.text = $text,
                c.chunk_index = $chunk_index,
                c.embedding = $embedding,
                c.metadata = $metadata
            MERGE (d)-[:HAS_CHUNK]->(c)
            """

            params = {
                "file_hash": file_hash,
                "chunk_id": chunk_id,
                "text": chunk["page_content"],
                "chunk_index": i,
                "embedding": embedding,
                "metadata": json.dumps(chunk.get("metadata", {}))
            }

            self.graph.query(query, params)
            progress.update()

        progress.complete()

    @timing_decorator
    def _build_entity_nodes_and_relationships(self, file_hash: str, graph_data: Dict, chunks_data: List[Dict]):
        """
        Create entity nodes and relationships with Graph-First metadata
        CRITICAL UPDATE: Links Entities to Chunks via text matching.
        """
        print_info("Building graph structure with Graph-First metadata...")

        # Extract from Graph-First format
        nodes_data = graph_data.get('nodes', [])
        relationships_data = graph_data.get('relationships', [])
        contradictions = graph_data.get('contradictions', [])

        print_info(f"Processing {len(nodes_data)} nodes, {len(relationships_data)} relationships, {len(contradictions)} contradictions")

        # 1. Create Entity Nodes with metadata AND Link to Chunks
        for node in nodes_data:
            # A. Merge Entity
            query = """
            MERGE (e:Entity {id: $entity_id})
            SET e.type = $entity_type,
                e.source_page = $source_page,
                e.source_text = $source_text,
                e.confidence = $confidence,
                e.version = $version,
                e.extracted_at = $extracted_at
            """
            self.graph.query(query, {
                "entity_id": node["id"],
                "entity_type": node["type"],
                "source_page": node.get("source_page"),
                "source_text": node.get("source_text"),
                "confidence": node.get("confidence", "stated"),
                "version": node.get("version"),
                "extracted_at": node.get("extracted_at")
            })

            # B. LINKAGE LOGIC: Connect Chunk -> Entity
            # We look for the chunk that contains this entity's source_text
            entity_source_text = node.get("source_text", "")
            if entity_source_text:
                # Normalize text: strip whitespace and take first 100 chars
                # (Entities often have truncated text compared to raw chunks, so we use a safe substring)
                clean_source = entity_source_text.strip()[:100]

                found_chunk_index = -1
                for i, chunk in enumerate(chunks_data):
                    if clean_source in chunk["page_content"]:
                        found_chunk_index = i
                        break

                if found_chunk_index != -1:
                    chunk_id = f"{file_hash}_chunk_{found_chunk_index}"

                    # Create the Critical Link: Chunk -> Entity
                    link_query = """
                    MATCH (c:Chunk {id: $chunk_id})
                    MATCH (e:Entity {id: $entity_id})
                    MERGE (c)-[:HAS_ENTITY]->(e)
                    """
                    self.graph.query(link_query, {
                        "chunk_id": chunk_id,
                        "entity_id": node["id"]
                    })

        # 2. Create Relationships with FULL metadata
        for rel in relationships_data:
            # Generate unique ID for this relationship
            version_str = rel.get('version') or 'default'
            rel_id = f"{rel['source_id']}:{rel['type']}:{rel['target_id']}:{version_str}"

            query = """
            MATCH (s:Entity {id: $source}), (t:Entity {id: $target})
            MERGE (s)-[r:RELATIONSHIP {id: $rel_id}]->(t)
            SET r.type = $rel_type,
                r.nature = $nature,
                r.route = $route,
                r.source_page = $source_page,
                r.source_text = $source_text,
                r.confidence = $confidence,
                r.version = $version,
                r.extracted_at = $extracted_at,
                r.contradicts = $contradicts
            """
            self.graph.query(query, {
                "source": rel["source_id"],
                "target": rel["target_id"],
                "rel_id": rel_id,
                "rel_type": rel["type"],
                "nature": rel.get("nature"),
                "route": rel.get("route"),
                "source_page": rel.get("source_page"),
                "source_text": rel.get("source_text"),
                "confidence": rel.get("confidence", "stated"),
                "version": rel.get("version"),
                "extracted_at": rel.get("extracted_at"),
                "contradicts": rel.get("contradicts")
            })

        print_success(f"Created {len(nodes_data)} entities and {len(relationships_data)} relationships")

        # 3. Create explicit CONTRADICTS nodes (Modified to fix Neo4j syntax error)
        if contradictions:
            print_info(f"Creating {len(contradictions)} contradiction links...")
            for i, contradiction in enumerate(contradictions):
                rel1 = contradiction['relationship_1']
                rel2 = contradiction['relationship_2']

                version1_str = rel1.get('version') or 'default'
                version2_str = rel2.get('version') or 'default'

                rel1_id = f"{rel1['source_id']}:{rel1['type']}:{rel1['target_id']}:{version1_str}"
                rel2_id = f"{rel2['source_id']}:{rel2['type']}:{rel2['target_id']}:{version2_str}"

                # Create a unique ID for this contradiction event
                contradiction_id = f"contra_{hash(rel1_id + rel2_id)}"

                # REVISED QUERY: Creates a Contradiction Node instead of a Relationship-to-Relationship link
                query = """
                MATCH (s1:Entity)-[r1:RELATIONSHIP {id: $rel1_id}]->(t1:Entity)
                MATCH (s2:Entity)-[r2:RELATIONSHIP {id: $rel2_id}]->(t2:Entity)

                MERGE (c:Contradiction {id: $cid})
                SET c.reason = $reason,
                    c.rel1_ref = $rel1_id,
                    c.rel2_ref = $rel2_id,
                    c.timestamp = datetime()

                MERGE (s1)-[:INVOLVED_IN]->(c)
                MERGE (s2)-[:INVOLVED_IN]->(c)
                """

                try:
                    self.graph.query(query, {
                        "rel1_id": rel1_id,
                        "rel2_id": rel2_id,
                        "cid": contradiction_id,
                        "reason": contradiction.get('reason', 'Conflicting statements')
                    })
                except Exception as e:
                    self.logger.warning(f"Could not create contradiction link: {e}")

            print_success(f"✅ Created {len(contradictions)} contradiction nodes")

    def _create_gds_shortcut(self, file_hash: str):
        """
        Creates the 'MENTIONS' relationship: Document -> Entity.
        This shortcuts the Chunk node to enable efficient GDS Algorithms (Leiden/Louvain).
        """
        print_info("Creating 'MENTIONS' shortcut for Graph Data Science...")
        query = """
        MATCH (d:Document {id: $file_hash})-[:HAS_CHUNK]->(c:Chunk)-[:HAS_ENTITY]->(e:Entity)
        MERGE (d)-[:MENTIONS]->(e)
        """
        self.graph.query(query, {"file_hash": file_hash})

    @timing_decorator
    def build_graph_from_file(self, file_hash: str) -> bool:
        """Main build orchestration"""
        self.metadata = DocumentMetadata(self.config.paths.processed_dir / "metadata.json")
        doc_metadata = self.metadata.get_document(file_hash)

        if not doc_metadata or doc_metadata["status"] != "extracted":
            print_error(f"File {file_hash} not ready for building")
            return False

        print_banner(f"Building Graph: {doc_metadata['filename']}")

        try:
            # 1. Ensure Indexes Exist (Idempotent)
            self.create_indexes()
            self.create_constraints()

            # 2. Load Data
            data = self._load_extraction_data(file_hash)
            if not data:
                print_error("Failed to load extraction data")
                return False

            # 3. Build Nodes (Doc -> Chunk + Embeddings)
            self._build_document_node(file_hash, doc_metadata)
            self._build_chunk_nodes(file_hash, data["chunks"])

            # 4. Build Graph (Entities -> Relationships)
            # CRITICAL UPDATE: Passing 'chunks' data to allow linking
            self._build_entity_nodes_and_relationships(
                file_hash,
                data["graph_data"],
                data["chunks"]
            )

            # 5. Create GDS Shortcut (Doc -> Entity)
            # CRITICAL UPDATE: Enables Leiden/Louvain algorithms
            self._create_gds_shortcut(file_hash)

            # 6. Finish
            self.metadata.update_processing_status(file_hash, "processed")
            print_success(f"Graph built with embeddings for {doc_metadata['filename']}")
            return True

        except Exception as e:
            self.logger.error(f"Build failed: {e}")
            print_error(f"Build failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def clear_database(self, confirm: bool = False):
        """Clear all data from Neo4j database"""
        if confirm:
            print_info("Clearing Neo4j database...")
            self.graph.query("MATCH (n) DETACH DELETE n")
            # Drop indexes too
            try:
                self.graph.query("DROP INDEX chunk_vector_index IF EXISTS")
                self.graph.query("DROP INDEX entity_fulltext IF EXISTS")
            except:
                pass
            print_success("Database cleared")
        else:
            print_error("Clear operation requires confirm=True")


def main():
    """Main function for standalone usage"""
    import argparse

    print_banner("GraphRAG Graph Builder (Graph-First)")

    parser = argparse.ArgumentParser(description="Build knowledge graph in Neo4j with Graph-First architecture")
    parser.add_argument("--hash", help="File hash to process")
    parser.add_argument("--all", action="store_true", help="Process all extracted files")
    parser.add_argument("--clear", action="store_true", help="Clear database (requires --confirm)")
    parser.add_argument("--confirm", action="store_true", help="Confirm destructive operations")
    parser.add_argument("--stats", action="store_true", help="Show database statistics")

    args = parser.parse_args()

    builder = GraphBuilder()
    metadata = DocumentMetadata(get_config().paths.processed_dir / "metadata.json")

    if args.clear:
        if args.confirm:
            builder.clear_database(confirm=True)
        else:
            print_error("Clear requires --confirm flag")
    elif args.stats:
        stats = builder.get_database_stats()
        print_section("Database Statistics")
        print(f"  Nodes: {stats.get('nodes', 0)}")
        print(f"  Relationships: {stats.get('relationships', 0)}")
    elif args.hash:
        # Process single file
        success = builder.build_graph_from_file(args.hash)
        if success:
            print_section("Build Complete")
            # Show stats
            stats = builder.get_database_stats()
            print(f"  Total Nodes: {stats.get('nodes', 0)}")
            print(f"  Total Relationships: {stats.get('relationships', 0)}")
    elif args.all:
        # Process all extracted files
        docs = metadata.list_documents(status="extracted")
        hashes = [doc["hash"] for doc in docs]
        if hashes:
            print_info(f"Building graphs for {len(hashes)} files...")
            success_count = 0
            for i, file_hash in enumerate(hashes, 1):
                print(f"\n{'='*60}")
                print(f"Processing {i}/{len(hashes)}")
                print(f"{'='*60}")
                if builder.build_graph_from_file(file_hash):
                    success_count += 1

            print_section("Batch Build Complete")
            print(f"  Successful: {success_count}/{len(hashes)}")
        else:
            print_info("No extracted files to process")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
