"""
graph_builder.py
Final Stage: Build complete graph in Neo4j from JSON files
Reads: processed.json + entities.json
Creates: Document → Chunks → Entities → Relationships
"""

import json
import sys
import os
from typing import Dict, List
from pathlib import Path
from neo4j import GraphDatabase
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Configuration
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
EMBED_MODEL = os.getenv('OLLAMA_EMBED_MODEL', 'nomic-embed-text')


class GraphBuilder:
    """Build complete knowledge graph from JSON files"""

    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        self.embedder = OllamaEmbeddings(model=EMBED_MODEL)

    def close(self):
        self.driver.close()

    # ============================================================
    # Layer 1: Document Node
    # ============================================================

    def create_document_node(self, doc_data: Dict):
        """Create Document node from processed.json metadata"""
        print(f"  [1/5] Creating Document node...", end=" ")
        
        with self.driver.session() as session:
            session.run("""
                MERGE (d:Document {file_hash: $file_hash})
                SET d.filename = $filename,
                    d.filepath = $filepath,
                    d.extension = $extension,
                    d.size_bytes = $size_bytes,
                    d.size_mb = $size_mb,
                    d.word_count = $word_count,
                    d.chunk_count = $chunk_count,
                    d.status = 'processing',
                    d.created_at = datetime()
            """,
                file_hash=doc_data['file_hash'],
                filename=doc_data['filename'],
                filepath=doc_data['filepath'],
                extension=doc_data['extension'],
                size_bytes=doc_data['size_bytes'],
                size_mb=doc_data['size_mb'],
                word_count=doc_data['word_count'],
                chunk_count=doc_data['chunk_count']
            )
        
        print("✓")

    # ============================================================
    # Layer 2: Chunk Nodes with Embeddings
    # ============================================================

    def create_chunk_nodes(self, chunks: List[Dict], file_hash: str):
        """Create Chunk nodes with embeddings"""
        print(f"  [2/5] Creating {len(chunks)} Chunk nodes with embeddings...")
        
        with self.driver.session() as session:
            for i, chunk in enumerate(chunks):
                # Generate embedding
                embedding = self.embedder.embed_query(chunk['text'])
                
                chunk_id = f"{file_hash}_{chunk['chunk_id']}"
                
                # Create chunk node
                session.run("""
                    CREATE (c:Chunk {
                        chunk_id: $chunk_id,
                        text: $text,
                        word_count: $word_count,
                        position: $position,
                        embedding: $embedding,
                        created_at: datetime()
                    })
                """,
                    chunk_id=chunk_id,
                    text=chunk['text'],
                    word_count=chunk['word_count'],
                    position=chunk['chunk_id'],
                    embedding=embedding
                )
                
                # Link to document
                session.run("""
                    MATCH (d:Document {file_hash: $file_hash})
                    MATCH (c:Chunk {chunk_id: $chunk_id})
                    MERGE (d)-[:HAS_CHUNK]->(c)
                """,
                    file_hash=file_hash,
                    chunk_id=chunk_id
                )
                
                if (i + 1) % 5 == 0 or i == len(chunks) - 1:
                    print(f"        Progress: {i + 1}/{len(chunks)}")
        
        print(f"        ✓ Created {len(chunks)} chunks")

    # ============================================================
    # Layer 3: Entity Nodes with Embeddings
    # ============================================================

    def create_entity_nodes(self, entities: List[Dict]):
        """Create Entity nodes with embeddings"""
        print(f"  [3/5] Creating {len(entities)} Entity nodes with embeddings...")
        
        # Generate all embeddings at once
        entity_texts = [f"{e['name']} ({e.get('type', 'Unknown')})" for e in entities]
        embeddings = self.embedder.embed_documents(entity_texts)
        
        with self.driver.session() as session:
            for i, (entity, embedding) in enumerate(zip(entities, embeddings)):
                session.run("""
                    MERGE (e:Entity {name: $name})
                    SET e.type = $type,
                        e.embedding = $embedding,
                        e.updated_at = datetime()
                """,
                    name=entity['name'],
                    type=entity.get('type', 'Unknown'),
                    embedding=embedding
                )
                
                if (i + 1) % 10 == 0 or i == len(entities) - 1:
                    print(f"        Progress: {i + 1}/{len(entities)}")
        
        print(f"        ✓ Created {len(entities)} entities")

    # ============================================================
    # Layer 4: Chunk → Entity Links (MENTIONS)
    # ============================================================

    def create_chunk_entity_links(self, entities: List[Dict], file_hash: str):
        """Create MENTIONS relationships"""
        print(f"  [4/5] Creating chunk→entity MENTIONS links...")
        
        mentions_count = 0
        
        with self.driver.session() as session:
            # For each chunk, link to all entities
            # Note: In production, you'd track which chunk mentioned which entity
            # For now, we link all entities to all chunks (simplified)
            for chunk_id in range(len(entities)):  # Simplified - link to corresponding chunks
                for entity in entities:
                    chunk_full_id = f"{file_hash}_{chunk_id}"
                    
                    result = session.run("""
                        MATCH (c:Chunk {chunk_id: $chunk_id})
                        MATCH (e:Entity {name: $entity_name})
                        MERGE (c)-[:MENTIONS]->(e)
                        RETURN count(*) as created
                    """,
                        chunk_id=chunk_full_id,
                        entity_name=entity['name']
                    )
                    
                    if result.single():
                        mentions_count += 1
        
        print(f"        ✓ Created {mentions_count} MENTIONS links")

    # ============================================================
    # Layer 5: Entity Relationships
    # ============================================================

    def create_entity_relationships(self, relationships: List[Dict]):
        """Create relationships between entities"""
        print(f"  [5/5] Creating {len(relationships)} entity relationships...")
        
        created = 0
        skipped = 0
        
        with self.driver.session() as session:
            for rel in relationships:
                # Validate
                if not rel.get('from') or not rel.get('to') or not rel.get('type'):
                    skipped += 1
                    continue
                
                try:
                    session.run("""
                        MATCH (from_entity:Entity {name: $from_name})
                        MATCH (to_entity:Entity {name: $to_name})
                        MERGE (from_entity)-[r:RELATIONSHIP {type: $rel_type}]->(to_entity)
                        SET r.created_at = coalesce(r.created_at, datetime())
                    """,
                        from_name=rel['from'],
                        to_name=rel['to'],
                        rel_type=rel['type']
                    )
                    created += 1
                except:
                    skipped += 1
        
        if skipped > 0:
            print(f"        ✓ Created {created} relationships ({skipped} skipped)")
        else:
            print(f"        ✓ Created {created} relationships")

    # ============================================================
    # Vector Indexes
    # ============================================================

    def create_vector_indexes(self):
        """Create vector indexes for similarity search"""
        print(f"  [6/6] Creating vector indexes...")
        
        with self.driver.session() as session:
            result = session.run("SHOW INDEXES")
            existing = [record['name'] for record in result]
            
            indexes = [
                ('chunk_embeddings', 'Chunk', 'embedding'),
                ('entity_embeddings', 'Entity', 'embedding')
            ]
            
            for index_name, label, prop in indexes:
                if index_name in existing:
                    print(f"        ✓ {index_name} exists")
                else:
                    try:
                        session.run(f"""
                            CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                            FOR (n:{label})
                            ON n.{prop}
                            OPTIONS {{
                                indexConfig: {{
                                    `vector.dimensions`: 768,
                                    `vector.similarity_function`: 'cosine'
                                }}
                            }}
                        """)
                        print(f"        ✓ Created {index_name}")
                    except Exception as e:
                        if "already exists" in str(e).lower():
                            print(f"        ✓ {index_name} exists")

    # ============================================================
    # Main Build Function
    # ============================================================

    def build_graph(self, processed_file: str, entities_file: str):
        """Build complete graph from JSON files"""
        
        print(f"\n{'='*70}")
        print(f"GraphRAG Knowledge Graph Builder")
        print(f"{'='*70}\n")
        
        # Load data
        print("Loading data files...")
        with open(processed_file, 'r') as f:
            processed_data = json.load(f)
        
        with open(entities_file, 'r') as f:
            entities_data = json.load(f)
        
        file_hash = processed_data['file_hash']
        filename = processed_data['filename']
        
        print(f"  File: {filename}")
        print(f"  Hash: {file_hash}")
        print(f"  Chunks: {len(processed_data['chunks'])}")
        print(f"  Entities: {len(entities_data.get('entities', []))}")
        print(f"  Relationships: {len(entities_data.get('relationships', []))}")
        
        # Build graph
        print(f"\nBuilding graph in Neo4j...")
        print("=" * 70)
        
        # 1. Create Document node
        self.create_document_node(processed_data)
        
        # 2. Create Chunk nodes with embeddings
        self.create_chunk_nodes(processed_data['chunks'], file_hash)
        
        # 3. Create Entity nodes with embeddings
        self.create_entity_nodes(entities_data.get('entities', []))
        
        # 4. Link chunks to entities
        self.create_chunk_entity_links(entities_data.get('entities', []), file_hash)
        
        # 5. Create entity relationships
        self.create_entity_relationships(entities_data.get('relationships', []))
        
        # 6. Create vector indexes
        self.create_vector_indexes()
        
        # Mark document as complete
        with self.driver.session() as session:
            session.run("""
                MATCH (d:Document {file_hash: $file_hash})
                SET d.status = 'completed',
                    d.completed_at = datetime()
            """, file_hash=file_hash)
        
        print("=" * 70)
        print(f"✓ Graph construction complete!\n")
        
        return file_hash

    # ============================================================
    # Statistics
    # ============================================================

    def get_document_statistics(self, file_hash: str) -> Dict:
        """Get statistics for specific document"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (d:Document {file_hash: $file_hash})
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
                OPTIONAL MATCH (e)-[r:RELATIONSHIP]->()
                RETURN 
                    d.filename as filename,
                    d.status as status,
                    count(DISTINCT c) as chunks,
                    count(DISTINCT e) as entities,
                    count(DISTINCT r) as relationships
            """, file_hash=file_hash)
            
            record = result.single()
            if record:
                return {
                    'filename': record['filename'],
                    'status': record['status'],
                    'chunks': record['chunks'],
                    'entities': record['entities'],
                    'relationships': record['relationships']
                }
            return None

    def get_graph_statistics(self) -> Dict:
        """Get overall graph statistics"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (d:Document)
                OPTIONAL MATCH (c:Chunk)
                OPTIONAL MATCH (e:Entity)
                OPTIONAL MATCH ()-[r:RELATIONSHIP]->()
                RETURN 
                    count(DISTINCT d) as documents,
                    count(DISTINCT c) as chunks,
                    count(DISTINCT e) as entities,
                    count(DISTINCT r) as relationships
            """)
            
            record = result.single()
            return {
                'documents': record['documents'],
                'chunks': record['chunks'],
                'entities': record['entities'],
                'relationships': record['relationships']
            }


# Command-line interface
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python graph_builder.py <processed.json> <entities.json>")
        print("\nExample:")
        print("  python graph_builder.py research_paper_processed.json research_paper_entities.json")
        print("\nReads JSON files and creates complete graph in Neo4j:")
        print("  - Document node (from processed.json metadata)")
        print("  - Chunk nodes with embeddings")
        print("  - Entity nodes with embeddings")
        print("  - All relationships")
        sys.exit(1)
    
    processed_file = sys.argv[1]
    entities_file = sys.argv[2]
    
    # Validate files
    if not os.path.exists(processed_file):
        print(f"✗ Error: File not found: {processed_file}")
        sys.exit(1)
    
    if not os.path.exists(entities_file):
        print(f"✗ Error: File not found: {entities_file}")
        sys.exit(1)
    
    # Build graph
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
        
        print("\nGlobal Graph Statistics:")
        print("-" * 70)
        global_stats = builder.get_graph_statistics()
        print(f"  Total documents: {global_stats['documents']}")
        print(f"  Total chunks: {global_stats['chunks']}")
        print(f"  Total entities: {global_stats['entities']}")
        print(f"  Total relationships: {global_stats['relationships']}")
        print()
        
    finally:
        builder.close()
