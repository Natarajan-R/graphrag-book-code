"""
hybrid_retriever.py
Complete hybrid retrieval: vector search + graph expansion (IMPROVED)
"""

from typing import List, Dict
from neo4j import GraphDatabase
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

# Configuration
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
EMBED_MODEL = os.getenv('OLLAMA_EMBED_MODEL', 'nomic-embed-text')


class HybridRetriever:
    """Hybrid retrieval combining vector search and graph traversal"""

    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        self.embedder = OllamaEmbeddings(model=EMBED_MODEL)

    def close(self):
        self.driver.close()

    def retrieve(
        self,
        query: str,
        chunk_k: int = 3,
        entity_k: int = 3,
        expand_hops: int = 1
    ) -> Dict:
        """
        Hybrid retrieval combining vector and graph

        Args:
            query: Natural language query
            chunk_k: Number of chunks to retrieve
            entity_k: Number of entities to retrieve
            expand_hops: Graph expansion distance

        Returns:
            Complete context dict
        """
        print(f"Retrieving context for: {query}")

        # Step 1: Vector search
        print("  [1/4] Vector search...", end=" ")
        query_vector = self.embedder.embed_query(query)
        
        chunks = self._vector_search_chunks(query_vector, chunk_k)
        entities = self._vector_search_entities(query_vector, entity_k)
        print(f"✓ ({len(chunks)} chunks, {len(entities)} entities)")

        # Step 2: Get entities from chunks (IMPROVED - limit and filter)
        print("  [2/4] Extracting entities from chunks...", end=" ")
        chunk_ids = [c['chunk_id'] for c in chunks]
        chunk_entities = self._get_entities_from_chunks(chunk_ids, limit=10)  # Limit to top 10
        
        # Merge entities from vector search and chunks, but prioritize vector search results
        seen_entities = set([e['name'] for e in entities])
        for entity in chunk_entities:
            if entity['name'] not in seen_entities and len(seen_entities) < 15:
                entities.append(entity)
                seen_entities.add(entity['name'])
        
        print(f"✓ ({len(entities)} entities)")

        # Combine all entity names for graph expansion
        all_entity_names = [e['name'] for e in entities]

        # Step 3: Graph expansion (only from relevant entities)
        print("  [3/4] Expanding graph context...", end=" ")
        expanded = self._expand_graph_context(all_entity_names, expand_hops)
        print(f"✓ ({len(expanded['relationships'])} relationships)")

        # Step 4: Get source chunks for entities
        print("  [4/4] Retrieving source chunks...", end=" ")
        entity_chunks = self._get_chunks_for_entities(all_entity_names)
        
        # Combine with original chunks (deduplicate)
        all_chunk_ids = set([c['chunk_id'] for c in chunks])
        for chunk in entity_chunks:
            if chunk['chunk_id'] not in all_chunk_ids:
                chunks.append(chunk)
                all_chunk_ids.add(chunk['chunk_id'])
        
        print(f"✓ ({len(chunks)} total chunks)")

        # Assemble final context
        context = {
            'query': query,
            'chunks': chunks,
            'entities': all_entity_names,
            'relationships': expanded['relationships'],
            'graph_neighbors': expanded['neighbors']
        }

        print("  ✓ Context retrieved")
        return context

    def _vector_search_chunks(self, query_vector: List[float], top_k: int) -> List[Dict]:
        """Search chunks by vector similarity"""
        with self.driver.session() as session:
            result = session.run("""
                CALL db.index.vector.queryNodes(
                    'chunk_embeddings',
                    $top_k,
                    $query_vector
                ) YIELD node, score

                RETURN 
                    node.chunk_id as chunk_id,
                    node.text as text,
                    score
                ORDER BY score DESC
            """, top_k=top_k, query_vector=query_vector)

            return [dict(record) for record in result]

    def _vector_search_entities(self, query_vector: List[float], top_k: int) -> List[Dict]:
        """Search entities by vector similarity"""
        with self.driver.session() as session:
            result = session.run("""
                CALL db.index.vector.queryNodes(
                    'entity_embeddings',
                    $top_k,
                    $query_vector
                ) YIELD node, score

                RETURN 
                    node.name as name,
                    node.type as type,
                    score
                ORDER BY score DESC
            """, top_k=top_k, query_vector=query_vector)

            return [dict(record) for record in result]

    def _get_entities_from_chunks(self, chunk_ids: List[str], limit: int = 10) -> List[Dict]:
        """Get entities mentioned in chunks (LIMITED to most relevant)"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
                WHERE c.chunk_id IN $chunk_ids
                
                // Count how many chunks mention each entity
                WITH e, count(DISTINCT c) as mention_count
                
                // Prioritize entities mentioned in multiple chunks
                RETURN DISTINCT
                    e.name as name,
                    e.type as type,
                    mention_count
                ORDER BY mention_count DESC
                LIMIT $limit
            """, chunk_ids=chunk_ids, limit=limit)

            return [{'name': record['name'], 'type': record['type']} for record in result]

    def _expand_graph_context(self, entity_names: List[str], hops: int) -> Dict:
        """Expand graph from entities"""
        with self.driver.session() as session:
            if hops == 1:
                # Single hop - can use simple relationship
                result = session.run("""
                    MATCH (e:Entity)
                    WHERE e.name IN $entity_names
                    
                    // Get direct neighbors
                    OPTIONAL MATCH (e)-[r:RELATIONSHIP]-(neighbor:Entity)
                    
                    RETURN 
                        COLLECT(DISTINCT neighbor.name) as neighbors,
                        COLLECT(DISTINCT {
                            from: startNode(r).name,
                            type: type(r),
                            to: endNode(r).name
                        }) as relationships
                """, entity_names=entity_names)
            else:
                # Multi-hop - need to handle path
                result = session.run(f"""
                    MATCH (e:Entity)
                    WHERE e.name IN $entity_names
                    
                    // Get neighbors within hops distance
                    OPTIONAL MATCH path = (e)-[*1..{hops}]-(neighbor:Entity)
                    WHERE neighbor.name IS NOT NULL
                    
                    // Unwind relationships in path
                    WITH DISTINCT neighbor, relationships(path) as rels
                    UNWIND rels as rel
                    
                    RETURN 
                        COLLECT(DISTINCT neighbor.name) as neighbors,
                        COLLECT(DISTINCT {{
                            from: startNode(rel).name,
                            type: type(rel),
                            to: endNode(rel).name
                        }}) as relationships
                """, entity_names=entity_names)

            record = result.single()
            if record:
                return {
                    'neighbors': [n for n in record['neighbors'] if n],
                    'relationships': [r for r in record['relationships'] if r and r.get('from')]
                }

            return {'neighbors': [], 'relationships': []}

    def _get_chunks_for_entities(self, entity_names: List[str]) -> List[Dict]:
        """Get chunks that mention entities"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
                WHERE e.name IN $entity_names
                
                RETURN DISTINCT
                    c.chunk_id as chunk_id,
                    c.text as text
                
                LIMIT 5
            """, entity_names=entity_names)

            return [dict(record) for record in result]


# Test the retriever
if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python hybrid_retriever.py <query>")
        sys.exit(1)

    query = ' '.join(sys.argv[1:])

    retriever = HybridRetriever()
    try:
        print(f"\n{'='*60}")
        print(f"Hybrid Retrieval Test")
        print(f"{'='*60}\n")

        context = retriever.retrieve(query)

        print(f"\n{'='*60}")
        print("Retrieved Context:")
        print(f"{'='*60}")
        print(f"\nChunks: {len(context['chunks'])}")
        print(f"Entities: {len(context['entities'])}")
        print(f"Relationships: {len(context['relationships'])}")

        print("\n--- Entities ---")
        for entity in context['entities'][:10]:
            print(f"  • {entity}")

        print("\n--- Sample Relationships ---")
        for rel in context['relationships'][:5]:
            print(f"  • {rel['from']} --[{rel['type']}]--> {rel['to']}")

        print("\n--- Sample Chunk ---")
        if context['chunks']:
            print(f"  {context['chunks'][0]['text'][:200]}...")

        print(f"\n{'='*60}\n")

    finally:
        retriever.close()