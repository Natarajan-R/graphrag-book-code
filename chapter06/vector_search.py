"""
vector_search.py
Vector similarity search in Neo4j
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


class VectorSearcher:
    """Vector similarity search in Neo4j"""

    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        self.embedder = OllamaEmbeddings(model=EMBED_MODEL)

    def close(self):
        self.driver.close()

    def search_chunks(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Search for similar chunks using vector similarity

        Args:
            query: Natural language query
            top_k: Number of results to return

        Returns:
            List of chunks with similarity scores
        """
        # Generate query embedding
        query_vector = self.embedder.embed_query(query)

        # Search vector index
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
                    node.word_count as word_count,
                    score

                ORDER BY score DESC
            """, top_k=top_k, query_vector=query_vector)

            chunks = []
            for record in result:
                chunks.append({
                    'chunk_id': record['chunk_id'],
                    'text': record['text'],
                    'word_count': record['word_count'],
                    'score': record['score']
                })

            return chunks

    def search_entities(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Search for similar entities using vector similarity

        Args:
            query: Natural language query
            top_k: Number of results to return

        Returns:
            List of entities with similarity scores
        """
        # Generate query embedding
        query_vector = self.embedder.embed_query(query)

        # Search vector index
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

            entities = []
            for record in result:
                entities.append({
                    'name': record['name'],
                    'type': record['type'],
                    'score': record['score']
                })

            return entities

    def search_hybrid(
        self,
        query: str,
        chunk_k: int = 3,
        entity_k: int = 3
    ) -> Dict:
        """
        Hybrid search: chunks + entities

        Args:
            query: Natural language query
            chunk_k: Top chunks to retrieve
            entity_k: Top entities to retrieve

        Returns:
            Dict with chunks and entities
        """
        chunks = self.search_chunks(query, chunk_k)
        entities = self.search_entities(query, entity_k)

        return {
            'chunks': chunks,
            'entities': entities
        }


# Test the searcher
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python vector_search.py <query>")
        sys.exit(1)

    query = ' '.join(sys.argv[1:])

    searcher = VectorSearcher()
    try:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}\n")

        # Search chunks
        print("Similar Chunks:")
        chunks = searcher.search_chunks(query, top_k=3)
        for i, chunk in enumerate(chunks, 1):
            print(f"\n{i}. Score: {chunk['score']:.3f}")
            print(f"   Text: {chunk['text'][:100]}...")

        # Search entities
        print(f"\n{'='*60}")
        print("Similar Entities:")
        entities = searcher.search_entities(query, top_k=3)
        for i, entity in enumerate(entities, 1):
            print(f"\n{i}. {entity['name']} ({entity['type']})")
            print(f"   Score: {entity['score']:.3f}")

        print(f"\n{'='*60}\n")

    finally:
        searcher.close()
    