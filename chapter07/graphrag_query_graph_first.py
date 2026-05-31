"""
graphrag_query_graph_first.py
Graph-First GraphRAG query system
"""

from hybrid_retriever import HybridRetriever
from answer_generator import AnswerGenerator
from neo4j import GraphDatabase
from dotenv import load_dotenv
from typing import Dict, List
import json
import os

load_dotenv()

# Configuration
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')


class GraphFirstQuery:
    """Graph-First GraphRAG query system"""

    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        self.retriever = HybridRetriever()
        self.generator = AnswerGenerator()

    def close(self):
        self.driver.close()
        self.retriever.close()

    def query(
        self,
        question: str,
        chunk_k: int = 3,
        entity_k: int = 3,
        verbose: bool = True
    ) -> Dict:
        """
        Graph-First query pipeline
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"GraphRAG Query (Graph-First)")
            print(f"{'='*60}\n")
            print(f"Question: {question}\n")

        # STEP 1: Extract key entities
        if verbose: print("[1/4] Extracting query entities...")
        query_entities = self._extract_query_entities(question)
        
        # STEP 2: Query graph for FACTS
        if verbose: print("[2/4] Extracting facts from knowledge graph...")
        graph_facts = self._extract_graph_facts(query_entities)
        
        # STEP 3: Get supporting text
        if verbose: print("[3/4] Retrieving supporting text chunks...")
        context = self.retriever.retrieve(
            query=question,
            chunk_k=chunk_k,
            entity_k=entity_k,
            expand_hops=1
        )

        # STEP 4: Generate answer (Returns FULL DICT now)
        if verbose: print("[4/4] Generating answer from structured facts...")
        
        result = self.generator.generate_graph_first_answer(
            question=question,
            graph_facts=graph_facts,
            supporting_chunks=context['chunks']
        )

        # Verbose Output (Using the returned dictionary)
        if verbose:
            print(f"\n{'='*60}")
            print("Answer:")
            print(f"{'='*60}\n")
            print(result['answer'])
            print(f"\n{'='*60}")
            print("Context Statistics:")
            print(f"{'='*60}")
            print(f"  Graph Facts: {result['context_used']['entities']} Entities, {result['context_used']['relationships']} Relationships")
            print(f"  Text Chunks: {result['context_used']['chunks']} Chunks")
            print(f"{'='*60}\n")

        return result

    def _extract_query_entities(self, question: str) -> List[str]:
        # (Same implementation as before - vector search for entities)
        query_vector = self.retriever.embedder.embed_query(question)
        with self.driver.session() as session:
            result = session.run("""
                CALL db.index.vector.queryNodes('entity_embeddings', 5, $query_vector) 
                YIELD node, score
                WHERE score > 0.5
                RETURN node.name as name
            """, query_vector=query_vector)
            return [record['name'] for record in result]

    def _extract_graph_facts(self, entity_names: List[str]) -> Dict:
        # (Same implementation as before - matching entities & relationships)
        if not entity_names:
            return {'entities': [], 'relationships': []}
        with self.driver.session() as session:
            entities = session.run("""
                MATCH (e:Entity) WHERE e.name IN $names
                RETURN e.name as name, e.type as type
            """, names=entity_names).data()
            relationships = session.run("""
                MATCH (e1:Entity)-[r:RELATIONSHIP]->(e2:Entity)
                WHERE e1.name IN $names OR e2.name IN $names
                RETURN e1.name as from, type(r) as type, e2.name as to
                LIMIT 20
            """, names=entity_names).data()
            return {'entities': entities, 'relationships': relationships}

# Command-line interface
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python graphrag_query_graph_first.py <question>")
        sys.exit(1)
    
    question = ' '.join(sys.argv[1:])
    system = GraphFirstQuery()
    try:
        result = system.query(question)
        # Save JSON output
        with open("query_result_graph_first.json", 'w') as f:
            json.dump(result, f, indent=2)
        print("Result saved to query_result_graph_first.json")
    finally:
        system.close()
        