"""
graphrag_query.py
Complete GraphRAG query system
"""

from hybrid_retriever import HybridRetriever
from answer_generator import AnswerGenerator
from typing import Dict
import json


class GraphRAGQuery:
    """Complete GraphRAG query system"""

    def __init__(self):
        self.retriever = HybridRetriever()
        self.generator = AnswerGenerator()

    def close(self):
        self.retriever.close()

    def query(
        self,
        question: str,
        chunk_k: int = 3,
        entity_k: int = 3,
        expand_hops: int = 1,
        verbose: bool = True
    ) -> Dict:
        """
        Complete query pipeline

        Args:
            question: Natural language question
            chunk_k: Chunks to retrieve
            entity_k: Entities to retrieve
            expand_hops: Graph expansion distance
            verbose: Print progress

        Returns:
            Complete result with answer and sources
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"GraphRAG Query")
            print(f"{'='*60}\n")
            print(f"Question: {question}\n")

        # Step 1: Retrieve context
        if verbose:
            print("Retrieving context...")
        
        context = self.retriever.retrieve(
            query=question,
            chunk_k=chunk_k,
            entity_k=entity_k,
            expand_hops=expand_hops
        )

        # Step 2: Generate answer
        if verbose:
            print("\nGenerating answer...")

        result = self.generator.generate_answer(question, context)

        if verbose:
            print(f"\n{'='*60}")
            print("Answer:")
            print(f"{'='*60}\n")
            print(result['answer'])
            print(f"\n{'='*60}")
            print("Context Used:")
            print(f"{'='*60}")
            print(f"  Chunks: {result['context_used']['chunks']}")
            print(f"  Entities: {result['context_used']['entities']}")
            print(f"  Relationships: {result['context_used']['relationships']}")
            print(f"{'='*60}\n")

        return result


# Command-line interface
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python graphrag_query.py <question>")
        print("Example: python graphrag_query.py 'What did Dr. Martinez develop?'")
        sys.exit(1)

    question = ' '.join(sys.argv[1:])

    # Run query
    system = GraphRAGQuery()
    try:
        result = system.query(question)

        # Save to file
        output_file = "query_result.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Full result saved to: {output_file}")

    finally:
        system.close()
        