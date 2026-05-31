"""
answer_generator.py
LLM-based answer generation with sources
"""

from typing import Dict, List
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os

load_dotenv()

# Configuration
LLM_MODEL = os.getenv('OLLAMA_LLM_MODEL', 'qwen2.5')


class AnswerGenerator:
    """Generate answers using retrieved context"""

    def __init__(self):
        self.llm = ChatOllama(
            model=LLM_MODEL,
            temperature=0.0  # Set to 0 for maximum strictness
        )

    def generate_answer(
        self,
        query: str,
        context: Dict
    ) -> Dict:
        """
        Generate answer from query and context
        """
        # Build prompt
        prompt = self._build_prompt(query, context)

        # Generate answer
        print("  Generating answer...", end=" ")
        try:
            response = self.llm.invoke(prompt)
            answer = response.content
        except Exception as e:
            answer = f"Error generating answer: {str(e)}"
        print("✓")

        # Extract sources
        sources = self._extract_sources(context)

        return {
            'query': query,
            'answer': answer,
            'sources': sources,
            'context_used': {
                'chunks': len(context.get('chunks', [])),
                'entities': len(context.get('entities', [])),
                'relationships': len(context.get('relationships', []))
            }
        }

    def _build_prompt(self, query: str, context: Dict) -> str:
        """Build a strict LLM prompt with clear separators"""

        # Format chunks (Limit text to avoid context overflow)
        chunks_list = context.get('chunks', [])[:5]
        chunks_text = "\n\n".join([
            f"--- Document Excerpt {i+1} ---\n{chunk.get('text', '')}"
            for i, chunk in enumerate(chunks_list)
        ])

        # Format entities
        entities_list = context.get('entities', [])[:15]
        entities_text = ", ".join(entities_list)

        # Format relationships
        rels_list = context.get('relationships', [])[:15]
        relationships_text = "\n".join([
            f"- {rel['from']} --[{rel['type']}]--> {rel['to']}"
            for rel in rels_list
        ])

        # --- THE FIX: IMPROVED PROMPT STRUCTURE ---
        prompt = f"""You are a helpful research assistant. Your goal is to answer the specific question asked by the user based ONLY on the provided context.

### SYSTEM INSTRUCTIONS
1. Answer the User Question below directly.
2. Do NOT generate new questions.
3. Do NOT make up examples.
4. Use the Context Data (Text and Graph) to support your answer.
5. If the answer is not in the context, state: "I cannot answer this based on the provided context."

### CONTEXT DATA
[Text Chunks]
{chunks_text}

[Knowledge Graph Entities]
{entities_text}

[Knowledge Graph Relationships]
{relationships_text}

### USER QUESTION
{query}

### YOUR ANSWER
"""
        return prompt

    def _extract_sources(self, context: Dict) -> Dict:
        """Extract source information"""
        return {
            'chunks': [
                {'chunk_id': c.get('chunk_id'), 'text': c.get('text', '')[:100] + '...'}
                for c in context.get('chunks', [])[:5]
            ],
            'entities': context.get('entities', [])[:10],
            'relationships': context.get('relationships', [])[:10]
        }


# Test the generator
if __name__ == "__main__":
    import sys
    from hybrid_retriever import HybridRetriever

    if len(sys.argv) < 2:
        print("Usage: python answer_generator.py <query>")
        sys.exit(1)

    query = ' '.join(sys.argv[1:])

    # Retrieve context
    print(f"Querying: {query}")
    retriever = HybridRetriever()
    context = retriever.retrieve(query)
    retriever.close()

    # Generate answer
    generator = AnswerGenerator()
    result = generator.generate_answer(query, context)

    print(f"\n{'='*60}")
    print("Answer:")
    print(f"{'='*60}\n")
    print(result['answer'])
    print(f"\n{'='*60}")
    print("Sources:")
    print(f"{'='*60}")
    print(f"Chunks used: {result['context_used']['chunks']}")
    print(f"Entities: {result['context_used']['entities']}")
    print(f"Relationships: {result['context_used']['relationships']}")
    print(f"{'='*60}\n")