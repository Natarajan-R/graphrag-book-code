"""
answer_generator.py
LLM-based answer generation with sources
"""




import sys
import os

# Add the parent directory (graphragsystem) to sys.path
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
#from config import get_config
from typing import Dict, List,Any
from dotenv import load_dotenv

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

    def generate_graph_first_answer(
            self, 
            question: str, 
            graph_facts: Dict, 
            supporting_chunks: List[Dict]
        ) -> Dict[str, Any]:
            """
            Graph-First Generation Strategy.
            Returns a structured Dictionary containing the answer, sources, and stats.
            """
            
            # 1. Format Graph Facts for the Prompt
            fact_str = ""
            if graph_facts.get('entities'):
                fact_str += "Key Entities:\n" + "\n".join([f"- {e['name']} ({e['type']})" for e in graph_facts['entities']]) + "\n"
            if graph_facts.get('relationships'):
                fact_str += "Key Relationships:\n" + "\n".join([f"- {r['from']} --[{r['type']}]--> {r['to']}" for r in graph_facts['relationships']])
            
            # 2. Format Supporting Text for the Prompt
            text_str = ""
            used_chunks = supporting_chunks[:3] # Limit to top 3 for prompt
            for i, chunk in enumerate(used_chunks, 1):
                text_str += f"Source {i}: {chunk.get('text', '')[:400]}\n\n"

            # 3. Construct the Graph-First Prompt
            prompt_text = f"""You are a Knowledge Graph Assistant. Answer the question using the provided context.

    STRATEGY:
    1. Use the **KNOWLEDGE GRAPH FACTS** to understand the specific entities and how they are related. This is your source of truth for structure.
    2. Use the **SUPPORTING TEXT** to fill in details, explanations, and nuance.
    3. Synthesize both sources. If the Text contradicts the Graph Relationships, trust the Graph.

    KNOWLEDGE GRAPH FACTS:
    {fact_str}

    SUPPORTING TEXT:
    {text_str}

    QUESTION:
    {question}

    ANSWER:"""

            # 4. Invoke LLM
            response = self.llm.invoke(prompt_text)
            answer_text = response.content if hasattr(response, 'content') else str(response)

            # 5. Construct and Return the Complete Result Dictionary
            result = {
                'query': question,
                'answer': answer_text,
                'approach': 'graph-first',
                'sources': {
                    'graph_facts': graph_facts,
                    'chunks': used_chunks
                },
                'context_used': {
                    'entities': len(graph_facts.get('entities', [])),
                    'relationships': len(graph_facts.get('relationships', [])),
                    'chunks': len(used_chunks)
                }
            }
            
            return result

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