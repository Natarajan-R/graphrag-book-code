"""
Query Engine Module - GRAPH-FIRST ARCHITECTURE (HYBRID)
Combines Graph Structures with Vector Search for high-accuracy RAG.
"""

from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import json

# LangChain Imports
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_neo4j import Neo4jGraph
from langchain_core.prompts import PromptTemplate

# Local Imports
from config import get_config
from utils import (
    setup_logger,
    timing_decorator,
    print_banner,
    print_section,
    print_success,
    print_error,
    print_info,
    print_warning,
    retry_on_failure
)


class QueryEngine:
    """
    Advanced query engine with Graph-First architecture.
    
    Strategy:
    1. Vector Search -> Find semantic entities (Skeleton)
    2. Graph Traversal -> Find relationships & contradictions (Structure)
    3. Vector Search -> Find text chunks (Flesh)
    4. Synthesis -> LLM combines Graph Truth + Text Details
    """

    def __init__(self):
        """Initialize the query engine"""
        self.config = get_config()

        # Setup logger
        log_file = self.config.paths.logs_dir / f"query_{datetime.now().strftime('%Y%m%d')}.log"
        self.logger = setup_logger("QueryEngine", log_file)

        # Initialize LLM & Embeddings
        self._initialize_ai()

        # Initialize Neo4j
        self._initialize_neo4j()

        # Query history tracking
        self.history_file = self.config.paths.logs_dir / "query_history.json"
        self.query_history = self._load_history()

        self.logger.info("QueryEngine initialized with Hybrid Graph-First architecture")

    @retry_on_failure(max_retries=3, delay=2.0)
    def _initialize_ai(self):
        """Initialize LLM and Embedding models"""
        try:
            # 1. Generation Model
            self.llm = ChatOllama(
                model=self.config.llm.model,
                temperature=self.config.llm.temperature,
                base_url=self.config.llm.base_url
            )
            
            # 2. Embedding Model (Crucial for Vector Search)
            # Hardcoded to match build_graph.py, or read from config if available
            self.embeddings = OllamaEmbeddings(
                model="nomic-embed-text", 
                base_url=self.config.llm.base_url
            )
            
            self.logger.info(f"AI Models initialized: {self.config.llm.model} + nomic-embed-text")
        except Exception as e:
            self.logger.error(f"Failed to initialize AI models: {e}")
            raise

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

    def _load_history(self) -> List[Dict]:
        """Load query history from disk"""
        if self.history_file.exists():
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def _save_history(self):
        """Save query history to disk"""
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(self.query_history, f, indent=2, ensure_ascii=False)

    def _add_to_history(self, question: str, answer: str, context_stats: Dict):
        """Add query to history"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "context_stats": context_stats
        }
        self.query_history.append(entry)
        self._save_history()

    # ===================================================================
    # STEP 1: VECTOR-BASED ENTITY EXTRACTION (The Fix)
    # ===================================================================

    def _detect_entities_by_vector(self, question: str, limit: int = 5) -> List[str]:
        """
        Find entities using semantic vector search.
        Much more robust than string matching or LLM extraction.
        """
        try:
            query_vector = self.embeddings.embed_query(question)
            
            # Note: Threshold set to 0.5 to capture relevant entities even if fuzzy
            query = """
            CALL db.index.vector.queryNodes('entity_embeddings', $k, $embedding)
            YIELD node, score
            WHERE score > 0.5
            RETURN DISTINCT node.id as name
            """
            
            results = self.graph.query(query, {
                "k": limit,
                "embedding": query_vector
            })
            
            entities = [r['name'] for r in results]
            return entities
            
        except Exception as e:
            self.logger.error(f"Vector entity detection failed: {e}")
            return []

    # ===================================================================
    # STEP 2: GRAPH TRAVERSAL & CONTRADICTIONS
    # ===================================================================

    def _get_explicit_contradictions(self, entity_names: List[str]) -> List[Dict]:
        """Find explicit Contradiction nodes linked to these entities"""
        if not entity_names:
            return []

        query = """
        MATCH (e:Entity)-[:INVOLVED_IN]->(c:Contradiction)
        WHERE e.id IN $names
        OPTIONAL MATCH (c)<-[:INVOLVED_IN]-(other:Entity)
        WHERE elementId(other) <> elementId(e)
        RETURN
            c.id as id,
            c.reason as reason,
            e.id as primary_entity,
            collect(other.id) as other_entities
        LIMIT 5
        """
        
        try:
            results = self.graph.query(query, {"names": entity_names})
            formatted = []
            for r in results:
                others = ", ".join(r['other_entities'])
                pair = f"{r['primary_entity']} & {others}" if others else r['primary_entity']
                formatted.append({
                    "entity_pair": pair,
                    "reason": r['reason'],
                    "is_explicit": True
                })
            return formatted
        except Exception as e:
            self.logger.error(f"Explicit contradiction query failed: {e}")
            return []

    def _query_graph_facts(self, entity_names: List[str]) -> List[Dict]:
        """Get structured relationships between found entities"""
        if not entity_names:
            return []
            
        query = """
        MATCH (e1:Entity)-[r:RELATIONSHIP]->(e2:Entity)
        WHERE e1.id IN $names OR e2.id IN $names
        RETURN
            e1.id as source,
            type(r) as relationship_type,
            r.type as relationship_subtype,
            r.nature as nature,
            r.version as version,
            e2.id as target
        LIMIT 20
        """
        try:
            return self.graph.query(query, {"names": entity_names})
        except Exception as e:
            self.logger.error(f"Graph fact query failed: {e}")
            return []

    # ===================================================================
    # STEP 3: TEXT CHUNK RETRIEVAL (The Missing Piece Restored)
    # ===================================================================

    def _retrieve_text_chunks(self, question: str, limit: int = 3) -> List[Dict]:
        """
        Retrieve source text chunks using vector search.
        Provides the 'Flesh' to the Graph's 'Skeleton'.
        """
        try:
            query_vector = self.embeddings.embed_query(question)
            
            query = """
            CALL db.index.vector.queryNodes('chunk_embeddings', $k, $embedding)
            YIELD node, score
            RETURN node.text as text, score
            """
            
            results = self.graph.query(query, {
                "k": limit,
                "embedding": query_vector
            })
            
            return results
        except Exception as e:
            self.logger.error(f"Chunk retrieval failed: {e}")
            return []

    # ===================================================================
    # STEP 4: ANSWER GENERATION (The "Synthesis" Prompt)
    # ===================================================================

    def _generate_answer(
        self, 
        question: str, 
        facts: List[Dict], 
        chunks: List[Dict], 
        contradictions: List[Dict]
    ) -> str:
        """
        Generate answer synthesizing Graph Facts AND Text Chunks.
        Prioritizes Graph for structure/truth, Text for detail.
        """
        
        # 1. Format Contradictions (Critical for your book)
        contra_text = ""
        if contradictions:
            contra_text = "\n⚠️ CONTRADICTION WARNING:\n"
            for c in contradictions:
                contra_text += f"- Conflict between {c['entity_pair']}: {c['reason']}\n"

        # 2. Format Graph Facts
        facts_text = "No direct relationships found in graph."
        if facts:
            facts_text = "\n".join([
                f"- {f['source']} --[{f.get('relationship_subtype', 'RELATED')}]--> {f['target']} (Context: {f.get('nature', 'General')})"
                for f in facts
            ])

        # 3. Format Text Chunks
        chunks_text = "No supporting text found."
        if chunks:
            chunks_text = "\n\n".join([
                f"Source Fragment: {c['text'][:400]}..." 
                for c in chunks
            ])

        # 4. Construct Prompt
        # This is the "Relaxed" prompt that allows synthesis
        prompt = f"""You are a Knowledge Graph Assistant. Answer the question using the provided context.

STRATEGY:
1. Use the **KNOWLEDGE GRAPH FACTS** to understand the specific entities and how they are related. This is your source of truth for structure.
2. Use the **SUPPORTING TEXT** to fill in details, explanations, and nuance that the graph might miss.
3. If there are **CONTRADICTIONS**, explicitly mention them to the user.
4. Synthesize all sources into a smooth, natural answer.

{contra_text}

=== KNOWLEDGE GRAPH FACTS ===
{facts_text}

=== SUPPORTING TEXT ===
{chunks_text}

QUESTION: {question}

ANSWER:"""

        try:
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            self.logger.error(f"LLM Generation failed: {e}")
            return "I'm sorry, I encountered an error while generating the answer."

    # ===================================================================
    # MAIN PIPELINE
    # ===================================================================

    @timing_decorator
    def query_natural_language(self, question: str, verbose: bool = False) -> Dict[str, Any]:
        """
        Execute the full Graph-First Hybrid Pipeline
        """
        self.logger.info(f"Processing: {question}")
        
        try:
            # 1. Vector Entity Detection
            if verbose: print_info("Step 1: Vector Entity Search...")
            entities = self._detect_entities_by_vector(question)
            if verbose: print(f"  Found: {entities}")

            # 2. Graph Retrieval
            if verbose: print_info("Step 2: Graph Traversal...")
            facts = self._query_graph_facts(entities)
            contradictions = self._get_explicit_contradictions(entities)
            
            # 3. Chunk Retrieval (The missing piece restored)
            if verbose: print_info("Step 3: Text Chunk Retrieval...")
            chunks = self._retrieve_text_chunks(question)

            # 4. Synthesis
            if verbose: print_info("Step 4: Answer Generation...")
            answer = self._generate_answer(question, facts, chunks, contradictions)

            # 5. History & Result
            stats = {
                "entities_found": len(entities),
                "facts_found": len(facts),
                "chunks_found": len(chunks),
                "contradictions": len(contradictions)
            }
            
            self._add_to_history(question, answer, stats)
            
            return {
                "question": question,
                "answer": answer,
                "stats": stats,
                "has_contradictions": len(contradictions) > 0
            }

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            return {"answer": f"Error: {e}", "success": False}

    # ===================================================================
    # UTILITY METHODS (Kept from original)
    # ===================================================================

    def get_query_history(self, limit: int = 10) -> List[Dict]:
        return self.query_history[-limit:]

    def clear_history(self):
        self.query_history = []
        self._save_history()
        print_success("Query history cleared")

    def interactive_mode(self):
        """Start interactive query mode"""
        print_banner("GraphRAG Interactive Query Mode (Hybrid)")
        print_info("Type 'quit', 'exit' to end, 'history' for logs")
        print_warning("⚠️  Graph-First mode: Graph logic is prioritized over raw text\n")

        while True:
            try:
                question = input("\n🔍 Question: ").strip()
                if not question: continue
                if question.lower() in ['quit', 'exit']:
                    print_success("Goodbye!")
                    break
                if question.lower() == 'history':
                    self._display_history()
                    continue

                result = self.query_natural_language(question, verbose=True)

                print_section("Answer")
                print(result["answer"])
                
                if result.get("has_contradictions"):
                    print_warning("⚠️  Contradictions detected in this answer.")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print_error(f"Error: {e}")

    def _display_history(self):
        history = self.get_query_history(10)
        if not history:
            print_info("No query history")
            return
        print_section("Recent Queries")
        for i, entry in enumerate(reversed(history), 1):
            print(f"\n{i}. {entry['question']}")
            print(f"   Answer: {entry['answer'][:100]}...")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="GraphRAG Query Engine")
    parser.add_argument("-q", "--question", help="Question to ask")
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--history", action="store_true", help="Show history")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    engine = QueryEngine()

    if args.interactive:
        engine.interactive_mode()
    elif args.question:
        result = engine.query_natural_language(args.question, verbose=args.verbose)
        print_section("Answer")
        print(result["answer"])
    elif args.history:
        engine._display_history()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()