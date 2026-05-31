#!/usr/bin/env python3
"""
Complete system integration test
Tests: Neo4j + Ollama LLM + Ollama Embeddings + Graph Write
"""

from neo4j import GraphDatabase
from langchain_ollama import ChatOllama, OllamaEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

def test_complete_integration():
    print("\n" + "="*60)
    print(" GraphRAG System Integration Test")
    print("="*60 + "\n")

    # Test 1: Neo4j Connection
    print("[1/5] Testing Neo4j connection...", end=" ")
    try:
        driver = GraphDatabase.driver(
            os.getenv('NEO4J_URI'),
            auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD'))
        )
        with driver.session() as session:
            session.run("RETURN 1")
        print("✓ PASS")
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False

    # Test 2: Ollama LLM
    print("[2/5] Testing Ollama LLM...", end=" ")
    try:
        llm = ChatOllama(model=os.getenv('OLLAMA_LLM_MODEL'))
        llm.invoke("test")
        print("✓ PASS")
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False

    # Test 3: Ollama Embeddings
    print("[3/5] Testing Ollama Embeddings...", end=" ")
    try:
        embedder = OllamaEmbeddings(model=os.getenv('OLLAMA_EMBED_MODEL'))
        emb = embedder.embed_query("test")
        assert len(emb) == 768, f"Wrong dimensions: {len(emb)}"
        print("✓ PASS")
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False

    # Test 4: Graph Write with Embedding
    print("[4/5] Testing Neo4j write with embedding...", end=" ")
    try:
        with driver.session() as session:
            # Create node with embedding
            session.run("""
                CREATE (n:TestNode {
                    id: 'integration_test',
                    text: 'Test node',
                    embedding: $embedding
                })
            """, embedding=emb)
            
            # Verify it was created
            result = session.run("""
                MATCH (n:TestNode {id: 'integration_test'})
                RETURN n.text as text, size(n.embedding) as dim
            """)
            record = result.single()
            assert record['text'] == 'Test node'
            assert record['dim'] == 768
            
            # Clean up
            session.run("MATCH (n:TestNode {id: 'integration_test'}) DELETE n")
        print("✓ PASS")
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False

    # Test 5: LLM + Graph Integration
    print("[5/5] Testing LLM + Graph integration...", end=" ")
    try:
        # Generate some text with LLM
        response = llm.invoke("Extract entity: 'John works at Microsoft'")
        
        # Embed it
        emb = embedder.embed_query(response.content)
        
        # Store in graph
        with driver.session() as session:
            session.run("""
                CREATE (n:TestEntity {
                    id: 'test_entity',
                    llm_output: $text,
                    embedding: $embedding
                })
            """, text=response.content, embedding=emb)
            
            # Clean up
            session.run("MATCH (n:TestEntity {id: 'test_entity'}) DELETE n")
        print("✓ PASS")
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False
    finally:
        driver.close()

    print("\n" + "="*60)
    print(" ✓ ALL TESTS PASSED")
    print(" Your GraphRAG system is ready!")
    print("="*60 + "\n")
    
    return True

if __name__ == "__main__":
    import sys
    success = test_complete_integration()
    sys.exit(0 if success else 1)

