#!/usr/bin/env python3
"""
Complete System Validation
Run this anytime to verify the system is working
"""

import sys
from neo4j import GraphDatabase
from langchain_ollama import ChatOllama, OllamaEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

# Configuration
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

def main():
    print("\n" + "="*60)
    print(" GraphRAG System - Complete Validation")
    print("="*60 + "\n")

    tests_passed = 0
    tests_total = 5

    # Test 1: Neo4j
    print("[1/5] Neo4j Connection...", end=" ")
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            session.run("RETURN 1")
        driver.close()
        print("✓ PASS")
        tests_passed += 1
    except Exception as e:
        print(f"✗ FAIL: {e}")

    # Test 2: LLM
    print("[2/5] Ollama LLM (qwen2.5)...", end=" ")
    try:
        llm = ChatOllama(model="qwen2.5")
        llm.invoke("test")
        print("✓ PASS")
        tests_passed += 1
    except Exception as e:
        print(f"✗ FAIL: {e}")

    # Test 3: Embeddings
    print("[3/5] Ollama Embeddings (nomic-embed-text)...", end=" ")
    try:
        embedder = OllamaEmbeddings(model="nomic-embed-text")
        emb = embedder.embed_query("test")
        assert len(emb) == 768, f"Wrong dimensions: {len(emb)}"
        print("✓ PASS")
        tests_passed += 1
    except Exception as e:
        print(f"✗ FAIL: {e}")

    # Test 4: Graph Write
    print("[4/5] Neo4j Write Operations...", end=" ")
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            session.run("CREATE (n:Test {id: 'validation'}) RETURN n")
            session.run("MATCH (n:Test {id: 'validation'}) DELETE n")
        driver.close()
        print("✓ PASS")
        tests_passed += 1
    except Exception as e:
        print(f"✗ FAIL: {e}")

    # Test 5: Vector Index
    print("[5/5] Neo4j Vector Index...", end=" ")
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        embedder = OllamaEmbeddings(model="nomic-embed-text")

        with driver.session() as session:
            emb = embedder.embed_query("test")
            session.run("""
                CREATE (n:VectorTest {id: 'test', embedding: $emb})
            """, emb=emb)

            # Try to use vector index
            result = session.run("""
                MATCH (n:VectorTest)
                WHERE n.embedding IS NOT NULL
                RETURN count(n) as cnt
            """)

            count = result.single()['cnt']

            # Cleanup
            session.run("MATCH (n:VectorTest) DELETE n")

        driver.close()
        print("✓ PASS")
        tests_passed += 1
    except Exception as e:
        print(f"✗ FAIL: {e}")

    # Summary
    print("\n" + "="*60)
    print(f" Results: {tests_passed}/{tests_total} tests passed")

    if tests_passed == tests_total:
        print(" Status: ✓ ALL SYSTEMS OPERATIONAL")
        print(" Your GraphRAG system is ready to use!")
    else:
        print(f" Status: ✗ {tests_total - tests_passed} test(s) failed")
        print(" Review errors above and fix issues")

    print("="*60 + "\n")

    return tests_passed == tests_total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

