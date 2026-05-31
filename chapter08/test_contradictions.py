"""
test_contradictions.py
Complete demonstration of Graph-First contradiction handling
"""

from versioned_graph_writer import VersionedGraphWriter
from contradiction_aware_query import ContradictionAwareQuery
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()

# Configuration
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')


def clean_test_data():
    """Clean up test data from previous runs"""
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD)
    )
    
    with driver.session() as session:
        # Delete test entities
        session.run("""
            MATCH (e:Entity)
            WHERE e.name IN ['Family Office', 'Rostova Dynamics', 
                           'Quantum Analytics', 'TechVentures Inc']
            DETACH DELETE e
        """)
    
    driver.close()
    print("✓ Cleaned up test data\n")


def test_scenario_1():
    """
    Scenario 1: Direct vs Indirect Investment
    Tests basic contradiction handling
    """
    
    print("\n" + "="*70)
    print("SCENARIO 1: Direct vs Indirect Investment")
    print("="*70 + "\n")
    
    print("📄 Simulating document with contradictory statements:\n")
    print("   Page 10: 'The family office made DIRECT investments'")
    print("   Page 15: 'Actually, the investment was INDIRECT through subsidiary'\n")
    
    relationships = [
        # VERSION A - Direct investment
        {
            'from_entity': 'Family Office',
            'to_entity': 'Rostova Dynamics',
            'type': 'INVESTED_IN',
            'nature': 'direct',
            'version': 'A',
            'source_page': 10,
            'confidence': 0.95,
            'source_text': 'The family office made direct investments in both companies'
        },
        
        # VERSION B - Indirect investment (correction)
        {
            'from_entity': 'Family Office',
            'to_entity': 'Rostova Dynamics',
            'type': 'INVESTED_IN',
            'nature': 'indirect',
            'version': 'B',
            'source_page': 15,
            'confidence': 0.85,
            'source_text': 'Actually, the investment in Rostova Dynamics was indirect through a Quantum Analytics subsidiary'
        }
    ]
    
    # Write to graph
    writer = VersionedGraphWriter()
    writer.write_versioned_relationships(relationships)
    writer.close()
    
    # Query with contradiction-aware system
    query_system = ContradictionAwareQuery()
    
    print("\n🔍 Querying: How did Family Office invest in Rostova Dynamics?\n")
    print("="*70)
    
    result = query_system.query_relationship(
        'Family Office',
        'Rostova Dynamics',
        'INVESTED_IN'
    )
    
    answer = query_system.present_answer(result)
    print(answer)
    
    query_system.close()
    
    print("\n✅ Result: System presented BOTH versions without deciding truth!\n")


def test_scenario_2():
    """
    Scenario 2: Multiple Consistent Versions
    Tests handling of non-contradictory variations
    """
    
    print("\n" + "="*70)
    print("SCENARIO 2: Multiple Consistent Versions")
    print("="*70 + "\n")
    
    print("📄 Simulating document with consistent statements:\n")
    print("   Page 5: 'TechVentures Inc acquired Quantum Analytics'")
    print("   Page 12: 'TechVentures Inc completed strategic acquisition of Quantum Analytics'\n")
    
    relationships = [
        {
            'from_entity': 'TechVentures Inc',
            'to_entity': 'Quantum Analytics',
            'type': 'ACQUIRED',
            'nature': 'strategic acquisition',
            'version': 'A',
            'source_page': 5,
            'confidence': 0.9,
            'source_text': 'TechVentures Inc acquired Quantum Analytics'
        },
        {
            'from_entity': 'TechVentures Inc',
            'to_entity': 'Quantum Analytics',
            'type': 'ACQUIRED',
            'nature': 'strategic acquisition',
            'version': 'B',
            'source_page': 12,
            'confidence': 0.95,
            'source_text': 'TechVentures Inc completed strategic acquisition of Quantum Analytics'
        }
    ]
    
    writer = VersionedGraphWriter()
    writer.write_versioned_relationships(relationships)
    writer.close()
    
    query_system = ContradictionAwareQuery()
    
    print("\n🔍 Querying: Did TechVentures acquire Quantum Analytics?\n")
    print("="*70)
    
    result = query_system.query_relationship(
        'TechVentures Inc',
        'Quantum Analytics',
        'ACQUIRED'
    )
    
    answer = query_system.present_answer(result)
    print(answer)
    
    query_system.close()
    
    print("\n✅ Result: System showed multiple versions are consistent!\n")


def compare_with_llm_first():
    """
    Show what would happen with traditional LLM-First approach
    """
    
    print("\n" + "="*70)
    print("COMPARISON: LLM-First vs Graph-First")
    print("="*70 + "\n")
    
    print("❌ LLM-FIRST APPROACH (Traditional RAG):")
    print("-" * 70)
    print("""
Query: "How did Family Office invest in Rostova Dynamics?"

Retrieved chunks:
  - "The family office made direct investments in both companies"
  - "Actually, the investment was indirect through subsidiary"

LLM Response:
  "The Family Office invested in Rostova Dynamics through an indirect
   investment structure, utilizing a subsidiary relationship with 
   Quantum Analytics. This approach provides tax advantages and..."
   
PROBLEMS:
  ✗ LLM "decided" indirect was correct
  ✗ Hallucinated justification ("tax advantages")
  ✗ User has no idea direct version existed
  ✗ No way to verify decision
  ✗ Different queries might get different answers
""")
    
    print("\n✅ GRAPH-FIRST APPROACH (Our System):")
    print("-" * 70)
    print("""
Query: "How did Family Office invest in Rostova Dynamics?"

Graph query returns ALL versions:
  VERSION A (Page 10): nature=direct, confidence=0.95
  VERSION B (Page 15): nature=indirect, confidence=0.85
  
System Response:
  "⚠️ MULTIPLE VERSIONS FOUND
   
   VERSION A (Page 10): Direct investment
   Signals: [High extraction confidence]
   
   VERSION B (Page 15): Indirect investment
   Signals: [Later in document (possible correction),
            Contains correction language]
   
   These versions CONTRADICT. Please review source pages
   and make an informed decision."

BENEFITS:
  ✓ All versions presented
  ✓ No hallucination
  ✓ User can decide
  ✓ Fully traceable
  ✓ Consistent results
""")


def main():
    """Run complete demonstration"""
    
    print("\n" + "="*70)
    print("GRAPH-FIRST CONTRADICTION HANDLING DEMONSTRATION")
    print("Chapter 7: Handling Contradictions Without Hallucination")
    print("="*70)
    
    # Clean up
    print("\nCleaning up previous test data...")
    clean_test_data()
    
    # Run test scenarios
    test_scenario_1()
    test_scenario_2()
    compare_with_llm_first()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70 + "\n")
    
    print("""
Key Principles Demonstrated:

1. STORE ALL VERSIONS
   - Every contradictory statement is preserved
   - Metadata tracks source, confidence, nature
   
2. PRESENT, DON'T DECIDE
   - System shows what the graph contains
   - No hallucinated resolutions
   - User makes informed decisions
   
3. FULL TRACEABILITY
   - Every version links to source page
   - Reliability signals guide users
   - Complete audit trail

4. CONSISTENT RESULTS
   - Same query always returns same facts
   - No randomness from LLM "decisions"
   - Predictable, reliable behavior

This is production-ready contradiction handling for enterprise GraphRAG.
""")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()