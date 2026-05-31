"""
versioned_graph_writer.py
Write versioned relationships to Neo4j graph (handles contradictions)
"""

from neo4j import GraphDatabase
from typing import List, Dict
from dotenv import load_dotenv
import os
import uuid

load_dotenv()

# Configuration
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')


class VersionedGraphWriter:
    """Write relationships with version tracking to handle contradictions"""
    
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
    
    def close(self):
        self.driver.close()
    
    def write_versioned_relationship(self, relationship: Dict):
        """
        Write a relationship with version metadata
        
        Args:
            relationship: Dict with keys:
                - from_entity: Source entity name
                - to_entity: Target entity name
                - type: Relationship type
                - nature: Characterization (e.g., "direct" vs "indirect")
                - confidence: 0.0-1.0
                - source_page: Page number
                - source_text: Optional quote
                - version: Version identifier
        """
        
        with self.driver.session() as session:
            # Generate unique version ID
            version_id = relationship.get('version_id', str(uuid.uuid4())[:8])
            
            # Create entities if they don't exist
            session.run("""
                MERGE (from:Entity {name: $from_name})
                MERGE (to:Entity {name: $to_name})
            """,
                from_name=relationship['from_entity'],
                to_name=relationship['to_entity']
            )
            
            # Create versioned relationship
            # FIXED: Use from_entity/to_entity instead of reserved keyword 'from'
            session.run("""
                MATCH (from:Entity {name: $from_name})
                MATCH (to:Entity {name: $to_name})
                
                CREATE (from)-[r:RELATIONSHIP {
                    type: $rel_type,
                    nature: $nature,
                    version: $version,
                    version_id: $version_id,
                    source_page: $source_page,
                    confidence: $confidence,
                    source_text: $source_text,
                    created_at: datetime()
                }]->(to)
            """,
                from_name=relationship['from_entity'],
                to_name=relationship['to_entity'],
                rel_type=relationship['type'],
                nature=relationship.get('nature', 'unknown'),
                version=relationship.get('version', 'A'),
                version_id=version_id,
                source_page=relationship.get('source_page', 0),
                confidence=relationship.get('confidence', 0.5),
                source_text=relationship.get('source_text', '')
            )
            
            print(f"✓ Created relationship: {relationship['from_entity']} "
                  f"-[{relationship['type']}]-> {relationship['to_entity']} "
                  f"(version: {relationship.get('version', 'A')}, "
                  f"nature: {relationship.get('nature', 'unknown')})")
    
    def write_versioned_relationships(self, relationships: List[Dict]):
        """Write multiple versioned relationships"""
        
        print(f"\n{'='*70}")
        print("Writing Versioned Relationships to Graph")
        print(f"{'='*70}\n")
        
        for i, rel in enumerate(relationships, 1):
            print(f"[{i}/{len(relationships)}] ", end="")
            self.write_versioned_relationship(rel)
        
        print(f"\n✓ Wrote {len(relationships)} relationships to graph\n")
    
    def get_contradictions(self, from_entity: str, to_entity: str) -> List[Dict]:
        """Get all versions of relationships between two entities"""
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (from:Entity {name: $from_name})
                      -[r:RELATIONSHIP]->
                      (to:Entity {name: $to_name})
                
                RETURN 
                    r.type as type,
                    r.nature as nature,
                    r.version as version,
                    r.source_page as source_page,
                    r.confidence as confidence,
                    r.source_text as source_text
                
                ORDER BY r.source_page
            """,
                from_name=from_entity,
                to_name=to_entity
            )
            
            versions = []
            for record in result:
                versions.append({
                    'type': record['type'],
                    'nature': record['nature'],
                    'version': record['version'],
                    'source_page': record['source_page'],
                    'confidence': record['confidence'],
                    'source_text': record['source_text']
                })
            
            return versions


# Example usage and testing
if __name__ == "__main__":
    print("\n" + "="*70)
    print("Versioned Graph Writer Demo")
    print("Demonstrates storing contradictory relationships")
    print("="*70)
    
    # Sample contradictory relationships
    relationships = [
        # VERSION A - Claims DIRECT investment
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
        {
            'from_entity': 'Family Office',
            'to_entity': 'Quantum Analytics',
            'type': 'INVESTED_IN',
            'nature': 'direct',
            'version': 'A',
            'source_page': 10,
            'confidence': 0.95,
            'source_text': 'The family office made direct investments in both companies'
        },
        
        # VERSION B - Claims INDIRECT investment for Rostova
        {
            'from_entity': 'Family Office',
            'to_entity': 'Rostova Dynamics',
            'type': 'INVESTED_IN',
            'nature': 'indirect',
            'version': 'B',
            'source_page': 15,
            'confidence': 0.85,
            'source_text': 'Actually, the investment in Rostova Dynamics was indirect through a Quantum Analytics subsidiary'
        },
        {
            'from_entity': 'Quantum Analytics',
            'to_entity': 'Rostova Dynamics',
            'type': 'HAS_SUBSIDIARY',
            'nature': 'subsidiary',
            'version': 'B',
            'source_page': 15,
            'confidence': 0.8,
            'source_text': 'through a Quantum Analytics subsidiary'
        }
    ]
    
    # Write to graph
    writer = VersionedGraphWriter()
    try:
        writer.write_versioned_relationships(relationships)
        
        # Check for contradictions
        print("="*70)
        print("Checking for contradictions...")
        print("="*70 + "\n")
        
        versions = writer.get_contradictions('Family Office', 'Rostova Dynamics')
        
        if len(versions) > 1:
            print(f"⚠️  CONTRADICTION DETECTED!")
            print(f"   Found {len(versions)} different versions:\n")
            
            for i, ver in enumerate(versions, 1):
                print(f"   VERSION {ver['version']} (Page {ver['source_page']}):")
                print(f"     Nature: {ver['nature']}")
                print(f"     Confidence: {ver['confidence']}")
                print(f"     Source: \"{ver['source_text'][:60]}...\"")
                print()
        else:
            print("✓ No contradictions found")
        
        print("="*70 + "\n")
        
    finally:
        writer.close()