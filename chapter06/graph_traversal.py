"""
graph_traversal.py
Graph expansion from entities
"""

from typing import List, Dict
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()

# Configuration
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')


class GraphTraverser:
    """Graph traversal and expansion"""

    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )

    def close(self):
        self.driver.close()

    def get_entity_neighbors(
        self,
        entity_name: str,
        hops: int = 1
    ) -> Dict:
        """
        Get neighbors of an entity within N hops

        Args:
            entity_name: Entity to start from
            hops: Number of relationship hops (1-3 recommended)

        Returns:
            Dict with entities and relationships
        """
        with self.driver.session() as session:
            # Fixed query - path variable properly handled in WITH clause
            query = f"""
                MATCH path = (start:Entity {{name: $entity_name}})
                             -[r:RELATIONSHIP*1..{hops}]-
                             (end:Entity)
                
                // Keep path variable in WITH to use relationships(path)
                WITH start, end, path, relationships(path) as rels
                
                RETURN 
                    end.name as entity_name,
                    end.type as entity_type,
                    [rel IN rels | type(rel)] as relationship_path,
                    length(path) as distance
                
                ORDER BY distance, entity_name
                LIMIT 50
            """
            
            result = session.run(query, entity_name=entity_name)

            neighbors = []
            for record in result:
                neighbors.append({
                    'name': record['entity_name'],
                    'type': record['entity_type'],
                    'path': record['relationship_path'],
                    'distance': record['distance']
                })

            return {
                'source': entity_name,
                'neighbors': neighbors
            }

    def get_relationships(
        self,
        entity_name: str
    ) -> List[Dict]:
        """
        Get all relationships for an entity

        Args:
            entity_name: Entity to query

        Returns:
            List of relationships
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:Entity {name: $entity_name})
                      -[r:RELATIONSHIP]->
                      (other:Entity)
                
                RETURN 
                    r.type as relationship_type,
                    other.name as target_name,
                    other.type as target_type,
                    r.confidence as confidence
                
                ORDER BY relationship_type, target_name
            """, entity_name=entity_name)

            relationships = []
            for record in result:
                relationships.append({
                    'type': record['relationship_type'],
                    'target': record['target_name'],
                    'target_type': record['target_type'],
                    'confidence': record['confidence']
                })

            return relationships

    def find_path(
        self,
        start_entity: str,
        end_entity: str,
        max_hops: int = 5
    ) -> List[Dict]:
        """
        Find paths connecting two entities

        Args:
            start_entity: Starting entity
            end_entity: Target entity
            max_hops: Maximum path length

        Returns:
            List of paths
        """
        with self.driver.session() as session:
            result = session.run(f"""
                MATCH path = shortestPath(
                    (start:Entity {{name: $start}})
                    -[r:RELATIONSHIP*1..{max_hops}]-
                    (end:Entity {{name: $end}})
                )
                
                RETURN 
                    [node IN nodes(path) | node.name] as node_names,
                    [rel IN relationships(path) | type(rel)] as rel_types,
                    length(path) as path_length
                
                ORDER BY path_length
                LIMIT 5
            """, start=start_entity, end=end_entity)

            paths = []
            for record in result:
                paths.append({
                    'nodes': record['node_names'],
                    'relationships': record['rel_types'],
                    'length': record['path_length']
                })

            return paths

    def expand_from_chunks(
        self,
        chunk_ids: List[str],
        hops: int = 1
    ) -> Dict:
        """
        Expand graph context from chunks

        Args:
            chunk_ids: List of chunk IDs
            hops: Expansion distance

        Returns:
            Dict with entities and relationships
        """
        with self.driver.session() as session:
            result = session.run(f"""
                // Start from chunks
                MATCH (c:Chunk)
                WHERE c.chunk_id IN $chunk_ids
                
                // Get mentioned entities
                MATCH (c)-[:MENTIONS]->(e:Entity)
                
                // Expand to neighbors
                OPTIONAL MATCH (e)-[r:RELATIONSHIP*1..{hops}]-(neighbor:Entity)
                
                WITH e, neighbor, r
                WHERE r IS NOT NULL
                
                RETURN 
                    COLLECT(DISTINCT e.name) as core_entities,
                    COLLECT(DISTINCT neighbor.name) as neighbor_entities,
                    COLLECT(DISTINCT {{
                        from: startNode(r).name,
                        type: type(r),
                        to: endNode(r).name
                    }}) as relationships
            """, chunk_ids=chunk_ids)

            record = result.single()
            if record:
                return {
                    'core_entities': record['core_entities'],
                    'neighbor_entities': record['neighbor_entities'],
                    'relationships': record['relationships']
                }

            return {
                'core_entities': [],
                'neighbor_entities': [],
                'relationships': []
            }


# Test the traverser
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python graph_traversal.py <entity_name>")
        print("\nExample:")
        print("  python graph_traversal.py 'Dr. Yazhini Martinez'")
        sys.exit(1)

    entity_name = ' '.join(sys.argv[1:])

    traverser = GraphTraverser()
    try:
        print(f"\n{'='*60}")
        print(f"Entity: {entity_name}")
        print(f"{'='*60}\n")

        # Get relationships first (simpler check)
        print("Direct Relationships:")
        rels = traverser.get_relationships(entity_name)
        
        if not rels:
            print("  No direct relationships found")
        else:
            for rel in rels:
                print(f"  {rel['type']} → {rel['target']} ({rel['target_type']})")
                if 'confidence' in rel and rel['confidence']:
                    print(f"    Confidence: {rel['confidence']:.2f}")

        # Get neighbors (1-hop)
        print(f"\n{'='*60}")
        print("1-Hop Neighbors:")
        neighbors = traverser.get_entity_neighbors(entity_name, hops=1)
        
        if not neighbors['neighbors']:
            print("  No neighbors found within 1 hop")
        else:
            for neighbor in neighbors['neighbors'][:10]:  # Limit to 10 for readability
                print(f"\n  {neighbor['name']} ({neighbor['type']})")
                if neighbor['path']:
                    print(f"  Path: {' → '.join(neighbor['path'])}")
                print(f"  Distance: {neighbor['distance']} hop(s)")

        print(f"\n{'='*60}")
        print(f"Total neighbors found: {len(neighbors['neighbors'])}")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure Neo4j is running: neo4j status")
        print("2. Check .env file has correct credentials")
        print("3. Verify entity exists: MATCH (e:Entity) RETURN e.name LIMIT 10")
        
    finally:
        traverser.close()
        