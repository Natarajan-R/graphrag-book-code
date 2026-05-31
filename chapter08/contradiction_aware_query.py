"""
contradiction_aware_query.py
Query graph and present contradictions without hallucinating
"""

from neo4j import GraphDatabase
from typing import Dict, List
from dotenv import load_dotenv
import os

load_dotenv()

# Configuration
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')


class ContradictionAwareQuery:
    """Query system that presents all versions without deciding truth"""
    
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
    
    def close(self):
        self.driver.close()
    
    def query_relationship(self, from_entity: str, to_entity: str, rel_type: str = None) -> Dict:
        """
        Query relationship and return ALL versions
        
        Args:
            from_entity: Source entity
            to_entity: Target entity
            rel_type: Optional relationship type filter
            
        Returns:
            Dict with all versions and contradiction analysis
        """
        
        with self.driver.session() as session:
            # Build query
            if rel_type:
                query = """
                    MATCH (from:Entity {name: $from_name})
                          -[r:RELATIONSHIP {type: $rel_type}]->
                          (to:Entity {name: $to_name})
                    RETURN r
                    ORDER BY r.source_page
                """
                params = {
                    'from_name': from_entity,
                    'to_name': to_entity,
                    'rel_type': rel_type
                }
            else:
                query = """
                    MATCH (from:Entity {name: $from_name})
                          -[r:RELATIONSHIP]->
                          (to:Entity {name: $to_name})
                    RETURN r
                    ORDER BY r.source_page
                """
                params = {
                    'from_name': from_entity,
                    'to_name': to_entity
                }
            
            result = session.run(query, params)
            
            versions = []
            for record in result:
                r = record['r']
                versions.append({
                    'type': r['type'],
                    'nature': r['nature'],
                    'version': r['version'],
                    'source_page': r['source_page'],
                    'confidence': r['confidence'],
                    'source_text': r['source_text']
                })
            
            return self._analyze_versions(from_entity, to_entity, versions)
    
    def _analyze_versions(self, from_entity: str, to_entity: str, versions: List[Dict]) -> Dict:
        """Analyze versions and detect contradictions"""
        
        if len(versions) == 0:
            return {
                'from': from_entity,
                'to': to_entity,
                'status': 'not_found',
                'message': 'No relationship found',
                'versions': []
            }
        
        if len(versions) == 1:
            return {
                'from': from_entity,
                'to': to_entity,
                'status': 'single_version',
                'message': 'Single version found',
                'versions': versions,
                'contradiction': False
            }
        
        # Check for contradictions
        natures = set(v['nature'] for v in versions)
        has_contradiction = len(natures) > 1
        
        # Calculate reliability signals
        for version in versions:
            version['reliability_signals'] = self._calculate_reliability_signals(
                version, versions
            )
        
        return {
            'from': from_entity,
            'to': to_entity,
            'status': 'multiple_versions',
            'message': f'Found {len(versions)} versions' + (' (CONTRADICTORY)' if has_contradiction else ''),
            'versions': versions,
            'contradiction': has_contradiction,
            'natures': list(natures)
        }
    
    def _calculate_reliability_signals(self, version: Dict, all_versions: List[Dict]) -> List[str]:
        """Calculate reliability signals for a version"""
        
        signals = []
        
        # Later source might be correction
        min_page = min(v['source_page'] for v in all_versions)
        if version['source_page'] > min_page:
            signals.append("Later in document (possible correction)")
        
        # High confidence
        if version['confidence'] > 0.9:
            signals.append("High extraction confidence")
        
        # Correction language
        correction_words = ['actually', 'correction', 'rather', 'instead', 'however']
        source_text = version.get('source_text', '').lower()
        if any(word in source_text for word in correction_words):
            signals.append("Contains correction language")
        
        # More detailed
        if len(source_text.split()) > 10:
            signals.append("More detailed description")
        
        return signals
    
    def present_answer(self, result: Dict) -> str:
        """
        Present query result WITHOUT deciding truth
        This is the key: present facts, don't judge
        """
        
        if result['status'] == 'not_found':
            return f"No relationship found between {result['from']} and {result['to']}."
        
        if result['status'] == 'single_version':
            version = result['versions'][0]
            return f"""
Based on the knowledge graph:

{result['from']} has a {version['nature']} {version['type']} relationship with {result['to']}.

Source: Page {version['source_page']}
Confidence: {version['confidence']:.2f}
Quote: "{version['source_text'][:100]}..."
"""
        
        # Multiple versions - present ALL without deciding
        if result['contradiction']:
            answer = f"""
⚠️  MULTIPLE VERSIONS FOUND - User decision required

The knowledge graph contains {len(result['versions'])} different characterizations 
of the relationship between {result['from']} and {result['to']}:

"""
        else:
            answer = f"""
Multiple consistent versions found for the relationship between 
{result['from']} and {result['to']}:

"""
        
        for i, version in enumerate(result['versions'], 1):
            answer += f"""
VERSION {version['version']} (Page {version['source_page']}):
  Nature: {version['nature']}
  Type: {version['type']}
  Confidence: {version['confidence']:.2f}
  Source: "{version['source_text'][:80]}..."
  
  Reliability signals:
"""
            for signal in version.get('reliability_signals', []):
                answer += f"    • {signal}\n"
        
        if result['contradiction']:
            answer += f"""
IMPORTANT: These versions CONTRADICT each other ({', '.join(result['natures'])}).
The system cannot determine which is correct. Please:
  1. Review the source pages ({', '.join(str(v['source_page']) for v in result['versions'])})
  2. Consider reliability signals above
  3. Consult authoritative sources if needed
  4. Make an informed decision
"""
        
        return answer


# Example usage and testing
if __name__ == "__main__":
    print("\n" + "="*70)
    print("Contradiction-Aware Query Demo")
    print("Graph-First approach: Present facts, don't decide truth")
    print("="*70 + "\n")
    
    query_system = ContradictionAwareQuery()
    
    try:
        # Query the contradictory relationship
        print("Querying: How did Family Office invest in Rostova Dynamics?\n")
        print("="*70)
        
        result = query_system.query_relationship(
            from_entity='Family Office',
            to_entity='Rostova Dynamics',
            rel_type='INVESTED_IN'
        )
        
        # Present answer (Graph-First way)
        answer = query_system.present_answer(result)
        print(answer)
        
        print("="*70)
        print("\n💡 Key Point: The system presents ALL versions and lets")
        print("   the user decide which is correct. No hallucination!")
        print("="*70 + "\n")
        
    finally:
        query_system.close()