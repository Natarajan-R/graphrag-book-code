"""
Enhanced Entity Extraction with Graph-First Architecture
Extracts ALL versions of facts, including contradictions
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import hashlib


@dataclass
class EntityNode:
    """Represents an entity (node) in the graph"""
    id: str
    type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    
    # Graph-First additions
    source_page: Optional[int] = None
    source_section: Optional[str] = None
    source_text: Optional[str] = None
    confidence: str = "stated"  # stated, implied, inferred
    version: Optional[str] = None
    extracted_at: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return {
            "id": self.id,
            "type": self.type,
            "properties": self.properties,
            "source_page": self.source_page,
            "source_section": self.source_section,
            "source_text": self.source_text,
            "confidence": self.confidence,
            "version": self.version,
            "extracted_at": self.extracted_at or datetime.now().isoformat()
        }


@dataclass
class EntityRelationship:
    """Represents a relationship (edge) in the graph"""
    source_id: str
    target_id: str
    type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    
    # Graph-First additions
    source_page: Optional[int] = None
    source_section: Optional[str] = None
    source_text: Optional[str] = None
    confidence: str = "stated"
    version: Optional[str] = None
    nature: Optional[str] = None  # For investment: "DIRECT", "INDIRECT"
    route: Optional[str] = None  # For indirect: "via subsidiary"
    extracted_at: Optional[str] = None
    contradicts: Optional[str] = None  # ID of contradicting relationship
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "type": self.type,
            "properties": self.properties,
            "source_page": self.source_page,
            "source_section": self.source_section,
            "source_text": self.source_text,
            "confidence": self.confidence,
            "version": self.version,
            "nature": self.nature,
            "route": self.route,
            "extracted_at": self.extracted_at or datetime.now().isoformat(),
            "contradicts": self.contradicts
        }
    
    def get_signature(self) -> str:
        """
        Generate signature for detecting contradictions
        Same source+target+type = potential contradiction
        """
        return f"{self.source_id}:{self.type}:{self.target_id}"


class GraphFirstExtractor:
    """
    Enhanced extractor that captures ALL versions of facts
    including contradictions
    """
    
    def __init__(self):
        self.extracted_nodes: List[EntityNode] = []
        self.extracted_relationships: List[EntityRelationship] = []
        self.relationship_signatures: Dict[str, List[EntityRelationship]] = {}
    
    def add_node(self, node: EntityNode):
        """Add a node to extracted entities"""
        self.extracted_nodes.append(node)
    
    def add_relationship(self, relationship: EntityRelationship):
        """
        Add a relationship and detect contradictions
        """
        # Add to list
        self.extracted_relationships.append(relationship)
        
        # Track by signature for contradiction detection
        sig = relationship.get_signature()
        if sig not in self.relationship_signatures:
            self.relationship_signatures[sig] = []
        self.relationship_signatures[sig].append(relationship)
    
    def detect_contradictions(self) -> List[Dict]:
        """
        Detect contradictory relationships
        
        Returns:
            List of contradiction pairs
        """
        contradictions = []
        
        for sig, rels in self.relationship_signatures.items():
            if len(rels) < 2:
                continue
            
            # Check for contradictions within this signature group
            for i, rel1 in enumerate(rels):
                for rel2 in rels[i+1:]:
                    # Check if they contradict
                    if self._are_contradictory(rel1, rel2):
                        contradictions.append({
                            "relationship_1": rel1.to_dict(),
                            "relationship_2": rel2.to_dict(),
                            "signature": sig,
                            "reason": self._get_contradiction_reason(rel1, rel2)
                        })
                        
                        # Mark them as contradicting each other
                        rel1.contradicts = f"{rel2.source_id}:{rel2.type}:{rel2.target_id}:{rel2.version}"
                        rel2.contradicts = f"{rel1.source_id}:{rel1.type}:{rel1.target_id}:{rel1.version}"
        
        return contradictions
    
    def _are_contradictory(self, rel1: EntityRelationship, rel2: EntityRelationship) -> bool:
        """
        Determine if two relationships contradict each other
        """
        # Same relationship type but different versions
        if rel1.version and rel2.version and rel1.version != rel2.version:
            return True
        
        # Same type but different nature (e.g., DIRECT vs INDIRECT)
        if rel1.nature and rel2.nature and rel1.nature != rel2.nature:
            return True
        
        # Explicitly marked as contradicting
        if rel1.properties.get("contradicts_version"):
            return True
        
        return False
    
    def _get_contradiction_reason(self, rel1: EntityRelationship, rel2: EntityRelationship) -> str:
        """Get human-readable reason for contradiction"""
        if rel1.nature and rel2.nature and rel1.nature != rel2.nature:
            return f"Nature differs: {rel1.nature} vs {rel2.nature}"
        
        if rel1.version and rel2.version:
            return f"Different versions: {rel1.version} vs {rel2.version}"
        
        return "Conflicting statements"
    
    def get_extraction_summary(self) -> Dict:
        """Get summary of extracted data"""
        contradictions = self.detect_contradictions()
        
        return {
            "nodes_count": len(self.extracted_nodes),
            "relationships_count": len(self.extracted_relationships),
            "contradictions_count": len(contradictions),
            "nodes": [n.to_dict() for n in self.extracted_nodes],
            "relationships": [r.to_dict() for r in self.extracted_relationships],
            "contradictions": contradictions
        }
    
    def clear(self):
        """Clear all extracted data"""
        self.extracted_nodes.clear()
        self.extracted_relationships.clear()
        self.relationship_signatures.clear()


# Enhanced prompts for LLM extraction

ENHANCED_EXTRACTION_PROMPT = """
You are extracting entities and relationships from a document to build a knowledge graph.

CRITICAL INSTRUCTIONS FOR GRAPH-FIRST ARCHITECTURE:

1. Extract ALL versions of facts, even if they contradict each other
2. If you find statements like "Version A: X" and "Version B: Y", extract BOTH
3. Include metadata for each extraction:
   - Where it came from (page number, section)
   - Confidence level (stated clearly, implied, inferred)
   - Version label if multiple versions exist

4. For relationships, capture:
   - Nature (e.g., for investments: DIRECT or INDIRECT)
   - Route (e.g., "via subsidiary", "through family office")
   - Any qualifiers or conditions

5. DO NOT try to resolve contradictions - extract everything as-is

Example Input:
"Version A (Page 10): The family office made DIRECT investments in both companies.
Version B (Page 15): The investment was INDIRECT through a subsidiary."

Example Output:
Relationship 1:
- Source: Family Office
- Target: Company
- Type: INVESTS_IN
- Nature: DIRECT
- Source: Page 10
- Version: A
- Confidence: stated

Relationship 2:
- Source: Family Office  
- Target: Company
- Type: INVESTS_IN
- Nature: INDIRECT
- Route: via subsidiary
- Source: Page 15
- Version: B
- Confidence: stated

Now extract from the following text:

{text}

Return as JSON with structure:
{{
  "entities": [
    {{
      "id": "entity_name",
      "type": "Entity_Type",
      "source_page": page_number,
      "confidence": "stated|implied|inferred"
    }}
  ],
  "relationships": [
    {{
      "source": "entity_1",
      "target": "entity_2", 
      "type": "RELATIONSHIP_TYPE",
      "nature": "qualifier if applicable",
      "source_page": page_number,
      "version": "A|B|etc if multiple versions",
      "confidence": "stated|implied|inferred"
    }}
  ]
}}
"""


def create_extraction_prompt(text: str, page_number: Optional[int] = None) -> str:
    """
    Create enhanced extraction prompt with metadata
    
    Args:
        text: Text to extract from
        page_number: Optional page number for context
    
    Returns:
        Formatted prompt
    """
    prompt = ENHANCED_EXTRACTION_PROMPT.format(text=text)
    
    if page_number is not None:
        prompt += f"\n\nThis text is from page {page_number}."
    
    return prompt


# Utility functions

def parse_llm_extraction(llm_response: Dict, page_number: Optional[int] = None) -> GraphFirstExtractor:
    """
    Parse LLM extraction response into GraphFirstExtractor
    
    Args:
        llm_response: Response from LLM with entities and relationships
        page_number: Optional page number for metadata
    
    Returns:
        GraphFirstExtractor with parsed data
    """
    extractor = GraphFirstExtractor()
    
    # Parse entities
    for entity_data in llm_response.get("entities", []):
        node = EntityNode(
            id=entity_data["id"],
            type=entity_data["type"],
            properties=entity_data.get("properties", {}),
            source_page=entity_data.get("source_page", page_number),
            confidence=entity_data.get("confidence", "stated"),
            version=entity_data.get("version")
        )
        extractor.add_node(node)
    
    # Parse relationships
    for rel_data in llm_response.get("relationships", []):
        relationship = EntityRelationship(
            source_id=rel_data["source"],
            target_id=rel_data["target"],
            type=rel_data["type"],
            properties=rel_data.get("properties", {}),
            source_page=rel_data.get("source_page", page_number),
            confidence=rel_data.get("confidence", "stated"),
            version=rel_data.get("version"),
            nature=rel_data.get("nature"),
            route=rel_data.get("route")
        )
        extractor.add_relationship(relationship)
    
    return extractor


def merge_extractors(*extractors: GraphFirstExtractor) -> GraphFirstExtractor:
    """
    Merge multiple extractors into one
    
    Args:
        extractors: Variable number of GraphFirstExtractor instances
    
    Returns:
        Merged extractor
    """
    merged = GraphFirstExtractor()
    
    for extractor in extractors:
        for node in extractor.extracted_nodes:
            merged.add_node(node)
        for rel in extractor.extracted_relationships:
            merged.add_relationship(rel)
    
    return merged
