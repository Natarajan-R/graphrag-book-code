"""
extract_entities.py
Stage 3: Entity and relationship extraction using LLM
"""

import json
import re
from typing import Dict, List
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os

load_dotenv()

# Configuration
LLM_MODEL = os.getenv('OLLAMA_LLM_MODEL', 'qwen2.5')


class EntityExtractor:
    """Extract entities and relationships using local LLM"""

    def __init__(self):
        self.llm = ChatOllama(
            model=LLM_MODEL,
            temperature=0.1  # Low temperature for consistent extraction
        )

    def create_extraction_prompt(self, text: str) -> str:
        """Create prompt for entity extraction"""
        prompt = f"""Extract entities and relationships from the following text.

INSTRUCTIONS:
1. Identify entities (people, companies, technologies, locations, dates)
2. Identify relationships between entities
3. Return ONLY valid JSON, no explanations
4. Use these entity types: Person, Company, Technology, Location, Date, Organization
5. Use clear relationship types: WORKS_AT, DEVELOPED, FOUNDED, LOCATED_IN, LICENSED_TO, etc.

TEXT:
{text}

OUTPUT FORMAT (JSON only):
{{
  "entities": [
    {{"name": "entity name", "type": "entity type"}}
  ],
  "relationships": [
    {{"from": "entity1", "type": "RELATIONSHIP_TYPE", "to": "entity2"}}
  ]
}}

JSON OUTPUT:"""

        return prompt

    def parse_llm_response(self, response: str) -> Dict:
        """Parse LLM response to extract JSON"""

        # Try to find JSON in response
        # Sometimes LLM adds text before/after JSON

        # Method 1: Direct JSON parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Method 2: Extract JSON from markdown code block
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Method 3: Find JSON object in text
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # If all fails, return empty structure
        print(f"Warning: Could not parse LLM response")
        print(f"Response: {response[:200]}...")
        return {"entities": [], "relationships": []}

    def validate_entity(self, entity: Dict) -> bool:
        """Validate entity has required fields"""
        if not isinstance(entity, dict):
            return False
        
        # Must have 'name' field and it must not be empty
        if 'name' not in entity or not entity['name']:
            return False
        
        # Must have 'type' field (can be empty, will default to 'Unknown')
        if 'type' not in entity:
            entity['type'] = 'Unknown'
        
        return True

    def validate_relationship(self, rel: Dict) -> bool:
        """Validate relationship has required fields"""
        if not isinstance(rel, dict):
            return False
        
        # Must have all three required fields
        required = ['from', 'to', 'type']
        for field in required:
            if field not in rel or not rel[field]:
                return False
        
        return True

    def clean_extraction(self, extraction: Dict) -> Dict:
        """Clean and validate extraction results"""
        
        # Validate entities
        valid_entities = []
        invalid_entity_count = 0
        
        for entity in extraction.get('entities', []):
            if self.validate_entity(entity):
                valid_entities.append(entity)
            else:
                invalid_entity_count += 1
        
        # Validate relationships
        valid_relationships = []
        invalid_rel_count = 0
        
        # Get set of valid entity names for relationship validation
        entity_names = set(e['name'].lower() for e in valid_entities)
        
        for rel in extraction.get('relationships', []):
            if self.validate_relationship(rel):
                # Additional check: make sure both entities exist
                from_exists = rel['from'].lower() in entity_names
                to_exists = rel['to'].lower() in entity_names
                
                if from_exists and to_exists:
                    valid_relationships.append(rel)
                else:
                    invalid_rel_count += 1
                    if not from_exists:
                        print(f"  ⚠️  Relationship references non-existent entity: '{rel['from']}'")
                    if not to_exists:
                        print(f"  ⚠️  Relationship references non-existent entity: '{rel['to']}'")
            else:
                invalid_rel_count += 1
        
        # Report issues if any
        if invalid_entity_count > 0 or invalid_rel_count > 0:
            print(f"  ⚠️  Cleaned: {invalid_entity_count} invalid entities, {invalid_rel_count} invalid relationships")
        
        return {
            'entities': valid_entities,
            'relationships': valid_relationships
        }

    def extract_from_text(self, text: str) -> Dict:
        """Extract entities and relationships from text"""

        # Create prompt
        prompt = self.create_extraction_prompt(text)

        # Query LLM
        try:
            response = self.llm.invoke(prompt)
            response_text = response.content

            # Parse response
            result = self.parse_llm_response(response_text)

            # Validate structure
            if 'entities' not in result:
                result['entities'] = []
            if 'relationships' not in result:
                result['relationships'] = []

            # Clean and validate extraction
            result = self.clean_extraction(result)

            return result

        except Exception as e:
            print(f"Error extracting entities: {e}")
            return {"entities": [], "relationships": []}

    def extract_from_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Extract entities from multiple chunks"""
        results = []

        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}...", end=" ")

            extraction = self.extract_from_text(chunk['text'])
            extraction['chunk_id'] = chunk['chunk_id']

            entity_count = len(extraction['entities'])
            rel_count = len(extraction['relationships'])

            print(f"✓ ({entity_count} entities, {rel_count} relationships)")

            results.append(extraction)

        return results

    def merge_extractions(self, extractions: List[Dict]) -> Dict:
        """Merge extractions from multiple chunks"""

        all_entities = []
        all_relationships = []

        # Collect all entities and relationships
        for extraction in extractions:
            all_entities.extend(extraction['entities'])
            all_relationships.extend(extraction['relationships'])

        # Deduplicate entities (by name, case-insensitive)
        seen_entities = {}
        unique_entities = []

        for entity in all_entities:
            name_lower = entity['name'].lower()
            if name_lower not in seen_entities:
                seen_entities[name_lower] = True
                unique_entities.append(entity)

        # Deduplicate relationships
        seen_rels = set()
        unique_relationships = []

        for rel in all_relationships:
            # Validate before adding
            if not self.validate_relationship(rel):
                continue
            
            rel_key = (
                rel['from'].lower(),
                rel['type'],
                rel['to'].lower()
            )
            if rel_key not in seen_rels:
                seen_rels.add(rel_key)
                unique_relationships.append(rel)

        # Final validation pass
        final_entities = [e for e in unique_entities if self.validate_entity(e)]
        
        # Validate relationships against final entity list
        entity_names = set(e['name'].lower() for e in final_entities)
        final_relationships = []
        
        for rel in unique_relationships:
            from_exists = rel['from'].lower() in entity_names
            to_exists = rel['to'].lower() in entity_names
            
            if from_exists and to_exists:
                final_relationships.append(rel)
            else:
                print(f"⚠️  Removed relationship (entity not found): {rel['from']} → {rel['to']}")

        return {
            'entities': final_entities,
            'relationships': final_relationships,
            'statistics': {
                'total_entities': len(all_entities),
                'unique_entities': len(final_entities),
                'total_relationships': len(all_relationships),
                'unique_relationships': len(final_relationships)
            }
        }


# Command-line interface
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python extract_entities.py <processed_json_file>")
        print("\nExample:")
        print("  python extract_entities.py research_paper_processed.json")
        sys.exit(1)

    input_file = sys.argv[1]

    # Validate input file exists
    if not os.path.exists(input_file):
        print(f"Error: File not found: {input_file}")
        sys.exit(1)

    # Load processed data
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {input_file}")
        print(f"  {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading {input_file}: {e}")
        sys.exit(1)

    # Validate data structure
    if 'chunks' not in data:
        print(f"Error: No 'chunks' key in {input_file}")
        print(f"  Expected format: {{'chunks': [...]}}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Extracting entities from: {input_file}")
    print(f"{'='*60}\n")
    print(f"Total chunks to process: {len(data['chunks'])}\n")

    # Extract entities
    extractor = EntityExtractor()
    results = extractor.extract_from_chunks(data['chunks'])
    merged = extractor.merge_extractions(results)

    # Print statistics
    print(f"\n{'='*60}")
    print("Extraction complete!")
    print(f"{'='*60}")
    print(f"Total entities found: {merged['statistics']['total_entities']}")
    print(f"Unique entities: {merged['statistics']['unique_entities']}")
    print(f"Total relationships: {merged['statistics']['total_relationships']}")
    print(f"Unique relationships: {merged['statistics']['unique_relationships']}")

    # Save results
    output_file = input_file.replace('_processed.json', '_entities.json')
    
    # If the filename doesn't contain '_processed', add '_entities' before extension
    if output_file == input_file:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_entities{ext}"
    
    try:
        with open(output_file, 'w') as f:
            json.dump(merged, f, indent=2)
        print(f"\n✓ Saved to: {output_file}")
    except Exception as e:
        print(f"\n✗ Error saving output: {e}")
        sys.exit(1)

    # Show sample of extracted data
    print(f"\n{'='*60}")
    print("Sample of extracted entities:")
    print(f"{'='*60}")
    for entity in merged['entities'][:5]:
        print(f"  - {entity['name']} ({entity['type']})")
    
    if len(merged['entities']) > 5:
        print(f"  ... and {len(merged['entities']) - 5} more")

    print(f"\n{'='*60}")
    print("Sample of extracted relationships:")
    print(f"{'='*60}")
    for rel in merged['relationships'][:5]:
        print(f"  - {rel['from']} → [{rel['type']}] → {rel['to']}")
    
    if len(merged['relationships']) > 5:
        print(f"  ... and {len(merged['relationships']) - 5} more")