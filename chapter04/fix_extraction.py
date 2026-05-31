#!/usr/bin/env python3
"""
fix_extraction.py
Inspect and fix extraction JSON file
"""

import json
import sys
import os


def inspect_extraction(filepath):
    """Inspect extraction file and report issues"""
    
    print(f"\n{'='*60}")
    print(f"Inspecting: {filepath}")
    print(f"{'='*60}\n")
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        return None
    
    # Check structure
    has_entities = 'entities' in data
    has_relationships = 'relationships' in data
    
    print("File structure:")
    print(f"  Has 'entities': {has_entities}")
    print(f"  Has 'relationships': {has_relationships}")
    
    if not has_entities:
        print("\n⚠️  WARNING: No 'entities' key found!")
        data['entities'] = []
    
    if not has_relationships:
        print("\n⚠️  WARNING: No 'relationships' key found!")
        data['relationships'] = []
    
    # Inspect entities
    print(f"\n{'='*60}")
    print("ENTITIES")
    print(f"{'='*60}")
    print(f"Total: {len(data['entities'])}\n")
    
    valid_entities = []
    invalid_entities = []
    
    for i, entity in enumerate(data['entities']):
        if not isinstance(entity, dict):
            print(f"Entity {i}: ✗ Not a dictionary")
            invalid_entities.append((i, entity, "Not a dictionary"))
            continue
        
        has_name = 'name' in entity and entity['name']
        has_type = 'type' in entity
        
        if not has_name:
            print(f"Entity {i}: ✗ Missing 'name'")
            print(f"  Data: {entity}")
            invalid_entities.append((i, entity, "Missing name"))
            continue
        
        valid_entities.append(entity)
        
        # Show sample of first 5 valid entities
        if len(valid_entities) <= 5:
            print(f"Entity {i}: ✓ {entity['name']}")
            if has_type:
                print(f"  Type: {entity['type']}")
    
    if len(valid_entities) > 5:
        print(f"... and {len(valid_entities) - 5} more valid entities")
    
    if invalid_entities:
        print(f"\n⚠️  Found {len(invalid_entities)} invalid entities:")
        for idx, entity, reason in invalid_entities[:10]:
            print(f"  {idx}: {reason}")
    
    # Inspect relationships
    print(f"\n{'='*60}")
    print("RELATIONSHIPS")
    print(f"{'='*60}")
    print(f"Total: {len(data['relationships'])}\n")
    
    valid_relationships = []
    invalid_relationships = []
    
    # Get entity names for validation
    entity_names = set(e['name'] for e in valid_entities)
    
    for i, rel in enumerate(data['relationships']):
        if not isinstance(rel, dict):
            print(f"Relationship {i}: ✗ Not a dictionary")
            invalid_relationships.append((i, rel, "Not a dictionary"))
            continue
        
        issues = []
        
        # Check required fields
        has_from = 'from' in rel and rel['from']
        has_to = 'to' in rel and rel['to']
        has_type = 'type' in rel and rel['type']
        
        if not has_from:
            issues.append("Missing 'from'")
        elif rel['from'] not in entity_names:
            issues.append(f"'from' entity not found: {rel['from']}")
        
        if not has_to:
            issues.append("Missing 'to'")
        elif rel['to'] not in entity_names:
            issues.append(f"'to' entity not found: {rel['to']}")
        
        if not has_type:
            issues.append("Missing 'type'")
        
        if issues:
            print(f"Relationship {i}: ✗ {', '.join(issues)}")
            print(f"  Data: {rel}")
            invalid_relationships.append((i, rel, issues))
            continue
        
        valid_relationships.append(rel)
        
        # Show sample of first 5 valid relationships
        if len(valid_relationships) <= 5:
            print(f"Relationship {i}: ✓ {rel['from']} → [{rel['type']}] → {rel['to']}")
    
    if len(valid_relationships) > 5:
        print(f"... and {len(valid_relationships) - 5} more valid relationships")
    
    if invalid_relationships:
        print(f"\n⚠️  Found {len(invalid_relationships)} invalid relationships:")
        for idx, rel, issues in invalid_relationships[:10]:
            print(f"  {idx}: {', '.join(issues) if isinstance(issues, list) else issues}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Valid entities: {len(valid_entities)} / {len(data['entities'])}")
    print(f"Valid relationships: {len(valid_relationships)} / {len(data['relationships'])}")
    
    if invalid_entities or invalid_relationships:
        print(f"\n⚠️  Found issues that need fixing")
        return {
            'data': data,
            'valid_entities': valid_entities,
            'valid_relationships': valid_relationships,
            'invalid_entities': invalid_entities,
            'invalid_relationships': invalid_relationships
        }
    else:
        print(f"\n✓ All data is valid!")
        return None


def fix_extraction(filepath, output_path=None):
    """Fix extraction file by removing invalid entries"""
    
    result = inspect_extraction(filepath)
    
    if not result:
        print("\n✓ No fixes needed!")
        return
    
    if not output_path:
        base, ext = os.path.splitext(filepath)
        output_path = f"{base}_fixed{ext}"
    
    # Create fixed version
    fixed_data = {
        'entities': result['valid_entities'],
        'relationships': result['valid_relationships']
    }
    
    print(f"\n{'='*60}")
    print("FIXING")
    print(f"{'='*60}")
    print(f"Writing fixed version to: {output_path}")
    print(f"  Valid entities: {len(fixed_data['entities'])}")
    print(f"  Valid relationships: {len(fixed_data['relationships'])}")
    
    try:
        with open(output_path, 'w') as f:
            json.dump(fixed_data, f, indent=2)
        
        print(f"\n✓ Fixed version saved to: {output_path}")
        print(f"\nYou can now run:")
        print(f"  python build_graph.py {output_path} <file_hash>")
        
    except Exception as e:
        print(f"\n✗ Error writing fixed file: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_extraction.py <extraction.json> [output.json]")
        print("\nExample:")
        print("  python fix_extraction.py research_paper_entities.json")
        print("  python fix_extraction.py research_paper_entities.json research_paper_entities_fixed.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(input_file):
        print(f"✗ Error: File not found: {input_file}")
        sys.exit(1)
    
    fix_extraction(input_file, output_file)
