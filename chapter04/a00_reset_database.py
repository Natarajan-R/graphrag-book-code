"""
reset_database.py
Utility to reset/clear Neo4j database - USE WITH CAUTION!
"""

import sys
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()

# Configuration
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')


class DatabaseResetter:
    """Reset Neo4j database"""
    
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
    
    def close(self):
        self.driver.close()
    
    def get_statistics(self):
        """Get current database statistics"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n)
                OPTIONAL MATCH ()-[r]->()
                RETURN 
                    count(DISTINCT n) as nodes,
                    count(DISTINCT r) as relationships,
                    labels(n) as labels
            """)
            
            record = result.single()
            if record:
                return {
                    'nodes': record['nodes'],
                    'relationships': record['relationships']
                }
            return {'nodes': 0, 'relationships': 0}
    
    def get_detailed_statistics(self):
        """Get detailed statistics by node type"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n)
                WITH labels(n)[0] as label, count(*) as count
                RETURN label, count
                ORDER BY count DESC
            """)
            
            stats = {}
            for record in result:
                stats[record['label']] = record['count']
            
            return stats
    
    def clear_all_data(self):
        """Delete all nodes and relationships"""
        with self.driver.session() as session:
            # Delete all nodes (relationships are automatically deleted)
            session.run("MATCH (n) DETACH DELETE n")
    
    def drop_all_indexes(self):
        """Drop all indexes"""
        with self.driver.session() as session:
            # Get all indexes
            result = session.run("SHOW INDEXES")
            indexes = [record['name'] for record in result]
            
            # Drop each index
            for index_name in indexes:
                try:
                    session.run(f"DROP INDEX {index_name} IF EXISTS")
                except Exception as e:
                    print(f"      ⚠️  Could not drop index {index_name}: {e}")
    
    def drop_all_constraints(self):
        """Drop all constraints"""
        with self.driver.session() as session:
            # Get all constraints
            result = session.run("SHOW CONSTRAINTS")
            constraints = [record['name'] for record in result]
            
            # Drop each constraint
            for constraint_name in constraints:
                try:
                    session.run(f"DROP CONSTRAINT {constraint_name} IF EXISTS")
                except Exception as e:
                    print(f"      ⚠️  Could not drop constraint {constraint_name}: {e}")
    
    def reset_database(self, drop_indexes: bool = True, drop_constraints: bool = True):
        """
        Complete database reset
        
        Args:
            drop_indexes: If True, drop all indexes
            drop_constraints: If True, drop all constraints
        """
        
        print(f"\n{'='*70}")
        print(f"Database Reset")
        print(f"{'='*70}\n")
        
        # Show current statistics
        print("Current database state:")
        print("-" * 70)
        stats = self.get_statistics()
        print(f"  Total nodes: {stats['nodes']}")
        print(f"  Total relationships: {stats['relationships']}")
        
        if stats['nodes'] > 0:
            print("\nBreakdown by node type:")
            detailed_stats = self.get_detailed_statistics()
            for label, count in detailed_stats.items():
                print(f"  {label}: {count}")
        
        print()
        
        # Delete data
        print("[1/3] Deleting all nodes and relationships...", end=" ")
        self.clear_all_data()
        print("✓")
        
        # Drop indexes
        if drop_indexes:
            print("[2/3] Dropping all indexes...", end=" ")
            self.drop_all_indexes()
            print("✓")
        else:
            print("[2/3] Skipping indexes (--keep-indexes)")
        
        # Drop constraints
        if drop_constraints:
            print("[3/3] Dropping all constraints...", end=" ")
            self.drop_all_constraints()
            print("✓")
        else:
            print("[3/3] Skipping constraints (--keep-constraints)")
        
        # Verify
        print("\nVerifying reset...")
        final_stats = self.get_statistics()
        print(f"  Remaining nodes: {final_stats['nodes']}")
        print(f"  Remaining relationships: {final_stats['relationships']}")
        
        print(f"\n{'='*70}")
        print("✓ Database reset complete!")
        print(f"{'='*70}\n")


def confirm_reset(stats: dict) -> bool:
    """Ask user for confirmation"""
    print(f"\n{'='*70}")
    print(f"⚠️  WARNING: DATABASE RESET")
    print(f"{'='*70}\n")
    
    print("This will permanently delete:")
    print(f"  • {stats['nodes']} nodes")
    print(f"  • {stats['relationships']} relationships")
    print(f"  • All indexes and constraints")
    
    print("\n⚠️  THIS CANNOT BE UNDONE!")
    print("\nType 'DELETE' to confirm: ", end="")
    
    confirmation = input().strip()
    return confirmation == "DELETE"


if __name__ == "__main__":
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Reset Neo4j database - USE WITH CAUTION!",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (asks for confirmation)
  python reset_database.py
  
  # Force mode (no confirmation - DANGEROUS!)
  python reset_database.py --confirm
  
  # Keep indexes and constraints
  python reset_database.py --confirm --keep-indexes --keep-constraints
  
  # Just show statistics (no deletion)
  python reset_database.py --stats-only
        """
    )
    
    parser.add_argument(
        '--confirm',
        action='store_true',
        help='Skip confirmation prompt (DANGEROUS!)'
    )
    
    parser.add_argument(
        '--keep-indexes',
        action='store_true',
        help='Keep indexes (only delete data)'
    )
    
    parser.add_argument(
        '--keep-constraints',
        action='store_true',
        help='Keep constraints (only delete data)'
    )
    
    parser.add_argument(
        '--stats-only',
        action='store_true',
        help='Only show statistics, do not delete anything'
    )
    
    args = parser.parse_args()
    
    # Initialize resetter
    resetter = DatabaseResetter()
    
    try:
        # Get current statistics
        stats = resetter.get_statistics()
        
        # Stats only mode
        if args.stats_only:
            print(f"\n{'='*70}")
            print(f"Database Statistics")
            print(f"{'='*70}\n")
            print(f"Total nodes: {stats['nodes']}")
            print(f"Total relationships: {stats['relationships']}")
            
            if stats['nodes'] > 0:
                print("\nBreakdown by node type:")
                detailed_stats = resetter.get_detailed_statistics()
                for label, count in detailed_stats.items():
                    print(f"  {label}: {count}")
            print()
            sys.exit(0)
        
        # Check if database is empty
        if stats['nodes'] == 0 and stats['relationships'] == 0:
            print("\n✓ Database is already empty. Nothing to reset.")
            sys.exit(0)
        
        # Confirm reset
        if not args.confirm:
            if not confirm_reset(stats):
                print("\n✗ Reset cancelled.")
                sys.exit(0)
        else:
            print("\n⚠️  Running in --confirm mode (skipping confirmation)")
        
        # Perform reset
        resetter.reset_database(
            drop_indexes=not args.keep_indexes,
            drop_constraints=not args.keep_constraints
        )
        
    except KeyboardInterrupt:
        print("\n\n✗ Reset cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        resetter.close()
