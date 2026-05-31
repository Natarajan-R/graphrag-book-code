from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()

# Connect to Neo4j
driver = GraphDatabase.driver(
    os.getenv('NEO4J_URI'),
    auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD'))
)

# Test query
with driver.session() as session:
    result = session.run("RETURN 'Connection successful!' as message")
    print(result.single()['message'])

driver.close()

