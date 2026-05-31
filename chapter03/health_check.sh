#!/bin/bash

echo "=================================="
echo "GraphRAG System Health Check"
echo "=================================="
echo ""

# Check Neo4j
echo -n "Neo4j Service: "
systemctl is-active neo4j && echo "✓ Running" || echo "✗ Not Running"

echo -n "Neo4j Port 7687: "
nc -z localhost 7687 && echo "✓ Open" || echo "✗ Closed"

echo -n "Neo4j Browser 7474: "
nc -z localhost 7474 && echo "✓ Open" || echo "✗ Closed"

# Check Ollama
echo -n "Ollama Service: "
systemctl is-active ollama && echo "✓ Running" || echo "✗ Not Running"

echo -n "Ollama API 11434: "
nc -z localhost 11434 && echo "✓ Open" || echo "✗ Closed"

# Check Models
echo ""
echo "Ollama Models:"
ollama list

# Check Disk Space
echo ""
echo "Disk Space:"
df -h ~ | awk 'NR==1 || NR==2'

# Check Memory
echo ""
echo "Available Memory:"
free -h | awk 'NR==1 || NR==2'

echo ""
echo "=================================="
echo "Health Check Complete"
echo "=================================="

