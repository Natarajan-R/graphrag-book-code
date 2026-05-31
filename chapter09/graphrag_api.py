"""
graphrag_api.py
Flask REST API server for GraphRAG queries
"""

from flask import Flask, request, jsonify
from graphrag_query import GraphRAGQuery

app = Flask(__name__)
system = GraphRAGQuery()


@app.route('/query', methods=['POST'])
def query():
    """
    Main query endpoint
    
    Request body:
    {
        "question": "What did Dr. Martinez develop?"
    }
    
    Response:
    {
        "query": "...",
        "answer": "...",
        "sources": {...},
        "context_used": {...}
    }
    """
    try:
        data = request.json
        question = data.get('question')
        
        if not question:
            return jsonify({
                'error': 'Question is required'
            }), 400
        
        result = system.query(question, verbose=False)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'graphrag-api',
        'version': '1.0'
    })


@app.route('/', methods=['GET'])
def home():
    """API information"""
    return jsonify({
        'name': 'GraphRAG API',
        'version': '1.0',
        'endpoints': {
            'POST /query': 'Submit a question to the knowledge graph',
            'GET /health': 'Health check',
            'GET /': 'This endpoint'
        },
        'example': {
            'url': '/query',
            'method': 'POST',
            'body': {
                'question': 'What did Dr. Martinez develop?'
            }
        }
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("GraphRAG REST API Server")
    print("="*60)
    print("\nEndpoints:")
    print("  POST /query  - Submit questions")
    print("  GET  /health - Health check")
    print("  GET  /       - API information")
    print("\nServer starting...")
    print("  URL: http://0.0.0.0:5000")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000)
