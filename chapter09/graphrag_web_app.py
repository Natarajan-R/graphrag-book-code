"""
graphrag_web_app.py
Flask web application (client) for GraphRAG API
"""

from flask import Flask, render_template, request, jsonify
import requests
import os

app = Flask(__name__)

# Configuration
GRAPHRAG_API_URL = os.getenv('GRAPHRAG_API_URL', 'http://localhost:5000/query')


@app.route('/')
def home():
    """Render the main query interface"""
    return render_template('index.html')


@app.route('/query', methods=['POST'])
def query():
    """
    Handle query requests from the web interface
    Forwards to GraphRAG API and returns response
    """
    try:
        # Get question from request
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({
                'error': 'Question is required'
            }), 400
        
        # Forward to GraphRAG API
        response = requests.post(
            GRAPHRAG_API_URL,
            json={'question': question},
            timeout=60
        )
        
        # Check if request was successful
        response.raise_for_status()
        
        # Return the API response
        return jsonify(response.json())
        
    except requests.exceptions.ConnectionError:
        return jsonify({
            'error': 'Cannot connect to GraphRAG API. Make sure it is running on port 5000.'
        }), 503
        
    except requests.exceptions.Timeout:
        return jsonify({
            'error': 'Request timed out. The query is taking too long.'
        }), 504
        
    except requests.exceptions.RequestException as e:
        return jsonify({
            'error': f'API Error: {str(e)}'
        }), 500
        
    except Exception as e:
        return jsonify({
            'error': f'Server Error: {str(e)}'
        }), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    try:
        # Check if GraphRAG API is accessible
        response = requests.get(
            GRAPHRAG_API_URL.replace('/query', '/health'),
            timeout=5
        )
        api_status = 'healthy' if response.status_code == 200 else 'unhealthy'
    except:
        api_status = 'unreachable'
    
    return jsonify({
        'status': 'healthy',
        'graphrag_api': api_status
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("GraphRAG Web Application")
    print("="*60)
    print(f"\nGraphRAG API URL: {GRAPHRAG_API_URL}")
    print("\nStarting web server...")
    print("Access the application at: http://localhost:8585")
    print("\nMake sure GraphRAG API is running on port 5000!")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=8585, debug=True)
