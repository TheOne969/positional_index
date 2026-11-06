from flask import Flask, render_template, request, jsonify
import time
import os
import query as query_mod

app = Flask(__name__)

from query import BinarySearchEngine

print("Initializing ultra-fast Binary Search Engine...")
engine = BinarySearchEngine()


@app.route('/')
def index():
    # Serve the main HTML interface
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    # Extract the JSON payload from the frontend AJAX request
    data = request.json
    search_type = data.get('type', 'phrase')
    query = data.get('query', '')
    
    # Return empty results immediately if the user submitted an empty search box
    if not query:
        return jsonify({"results": []})
        
    try:
        start_time = time.perf_counter()
        
        if search_type == 'phrase':
            results = engine.phrase_search(query)
        elif search_type == 'proximity':
            try:
                dist = int(data.get('distance', 1))
            except ValueError:
                dist = 1
            from query import tokenize
            terms = tokenize(query)
            if len(terms) >= 2:
                results = engine.proximity_search(terms[0], terms[1], dist)
            else:
                results = []
        else:
            results = []
            
        end_time = time.perf_counter()
        # Convert to microseconds for frontend display since it's so fast!
        time_ms = round((end_time - start_time) * 1000, 4)
        
        # Send the results and performance metadata back to the frontend
        return jsonify({"results": results, "count": len(results), "time_ms": time_ms})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run the development web server accessible to the local network
    app.run(debug=True, host='0.0.0.0', port=5000)
