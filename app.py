from typing import List, Dict, Any
import json
import numpy as np
from flask import Flask, request, jsonify
import os
assessments: List[Dict[str, Any]] = []
model = None
index = None

def create_app():
    app = Flask(__name__)
    
    @app.route('/api/recommend', methods=['GET'])
    def recommend():
        query = request.args.get('query', '')
        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        global model, index, assessments
        if model is None or index is None:
            initialize_resources()
            
        results = search_assessments(query)
        return jsonify(results)
    
    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({"status": "ok"})
    
    return app

def initialize_resources():
    global assessments, model, index
    from sentence_transformers import SentenceTransformer
    import faiss
    
    print("Loading assessments...")
    assessments = load_assessments('assessments.json')
    
    print("Loading model...")
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    except (NameError, ImportError) as e:
        print(f"Error initializing model: {str(e)}")
        model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
    
    print("Creating embeddings...")
    embeddings = create_embeddings(assessments)
    
    print("Setting up FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    print("Resources initialized successfully")

def load_assessments(file_path: str) -> List[Dict[str, Any]]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading assessments: {str(e)}")
        return []

def create_embeddings(assessments: List[Dict[str, Any]]) -> np.ndarray:
    global model
    texts = [f"{a['name']} {a['description']} {a['test_type']}" for a in assessments]
    return model.encode(texts)

def search_assessments(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    global model, index, assessments
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(assessments):
            assessment = assessments[idx].copy()
            assessment['score'] = float(distances[0][i])
            results.append(assessment)
    return results

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)