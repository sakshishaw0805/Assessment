from typing import List, Dict, Any
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from flask import Flask, request, jsonify

# Global variables
assessments: List[Dict[str, Any]] = []
model = SentenceTransformer('all-MiniLM-L6-v2')
index = None

def load_assessments(file_path: str) -> List[Dict[str, Any]]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading assessments: {str(e)}")
        return []

def create_embeddings(assessments: List[Dict[str, Any]]) -> np.ndarray:
    texts = [f"{a['name']} {a['description']} {a['test_type']}" for a in assessments]
    return model.encode(texts)

def setup_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def search_assessments(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(assessments):
            assessment = assessments[idx].copy()
            assessment['score'] = float(distances[0][i])
            results.append(assessment)
    return results

def create_app():
    app = Flask(__name__)
    
    global assessments, index
    assessments = load_assessments('assessments.json')
    embeddings = create_embeddings(assessments)
    index = setup_faiss_index(embeddings)
    
    @app.route('/api/recommend', methods=['GET'])
    def recommend():
        query = request.args.get('query', '')
        if not query:
            return jsonify({"error": "No query provided"}), 400
        results = search_assessments(query)
        return jsonify(results)
    
    return app
app = create_app()
if __name__ == "__main__":
    app.run(debug=True)