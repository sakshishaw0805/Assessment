from typing import List, Dict, Any
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from flask import Flask, request, jsonify
import os

app = Flask(__name__)
assessments = []
model = None
index = None

def load_assessments(file_path: str) -> List[Dict[str, Any]]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading assessments: {str(e)}")
        return []

def create_embeddings(assessments: List[Dict[str, Any]], model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    model = SentenceTransformer(model_name)
    texts = [f"{assessment['name']} {assessment['description']} {assessment['test_type']}" 
             for assessment in assessments]
    return model.encode(texts)

def setup_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def search_assessments(query: str, model, index, assessments: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(assessments):
            assessment = assessments[idx].copy()
            assessment['score'] = float(distances[0][i])
            results.append(assessment)
    return results

@app.before_request
def initialize():
    global assessments, model, index
    assessments = load_assessments('assessments.json')  # Use the output from your scraper
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = create_embeddings(assessments)
    index = setup_faiss_index(embeddings)

@app.route('/api/recommend', methods=['GET'])
def recommend():
    query = request.args.get('query', '')
    if not query:
        return jsonify({"error": "No query provided"}), 400
    results = search_assessments(query, model, index, assessments)
    return jsonify(results)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
