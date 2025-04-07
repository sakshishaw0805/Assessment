from typing import List, Dict, Any
import os
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from flask import Flask, request, jsonify
import logging

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

assessments = []
model = None
index = None




    
def load_assessments(file_path: str) -> List[Dict[str, Any]]:
        with open(file_path, 'r') as f:
            assessments = json.load(f)
        return assessments

def create_embeddings(assessments: List[Dict[str, Any]], model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
        model = SentenceTransformer(model_name)
        texts = [f"{assessment['name']} {assessment['description']} {assessment['test_type']}" 
                for assessment in assessments]
        embeddings = model.encode(texts)
        return embeddings
def setup_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return index
def search_assessments(query: str, 
                        model, 
                        index, 
                        assessments: List[Dict[str, Any]], 
                        top_k: int = 10) -> List[Dict[str, Any]]:
        query_embedding = model.encode([query])
        distances, indices = index.search(query_embedding, top_k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(assessments):
                assessment = assessments[idx]
                assessment_with_score = assessment.copy()
                assessment_with_score['score'] = float(distances[0][i])
                results.append(assessment_with_score)
    
        return results





app = Flask(__name__)
    
@app.before_request
def initialize():
        global assessments, model, index
        if not app._got_first_request:
            assessments = load_assessments('assessments.json')
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = create_embeddings(assessments)
            index = setup_faiss_index(embeddings)
            app._got_first_request = True


@app.route('/api/recommend', methods=['GET'])
def recommend():
        query = request.args.get('query', '')
        if not query:
            return jsonify({"error": "No query provided"}), 400
    
        results = search_assessments(query, model, index, assessments)
        return jsonify(results)

@app.route('/', methods=['GET'])
def home():
        logger.info("Root endpoint accessed")
        return jsonify({
            "status": "online", 
            "endpoints": ["/", "/api/recommend", "/health", "/simple"],
            "message": "Assessment recommendation API is running"
        })

@app.route('/simple', methods=['GET'])
def simple():
        logger.info("Simple endpoint accessed")
        return jsonify({"status": "working", "message": "Simple endpoint works"})

@app.route('/health', methods=['GET'])
def health_check():
        logger.info("Health check endpoint accessed")
        return jsonify({"status": "ok"})
    



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting development server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True)