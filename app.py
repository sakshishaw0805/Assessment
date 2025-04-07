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
initialized = False

def load_assessments(file_path: str) -> List[Dict[str, Any]]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            assessments = json.load(f)
        return assessments
    except UnicodeDecodeError:
        logger.warning("UTF-8 encoding failed, trying with utf-8-sig")
        try:
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                assessments = json.load(f)
            return assessments
        except Exception as e:
            logger.error(f"Failed to load assessments with alternate encoding: {str(e)}")
            raise
    except FileNotFoundError:
        logger.error(f"Assessment file not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in assessment file: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading assessments: {str(e)}")
        raise

def create_embeddings(assessments: List[Dict[str, Any]], model_name: str = 'all-MiniLM-L6-v2') -> tuple:
    model = SentenceTransformer(model_name)
    texts = [f"{assessment['name']} {assessment['description']} {assessment['test_type']}" 
            for assessment in assessments]
    embeddings = model.encode(texts)
    return embeddings, model

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

def initialize_app():
    global assessments, model, index, initialized
    
    if initialized:
        return
        
    logger.info("Initializing application components...")
    try:
        assessments = load_assessments('assessments.json')
        logger.info(f"Loaded {len(assessments)} assessments")
        
        embeddings, sentence_model = create_embeddings(assessments)
        model = sentence_model
        logger.info(f"Created embeddings with shape {embeddings.shape}")
        
        index = setup_faiss_index(embeddings)
        logger.info("FAISS index created successfully")
        
        initialized = True
    except Exception as e:
        logger.error(f"Initialization error: {str(e)}")
        raise

app = Flask(__name__)
@app.route('/initialize', methods=['GET'])
def init_route():
    initialize_app()
    return jsonify({"status": "initialization complete"})

@app.route('/api/recommend', methods=['GET'])
def recommend():
    global model, index, assessments, initialized
    if not initialized:
        logger.warning("Components not initialized, attempting to initialize now")
        initialize_app()
        
    query = request.args.get('query', '')
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    try:
        results = search_assessments(query, model, index, assessments)
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error in recommendation: {str(e)}")
        return jsonify({"error": f"Failed to process query: {str(e)}"}), 500

@app.route('/', methods=['GET'])
def home():
    if not initialized:
        initialize_app()
    
    logger.info("Root endpoint accessed")
    return jsonify({
        "status": "online", 
        "endpoints": ["/", "/api/recommend", "/health", "/simple", "/initialize"],
        "message": "Assessment recommendation API is running"
    })

@app.route('/simple', methods=['GET'])
def simple():
    logger.info("Simple endpoint accessed")
    return jsonify({"status": "working", "message": "Simple endpoint works"})

@app.route('/health', methods=['GET'])
def health_check():
    logger.info("Health check endpoint accessed")
    components_status = "initialized" if initialized else "not initialized"
    return jsonify({"status": "ok", "components": components_status})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting development server on port {port}")
    initialize_app()
    app.run(host="0.0.0.0", port=port, debug=True)