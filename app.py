from typing import List, Dict, Any
import json
import numpy as np
from flask import Flask, request, jsonify
import os
import logging
import pickle
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
assessments: List[Dict[str, Any]] = []
model = None
index = None

def create_app():
    app = Flask(__name__)
    
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
    
    @app.route('/api/recommend', methods=['GET'])
    def recommend():
        logger.info("Received request to /api/recommend")
        query = request.args.get('query', '')
        logger.info(f"Query: {query}")
        
        if not query:
            logger.warning("No query provided")
            return jsonify({"error": "No query provided"}), 400
        
        global model, index, assessments
        if model is None or index is None:
            logger.info("Initializing resources...")
            try:
                initialize_resources()
            except Exception as e:
                logger.error(f"Failed to initialize resources: {str(e)}")
                return jsonify({"error": "Server initialization error", "details": str(e)}), 500
        
        logger.info(f"Searching assessments, found {len(assessments)} items")
        try:
            results = search_assessments(query)
            logger.info(f"Search complete, found {len(results)} results")
            return jsonify(results)
        except Exception as e:
            logger.error(f"Error searching assessments: {str(e)}")
            return jsonify({"error": "Search error", "details": str(e)}), 500
    
    @app.route('/health', methods=['GET'])
    def health_check():
        logger.info("Health check endpoint accessed")
        return jsonify({"status": "ok"})
    
    return app

def initialize_resources():
    global assessments, model, index
    
    try:
        logger.info("Loading pre-generated resources from disk...")
        try:
            with open('model.pkl', 'rb') as f:
                model = pickle.load(f)
            logger.info("Loaded model from pickle file")
        except Exception as e:
            logger.warning(f"Could not load model from pickle: {e}")
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
            logger.info("Loaded fresh model instance")
        try:
            with open('assessments_processed.json', 'r') as f:
                assessments = json.load(f)
            logger.info(f"Loaded {len(assessments)} processed assessments")
        except Exception as e:
            logger.warning(f"Could not load processed assessments: {e}")
            assessments = load_assessments('assessments.json')
        try:
            import faiss
            with open('index.pkl', 'rb') as f:
                index = pickle.load(f)
            logger.info("Loaded index from pickle file")
        except Exception as e:
            logger.warning(f"Could not load index from pickle: {e}")
            logger.info("Creating new embeddings and index...")
            embeddings = create_embeddings(assessments)
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)
            
        logger.info("Resources initialized successfully")
    except Exception as e:
        logger.error(f"Error in initialize_resources: {e}")
        raise

def load_assessments(file_path: str) -> List[Dict[str, Any]]:
    try:
        logger.info(f"Attempting to load assessments from {file_path}")
        logger.info(f"Current directory: {os.getcwd()}")
        try:
            logger.info(f"Directory contents: {os.listdir('.')}")
        except Exception as e:
            logger.warning(f"Could not list directory: {str(e)}")
        
        if not os.path.exists(file_path):
            logger.warning(f"File {file_path} not found, using dummy data")
            return [
                {"id": 1, "name": "Dummy Assessment 1", "description": "A placeholder assessment", "test_type": "General"},
                {"id": 2, "name": "Dummy Assessment 2", "description": "Another placeholder", "test_type": "Technical"}
            ]
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logger.info(f"Successfully loaded {len(data)} assessments")
            return data
    except Exception as e:
        logger.error(f"Error loading assessments: {str(e)}")
        return [
            {"id": 1, "name": "Dummy Assessment 1", "description": "A placeholder assessment", "test_type": "General"},
            {"id": 2, "name": "Dummy Assessment 2", "description": "Another placeholder", "test_type": "Technical"}
        ]

def create_embeddings(assessments: List[Dict[str, Any]]) -> np.ndarray:
    global model
    texts = [f"{a.get('name', '')} {a.get('description', '')} {a.get('test_type', '')}" for a in assessments]
    logger.info(f"Encoding {len(texts)} texts")
    return model.encode(texts)

def search_assessments(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    global model, index, assessments
    logger.info(f"Encoding search query: {query}")
    query_embedding = model.encode([query])
    logger.info("Searching FAISS index")
    distances, indices = index.search(query_embedding, min(top_k, len(assessments)))
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(assessments):
            assessment = assessments[idx].copy()
            assessment['score'] = float(distances[0][i])
            results.append(assessment)
    logger.info(f"Found {len(results)} matching assessments")
    return results

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting development server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True)