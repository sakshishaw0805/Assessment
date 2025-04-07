from typing import List, Dict, Any
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from flask import Flask, request, jsonify
import logging
import gc
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

assessments = None
model = None
index = None
initialized = False

def load_assessments(file_path: str) -> List[Dict[str, Any]]:
    try:
        logger.info(f"Attempting to load assessments from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load assessments: {str(e)}")
        raise

def create_embeddings_batched(assessments: List[Dict[str, Any]], model_name: str = 'all-MiniLM-L6-v2', batch_size: int = 16) -> tuple:
    logger.info(f"Loading model: {model_name}")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(1)
    model = SentenceTransformer(model_name, device='cpu')
    texts = [f"{assessment.get('name', '')} {assessment.get('description', '')} {assessment.get('test_type', '')}" 
             for assessment in assessments]
    logger.info(f"Processing {len(texts)} texts in batches of {batch_size}")
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
        all_embeddings.append(batch_embeddings)
        gc.collect()
    embeddings = np.vstack(all_embeddings)
    logger.info(f"Created embeddings with shape {embeddings.shape}")
    return embeddings, model

def setup_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    logger.info("FAISS index created")
    return index

def search_assessments(query: str, model, index, assessments: List[Dict[str, Any]], top_k: int = 10):
    query_embedding = model.encode([query], show_progress_bar=False)
    distances, indices = index.search(query_embedding, top_k)
    results = [assessments[idx].copy() 
               for i, idx in enumerate(indices[0]) if idx < len(assessments)]
    gc.collect()
    return results

def initialize_app():
    global assessments, model, index, initialized
    if initialized:
        logger.info("Already initialized")
        return
    logger.info("Starting initialization...")
    try:
        gc.collect()
        assessments = load_assessments('assessments.json')
        logger.info(f"Loaded {len(assessments)} assessments")
        embeddings, sentence_model = create_embeddings_batched(assessments, batch_size=16)
        model = sentence_model
        index = setup_faiss_index(embeddings)
        initialized = True
        logger.info("Initialization completed successfully")
    except Exception as e:
        logger.error(f"Initialization failed: {str(e)}")
        initialized = False
        raise

@app.route('/api/recommend', methods=['GET'])
def recommend():
    global model, index, assessments, initialized
    if not initialized:
        logger.warning("Not initialized yet, attempting to initialize now")
        try:
            initialize_app()
        except Exception as e:
            return jsonify({"error": f"Initialization failed: {str(e)}"}), 503
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
    return jsonify({"status": "online" if initialized else "initializing", "message": "API running"})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok" if initialized else "initializing"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting server on port {port}")
    try:
        initialize_app() 
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        exit(1)
    app.run(host="0.0.0.0", port=port, debug=False)