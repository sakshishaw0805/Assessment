from typing import List, Dict, Any
import os
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from flask import Flask, request, jsonify
import logging
import gc
import torch

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables
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
        # Fallback to different encodings if UTF-8 fails
        logger.warning("UTF-8 encoding failed, trying with utf-8-sig")
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            assessments = json.load(f)
        return assessments

def create_embeddings_batched(assessments: List[Dict[str, Any]], model_name: str = 'all-MiniLM-L6-v2', batch_size: int = 32) -> tuple:
    """Create embeddings in batches to reduce memory usage"""
    logger.info(f"Loading model: {model_name}")
    
    # Set PyTorch to use CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(1)  # Reduce thread count
    
    # Load the model with minimal memory usage
    model = SentenceTransformer(model_name, device='cpu')
    
    # Create texts for embedding
    texts = [f"{assessment.get('name', '')} {assessment.get('description', '')} {assessment.get('test_type', '')}" 
             for assessment in assessments]
    
    logger.info(f"Processing {len(texts)} texts in batches of {batch_size}")
    
    # Process in batches
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        # Generate embeddings for this batch
        batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
        all_embeddings.append(batch_embeddings)
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Combine all batch embeddings
    embeddings = np.vstack(all_embeddings)
    logger.info(f"Created embeddings with shape {embeddings.shape}")
    
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
    # Create query embedding
    query_embedding = model.encode([query], show_progress_bar=False)
    
    # Search the index
    distances, indices = index.search(query_embedding, top_k)
    
    # Create results
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(assessments):
            assessment = assessments[idx]
            assessment_with_score = assessment.copy()
            assessment_with_score['score'] = float(distances[0][i])
            results.append(assessment_with_score)
    
    # Force garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results

def initialize_app():
    """Initialize application components with memory optimization"""
    global assessments, model, index, initialized
    
    if initialized:
        return
        
    logger.info("Initializing application components...")
    try:
        # Clear memory before starting
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Load assessments
        assessments = load_assessments('assessments.json')
        logger.info(f"Loaded {len(assessments)} assessments")
        
        # Create embeddings with batch processing
        embeddings, sentence_model = create_embeddings_batched(
            assessments, 
            batch_size=32  # Adjust batch size based on available memory
        )
        model = sentence_model
        
        # Create index
        index = setup_faiss_index(embeddings)
        logger.info("FAISS index created successfully")
        
        initialized = True
        
        # Clean up any temporary objects
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        logger.error(f"Initialization error: {str(e)}")
        raise

app = Flask(__name__)

@app.route('/api/recommend', methods=['GET'])
def recommend():
    global model, index, assessments, initialized
    
    # Check if components are initialized
    if not initialized:
        return jsonify({"error": "System is initializing. Please try again later."}), 503
        
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
    logger.info("Root endpoint accessed")
    return jsonify({
        "status": "online" if initialized else "initializing", 
        "endpoints": ["/", "/api/recommend", "/health"],
        "message": "Assessment recommendation API is running"
    })

@app.route('/health', methods=['GET'])
def health_check():
    logger.info("Health check endpoint accessed")
    memory_info = {}
    
    # Get Python garbage collector stats
    memory_info["gc_count"] = gc.get_count()
    
    # Get PyTorch memory stats if available
    if torch.cuda.is_available():
        memory_info["cuda_allocated"] = f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB"
        memory_info["cuda_reserved"] = f"{torch.cuda.memory_reserved() / 1024**2:.2f} MB"
    
    return jsonify({
        "status": "ok" if initialized else "initializing", 
        "components": "initialized" if initialized else "not initialized",
        "memory": memory_info
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    workers = int(os.environ.get("WEB_CONCURRENCY", 1))
    
    logger.info(f"Starting server on port {port} with {workers} workers")
    
    # Only initialize in the main process, not in the worker processes
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        initialize_app()
    
    app.run(host="0.0.0.0", port=port, debug=False)