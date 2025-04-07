import os
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import faiss
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_and_index():
    try:
        logger.info("Preloading model and creating index...")
        model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
        try:
            with open('assessments.json', 'r') as f:
                assessments = json.load(f)
            logger.info(f"Loaded {len(assessments)} assessments")
        except Exception as e:
            logger.warning(f"Could not load assessments: {e}")
            assessments = [
                {"id": 1, "name": "Dummy Assessment", "description": "Placeholder", "test_type": "General"}
            ]
        texts = [f"{a.get('name', '')} {a.get('description', '')} {a.get('test_type', '')}" for a in assessments]
        embeddings = model.encode(texts)
        
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        with open('index.pkl', 'wb') as f:
            pickle.dump(index, f)
            
        with open('assessments_processed.json', 'w') as f:
            json.dump(assessments, f)
            
        logger.info("Preloading complete, resources saved to disk")
        return True
    except Exception as e:
        logger.error(f"Error in preloading: {e}")
        return False

if __name__ == "__main__":
    load_model_and_index()