from typing import List, Dict, Any
import os
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from flask import Flask, request, jsonify
import streamlit as st

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
assessments = []
model = None
index = None

@app.before_request
def initialize():
    global assessments, model, index
    if not app._got_first_request:
        assessments = load_assessments('shl_assessments_data.json')
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
def streamlit_app():
    st.title("SHL Assessment Recommendation System")
    assessments = load_assessments(r'C:\Users\Sakshi\OneDrive\Desktop\SHL\shl_assessments_data.json')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = create_embeddings(assessments)
    index = setup_faiss_index(embeddings)
    query = st.text_area("Enter job description or requirements:", height=150)
    
    if st.button("Get Recommendations"):
        if query:
            results = search_assessments(query, model, index, assessments)
            if results:
                st.subheader("Recommended Assessments")
                table_data = []
                for result in results:
                    table_data.append({
                        "Assessment Name": f"[{result['name']}]({result['url']})",
                        "Test Type": result['test_type'],
                        "Duration": result['duration'],
                        "Remote Testing": result['remote_testing'],
                        "Adaptive/IRT": result['adaptive_irt']
                    })
                from typing import List, Dict, Any
import os
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from flask import Flask, request, jsonify
import streamlit as st

def load_assessments(file_path: str) -> List[Dict[str, Any]]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:  # Explicit UTF-8 encoding
            assessments = json.load(f)
        return assessments
    except Exception as e:
        st.error(f"Error loading assessments: {str(e)}")
        return []

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
assessments = []
model = None
index = None

@app.before_request
def initialize():
    global assessments, model, index
    if not app._got_first_request:
        assessments = load_assessments('562data.json')
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
def streamlit_app():
    st.title("SHL Assessment Recommendation System")
    assessments = load_assessments(r'C:\Users\Sakshi\OneDrive\Desktop\SHL\562data.json')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = create_embeddings(assessments)
    index = setup_faiss_index(embeddings)
    query = st.text_area("Enter job description or requirements:", height=150)
    
    if st.button("Get Recommendations"):
        if query:
            results = search_assessments(query, model, index, assessments)
            if results:
                st.subheader("Recommended Assessments")
                table_data = []
                for result in results:
                    table_data.append({
                        "Assessment Name": f"[{result['name']}]({result['url']})",
                        "Test Type": result['test_type'],
                        "Duration": result['duration'],
                        "Remote Testing": result['remote_testing'],
                        "Adaptive / IRT": result['adaptive_irt']
                    })
                st.markdown(
                    """
                    <style>
                        .block-container {
                            max-width: 100% !important;
                            padding-left: 2rem;
                            padding-right: 2rem;
                        }
                        table td:nth-child(2) {
                            width: 300px;
                            word-wrap: break-word;
                            white-space: normal;
                        }
                        table td:nth-child(1) {
                            width: 300px !important;
                        }
                        table td:nth-child(4) {
                            width: 250px !important;
                        }
                        table td:nth-child(3) {
                            width: 400px !important;
                        }
                        table td:nth-child(5) {
                            width: 100px !important;
                        }
                        table td:nth-child(6) {
                            width: 100px !important;
                        }
                    </style>
                    """,
                    unsafe_allow_html=True
                )

                st.table(table_data)
        else:
            st.info("No matching assessments found.")
    else:
        st.warning("Please enter a query.")
        
if __name__ == "__main__":
    streamlit_app()
