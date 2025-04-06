import streamlit as st
import requests
import pandas as pd

def fetch_recommendations(query):
    url = f"http://localhost:5000/api/recommend?query={query}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching recommendations: {e}")
        return []

def streamlit_app():
    st.title("SHL Assessment Recommendation System")
    query = st.text_area("Enter job description or requirements:", height=150)
    
    if st.button("Get Recommendations"):
        if query:
            results = fetch_recommendations(query)
            if results and isinstance(results, list):
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