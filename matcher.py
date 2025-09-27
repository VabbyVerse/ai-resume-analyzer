from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

def get_embedding(text, model):
    """Generates an embedding for a given text."""
    if model is None:
        raise ValueError("SentenceTransformer model is not loaded.")
    return model.encode(text)

def find_best_matches(resume_embedding, jobs_df):
    """Finds the best job matches from the DataFrame."""
    if 'embedding' not in jobs_df.columns or jobs_df.empty:
        return []

    job_embeddings = np.array(jobs_df['embedding'].tolist())
    
    similarity_scores = cosine_similarity(resume_embedding.reshape(1, -1), job_embeddings)[0]
    
    jobs_df['score'] = similarity_scores
    
    top_matches_df = jobs_df.sort_values(by='score', ascending=False).head(5)
    
    return top_matches_df.to_dict('records')