import os
import gc
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import linear_kernel

# ==========================================
# CINEIQ - HYBRID ENSEMBLE (STACKING) PIPELINE
# ==========================================

# Path configurations
SVD_PARQUET = 'processed/svd_view.parquet'
CONTENT_PARQUET = 'processed/content_view.parquet'

MODEL_SVD = 'models/svd_model.pkl'
TFIDF_MATRIX = 'models/tfidf_matrix.pkl'
# Note: tfidf_vectorizer.pkl is available, but the precomputed matrix is what we need for similarities.

COLLAB_SCORES = 'processed/collaborative_scores.pkl'
CONTENT_SCORES = 'processed/content_scores.pkl'

META_MODEL_OUT = 'models/stacking_meta_model.pkl'
HYBRID_SCORES_OUT = 'processed/hybrid_scores.pkl'

RANDOM_STATE = 42

def load_pkl(path):
    """Utility to safely load pickle objects."""
    with open(path, 'rb') as f:
        return pickle.load(f)

def generate_user_taste_vectors(active_users, df_interactions, tfidf_matrix, movie_id_to_idx):
    """
    Precompute User Taste Vectors for active users to speed up dynamic content scoring.
    """
    print("Precomputing User Taste Vectors for historical meta-dataset...")
    taste_vectors = {}
    
    # Filter interactions for active users only to speed up pandas operations
    df_active = df_interactions[df_interactions['userId'].isin(active_users)]
    
    for user_id in active_users:
        user_history = df_active[df_active['userId'] == user_id]
        positive_history = user_history[user_history['rating'] >= 3.5]
        
        if positive_history.empty:
            continue
            
        movie_indices = []
        weights = []
        
        for _, row in positive_history.iterrows():
            mid = row['movieId']
            if mid in movie_id_to_idx:
                movie_indices.append(movie_id_to_idx[mid])
                weights.append(row['rating'])
                
        if not movie_indices:
            continue
            
        # Extract sparse vectors and calculate weighted sum
        movie_vectors = tfidf_matrix[movie_indices]
        weights = np.array(weights).reshape(-1, 1)
        weighted_sum = movie_vectors.T.dot(weights).ravel()
        
        # Normalize and store
        user_taste_vector = weighted_sum / np.sum(weights)
        taste_vectors[user_id] = user_taste_vector.reshape(1, -1)
        
    return taste_vectors

def build_meta_dataset(df_interactions, active_users, svd_model, tfidf_matrix, movie_id_to_idx, sample_size=50000):
    """
    Build the training dataset for the Stacking Meta-Model.
    Extracts a sample of historical ratings, then computes independent feature predictions.
    """
    print(f"Building meta-dataset with ~{sample_size} historical samples...")
    
    # 1. Filter historical ground truth ratings down to our active user batch
    df_meta = df_interactions[df_interactions['userId'].isin(active_users)].copy()
    
    # 2. Sample down to manage memory footprint (16GB constraints) and training time
    if len(df_meta) > sample_size:
        df_meta = df_meta.sample(n=sample_size, random_state=RANDOM_STATE)
        
    # Precalculate content taste vectors to avoid O(N^2) inner loop overhead
    taste_vectors = generate_user_taste_vectors(active_users, df_interactions, tfidf_matrix, movie_id_to_idx)
    
    features = []
    targets = []
    
    for _, row in df_meta.iterrows():
        user_id = row['userId']
        movie_id = row['movieId']
        actual_rating = row['rating']
        
        # -- Generate Feature 1: Collaborative Score (SVD) --
        # SVD inherently provides estimates for any valid user-item pair
        collab_score = svd_model.predict(uid=user_id, iid=movie_id).est
        
        # -- Generate Feature 2: Content Score (TF-IDF & Cosine Similarity) --
        content_score = 0.0 # Default missing-data fallback
        if user_id in taste_vectors and movie_id in movie_id_to_idx:
            u_vec = taste_vectors[user_id]
            m_vec = tfidf_matrix[movie_id_to_idx[movie_id]]
            # Dot product (linear kernel) against scaled matrices functions identically to cosine similarity.
            # We scale the 0.0-1.0 probability up to a 0.0-5.0 continuum to match collaborative inputs.
            sim = linear_kernel(u_vec, m_vec).flatten()[0]
            content_score = sim * 5.0
            
        features.append([collab_score, content_score])
        targets.append(actual_rating)
        
    X = np.array(features)
    y = np.array(targets)
    
    return X, y

def train_stacking_model(X, y):
    """
    Train LinearRegression meta-model to learn optimal ensemble weights.
    
    Translating Empirical Behavior to Policy:
    Instead of manually guessing weights (e.g., w1 = 0.6 and w2 = 0.4), Feature-Weighted Linear Stacking 
    uses Ordinary Least Squares (OLS) Regression. It objectively evaluates how often the Collaborative Model 
    versus the Content Model predicted correctly against the ground truth labels in the historical data. 
    It calculates coefficients that minimize the Residual Sum of Squares, yielding mathematically optimal weighting.
    """
    print("Training LinearRegression Meta-Model...")
    meta_model = LinearRegression()
    meta_model.fit(X, y)
    
    w1, w2 = meta_model.coef_
    intercept = meta_model.intercept_
    
    print("\n=== STACKING META-MODEL POLICY ===")
    print(f"Learned Collaborative Weight (w1): {w1:.4f}")
    print(f"Learned Content Weight (w2):       {w2:.4f}")
    print(f"Learned Intercept:                 {intercept:.4f}")
    print("==================================\n")
    
    return meta_model

def candidate_inference_combinator(active_users, collab_scores_dict, content_scores_dict, meta_model, svd_model, top_n=100):
    """
    Calculate final hybrid scores for unrated candidates using learned coefficients.
    Uses a Union strategy to resolve the Top-100 Intersection Paradox.
    """
    print("Executing Candidate Inference Combinator...")
    w1, w2 = meta_model.coef_
    intercept = meta_model.intercept_
    
    hybrid_scores_dict = {}
    
    for i, user_id in enumerate(active_users):
        if i > 0 and i % 100 == 0:
            print(f"Combined candidates for {i}/{len(active_users)} users...")
            
        c_scores = collab_scores_dict.get(user_id, {})
        t_scores = content_scores_dict.get(user_id, {})
        
        # Identify the full pool of candidate movies surfaced by EITHER model
        all_candidates = set(c_scores.keys()).union(set(t_scores.keys()))
        
        user_hybrid_scores = []
        for movie_id in all_candidates:
            # Dynamic Fallback 1: Calculate missing collaborative score on the fly
            if movie_id in c_scores:
                s_collab = c_scores[movie_id]
            else:
                s_collab = svd_model.predict(uid=user_id, iid=movie_id).est
                
            # Dynamic Fallback 2: Safely default missing text similarity to 0.0
            s_content = t_scores.get(movie_id, 0.0)
            
            # Apply learned meta-model policy formula dynamically
            final_score = (w1 * s_collab) + (w2 * s_content) + intercept
            user_hybrid_scores.append((movie_id, final_score))
            
        # Sort combined results descending and slice top N
        user_hybrid_scores.sort(key=lambda x: x[1], reverse=True)
        top_candidates = user_hybrid_scores[:top_n]
        
        # Enforce expected O(1) queryable nested dictionary layout
        hybrid_scores_dict[user_id] = {m: float(score) for m, score in top_candidates}
        
    return hybrid_scores_dict

def main():
    print("Loading models and candidate dictionaries...")
    svd_model = load_pkl(MODEL_SVD)
    tfidf_matrix = load_pkl(TFIDF_MATRIX)
    
    collab_scores_dict = load_pkl(COLLAB_SCORES)
    content_scores_dict = load_pkl(CONTENT_SCORES)
    
    # Establish the ground truth active user index
    active_users = list(collab_scores_dict.keys())
    
    print("Loading datasets...")
    df_interactions = pd.read_parquet(SVD_PARQUET)
    df_content = pd.read_parquet(CONTENT_PARQUET)
    
    # Map movieId to TF-IDF matrix index explicitly
    df_content = df_content.reset_index(drop=True)
    movie_id_to_idx = {row['movieId']: idx for idx, row in df_content.iterrows()}
    
    # Aggressively clear df_content as we only needed the mapping index
    del df_content
    gc.collect()
    
    # 1. Build Meta Dataset (X, y)
    X, y = build_meta_dataset(
        df_interactions, active_users, svd_model, tfidf_matrix, movie_id_to_idx, sample_size=50000
    )
    
    # Aggressively clear interactions dataframe before training to protect RAM
    del df_interactions
    gc.collect()
    
    # 2. Train Stacking Model
    meta_model = train_stacking_model(X, y)
    
    # Save Stacking Model Output
    os.makedirs(os.path.dirname(META_MODEL_OUT), exist_ok=True)
    with open(META_MODEL_OUT, 'wb') as f:
        pickle.dump(meta_model, f)
        
    # 3. Combine Precomputed Candidates using the new weights
    hybrid_scores_dict = candidate_inference_combinator(
        active_users, collab_scores_dict, content_scores_dict, meta_model, svd_model, top_n=100
    )
    
    # 4. Save Final Hybrid Scores Output
    os.makedirs(os.path.dirname(HYBRID_SCORES_OUT), exist_ok=True)
    print(f"Saving final hybrid recommendations to {HYBRID_SCORES_OUT}...")
    with open(HYBRID_SCORES_OUT, 'wb') as f:
        pickle.dump(hybrid_scores_dict, f)
        
    print("CineIQ Hybrid Ensemble Pipeline Complete.")

if __name__ == "__main__":
    main()
