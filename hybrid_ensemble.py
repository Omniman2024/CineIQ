import os
import gc
import time
import pickle
import numpy as np
import pandas as pd
import mlflow
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import linear_kernel

# Zero setup local backend tracking configuration
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("CineIQ_Hybrid_Pipeline")

SVD_PARQUET = 'processed/svd_view.parquet'
CONTENT_PARQUET = 'processed/content_view.parquet'

MODEL_SVD = 'models/svd_model.pkl'
TFIDF_MATRIX = 'models/tfidf_matrix.pkl'

COLLAB_SCORES = 'processed/collaborative_scores.pkl'
CONTENT_SCORES = 'processed/content_scores.pkl'

META_MODEL_OUT = 'models/stacking_meta_model.pkl'
HYBRID_SCORES_OUT = 'processed/hybrid_scores.pkl'

RANDOM_STATE = 42

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def generate_user_taste_vectors(active_users, df_interactions, tfidf_matrix, movie_id_to_idx):
    print("Precomputing User Taste Vectors for historical meta-dataset...")
    taste_vectors = {}
    
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
            
        movie_vectors = tfidf_matrix[movie_indices]
        weights = np.array(weights).reshape(-1, 1)
        weighted_sum = movie_vectors.T.dot(weights).ravel()
        
        user_taste_vector = weighted_sum / np.sum(weights)
        taste_vectors[user_id] = user_taste_vector.reshape(1, -1)
        
    return taste_vectors

def build_meta_dataset(df_interactions, active_users, svd_model, tfidf_matrix, movie_id_to_idx, sample_size=50000):
    print(f"Building meta-dataset with ~{sample_size} historical samples...")
    
    df_meta = df_interactions[df_interactions['userId'].isin(active_users)].copy()
    
    if len(df_meta) > sample_size:
        df_meta = df_meta.sample(n=sample_size, random_state=RANDOM_STATE)
        
    taste_vectors = generate_user_taste_vectors(active_users, df_interactions, tfidf_matrix, movie_id_to_idx)
    
    features = []
    targets = []
    
    for _, row in df_meta.iterrows():
        user_id = row['userId']
        movie_id = row['movieId']
        actual_rating = row['rating']
        
        collab_score = svd_model.predict(uid=user_id, iid=movie_id).est
        
        content_score = 0.0 
        if user_id in taste_vectors and movie_id in movie_id_to_idx:
            u_vec = taste_vectors[user_id]
            m_vec = tfidf_matrix[movie_id_to_idx[movie_id]]
            sim = linear_kernel(u_vec, m_vec).flatten()[0]
            content_score = sim * 5.0
            
        features.append([collab_score, content_score])
        targets.append(actual_rating)
        
    X = np.array(features)
    y = np.array(targets)
    
    return X, y

def train_stacking_model(X, y):
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
    
    # Log optimization coefficients directly to active tracking telemetry
    mlflow.log_metric("w1", float(w1))
    mlflow.log_metric("w2", float(w2))
    mlflow.log_metric("intercept", float(intercept))
    
    return meta_model

def candidate_inference_combinator(active_users, collab_scores_dict, content_scores_dict, meta_model, svd_model, top_n=100):
    print("Executing Candidate Inference Combinator...")
    w1, w2 = meta_model.coef_
    intercept = meta_model.intercept_
    
    hybrid_scores_dict = {}
    
    for i, user_id in enumerate(active_users):
        if i > 0 and i % 100 == 0:
            print(f"Combined candidates for {i}/{len(active_users)} users...")
            
        c_scores = collab_scores_dict.get(user_id, {})
        t_scores = content_scores_dict.get(user_id, {})

        all_candidates = set(c_scores.keys()).union(set(t_scores.keys()))
        
        user_hybrid_scores = []
        for movie_id in all_candidates:
            if movie_id in c_scores:
                s_collab = c_scores[movie_id]
            else:
                s_collab = svd_model.predict(uid=user_id, iid=movie_id).est
                
            s_content = t_scores.get(movie_id, 0.0)
            
            final_score = (w1 * s_collab) + (w2 * s_content) + intercept
            user_hybrid_scores.append((movie_id, final_score))
            
        user_hybrid_scores.sort(key=lambda x: x[1], reverse=True)
        top_candidates = user_hybrid_scores[:top_n]
        
        hybrid_scores_dict[user_id] = {m: float(score) for m, score in top_candidates}
        
    return hybrid_scores_dict

def main():
    # Start tracking context for the Stacking Ensemble Meta-Model policy
    with mlflow.start_run():
        start_pipeline_time = time.time()
        
        # Log metadata parameters
        sample_size_param = 50000
        mlflow.log_param("sample_size", sample_size_param)
        mlflow.log_param("random_state", RANDOM_STATE)
        
        print("Loading models and candidate dictionaries...")
        svd_model = load_pkl(MODEL_SVD)
        tfidf_matrix = load_pkl(TFIDF_MATRIX)
        
        collab_scores_dict = load_pkl(COLLAB_SCORES)
        content_scores_dict = load_pkl(CONTENT_SCORES)
        
        active_users = list(collab_scores_dict.keys())
        mlflow.log_metric("active_users_count", len(active_users))
        
        print("Loading datasets...")
        df_interactions = pd.read_parquet(SVD_PARQUET)
        df_content = pd.read_parquet(CONTENT_PARQUET)
        
        df_content = df_content.reset_index(drop=True)
        movie_id_to_idx = {row['movieId']: idx for idx, row in df_content.iterrows()}
        
        del df_content
        gc.collect()
        
        X, y = build_meta_dataset(
            df_interactions, active_users, svd_model, tfidf_matrix, movie_id_to_idx, sample_size=sample_size_param
        )
        
        del df_interactions
        gc.collect()
        
        # Train and automatically log w1, w2, intercept
        meta_model = train_stacking_model(X, y)
        
        os.makedirs(os.path.dirname(META_MODEL_OUT), exist_ok=True)
        with open(META_MODEL_OUT, 'wb') as f:
            pickle.dump(meta_model, f)
            
        start_combinator_time = time.time()
        hybrid_scores_dict = candidate_inference_combinator(
            active_users, collab_scores_dict, content_scores_dict, meta_model, svd_model, top_n=100
        )
        end_combinator_time = time.time()
        mlflow.log_metric("combinator_inference_time_seconds", end_combinator_time - start_combinator_time)
        
        os.makedirs(os.path.dirname(HYBRID_SCORES_OUT), exist_ok=True)
        print(f"Saving final hybrid recommendations to {HYBRID_SCORES_OUT}...")
        with open(HYBRID_SCORES_OUT, 'wb') as f:
            pickle.dump(hybrid_scores_dict, f)
            
        # Log artifacts directly to local experiment payload tracker
        mlflow.log_artifact(META_MODEL_OUT)
        mlflow.log_artifact(HYBRID_SCORES_OUT)
        
        end_pipeline_time = time.time()
        mlflow.log_metric("total_pipeline_time_seconds", end_pipeline_time - start_pipeline_time)
        
        print("CineIQ Hybrid Ensemble Pipeline Complete. Telemetry tracking logged.")

if __name__ == "__main__":
    main()