import os
import gc
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

CONTENT_PARQUET = 'processed/content_view.parquet'
SVD_PARQUET = 'processed/svd_view.parquet'
COLLAB_SCORES = 'processed/collaborative_scores.pkl'

VECTORIZER_OUT = 'models/tfidf_vectorizer.pkl'
TFIDF_MATRIX_OUT = 'models/tfidf_matrix.pkl'
SCORES_OUT = 'processed/content_scores.pkl'

def load_data():
    print(f"Loading content data from {CONTENT_PARQUET}...")
    df_content = pd.read_parquet(CONTENT_PARQUET)
    
    print(f"Loading interaction data from {SVD_PARQUET}...")
    df_interactions = pd.read_parquet(SVD_PARQUET)
    
    return df_content, df_interactions

def get_active_users(df_interactions):
    if os.path.exists(COLLAB_SCORES):
        print(f"Loading active users from {COLLAB_SCORES}...")
        with open(COLLAB_SCORES, 'rb') as f:
            collab_scores = pickle.load(f)
        return list(collab_scores.keys())
    else:
        print("Collaborative scores not found. Defaulting to top 500 active users.")
        user_counts = df_interactions['userId'].value_counts()
        return user_counts.head(500).index.tolist()

def build_tfidf_features(df_content, max_features=5000):
    print("Fitting TF-IDF Vectorizer...")
    tfidf = TfidfVectorizer(stop_words='english', max_features=max_features)
    
    tfidf_matrix = tfidf.fit_transform(df_content['soup'])
    
    print(f"TF-IDF Matrix shape: {tfidf_matrix.shape}")
    return tfidf, tfidf_matrix

def generate_user_taste_vector(user_id, df_interactions, tfidf_matrix, movie_id_to_idx):
    user_history = df_interactions[df_interactions['userId'] == user_id]
    
    positive_history = user_history[user_history['rating'] >= 3.5]
    
    rated_movie_ids = set(user_history['movieId'].unique())
    
    if positive_history.empty:
        return None, rated_movie_ids
        
    movie_indices = []
    weights = []
    
    for _, row in positive_history.iterrows():
        mid = row['movieId']
        if mid in movie_id_to_idx:
            movie_indices.append(movie_id_to_idx[mid])
            weights.append(row['rating'])
            
    if not movie_indices:
        return None, rated_movie_ids
        
    movie_vectors = tfidf_matrix[movie_indices]
    
    weights = np.array(weights).reshape(-1, 1)
    
    weighted_sum = movie_vectors.T.dot(weights).ravel()
    
    user_taste_vector = weighted_sum / np.sum(weights)
    
    user_taste_vector = user_taste_vector.reshape(1, -1)
    
    return user_taste_vector, rated_movie_ids

def compute_user_content_scores(user_id, df_interactions, tfidf_matrix, df_content, movie_id_to_idx, top_n=100):
    user_taste_vector, rated_movie_ids = generate_user_taste_vector(
        user_id, df_interactions, tfidf_matrix, movie_id_to_idx
    )
    
    if user_taste_vector is None:
        return {}
        
    sim_scores = linear_kernel(user_taste_vector, tfidf_matrix).flatten()
    
    sim_scores = sim_scores * 5.0
    
    k = min(len(sim_scores) - 1, top_n + len(rated_movie_ids))
    if k <= 0:
        return {}
        
    top_indices = np.argpartition(sim_scores, -k)[-k:]
    
    top_indices = top_indices[np.argsort(sim_scores[top_indices])[::-1]]
    
    final_scores = {}
    for idx in top_indices:
        movie_id = df_content.iloc[idx]['movieId']
        if movie_id not in rated_movie_ids:
            final_scores[movie_id] = float(sim_scores[idx])
            if len(final_scores) == top_n:
                break
                
    return final_scores

def main():
    df_content, df_interactions = load_data()
    
    df_content = df_content.reset_index(drop=True)
    movie_id_to_idx = {row['movieId']: idx for idx, row in df_content.iterrows()}
    
    tfidf_vectorizer, tfidf_matrix = build_tfidf_features(df_content, max_features=5000)
    
    os.makedirs(os.path.dirname(VECTORIZER_OUT), exist_ok=True)
    print(f"Saving TF-IDF vectorizer to {VECTORIZER_OUT}...")
    with open(VECTORIZER_OUT, 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
        
    print(f"Saving TF-IDF matrix to {TFIDF_MATRIX_OUT}...")
    with open(TFIDF_MATRIX_OUT, 'wb') as f:
        pickle.dump(tfidf_matrix, f)
        
    active_users = get_active_users(df_interactions)
        
    print(f"Pre-computing content scores for {len(active_users)} active users...")
    batch_content_scores = {}
    
    for i, user_id in enumerate(active_users):
        if i > 0 and i % 50 == 0:
            print(f"Processed {i}/{len(active_users)} users...")
            gc.collect()
            
        scores = compute_user_content_scores(
            user_id, df_interactions, tfidf_matrix, df_content, movie_id_to_idx, top_n=100
        )
        batch_content_scores[user_id] = scores
        
    os.makedirs(os.path.dirname(SCORES_OUT), exist_ok=True)
    print(f"Saving precomputed content scores to {SCORES_OUT}...")
    with open(SCORES_OUT, 'wb') as f:
        pickle.dump(batch_content_scores, f)
        
    print("CineIQ Content-Based Pipeline Complete.")

if __name__ == "__main__":
    main()
