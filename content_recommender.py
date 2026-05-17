import os
import gc
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# ==========================================
# CINEIQ - CONTENT-BASED FILTERING PIPELINE
# ==========================================

# Path configurations
CONTENT_PARQUET = 'processed/content_view.parquet'
SVD_PARQUET = 'processed/svd_view.parquet'
COLLAB_SCORES = 'processed/collaborative_scores.pkl'

VECTORIZER_OUT = 'models/tfidf_vectorizer.pkl'
TFIDF_MATRIX_OUT = 'models/tfidf_matrix.pkl'
SCORES_OUT = 'processed/content_scores.pkl'

def load_data():
    """
    Safely load content and interaction data.
    """
    print(f"Loading content data from {CONTENT_PARQUET}...")
    df_content = pd.read_parquet(CONTENT_PARQUET)
    
    print(f"Loading interaction data from {SVD_PARQUET}...")
    df_interactions = pd.read_parquet(SVD_PARQUET)
    
    return df_content, df_interactions

def get_active_users(df_interactions):
    """
    Load the exact same batch of active users used in the collaborative pipeline
    to ensure perfect alignment for the hybrid ensemble later.
    """
    if os.path.exists(COLLAB_SCORES):
        print(f"Loading active users from {COLLAB_SCORES}...")
        with open(COLLAB_SCORES, 'rb') as f:
            collab_scores = pickle.load(f)
        return list(collab_scores.keys())
    else:
        print("Collaborative scores not found. Defaulting to top 500 active users.")
        # Fallback if collab scores haven't been generated
        user_counts = df_interactions['userId'].value_counts()
        return user_counts.head(500).index.tolist()

def build_tfidf_features(df_content, max_features=5000):
    """
    Vectorize the 'soup' text using TF-IDF.
    """
    print("Fitting TF-IDF Vectorizer...")
    # Limiting max_features to 5000 caps the vocabulary size, ensuring the 
    # sparse matrix stays compact in memory (critical for 16GB RAM limit).
    # stop_words='english' removes uninformative common words (e.g., 'the', 'is').
    tfidf = TfidfVectorizer(stop_words='english', max_features=max_features)
    
    # Fit and transform the text soup into a sparse TF-IDF matrix (V_m).
    # Each row corresponds to a movie, each column represents a word feature.
    # Values represent the term's frequency offset by its document frequency (rarity/importance).
    tfidf_matrix = tfidf.fit_transform(df_content['soup'])
    
    print(f"TF-IDF Matrix shape: {tfidf_matrix.shape}")
    return tfidf, tfidf_matrix

def generate_user_taste_vector(user_id, df_interactions, tfidf_matrix, movie_id_to_idx):
    """
    Create a 'User Taste Vector' (V_u) by taking the weighted average
    of the TF-IDF vectors for movies the user has rated highly.
    """
    # 1. Isolate user's interactions
    user_history = df_interactions[df_interactions['userId'] == user_id]
    
    # 2. Filter highly rated movies (>= 3.5 stars as an indicator of positive preference)
    positive_history = user_history[user_history['rating'] >= 3.5]
    
    rated_movie_ids = set(user_history['movieId'].unique())
    
    if positive_history.empty:
        return None, rated_movie_ids
        
    # Get indices of positively rated movies in the TF-IDF matrix and their weights
    movie_indices = []
    weights = []
    
    for _, row in positive_history.iterrows():
        mid = row['movieId']
        if mid in movie_id_to_idx:
            movie_indices.append(movie_id_to_idx[mid])
            weights.append(row['rating']) # Use the rating as the weight
            
    if not movie_indices:
        return None, rated_movie_ids
        
    # Retrieve the sparse vectors for these highly-rated movies
    movie_vectors = tfidf_matrix[movie_indices]
    
    # Compute weighted average to form the User Taste Vector (V_u)
    # V_u = sum(weight_i * V_m_i) / sum(weight_i)
    weights = np.array(weights).reshape(-1, 1)
    
    # Perform sparse matrix multiplication:
    # movie_vectors is (N_movies, features), weights is (N_movies, 1)
    # We transpose movie_vectors to multiply with weights: (features, N) x (N, 1) -> (features, 1)
    weighted_sum = movie_vectors.T.dot(weights).ravel()
    
    # Normalize by the sum of all weights
    user_taste_vector = weighted_sum / np.sum(weights)
    
    # Reshape to (1, features) for compatibility with sklearn's similarity functions
    user_taste_vector = user_taste_vector.reshape(1, -1)
    
    return user_taste_vector, rated_movie_ids

def compute_user_content_scores(user_id, df_interactions, tfidf_matrix, df_content, movie_id_to_idx, top_n=100):
    """
    Compute similarity between the User Taste Vector and all unrated movies dynamically.
    
    Linear Algebra & Memory Optimization:
    Cosine Similarity = (V_u * V_m) / (||V_u|| * ||V_m||)
    Because TfidfVectorizer naturally L2-normalizes its output rows, taking the dot product 
    is mathematically equivalent to cosine similarity. 
    Using `linear_kernel` (dot product) rather than computing an NxN dense cosine similarity matrix 
    avoids O(N^2) memory scaling, performing this query in O(N_movies * features) space dynamically per user.
    """
    user_taste_vector, rated_movie_ids = generate_user_taste_vector(
        user_id, df_interactions, tfidf_matrix, movie_id_to_idx
    )
    
    if user_taste_vector is None:
        # User has no positive history or no overlapping movies; return empty dict
        return {}
        
    # Calculate dot product (cosine similarity) against all movies
    # Returns an array of shape (1, N_movies) -> flatten to (N_movies,)
    sim_scores = linear_kernel(user_taste_vector, tfidf_matrix).flatten()
    
    # Scale similarity scores (0.0 to 1.0) to match collaborative ratings scale (0.0 to 5.0)
    # This prepares the data for a cleaner weighted sum in the final hybrid layer.
    sim_scores = sim_scores * 5.0
    
    # Identify the indices of the top N + len(rated_movies) highest scores.
    # np.argpartition is O(n), significantly faster than a full O(n log n) sort.
    k = min(len(sim_scores) - 1, top_n + len(rated_movie_ids))
    if k <= 0:
        return {}
        
    top_indices = np.argpartition(sim_scores, -k)[-k:]
    
    # Sort only those top k elements in descending order
    top_indices = top_indices[np.argsort(sim_scores[top_indices])[::-1]]
    
    # Filter out already rated movies and populate the final lookup dictionary
    final_scores = {}
    for idx in top_indices:
        movie_id = df_content.iloc[idx]['movieId']
        if movie_id not in rated_movie_ids:
            final_scores[movie_id] = float(sim_scores[idx]) # Cast to float for standard serialization
            if len(final_scores) == top_n:
                break
                
    return final_scores

def main():
    # 1. Load Data
    df_content, df_interactions = load_data()
    
    # Map movieId to its corresponding row index in the TF-IDF matrix
    df_content = df_content.reset_index(drop=True)
    movie_id_to_idx = {row['movieId']: idx for idx, row in df_content.iterrows()}
    
    # 2. Build TF-IDF Feature Matrix
    tfidf_vectorizer, tfidf_matrix = build_tfidf_features(df_content, max_features=5000)
    
    # Save the vectorizer and matrix (Output 1)
    os.makedirs(os.path.dirname(VECTORIZER_OUT), exist_ok=True)
    print(f"Saving TF-IDF vectorizer to {VECTORIZER_OUT}...")
    with open(VECTORIZER_OUT, 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
        
    print(f"Saving TF-IDF matrix to {TFIDF_MATRIX_OUT}...")
    with open(TFIDF_MATRIX_OUT, 'wb') as f:
        pickle.dump(tfidf_matrix, f)
        
    # 3. Precompute Content Scores for the Active User Batch (Output 2)
    active_users = get_active_users(df_interactions)
        
    print(f"Pre-computing content scores for {len(active_users)} active users...")
    batch_content_scores = {}
    
    for i, user_id in enumerate(active_users):
        if i > 0 and i % 50 == 0:
            print(f"Processed {i}/{len(active_users)} users...")
            # Aggressive garbage collection prevents gradual memory creep during iteration
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
