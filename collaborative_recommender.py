import os
import gc
import pickle
import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

# ==========================================
# CINEIQ - COLLABORATIVE FILTERING PIPELINE
# ==========================================

# Path configurations
INPUT_PARQUET = 'processed/svd_view.parquet'
MODEL_OUT = 'models/svd_model.pkl'
SCORES_OUT = 'processed/collaborative_scores.pkl'

# Random state for reproducibility
RANDOM_STATE = 42

def load_and_sample_data(filepath, sample_frac=0.2):
    """
    Load data efficiently and sample to avoid OOM errors.
    25M rows into Surprise can consume substantial memory on a 16GB system.
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_parquet(filepath)
    print(f"Original shape: {df.shape}")
    
    # We take a random sample to fit comfortably in 16GB RAM while 
    # building the interaction matrices in Surprise. 
    # To maintain distribution, we could stratify, but random sampling is usually sufficient 
    # for such large-scale uniform interaction data.
    df_sampled = df.sample(frac=sample_frac, random_state=RANDOM_STATE)
    print(f"Sampled shape: {df_sampled.shape}")
    
    # Cleanup memory
    del df
    gc.collect()
    
    return df_sampled

def train_svd_model(df):
    """
    Train SVD Model using Surprise.
    
    Linear Algebra Mapping:
    SVD performs matrix factorization, splitting the sparse rating matrix R into:
    R ≈ µ + b_u + b_i + q_i^T * p_u
    
    where:
    - µ is the global mean of all ratings
    - b_u and b_i are user and item biases
    - q_i is the item latent feature vector (movie's hidden characteristics)
    - p_u is the user latent feature vector (user's hidden preferences)
    
    The algorithm minimizes squared error with a regularization term (λ) 
    to prevent overfitting the latent dimensions.
    """
    print("Preparing dataset for Surprise...")
    # Reader defines the rating scale. 0.5 to 5.0 is standard for movie ratings.
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
    
    print("Splitting train/test sets...")
    trainset, testset = train_test_split(data, test_size=0.2, random_state=RANDOM_STATE)
    
    print("Training SVD Model...")
    # SVD inherently provides 'Mean Normalization' via its baseline estimates (µ + b_u + b_i).
    # If a user is unseen (missing entirely), their baseline falls back to µ + b_i.
    algo = SVD(n_factors=100, random_state=RANDOM_STATE)
    algo.fit(trainset)
    
    print("Evaluating SVD Model...")
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions)
    print(f"Validation RMSE: {rmse:.4f}")
    
    # Train on full data for the final production model
    print("Retraining on full sampled dataset for production use...")
    trainset_full = data.build_full_trainset()
    algo.fit(trainset_full)
    
    return algo, trainset_full

def predict_user_ratings(user_id, algo, trainset_full, all_movie_ids, top_n=100):
    """
    Generate predictions for a specific user for unrated movies.
    """
    # Find movies the user has already rated in the training set
    try:
        inner_user_id = trainset_full.to_inner_uid(user_id)
        user_rated_inner_ids = set([j for (j, _) in trainset_full.ur[inner_user_id]])
        user_rated_raw_ids = set([trainset_full.to_raw_iid(inner_id) for inner_id in user_rated_inner_ids])
    except ValueError:
        # Unseen user (cold start) -> no rated movies in the training set
        user_rated_raw_ids = set()
        
    # Unrated movies are candidates for prediction
    unrated_movies = [m for m in all_movie_ids if m not in user_rated_raw_ids]
    
    # Predict scores for unrated movies
    predictions = []
    for movie_id in unrated_movies:
        # Predict method uses µ + b_u + b_i + dot(q_i, p_u)
        # Inherently handles Mean Normalization for missing/sparse user data
        pred = algo.predict(uid=user_id, iid=movie_id)
        predictions.append((movie_id, pred.est))
        
    # Sort and get top N
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_predictions = predictions[:top_n]
    
    # Format into DataFrame
    res_df = pd.DataFrame(top_predictions, columns=['movieId', 'collaborative_score'])
    return res_df

def precompute_batch_scores(algo, trainset_full, users, all_movie_ids, top_n=100):
    """
    Pre-compute candidate lists for a batch of active users to persist.
    Yields dictionary: {user_id: {movie_id: collaborative_score}}
    This format enables instant lookup during Hybrid Ensemble calculations:
    Final_Score = (w1 * Collaborative_Score) + (w2 * Content_Score)
    """
    print(f"Pre-computing collaborative scores for {len(users)} users...")
    batch_scores = {}
    for i, u in enumerate(users):
        if i > 0 and i % 100 == 0:
            print(f"Processed {i}/{len(users)} users...")
            
        preds = predict_user_ratings(u, algo, trainset_full, all_movie_ids, top_n)
        # Convert DataFrame to dictionary for fast O(1) queryable mapping
        batch_scores[u] = dict(zip(preds['movieId'], preds['collaborative_score']))
        
    return batch_scores

def main():
    # 1. Load and sample data
    # Loading ~20% of 25M is ~5M rows.
    # We sample to ensure our memory usage across Pandas -> Surprise stays well within 16GB.
    df = load_and_sample_data(INPUT_PARQUET, sample_frac=0.2)
    
    # Cache all unique movie IDs for candidate generation
    all_movie_ids = df['movieId'].unique()
    
    # 2. Train model and evaluate
    algo, trainset_full = train_svd_model(df)
    
    # Aggressively cleanup the dataframe to free memory before final predictions
    del df
    gc.collect()
    
    # 3. Save Production Model (Output 1)
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    print(f"Saving SVD model to {MODEL_OUT}...")
    with open(MODEL_OUT, 'wb') as f:
        pickle.dump(algo, f)
        
    # 4. Generate & Save Batch Predictions (Output 2)
    # We simulate a batch of active users to precompute scores. 
    # In a real setup, this list could come from a redis queue or latest logins.
    print("Selecting a batch of users for pre-computation aligned with dashboard_view.parquet...")
    # Load Dashboard Users First
    df_dash = pd.read_parquet('processed/dashboard_view.parquet', columns=['userId'])
    dashboard_users = df_dash['userId'].unique()
    
    # Filter & Map Safely
    active_users_raw = []
    for user_id in dashboard_users:
        try:
            # Check if user exists in the trained SVD model
            trainset_full.to_inner_uid(user_id)
            active_users_raw.append(user_id)
            if len(active_users_raw) >= 500:
                break
        except ValueError:
            continue
    
    batch_scores = precompute_batch_scores(algo, trainset_full, active_users_raw, all_movie_ids, top_n=100)
    
    os.makedirs(os.path.dirname(SCORES_OUT), exist_ok=True)
    print(f"Saving precomputed batch scores to {SCORES_OUT}...")
    with open(SCORES_OUT, 'wb') as f:
        pickle.dump(batch_scores, f)
        
    print("CineIQ Collaborative Pipeline Complete.")

if __name__ == "__main__":
    main()
