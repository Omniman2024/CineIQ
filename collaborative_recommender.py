import os
import gc
import time
import pickle
import numpy as np
import pandas as pd
import mlflow
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("CineIQ_Collaborative")

INPUT_PARQUET = 'processed/svd_view.parquet'
MODEL_OUT = 'models/svd_model.pkl'
SCORES_OUT = 'processed/collaborative_scores.pkl'
RANDOM_STATE = 42

def load_and_sample_data(filepath, sample_frac=0.2):
    print(f"Loading data from {filepath}...")
    df = pd.read_parquet(filepath)
    print(f"Original shape: {df.shape}")
    df_sampled = df.sample(frac=sample_frac, random_state=RANDOM_STATE)
    print(f"Sampled shape: {df_sampled.shape}")
    
    del df
    gc.collect()
    
    return df_sampled

def train_svd_model(df):
    print("Preparing dataset for Surprise...")
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
    
    print("Splitting train/test sets...")
    trainset, testset = train_test_split(data, test_size=0.2, random_state=RANDOM_STATE)
    
    print("Training SVD Model...")
    n_factors = 100
    algo = SVD(n_factors=n_factors, random_state=RANDOM_STATE)
    
    mlflow.log_param("n_factors", n_factors)
    mlflow.log_param("random_state", RANDOM_STATE)
    
    start_time_cv = time.time()
    algo.fit(trainset)
    end_time_cv = time.time()
    mlflow.log_metric("cv_training_time_seconds", end_time_cv - start_time_cv)
    
    print("Evaluating SVD Model...")
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions)
    print(f"Validation RMSE: {rmse:.4f}")
    
    mlflow.log_metric("validation_rmse", rmse)
    
    print("Retraining on full sampled dataset for production use...")
    trainset_full = data.build_full_trainset()
    
    start_time_full = time.time()
    algo.fit(trainset_full)
    end_time_full = time.time()
    mlflow.log_metric("production_retrain_time_seconds", end_time_full - start_time_full)
    
    return algo, trainset_full

def predict_user_ratings(user_id, algo, trainset_full, all_movie_ids, top_n=100):
    try:
        inner_user_id = trainset_full.to_inner_uid(user_id)
        user_rated_inner_ids = set([j for (j, _) in trainset_full.ur[inner_user_id]])
        user_rated_raw_ids = set([trainset_full.to_raw_iid(inner_id) for inner_id in user_rated_inner_ids])
    except ValueError:
        user_rated_raw_ids = set()
        
    unrated_movies = [m for m in all_movie_ids if m not in user_rated_raw_ids]
    
    predictions = []
    for movie_id in unrated_movies:
        pred = algo.predict(uid=user_id, iid=movie_id)
        predictions.append((movie_id, pred.est))
        
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_predictions = predictions[:top_n]
    
    res_df = pd.DataFrame(top_predictions, columns=['movieId', 'collaborative_score'])
    return res_df

def precompute_batch_scores(algo, trainset_full, users, all_movie_ids, top_n=100):
    print(f"Pre-computing collaborative scores for {len(users)} users...")
    batch_scores = {}
    for i, u in enumerate(users):
        if i > 0 and i % 100 == 0:
            print(f"Processed {i}/{len(users)} users...")
            
        preds = predict_user_ratings(u, algo, trainset_full, all_movie_ids, top_n)
        batch_scores[u] = dict(zip(preds['movieId'], preds['collaborative_score']))
        
    return batch_scores

def main():
    with mlflow.start_run():
        start_pipeline_time = time.time()
        
        df = load_and_sample_data(INPUT_PARQUET, sample_frac=0.2)
        mlflow.log_metric("sampled_training_rows", df.shape[0])
        
        all_movie_ids = df['movieId'].unique()
        mlflow.log_metric("unique_movies_count", len(all_movie_ids))
        
        algo, trainset_full = train_svd_model(df)
        
        del df
        gc.collect()
        
        os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
        print(f"Saving SVD model to {MODEL_OUT}...")
        with open(MODEL_OUT, 'wb') as f:
            pickle.dump(algo, f)
            
        print("Selecting a batch of users for pre-computation aligned with dashboard_view.parquet...")
        df_dash = pd.read_parquet('processed/dashboard_view.parquet', columns=['userId'])
        dashboard_users = df_dash['userId'].unique()
        
        active_users_raw = []
        for user_id in dashboard_users:
            try:
                trainset_full.to_inner_uid(user_id)
                active_users_raw.append(user_id)
                if len(active_users_raw) >= 500:
                    break
            except ValueError:
                continue
        
        mlflow.log_metric("precomputed_users_count", len(active_users_raw))
        
        batch_scores = precompute_batch_scores(algo, trainset_full, active_users_raw, all_movie_ids, top_n=100)
        
        os.makedirs(os.path.dirname(SCORES_OUT), exist_ok=True)
        print(f"Saving precomputed batch scores to {SCORES_OUT}...")
        with open(SCORES_OUT, 'wb') as f:
            pickle.dump(batch_scores, f)
            
        mlflow.log_artifact(MODEL_OUT)
        mlflow.log_artifact(SCORES_OUT)
        
        end_pipeline_time = time.time()
        mlflow.log_metric("total_pipeline_time_seconds", end_pipeline_time - start_pipeline_time)
        
        print("CineIQ Collaborative Pipeline Complete. Run metrics and artifacts registered.")

if __name__ == "__main__":
    main()
