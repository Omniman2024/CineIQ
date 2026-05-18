import os
import gc
import pickle
import pandas as pd

MOVIES_CSV = 'datasets/movie25lens/movies.csv'
SVD_PARQUET = 'processed/svd_view.parquet'
HYBRID_SCORES = 'processed/hybrid_scores.pkl'

def load_movie_metadata():
    if not os.path.exists(MOVIES_CSV):
        raise FileNotFoundError(f"Movie metadata file not found at {MOVIES_CSV}")
        
    df_movies = pd.read_csv(MOVIES_CSV, usecols=['movieId', 'title', 'genres'])
    
    df_movies['movieId'] = df_movies['movieId'].astype(int)
    df_movies['title'] = df_movies['title'].astype(str)
    df_movies['genres'] = df_movies['genres'].astype(str)
    
    movie_titles = dict(zip(df_movies['movieId'], df_movies['title']))
    movie_genres = dict(zip(df_movies['movieId'], df_movies['genres']))
    
    del df_movies
    gc.collect()
    
    return movie_titles, movie_genres

def load_hybrid_scores():
    if not os.path.exists(HYBRID_SCORES):
        raise FileNotFoundError(f"Hybrid scores not found at {HYBRID_SCORES}. Please run hybrid_ensemble.py first.")
        
    with open(HYBRID_SCORES, 'rb') as f:
        scores = pickle.load(f)
        
    if not isinstance(scores, dict):
        raise TypeError("Hybrid scores structure violation: Must be a nested dictionary.")
        
    return scores

def main():
    movie_titles, movie_genres = load_movie_metadata()
    hybrid_scores = load_hybrid_scores()
    
    sample_users = list(hybrid_scores.keys())[:4]
    if not sample_users:
        print("Audit aborted: No users found in hybrid scores.")
        return
        
    if not os.path.exists(SVD_PARQUET):
        raise FileNotFoundError(f"Historical interactions not found at {SVD_PARQUET}")
        
    df_interactions = pd.read_parquet(SVD_PARQUET, columns=['userId', 'movieId', 'rating'])
    
    df_sample_history = df_interactions[df_interactions['userId'].isin(sample_users)].copy()
    del df_interactions
    gc.collect()
    
    df_sample_history['rating'] = df_sample_history['rating'].astype(float)
    df_sample_history['movieId'] = df_sample_history['movieId'].astype(int)
    
    for user_id in sample_users:
        print("\n============================================================")
        print(f"CINEIQ RECOMMENDATION AUDIT FOR USER: {user_id}")
        print("============================================================")
        
        print("\n WHAT THIS USER LOVED IN THE PAST:")
        user_history = df_sample_history[df_sample_history['userId'] == user_id]
        
        top_history = user_history.sort_values(by='rating', ascending=False).head(7)
        
        for _, row in top_history.iterrows():
            m_id = int(row['movieId'])
            rating = row['rating']
            title = movie_titles.get(m_id, "Unknown Title")
            genres = movie_genres.get(m_id, "Unknown Genres")
            print(f" {rating:.1f} | {title} ({genres})")
            
        print("\nTOP 7 HYBRID RECOMMENDATIONS SURFACE-LEVEL PREVIEW:")
        user_recs = hybrid_scores.get(user_id, {})
        
        sorted_recs = sorted(user_recs.items(), key=lambda x: x[1], reverse=True)[:7]
        
        for m_id, score in sorted_recs:
            m_id = int(m_id) 
            title = movie_titles.get(m_id, "Unknown Title")
            genres = movie_genres.get(m_id, "Unknown Genres")
            print(f"   Score: {score:.2f} | {title} ({genres})")
            
        print("============================================================")

if __name__ == "__main__":
    main()
