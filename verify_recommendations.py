import os
import gc
import pickle
import pandas as pd

# Path configurations mapped to existing CineIQ structure
MOVIES_CSV = 'datasets/movie25lens/movies.csv'
SVD_PARQUET = 'processed/svd_view.parquet'
HYBRID_SCORES = 'processed/hybrid_scores.pkl'

def load_movie_metadata():
    """
    Load movie titles and genres into fast O(1) memory dictionaries.
    Guarantees movieId is cast to int for consistent dictionary mapping across structures.
    """
    # Verify file exists to prevent runtime failures
    if not os.path.exists(MOVIES_CSV):
        raise FileNotFoundError(f"Movie metadata file not found at {MOVIES_CSV}")
        
    # Read core columns only to minimize initial RAM spike
    df_movies = pd.read_csv(MOVIES_CSV, usecols=['movieId', 'title', 'genres'])
    
    # Explicit type casting to ensure robust mapping keys and string safety
    df_movies['movieId'] = df_movies['movieId'].astype(int)
    df_movies['title'] = df_movies['title'].astype(str)
    df_movies['genres'] = df_movies['genres'].astype(str)
    
    # Construct lightweight O(1) lookup dictionaries
    movie_titles = dict(zip(df_movies['movieId'], df_movies['title']))
    movie_genres = dict(zip(df_movies['movieId'], df_movies['genres']))
    
    # Enforce strict 16GB memory constraints by wiping the massive DataFrame immediately
    del df_movies
    gc.collect()
    
    return movie_titles, movie_genres

def load_hybrid_scores():
    """
    Load precomputed hybrid scores. Expected format:
    { user_id (int): { movie_id (int): hybrid_score (float) } }
    """
    if not os.path.exists(HYBRID_SCORES):
        raise FileNotFoundError(f"Hybrid scores not found at {HYBRID_SCORES}. Please run hybrid_ensemble.py first.")
        
    with open(HYBRID_SCORES, 'rb') as f:
        scores = pickle.load(f)
        
    # Structural verification check to prevent silent iteration bugs
    if not isinstance(scores, dict):
        raise TypeError("Hybrid scores structure violation: Must be a nested dictionary.")
        
    return scores

def main():
    # 1. Load lightweight O(1) movie lookups
    movie_titles, movie_genres = load_movie_metadata()
    
    # 2. Load nested dictionary of hybrid recommendations
    hybrid_scores = load_hybrid_scores()
    
    # 3. Sample Selection: Extract first 4 unique users to act as audit subjects
    sample_users = list(hybrid_scores.keys())[:4]
    if not sample_users:
        print("Audit aborted: No users found in hybrid scores.")
        return
        
    # 4. Load Interaction History & Extract Historical Profiles
    if not os.path.exists(SVD_PARQUET):
        raise FileNotFoundError(f"Historical interactions not found at {SVD_PARQUET}")
        
    # The parquet file is massive (25M rows), so we load only necessary columns
    df_interactions = pd.read_parquet(SVD_PARQUET, columns=['userId', 'movieId', 'rating'])
    
    # Filter aggressively down to only our sample users to instantly reclaim RAM
    df_sample_history = df_interactions[df_interactions['userId'].isin(sample_users)].copy()
    
    # Immediate cleanup of the 25M row DataFrame
    del df_interactions
    gc.collect()
    
    # Ensure types are strict for accurate sorting and dictionary lookups
    df_sample_history['rating'] = df_sample_history['rating'].astype(float)
    df_sample_history['movieId'] = df_sample_history['movieId'].astype(int)
    
    # 5. Generate and Format Terminal Audit Report
    for user_id in sample_users:
        print("\n============================================================")
        print(f"CINEIQ RECOMMENDATION AUDIT FOR USER: {user_id}")
        print("============================================================")
        
        # -- Historical Profile Extraction --
        print("\n📜 WHAT THIS USER LOVED IN THE PAST:")
        user_history = df_sample_history[df_sample_history['userId'] == user_id]
        
        # Sort descending by ground truth rating to find top 7
        top_history = user_history.sort_values(by='rating', ascending=False).head(7)
        
        for _, row in top_history.iterrows():
            m_id = int(row['movieId'])
            rating = row['rating']
            title = movie_titles.get(m_id, "Unknown Title")
            genres = movie_genres.get(m_id, "Unknown Genres")
            print(f"  ⭐ {rating:.1f} | {title} ({genres})")
            
        # -- Recommendation Preview --
        print("\n🚀 TOP 7 HYBRID RECOMMENDATIONS SURFACE-LEVEL PREVIEW:")
        user_recs = hybrid_scores.get(user_id, {})
        
        # Sort precomputed recommendations descending by hybrid score to find top 7
        sorted_recs = sorted(user_recs.items(), key=lambda x: x[1], reverse=True)[:7]
        
        for m_id, score in sorted_recs:
            m_id = int(m_id) # Type safety check
            title = movie_titles.get(m_id, "Unknown Title")
            genres = movie_genres.get(m_id, "Unknown Genres")
            print(f"  🎬 Score: {score:.2f} | {title} ({genres})")
            
        print("============================================================")

if __name__ == "__main__":
    main()
