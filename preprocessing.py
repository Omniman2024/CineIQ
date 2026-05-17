import numpy as np 
import pandas as pd 

def preview_datasets():
    print("--- IMDB Reviews ---")
    imdb_df = pd.read_csv('datasets/imdb.csv')
    print(imdb_df.head(), "\n")

    print("--- MovieLens 25M: Movies ---")
    ml_movies_df = pd.read_csv('datasets/movie25lens/movies.csv')
    print(ml_movies_df.head(), "\n")

    print("--- MovieLens 25M: Ratings ---")
    ml_ratings_df = pd.read_csv('datasets/movie25lens/ratings.csv')
    print(ml_ratings_df.head(), "\n")

    print("--- MovieLens 25M: Tags ---")
    ml_tags_df = pd.read_csv('datasets/movie25lens/tags.csv')
    print(ml_tags_df.head(), "\n")

    print("--- MovieLens 25M: Links ---")
    ml_links_df = pd.read_csv('datasets/movie25lens/links.csv')
    print(ml_links_df.head(), "\n")

    print("--- MovieLens 25M: Genome_Scores ---")
    ml_genome_scores_df = pd.read_csv('datasets/movie25lens/genome-scores.csv')
    print(ml_genome_scores_df.head(), "\n")

    print("--- MovieLens 25M: Genome_Tags ---")
    ml_genome_tags_df = pd.read_csv('datasets/movie25lens/genome-tags.csv')
    print(ml_genome_tags_df.head(), "\n")

    print("--- TMDB: Movies Metadata ---")
    tmdb_movies_df = pd.read_csv('datasets/tmdb/movies_metadata.csv')
    print(tmdb_movies_df.head(), "\n")

    print("--- TMDB: Credits ---")
    tmdb_credits_df = pd.read_csv('datasets/tmdb/credits.csv')
    print(tmdb_credits_df.head(), "\n")

    print("--- TMDB: Links ---")
    tmdb_links_df = pd.read_csv('datasets/tmdb/links.csv')
    print(tmdb_links_df.head(), "\n")

    print("--- TMDB: Keywords ---")
    tmdb_keywords_df = pd.read_csv('datasets/tmdb/keywords.csv')
    print(tmdb_keywords_df.head(), "\n")


def get_collaborative_view():
    print("Loading Collaborative View (SVD)...")
    svd_df = pd.read_csv('datasets/movie25lens/ratings.csv', 
                         usecols=['userId', 'movieId', 'rating'],
                         dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})
    
    print("Collaborative View ready. First few rows:")
    print(svd_df.head())
    return svd_df

def get_content_view():
    import ast
    print("Loading datasets for Content View...")
    # Load links to map tmdbId to movieId
    links_df = pd.read_csv('datasets/movie25lens/links.csv', usecols=['movieId', 'tmdbId'])
    links_df = links_df.dropna(subset=['tmdbId'])
    links_df['tmdbId'] = links_df['tmdbId'].astype('int32')
    
    # Load metadata (genres, overview)
    metadata_df = pd.read_csv('datasets/tmdb/movies_metadata.csv', usecols=['id', 'genres', 'overview'], low_memory=False)
    metadata_df['id'] = pd.to_numeric(metadata_df['id'], errors='coerce')
    metadata_df = metadata_df.dropna(subset=['id'])
    metadata_df['id'] = metadata_df['id'].astype('int32')
    
    # Load keywords
    keywords_df = pd.read_csv('datasets/tmdb/keywords.csv')
    keywords_df['id'] = pd.to_numeric(keywords_df['id'], errors='coerce')
    keywords_df = keywords_df.dropna(subset=['id'])
    keywords_df['id'] = keywords_df['id'].astype('int32')
    
    # Load credits (cast, crew)
    credits_df = pd.read_csv('datasets/tmdb/credits.csv')
    credits_df['id'] = pd.to_numeric(credits_df['id'], errors='coerce')
    credits_df = credits_df.dropna(subset=['id'])
    credits_df['id'] = credits_df['id'].astype('int32')
    
    print("Merging dataframes on tmdbId...")
    content_df = pd.merge(links_df, metadata_df, left_on='tmdbId', right_on='id', how='inner').drop(columns=['id'])
    content_df = pd.merge(content_df, keywords_df, left_on='tmdbId', right_on='id', how='inner').drop(columns=['id'])
    content_df = pd.merge(content_df, credits_df, left_on='tmdbId', right_on='id', how='inner').drop(columns=['id'])
    
    # Free up memory
    del links_df, metadata_df, keywords_df, credits_df
    
    def extract_features(row):
        genres = []
        if isinstance(row.get('genres'), str):
            try:
                for i in ast.literal_eval(row['genres']):
                    genres.append(i['name'].replace(" ", "").lower())
            except:
                pass
                
        keywords = []
        if isinstance(row.get('keywords'), str):
            try:
                for i in ast.literal_eval(row['keywords']):
                    keywords.append(i['name'].replace(" ", "").lower())
            except:
                pass
                
        director = ""
        if isinstance(row.get('crew'), str):
            try:
                for i in ast.literal_eval(row['crew']):
                    if i.get('job') == 'Director':
                        director = i['name'].replace(" ", "").lower()
                        break
            except:
                pass
                
        cast = []
        if isinstance(row.get('cast'), str):
            try:
                for i in ast.literal_eval(row['cast'])[:3]:
                    cast.append(i['name'].replace(" ", "").lower())
            except:
                pass
                
        overview = str(row.get('overview', '')) if pd.notnull(row.get('overview')) else ""
        
        soup_elements = genres + keywords + cast
        if director:
            soup_elements.append(director)
            
        return " ".join(soup_elements) + " " + overview

    print("Creating 'soup' column (this will take a minute parsing strings)...")
    content_df['soup'] = content_df.apply(extract_features, axis=1)
    
    final_content_df = content_df[['movieId', 'soup']]
    
    print("Content View ready. First few rows:")
    print(final_content_df.head())
    return final_content_df

def get_sentiment_view():
    print("Loading IMDB 50K dataset for Sentiment View...")
    # Load the raw text reviews and sentiments
    imdb_df = pd.read_csv('datasets/imdb.csv')
    
    # Map the sentiment to a binary label (1 for positive, 0 for negative)
    imdb_df['sentiment'] = imdb_df['sentiment'].map({'positive': 1, 'negative': 0})
    
    print("Sentiment View ready. First few rows:")
    print(imdb_df.head())
    
    # Note: Inference columns (imdbId, sentiment_score) will be generated dynamically 
    # during the engine's inference phase using DistilBERT/VADER on specific movie reviews.
    
    return imdb_df

def get_taste_dashboard_view():
    import ast
    print("Loading datasets for Taste Dashboard...")
    
    # Load ratings (limit to 10,000 rows to prevent OOM during preview)
    ratings_df = pd.read_csv('datasets/movie25lens/ratings.csv', 
                             usecols=['userId', 'movieId'],
                             dtype={'userId': 'int32', 'movieId': 'int32'},
                             nrows=10000)
    
    # Load links to map movieId -> tmdbId
    links_df = pd.read_csv('datasets/movie25lens/links.csv', usecols=['movieId', 'tmdbId'])
    links_df = links_df.dropna(subset=['tmdbId'])
    links_df['tmdbId'] = links_df['tmdbId'].astype('int32')
    
    # Merge ratings with tmdbId
    taste_df = pd.merge(ratings_df, links_df, on='movieId', how='inner')
    del ratings_df, links_df
    
    # Load metadata (title, release_date, genres, revenue)
    metadata_df = pd.read_csv('datasets/tmdb/movies_metadata.csv', 
                              usecols=['id', 'title', 'release_date', 'genres', 'revenue'], 
                              low_memory=False)
    metadata_df['id'] = pd.to_numeric(metadata_df['id'], errors='coerce')
    metadata_df = metadata_df.dropna(subset=['id'])
    metadata_df['id'] = metadata_df['id'].astype('int32')
    
    # Merge with metadata
    taste_df = pd.merge(taste_df, metadata_df, left_on='tmdbId', right_on='id', how='inner').drop(columns=['id'])
    del metadata_df
    
    # Load credits (cast, crew)
    credits_df = pd.read_csv('datasets/tmdb/credits.csv', usecols=['id', 'cast', 'crew'])
    credits_df['id'] = pd.to_numeric(credits_df['id'], errors='coerce')
    credits_df = credits_df.dropna(subset=['id'])
    credits_df['id'] = credits_df['id'].astype('int32')
    
    # Merge with credits
    taste_df = pd.merge(taste_df, credits_df, left_on='tmdbId', right_on='id', how='inner').drop(columns=['id'])
    del credits_df
    
    # Helper to parse JSON strings
    def parse_taste_features(row):
        genres = []
        if isinstance(row.get('genres'), str):
            try:
                for i in ast.literal_eval(row['genres']):
                    genres.append(i['name'])
            except:
                pass
                
        director = ""
        if isinstance(row.get('crew'), str):
            try:
                for i in ast.literal_eval(row['crew']):
                    if i.get('job') == 'Director':
                        director = i['name']
                        break
            except:
                pass
                
        cast = []
        if isinstance(row.get('cast'), str):
            try:
                for i in ast.literal_eval(row['cast'])[:5]:
                    cast.append(i['name'])
            except:
                pass
                
        return pd.Series([genres, cast, director])

    print("Parsing JSON fields for Dashboard (this may take a moment)...")
    taste_df[['genres', 'cast', 'director']] = taste_df.apply(parse_taste_features, axis=1)
    
    # Drop the raw 'crew' column since we extracted the director
    taste_df = taste_df.drop(columns=['crew'])
    
    # Ensure revenue is numeric
    taste_df['revenue'] = pd.to_numeric(taste_df['revenue'], errors='coerce').fillna(0)
    
    # Select final columns requested
    final_taste_df = taste_df[['userId', 'title', 'release_date', 'tmdbId', 'genres', 'cast', 'director', 'revenue']]
    
    print("Taste Dashboard View ready. First few rows:")
    print(final_taste_df.head())
    
    return final_taste_df

if __name__ == "__main__":
    svd_data = get_collaborative_view()
    content_data = get_content_view()
    sentiment_data = get_sentiment_view()
    taste_dashboard_data = get_taste_dashboard_view()

    # Save them to a 'processed' folder
    import os
    os.makedirs('processed', exist_ok=True)

    print("Saving processed dataframes...")
    # Using Parquet for speed and efficiency
    svd_data.to_parquet('processed/svd_view.parquet')
    content_data.to_parquet('processed/content_view.parquet')
    sentiment_data.to_parquet('processed/sentiment_view.parquet')
    taste_dashboard_data.to_parquet('processed/dashboard_view.parquet')
    
    print("All files saved in 'processed/' folder.")
