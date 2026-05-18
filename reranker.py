import os
import gc
import pickle
import numpy as np
import pandas as pd

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

try:
    import torch
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

HYBRID_SCORES = 'processed/hybrid_scores.pkl'
CONTENT_PARQUET = 'processed/content_view.parquet'
IMDB_CSV = 'datasets/imdb.csv'
FINAL_RANKED_OUT = 'processed/final_ranked_scores.pkl'

USE_TRANSFORMER = False  
ALPHA = 0.1

nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

def get_distilbert_pipeline():
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("Transformers or PyTorch not installed.")
    
    device = 0 if torch.cuda.is_available() else -1
    print(f"Loading DistilBERT model (device={device})...")
    
    classifier = pipeline(
        "sentiment-analysis", 
        model="distilbert-base-uncased-finetuned-sst-2-english", 
        device=device,
        truncation=True, 
        max_length=512
    )
    return classifier

def benchmark_models(sample_size=2000):
    print(f"Loading {IMDB_CSV} for validation benchmark...")
    if not os.path.exists(IMDB_CSV):
        print(f"Warning: {IMDB_CSV} not found. Skipping benchmark.")
        return
        
    df_imdb = pd.read_csv(IMDB_CSV)
    
    if len(df_imdb) > sample_size:
        df_imdb = df_imdb.sample(n=sample_size, random_state=42)
        
    reviews = df_imdb['review'].astype(str).tolist()
    
    if df_imdb['sentiment'].dtype == object:
        labels = (df_imdb['sentiment'].str.lower() == 'positive').astype(int).values
    else:
        labels = df_imdb['sentiment'].astype(int).values
    
    # -- Option A: VADER Benchmark --
    print("Running Option A: VADER benchmark (CPU)...")
    vader_preds = []
    for text in reviews:
        score = sia.polarity_scores(text)['compound']
        vader_preds.append(1 if score > 0 else 0)
    vader_acc = np.mean(np.array(vader_preds) == labels)
    
    # -- Option B: DistilBERT Benchmark --
    distilbert_acc = 0.0
    if TRANSFORMERS_AVAILABLE:
        try:
            print("Running Option B: DistilBERT benchmark (GPU)...")
            classifier = get_distilbert_pipeline()
            
            results = classifier(reviews, batch_size=32)
            bert_preds = [1 if r['label'] == 'POSITIVE' else 0 for r in results]
            distilbert_acc = np.mean(np.array(bert_preds) == labels)
            
            del classifier
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"DistilBERT benchmark failed: {e}")
            
    print("\n=== SENTIMENT BENCHMARK SUMMARY ===")
    print(f"VADER Accuracy:      {vader_acc*100:.2f}%")
    if TRANSFORMERS_AVAILABLE:
        print(f"DistilBERT Accuracy: {distilbert_acc*100:.2f}%")
    print("===================================\n")

def compute_sentiment_scores(unique_movie_ids, use_transformer=False):
    print(f"Extracting movie text profiles for {len(unique_movie_ids)} unique hybrid candidates...")
    
    df_content = pd.read_parquet(CONTENT_PARQUET, columns=['movieId', 'soup'])
    df_candidates = df_content[df_content['movieId'].isin(unique_movie_ids)].copy()
    
    del df_content
    gc.collect()
    
    movie_texts = dict(zip(df_candidates['movieId'], df_candidates['soup'].astype(str)))
    sentiment_scores = {}
    
    if use_transformer and TRANSFORMERS_AVAILABLE:
        print("Scoring candidates using Option B: DistilBERT (GPU Batch Inference)...")
        classifier = get_distilbert_pipeline()
        
        m_ids = list(movie_texts.keys())
        texts = list(movie_texts.values())
        
        results = classifier(texts, batch_size=32)
        
        for m_id, result in zip(m_ids, results):
            if result['label'] == 'POSITIVE':
                prob_pos = result['score']
            else:
                prob_pos = 1.0 - result['score']
                
            scaled_score = (prob_pos * 2.0) - 1.0
            sentiment_scores[m_id] = scaled_score
            
        del classifier
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    else:
        print("Scoring candidates using Option A: VADER (CPU O(N) Inference)...")
        for m_id, text in movie_texts.items():
            score = sia.polarity_scores(text)['compound']
            sentiment_scores[m_id] = score
            
    return sentiment_scores

def apply_sentiment_reranking(hybrid_scores_dict, sentiment_scores, alpha=0.1, top_n=100):
    print(f"Applying Re-Ranking logic (alpha={alpha})...")
    final_ranked_dict = {}
    
    for user_id, candidates in hybrid_scores_dict.items():
        user_final_scores = []
        
        for movie_id, hybrid_score in candidates.items():
            s_score = sentiment_scores.get(movie_id, 0.0)
            
            final_score = hybrid_score * (1.0 + (alpha * s_score))
            user_final_scores.append((movie_id, final_score))
            
        user_final_scores.sort(key=lambda x: x[1], reverse=True)
        top_candidates = user_final_scores[:top_n]
        
        final_ranked_dict[user_id] = {m: float(score) for m, score in top_candidates}
        
    return final_ranked_dict

def main():
    benchmark_models(sample_size=2000)
    
    print(f"Loading hybrid candidate scores from {HYBRID_SCORES}...")
    with open(HYBRID_SCORES, 'rb') as f:
        hybrid_scores_dict = pickle.load(f)
        
    unique_movie_ids = set()
    for candidates in hybrid_scores_dict.values():
        unique_movie_ids.update(candidates.keys())
        
    sentiment_scores = compute_sentiment_scores(unique_movie_ids, use_transformer=USE_TRANSFORMER)
    
    final_ranked_dict = apply_sentiment_reranking(hybrid_scores_dict, sentiment_scores, alpha=ALPHA, top_n=100)
    
    os.makedirs(os.path.dirname(FINAL_RANKED_OUT), exist_ok=True)
    print(f"Saving finalized sentiment-aware recommendations to {FINAL_RANKED_OUT}...")
    with open(FINAL_RANKED_OUT, 'wb') as f:
        pickle.dump(final_ranked_dict, f)
        
    print("CineIQ Re-Ranking Pipeline Complete.")

if __name__ == "__main__":
    main()