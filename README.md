#  CineIQ: Enterprise-Grade Hybrid Recommendation Engine

CineIQ is an enterprise-grade, multi-stage hybrid recommendation pipeline built to handle large-scale movie and interaction datasets within localized consumer hardware constraints (16GB RAM / GTX 1660 Ti). The core architecture shifts seamlessly from raw vector parsing to an optimized machine learning ensemble and a contextual transformer re-ranking stage.

##  Architectural Summary
The pipeline transitions through 4 modular steps:

1. **Candidate Retrieval Phase:** Dual-track engine leveraging Matrix Factorization (Surprise SVD) for behavioral collaborative filtering and TF-IDF Text Vectorization with Linear Kernel similarity for metadata content-based filtering.
2. **Feature-Weighted Linear Stacking:** An Ordinary Least Squares (OLS) Linear Regression meta-model that dynamically evaluates baseline scores against historical user interactions to eliminate human weighting guesswork.
3. **Contextual Sentiment Re-Ranking:** A secondary transformer-driven scoring step using a pre-trained DistilBERT pipeline to audit public cinematic reception and nudge final item rankings.
4. **UI Visual Analytics & Explainability Layer:** A production front-end dashboard featuring local feature-importance routing to break open the "black box" model and display 5 distinct types of human-readable matching logic.

##  Workspace Directory Structure

```text
CineIQ/
├── datasets/                            # [Ignored via .gitignore] 
│   ├── imdb.csv                         # Validation set (50K reviews) for sentiment benchmarking
│   ├── movie25lens/
│   │   ├── genome-scores.csv            # Tag relevance scores for movies
│   │   ├── genome-tags.csv              # Tag descriptions for the genome scores
│   │   ├── links.csv                    # Identifiers linking MovieLens to IMDB/TMDB
│   │   ├── movies.csv                   # Raw MovieLens database mapping IDs to human-readable titles/genres
│   │   ├── ratings.csv                  # 25 million user-movie rating interactions
│   │   └── tags.csv                     # User-generated tags for movies
│   └── tmdb/
│       ├── credits.csv                  # Cast and crew information for movies
│       ├── keywords.csv                 # Plot keywords and descriptive tags
│       ├── links.csv                    # Movie identifiers mapping to IMDB
│       └── movies_metadata.csv          # Comprehensive TMDB movie metadata (revenue, overview, etc.)
├── models/                              # [Ignored via .gitignore] 
│   ├── stacking_meta_model.pkl          # OLS Linear Regression coefficients for the hybrid policy
│   ├── svd_model.pkl                    # Serialized Surprise Matrix Factorization weights
│   ├── tfidf_matrix.pkl                 # Precomputed sparse matrix of movie text feature vectors
│   └── tfidf_vectorizer.pkl             # Trained TF-IDF vocabulary schema
├── processed/                           # [Ignored via .gitignore] 
│   ├── collaborative_scores.pkl         # SVD candidate prediction dictionaries {userId: {movieId: score}}
│   ├── content_scores.pkl               # Cosine similarity candidate dictionaries
│   ├── content_view.parquet             # Memory-optimized textual "soup" profiles for TF-IDF
│   ├── dashboard_view.parquet           # Flattened historical watch profiles for UI analytics
│   ├── final_ranked_scores.pkl          # Definitive recommendation payload (Sentiment-adjusted)
│   ├── hybrid_scores.pkl                # Pre-sentiment stacked ensemble candidate scores
│   └── svd_view.parquet                 # Downcasted [userId, movieId, rating] matrix for Surprise
├── .gitignore                           # Configuration explicitly preventing heavy data uploads
├── app.py                               # Streamlit Dashboard UI & 5-Tier Explainability Routing Engine
├── collaborative_recommender.py         # Step 1A: Trains SVD & generates Collaborative candidates
├── content_recommender.py               # Step 1B: Trains TF-IDF & generates Content-based candidates
├── hybrid_ensemble.py                   # Step 2: Trains OLS Stacking Policy & executes Union candidate merge
├── preprocessing.py                     # Step 0: Data Engineering, Downcasting, and Parquet View Pre-computation
├── requirements.txt                     # Filtered production dependencies for environment replication
├── reranker.py                          # Step 3: DistilBERT Validation Benchmarking & Sentiment Adjustment Script
└── verify_recommendations.py            # Terminal utility for qualitative ground-truth vibe checks
```

##  Core Algorithmic Metrics & Empirical Wins
Our pipeline execution yielded the following empirical results during validation:

### Stacking Meta-Model Policy Coefficients
Instead of manual tuning, the OLS meta-model learned the optimal ensemble policy:
- **Learned Collaborative Weight ($w_1$):** `0.9578`
- **Learned Content Weight ($w_2$):** `4.4236`
- **Learned Intercept:** `-0.2676`

> *The significantly higher content weight is an elegant mathematical scaling response to the natural numerical sparsity of high-dimensional TF-IDF dot products, ensuring both signals contribute equitably to the final hybrid score.*

### Sentiment Benchmarking Accuracy
Before deploying our sentiment catalyst logic, we benchmarked NLP analyzers against 50K human reviews:
- **Lexical VADER Baseline:** `69.70%`
- **Contextual DistilBERT Pipeline:** `88.25%`

## ⚡ Big Data & Hardware Optimization Highlights
Handling 25 million interaction rows on a 16GB RAM laptop requires strict memory-safeguard engineering. The following optimizations allowed CineIQ to scale efficiently:

- **Downcasting & Parquet Strategy:** Converting raw datasets to stringified text "soups" and aggressively downcasting numerical interaction datatypes to dense Parquet arrays to save substantial cold-storage overhead.
- **Unique Candidate Filtering:** By enforcing a distinct key-union strategy across targeted active profiles, we reduced the final NLP re-ranking inference workload by **over 90%** (shrinking the payload from 50,000 potential rows down to just 3,554 unique movie entities).
- **Streamlit Cache Aggregation:** Front-end lookup loaders are decorated with `@st.cache_data` and trigger explicit garbage collection (`gc.collect()`) after serialization. This prevents memory leaks and ensures near-zero interface transition latency during dynamic user filtering.

##  Execution & Replication Guide
To stand up the CineIQ environment and dashboard on your local machine, run the following commands sequentially:

**1. Clone the repository:**
```bash
git clone https://github.com/Omniman2024/CineIQ.git
cd CineIQ
```

**2. Install production dependencies:**
```bash
pip install -r requirements.txt
```

**3. Execute the backend machine learning pipeline sequentially:**
```bash
python3 preprocessing.py
python3 collaborative_recommender.py
python3 content_recommender.py
python3 hybrid_ensemble.py
python3 reranker.py
```

**4. Launch the interactive dashboard UI:**
```bash
streamlit run app.py
```
