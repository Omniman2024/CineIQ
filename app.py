import os
import gc
import ast
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ==========================================
# CINEIQ - USER TASTE DASHBOARD
# ==========================================

# Configure overall Streamlit page layout and theme structure
st.set_page_config(
    page_title="CineIQ Dashboard", 
    layout="wide", 
    page_icon="🍿",
    initial_sidebar_state="expanded"
)

# --- UTILITY PARSERS ---
def parse_list_column(val):
    """Safely parse list-like columns that might be strings, NaN, or arrays."""
    # Prevent ValueError on multi-element numpy arrays by intercepting the type first
    if isinstance(val, (list, np.ndarray)):
        return list(val) if len(val) > 0 else []
        
    # Safe scalar check for missing data
    if pd.isna(val):
        return []
        
    if isinstance(val, str):
        # Handle stringified lists "['Action', 'Comedy']"
        if val.strip().startswith('['):
            try:
                parsed = ast.literal_eval(val)
                return parsed if isinstance(parsed, list) else []
            except (ValueError, SyntaxError):
                return []
        # Handle pipe-separated genres "Action|Comedy"
        elif '|' in val:
            return val.split('|')
        else:
            return [val]
    return []

# --- EXPLAINABILITY ENGINE ---
def generate_explanation(user_id, movie_id, movie_genres_str, user_history, collab_scores, content_scores, meta_model, hybrid_scores, final_scores, top_directors_list, top_actors_list, movie_director, movie_cast):
    """
    Mathematical Interpretability Logic:
    Determines Local Feature Importance by inspecting the raw coefficients of the stacking model.
    """
    try:
        w1, w2 = meta_model.coef_
        c_score = collab_scores.get(user_id, {}).get(movie_id, 0.0)
        t_score = content_scores.get(user_id, {}).get(movie_id, 0.0)
        
        # Calculate relative structural strength
        collab_contrib = w1 * c_score
        content_contrib = w2 * t_score
        
        h_score = hybrid_scores.get(user_id, {}).get(movie_id, collab_contrib + content_contrib + meta_model.intercept_)
        f_score = final_scores.get(user_id, {}).get(movie_id, h_score)
        
        # Define an exceptionally high sentiment bump (>8% increase) as a savior
        sentiment_bump = (f_score - h_score) / h_score if h_score > 0 else 0.0
        is_sentiment_savior = (sentiment_bump > 0.08 and f_score > h_score)
        
        # Tier 1: The Director Connection (High Content Match)
        if content_contrib > collab_contrib and movie_director and movie_director in top_directors_list:
            return f"🎬 *Because you are a big fan of work directed by {movie_director}.*"
            
        # Tier 2: The Star Power Connection (High Content Match)
        if content_contrib > collab_contrib and isinstance(movie_cast, list):
            matched_actor = next((actor for actor in movie_cast if actor in top_actors_list), None)
            if matched_actor:
                return f"🍿 *Features {matched_actor}, one of your most-watched actors based on your history.*"
                
        # Tier 3: The Perfect Hybrid Convergence (High Shared Confidence)
        if c_score >= 4.0 and t_score > 3.5:
            return "🎯 *An absolute match—perfectly hits your personal niche while being a certified favorite among similar viewers.*"
            
        # Tier 4: Standard Genre Match (Fallback Content)
        if t_score > 0.0 and content_contrib > collab_contrib:
            # Look up shared genres safely
            history_subset = user_history[user_history['rating'] >= 4.0] if 'rating' in user_history.columns else user_history
            
            history_subset = history_subset.copy()
            history_subset['parsed_genres'] = history_subset['genres'].apply(parse_list_column)
            genres_exploded = history_subset.explode('parsed_genres')
            genre_counts = genres_exploded['parsed_genres'].dropna().value_counts()
            
            top_genre = "cinematic"
            if not genre_counts.empty and isinstance(movie_genres_str, str):
                movie_genres = [g.strip() for g in movie_genres_str.split(',')]
                for g in genre_counts.index:
                    if g in movie_genres:
                        top_genre = g
                        break
                        
            return f"🎬 *Aligns perfectly with your preference for {top_genre} elements seen in your watch history.*"
            
        # Tier 5: Sentiment Savior & Peer Fallbacks
        if is_sentiment_savior:
            return "🔥 *Trending upward due to overwhelmingly positive recent audience reviews.*"
            
        return "🎯 *Highly recommended by cinephiles with tasting profiles matching yours.*"
        
    except Exception as e:
        return "✨ *Recommended based on CineIQ multi-layered analysis.*"

# --- DATA PIPELINE (CACHED) ---
@st.cache_data(show_spinner=False)
def load_pipeline_data():
    """
    Load data strictly once upon app initialization. 
    Decorated with @st.cache_data to protect 16GB RAM constraints during UI interactions.
    """
    try:
        # 1. Historical Dashboard View
        df_dash = pd.read_parquet('processed/dashboard_view.parquet')
        
        # 2. Movie Metadata Database
        df_movies = pd.read_csv('datasets/movie25lens/movies.csv', usecols=['movieId', 'title', 'genres'])
        df_movies['movieId'] = df_movies['movieId'].astype(int)
        
        # 3. Final Recommendation Payload
        with open('processed/final_ranked_scores.pkl', 'rb') as f:
            final_scores = pickle.load(f)
            
        # 4. Explainability Payloads (Lightweight dictionaries)
        with open('processed/collaborative_scores.pkl', 'rb') as f:
            collab_scores = pickle.load(f)
        with open('processed/content_scores.pkl', 'rb') as f:
            content_scores = pickle.load(f)
        with open('processed/hybrid_scores.pkl', 'rb') as f:
            hybrid_scores = pickle.load(f)
        with open('models/stacking_meta_model.pkl', 'rb') as f:
            meta_model = pickle.load(f)
            
        # Reclaim temporary memory overhead
        gc.collect()
        
        return df_dash, df_movies, final_scores, collab_scores, content_scores, hybrid_scores, meta_model
    except Exception as e:
        st.error(f"Critical Data Load Failure: {e}")
        st.stop()

# Execute cached load
df_dash, df_movies, final_scores, collab_scores, content_scores, hybrid_scores, meta_model = load_pipeline_data()

# --- SIDEBAR CONFIGURATION ---
st.sidebar.title("🎬 CineIQ Control")
st.sidebar.markdown("Analyze historical taste profiles and generate smart recommendations.")

active_users = list(final_scores.keys())
if not active_users:
    st.sidebar.error("No active users found in final_ranked_scores.pkl")
    st.stop()

selected_user = st.sidebar.selectbox("Select Active User ID:", options=active_users)

# Add dynamic control for recommendation volume
num_recs = st.sidebar.slider("Number of Recommendations", min_value=5, max_value=50, value=15)

# On-the-fly dynamic filtering (Lightning fast because dataframe is cached in RAM)
user_history = df_dash[df_dash['userId'] == selected_user].copy()

# --- MAIN PANEL LAYOUT ---
st.title(f"User Taste Dashboard: Profile #{selected_user}")
st.markdown("---")

# Layout segmentation
tab1, tab2 = st.tabs(["📊 User Taste Analytics", "🚀 CineIQ Smart Recommendations"])

# Track top affinities to pass down to explainability layer
top_directors_list = []
top_actors_list = []

# ==========================================
# TAB 1: HISTORICAL TASTE ANALYTICS
# ==========================================
with tab1:
    if user_history.empty:
        st.info("No historical watch data found for this user in dashboard_view.parquet.")
    else:
        # Top Row Analytics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Genre Footprint")
            try:
                # Dynamically explode parsed genres to count frequencies
                user_history['parsed_genres'] = user_history['genres'].apply(parse_list_column)
                genres_exploded = user_history.explode('parsed_genres')
                genre_counts = genres_exploded['parsed_genres'].dropna().value_counts().reset_index()
                genre_counts.columns = ['Genre', 'Count']
                
                if not genre_counts.empty:
                    # Beautiful Radar Chart
                    fig_radar = px.line_polar(
                        genre_counts.head(8), # Top 8 prevents chart clutter
                        r='Count', 
                        theta='Genre', 
                        line_close=True,
                        color_discrete_sequence=['#00f5d4']
                    )
                    fig_radar.update_traces(fill='toself', line=dict(width=2))
                    fig_radar.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)', 
                        plot_bgcolor='rgba(0,0,0,0)',
                        polar=dict(radialaxis=dict(visible=False))
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)
                else:
                    st.warning("Insufficient genre data to map footprint.")
            except Exception as e:
                st.error(f"Render Error (Genres): {e}")

        with col2:
            st.subheader("Era Preferences")
            try:
                # Safely parse release dates and extract decades
                user_history['parsed_date'] = pd.to_datetime(user_history['release_date'], errors='coerce')
                user_history['decade'] = (user_history['parsed_date'].dt.year // 10 * 10).astype('Int64')
                
                decade_counts = user_history['decade'].dropna().value_counts().reset_index()
                decade_counts.columns = ['Decade', 'Movies Watched']
                decade_counts['Decade'] = decade_counts['Decade'].astype(str) + "s"
                decade_counts = decade_counts.sort_values('Decade')
                
                if not decade_counts.empty:
                    # Modern Bar Chart
                    fig_bar = px.bar(
                        decade_counts, 
                        x='Decade', 
                        y='Movies Watched',
                        color_discrete_sequence=['#9b5de5'],
                        text_auto=True
                    )
                    fig_bar.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)', 
                        plot_bgcolor='rgba(0,0,0,0)',
                        xaxis_title="", 
                        yaxis_title=""
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.warning("Insufficient release date data to map era preferences.")
            except Exception as e:
                st.error(f"Render Error (Decades): {e}")

        st.markdown("---")
        
        # Bottom Row Affinities
        st.subheader("Cinematic Affinities")
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("#### **Top Directors**")
            try:
                # Some datasets store director as a string or list
                if 'director' in user_history.columns:
                    user_history['parsed_director'] = user_history['director'].apply(
                        lambda x: x[0] if isinstance(x, (list, np.ndarray)) and len(x) > 0 else (x if isinstance(x, str) else None)
                    )
                    directors = user_history['parsed_director'].dropna()
                    if not directors.empty:
                        dir_counts = directors.value_counts().head(5)
                        top_directors_list = dir_counts.index.tolist()
                        for d, count in dir_counts.items():
                            st.metric(label=str(d), value=f"{count} movies")
                    else:
                        st.write("No distinct director patterns found.")
            except Exception as e:
                st.error(f"Render Error (Directors): {e}")
                
        with col4:
            st.markdown("#### **Top Actors**")
            try:
                if 'cast' in user_history.columns:
                    user_history['parsed_cast'] = user_history['cast'].apply(parse_list_column)
                    cast_exploded = user_history.explode('parsed_cast')
                    cast_counts = cast_exploded['parsed_cast'].dropna().value_counts().head(5)
                    
                    if not cast_counts.empty:
                        top_actors_list = cast_counts.index.tolist()
                        for actor, count in cast_counts.items():
                            st.metric(label=str(actor), value=f"{count} movies")
                    else:
                        st.write("No distinct actor patterns found.")
            except Exception as e:
                st.error(f"Render Error (Actors): {e}")


# ==========================================
# TAB 2: CINEIQ RECOMMENDATIONS
# ==========================================
with tab2:
    st.subheader(f"Top Sentiment-Adjusted Picks for User {selected_user}")
    
    user_recs = final_scores.get(selected_user, {})
    
    if not user_recs:
        st.warning("No recommendations available for this user.")
    else:
        # Extract top N highest scores based on slider selection
        sorted_recs = sorted(user_recs.items(), key=lambda x: x[1], reverse=True)[:num_recs]
        
        for rank, (m_id, score) in enumerate(sorted_recs, start=1):
            # Look up human-readable metadata
            movie_info = df_movies[df_movies['movieId'] == m_id]
            
            if not movie_info.empty:
                title = movie_info.iloc[0]['title']
                raw_genres = str(movie_info.iloc[0]['genres'])
                # Clean pipe-separated genres for cleaner UI
                if '|' in raw_genres:
                    genres = ", ".join(raw_genres.split('|'))
                else:
                    genres = raw_genres
            else:
                title = f"Unknown Movie ID: {m_id}"
                genres = "Unknown"
                
            # Get movie cast and director from df_dash
            movie_dash_data = df_dash[df_dash['title'] == title].head(1)
            movie_director = None
            movie_cast = []
            if not movie_dash_data.empty:
                if 'director' in movie_dash_data.columns:
                    raw_dir = movie_dash_data.iloc[0]['director']
                    movie_director = raw_dir[0] if isinstance(raw_dir, (list, np.ndarray)) and len(raw_dir)>0 else (raw_dir if isinstance(raw_dir, str) else None)
                if 'cast' in movie_dash_data.columns:
                    movie_cast = parse_list_column(movie_dash_data.iloc[0]['cast'])

            # Dynamic Explainability Generation
            explanation = generate_explanation(
                user_id=selected_user,
                movie_id=m_id,
                movie_genres_str=genres,
                user_history=user_history,
                collab_scores=collab_scores,
                content_scores=content_scores,
                meta_model=meta_model,
                hybrid_scores=hybrid_scores,
                final_scores=final_scores,
                top_directors_list=top_directors_list,
                top_actors_list=top_actors_list,
                movie_director=movie_director,
                movie_cast=movie_cast
            )
            
            # Render using native Streamlit commercial layout blocks
            with st.container():
                col_a, col_b = st.columns([1, 6])
                with col_a:
                    st.metric(label=f"Rank #{rank}", value=f"{score:.2f}")
                with col_b:
                    st.markdown(f"### {title}")
                    st.markdown(f"**Genres:** {genres}")
                    st.caption(explanation)
                st.divider()

# --- GLOBAL MEMORY CLEANUP ---
del user_history
gc.collect()
