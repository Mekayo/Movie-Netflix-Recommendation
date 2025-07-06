# app.py
import streamlit as st
from reco_part import ContentBasedRecommender, recommend_content_based
import pandas as pd
import time

# Load the dataset to get movie titles
df = pd.read_csv(r"C:\Users\monst\Desktop\Movie-Netflix-Recommendation\data\netflix_titles_cleaned.csv")
movie_list = df['title'].dropna().unique().tolist()
movie_list.sort()

# Initialize the recommender (lazy loading)
@st.cache_resource
def get_recommender():
    """Initialize and cache the recommender"""
    recommender = ContentBasedRecommender(r"C:\Users\monst\Desktop\Movie-Netflix-Recommendation\data\netflix_titles_cleaned.csv")
    with st.spinner("🚀 Initializing recommendation model..."):
        recommender.initialize_model()
    return recommender

# --- Page Config ---
st.set_page_config(
    page_title="Netflix Movie Recommender 🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Background + Styling ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #000000 0%, #1a1a1a 50%, #000000 100%);
        background-size: cover;
        background-attachment: fixed;
        color: white;
    }
        
    .main-container {
        background-color: rgba(0, 0, 0, 0.8);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .recommendation-card {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #E50914;
        backdrop-filter: blur(5px);
        transition: all 0.3s ease;
    }
    
    .recommendation-card:hover {
        background-color: rgba(255, 255, 255, 0.15);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }
    
    h1, h2, h3 {
        color: #E50914;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
        font-weight: bold;
    }
    
    .metric-card {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(5px);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        background-color: rgba(255, 255, 255, 0.15);
        transform: scale(1.05);
    }
    
    .sidebar .sidebar-content {
        background-color: rgba(0, 0, 0, 0.95);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stSelectbox, .stSlider {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #E50914, #B81D24);
        border: none;
        border-radius: 8px;
        color: white;
        font-weight: bold;
        padding: 0.75rem 2rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, #B81D24, #E50914);
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(229, 9, 20, 0.4);
    }
    
    .netflix-header {
        text-align: center;
        margin-bottom: 2rem;
        padding: 2rem 0;
    }
    
    .netflix-header h1 {
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(45deg, #E50914, #B81D24);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: none;
        margin-bottom: 0.5rem;
    }
    
    .movie-info-section {
        background: rgba(0, 0, 0, 0.7);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(5px);
        margin: 1rem 0;
    }
    
    .stats-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-top: 2rem;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("## 🎬 Netflix Recommender")
    st.markdown("---")
    
    # Recommendation Type
    rec_type = st.selectbox(
        "🎯 Recommendation Type",
        ["Content-Based", "Genre-Based", "Director-Based"]
    )
    
    # Movie Search
    selected_movie = st.selectbox("🔍 Select a movie you like:", movie_list)
    
    # Top N Slider
    top_n = st.slider("🎯 Number of Recommendations", 3, 15, 5)
    
    # Similarity Threshold (for content-based)
    if rec_type == "Content-Based":
        similarity_threshold = st.slider("📊 Similarity Threshold", 0.1, 0.9, 0.2, 0.1)
    
    # Genre Selection (for genre-based)
    if rec_type == "Genre-Based":
        genres = df['genres'].str.split(', ').explode().unique()
        genres = [g for g in genres if g and g != 'nan']
        selected_genre = st.selectbox("🎭 Select Genre:", sorted(genres))
    
    # Director Selection (for director-based)
    if rec_type == "Director-Based":
        directors = df['director'].dropna().unique()
        directors = [d for d in directors if d and d != '']
        selected_director = st.selectbox("🎭 Select Director:", sorted(directors))
    
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown(f"**Total Movies:** {len(df):,}")
    st.markdown(f"**Total Genres:** {len(df['genres'].str.split(', ').explode().unique())}")
    st.markdown(f"**Total Directors:** {len(df['director'].dropna().unique())}")

# --- Main Content ---
st.markdown('<div class="main-container fade-in">', unsafe_allow_html=True)

# Netflix-style header
st.markdown("""
<div class="netflix-header">
    <h1 class="pulse">NETFLIX</h1>
    <h2>Movie Recommendation System</h2>
    <p style="color: #999; font-size: 1.1rem;">Powered by Advanced Content-Based Filtering ❤️</p>
</div>
""", unsafe_allow_html=True)

# Initialize recommender
recommender = get_recommender()

# --- Recommendation Section ---
if st.button("🎉 Get Recommendations", type="primary"):
    with st.spinner("🔍 Finding the best matches for you..."):
        start_time = time.time()
        
        if rec_type == "Content-Based":
            recommendations = recommender.recommend_content_based(
                selected_movie, 
                top_n=top_n, 
                min_similarity=similarity_threshold
            )
            
            # Show performance metrics
            end_time = time.time()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("⏱️ Response Time", f"{end_time - start_time:.2f}s")
            with col2:
                st.metric("🎯 Recommendations", len(recommendations))
            with col3:
                if recommendations and 'error' not in recommendations[0]:
                    avg_score = sum(rec['similarity_score'] for rec in recommendations) / len(recommendations)
                    st.metric("📊 Avg Similarity", f"{avg_score:.3f}")
            
            # Display recommendations
            st.markdown("---")
            st.subheader("✨ Recommended Movies:")
            
            if 'error' in recommendations[0]:
                st.error(recommendations[0]['error'])
            else:
                for i, rec in enumerate(recommendations, 1):
                    with st.container():
                        st.markdown(f"""
                        <div class="recommendation-card fade-in">
                            <h4>#{i} 🎬 {rec['title']}</h4>
                            <p><strong>Similarity Score:</strong> {rec['similarity_score']}</p>
                            <p><strong>Genres:</strong> {rec['genres']}</p>
                            <p><strong>Director:</strong> {rec['director']}</p>
                            <p><strong>Year:</strong> {rec['release_year']} | <strong>Rating:</strong> {rec['rating']} | <strong>Duration:</strong> {rec['duration']}</p>
                            <p><strong>Country:</strong> {rec['country']}</p>
                            <p><strong>Description:</strong> {rec['description']}</p>
                        </div>
                        """, unsafe_allow_html=True)
        
        elif rec_type == "Genre-Based":
            recommendations = recommender.get_similar_movies_by_genre(selected_genre, top_n=top_n)
            
            st.markdown("---")
            st.subheader(f"🎭 Movies in {selected_genre} Genre:")
            
            if 'error' in recommendations[0]:
                st.error(recommendations[0]['error'])
            else:
                for i, rec in enumerate(recommendations, 1):
                    with st.container():
                        st.markdown(f"""
                        <div class="recommendation-card fade-in">
                            <h4>#{i} 🎬 {rec['title']}</h4>
                            <p><strong>Year:</strong> {rec['release_year']} | <strong>Rating:</strong> {rec['rating']}</p>
                            <p><strong>Director:</strong> {rec['director']}</p>
                            <p><strong>Genres:</strong> {rec['genres']}</p>
                            <p><strong>Description:</strong> {rec['description']}</p>
                        </div>
                        """, unsafe_allow_html=True)
        
        elif rec_type == "Director-Based":
            recommendations = recommender.get_movies_by_director(selected_director, top_n=top_n)
            
            st.markdown("---")
            st.subheader(f"🎭 Movies by {selected_director}:")
            
            if 'error' in recommendations[0]:
                st.error(recommendations[0]['error'])
            else:
                for i, rec in enumerate(recommendations, 1):
                    with st.container():
                        st.markdown(f"""
                        <div class="recommendation-card fade-in">
                            <h4>#{i} 🎬 {rec['title']}</h4>
                            <p><strong>Year:</strong> {rec['release_year']} | <strong>Rating:</strong> {rec['rating']}</p>
                            <p><strong>Genres:</strong> {rec['genres']}</p>
                            <p><strong>Description:</strong> {rec['description']}</p>
                        </div>
                        """, unsafe_allow_html=True)

# --- Movie Information Section ---
st.markdown("---")
st.subheader("📋 Movie Information")

if selected_movie:
    movie_info = recommender.get_movie_info(selected_movie)
    
    if movie_info:
        st.markdown('<div class="movie-info-section fade-in">', unsafe_allow_html=True)
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            **🎬 Title:** {movie_info['title']}  
            **🎭 Director:** {movie_info['director']}  
            **🧾 Genres:** {movie_info['genres']}  
            **🌍 Country:** {movie_info['country']}  
            **📅 Year:** {movie_info['release_year']}  
            **📊 Rating:** {movie_info['rating']}  
            **⏱️ Duration:** {movie_info['duration']}
            """)
        
        with col2:
            st.markdown("**📃 Description:**")
            st.text_area("", movie_info['description'], height=150, disabled=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("❌ Movie information not found")

# --- Model Statistics ---
st.markdown("---")
st.subheader("📊 Model Statistics")

st.markdown('<div class="stats-container">', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card fade-in">
        <h3>🎬</h3>
        <h4>Total Movies</h4>
        <h2>{:,}</h2>
    </div>
    """.format(len(df)), unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card fade-in">
        <h3>🎭</h3>
        <h4>Genres</h4>
        <h2>{}</h2>
    </div>
    """.format(len(df['genres'].str.split(', ').explode().unique())), unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card fade-in">
        <h3>🌍</h3>
        <h4>Countries</h4>
        <h2>{}</h2>
    </div>
    """.format(len(df['country'].dropna().unique())), unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card fade-in">
        <h3>📊</h3>
        <h4>TF-IDF Features</h4>
        <h2>{:,}</h2>
    </div>
    """.format(recommender.tfidf_matrix.shape[1]), unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
