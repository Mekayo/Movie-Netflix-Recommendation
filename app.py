# app.py
import streamlit as st
from reco_part import recommend_content_based
import pandas as pd

# Load the dataset to get movie titles
df = pd.read_csv(r"C:\Users\monst\Desktop\project\Movie-Netflix-Recommendation\data\netflix_titles_cleaned.csv")
movie_list = df['title'].dropna().unique().tolist()
movie_list.sort()

# --- Page Config ---
st.set_page_config(
    page_title="Netflix Movie Recommender 🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom Background + Styling ---
st.markdown("""
    <style>
    .stApp {
        background-size: cover;
        background-attachment: fixed;
        color: white;
    }
        
    .block-container {
        background-color: rgba(0, 0, 0, 0.7);
        padding: 2rem;
        border-radius: 15px;
    }
    h1, h2, h3 {
        color: #FF4C4B;
    }
    </style>
""", unsafe_allow_html=True)

# --- Main UI ---
st.markdown("<div class='block-container'>", unsafe_allow_html=True)
st.title("🎬 Welcome to the Netflix Movie Recommender")
st.subheader("Find similar movies based on what you love ❤️")

# Movie Search
selected_movie = st.selectbox("🔍 Select a movie you like:", movie_list)

# Top N Slider
top_n = st.slider("🎯 Number of Recommendations", 3, 10, 5)

# Recommend Button
if st.button("🎉 Recommend"):
    with st.spinner("Finding the best matches..."):
        recommendations = recommend_content_based(selected_movie, top_n)
        st.markdown("---")
        st.subheader("✨ Recommended for you:")
        for rec in recommendations:
            st.markdown(rec)
            st.markdown("---")

st.markdown("</div>", unsafe_allow_html=True)
