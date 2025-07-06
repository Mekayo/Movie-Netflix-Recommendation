#!/usr/bin/env python3
"""
Test Script for Enhanced Content-Based Filtering
Demonstrates various features of the recommendation system
"""

import pandas as pd
from reco_part import ContentBasedRecommender, recommend_content_based
import time

def test_basic_recommendations():
    """Test basic movie recommendations"""
    print("=" * 60)
    print("🎬 TESTING BASIC MOVIE RECOMMENDATIONS")
    print("=" * 60)
    
    # Test movies
    test_movies = [
        "The Dark Knight",
        "Inception", 
        "Pulp Fiction",
        "The Shawshank Redemption",
        "Fight Club"
    ]
    
    for movie in test_movies:
        print(f"\n🔍 Finding recommendations for: {movie}")
        print("-" * 40)
        
        try:
            recommendations = recommend_content_based(movie, top_n=3)
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec}")
                print()
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 40)

def test_enhanced_features():
    """Test enhanced recommender features"""
    print("\n" + "=" * 60)
    print("🚀 TESTING ENHANCED FEATURES")
    print("=" * 60)
    
    # Initialize the enhanced recommender
    recommender = ContentBasedRecommender(r"C:\Users\monst\Desktop\Movie-Netflix-Recommendation\data\netflix_titles_cleaned.csv")
    recommender.initialize_model()
    
    # Test movie information retrieval
    print("\n📋 MOVIE INFORMATION RETRIEVAL")
    print("-" * 40)
    
    test_movie = "The Dark Knight"
    movie_info = recommender.get_movie_info(test_movie)
    
    if movie_info:
        print(f"🎬 Title: {movie_info['title']}")
        print(f"🎭 Director: {movie_info['director']}")
        print(f"🧾 Genres: {movie_info['genres']}")
        print(f"🌍 Country: {movie_info['country']}")
        print(f"📅 Year: {movie_info['release_year']}")
        print(f"📊 Rating: {movie_info['rating']}")
        print(f"⏱️ Duration: {movie_info['duration']}")
        print(f"📃 Description: {movie_info['description'][:100]}...")
    else:
        print(f"❌ Movie '{test_movie}' not found")
    
    # Test genre-based recommendations
    print("\n🎭 GENRE-BASED RECOMMENDATIONS")
    print("-" * 40)
    
    test_genres = ["Action", "Comedy", "Drama", "Horror"]
    
    for genre in test_genres:
        print(f"\n🎬 Movies in {genre} genre:")
        genre_movies = recommender.get_similar_movies_by_genre(genre, top_n=3)
        
        if 'error' not in genre_movies[0]:
            for i, movie in enumerate(genre_movies, 1):
                print(f"  {i}. {movie['title']} ({movie['release_year']}) - {movie['rating']}")
        else:
            print(f"  {genre_movies[0]['error']}")
    
    # Test director-based recommendations
    print("\n🎭 DIRECTOR-BASED RECOMMENDATIONS")
    print("-" * 40)
    
    test_directors = ["Christopher Nolan", "Quentin Tarantino", "Martin Scorsese"]
    
    for director in test_directors:
        print(f"\n🎬 Movies by {director}:")
        director_movies = recommender.get_movies_by_director(director, top_n=3)
        
        if 'error' not in director_movies[0]:
            for i, movie in enumerate(director_movies, 1):
                print(f"  {i}. {movie['title']} ({movie['release_year']}) - {movie['rating']}")
        else:
            print(f"  {director_movies[0]['error']}")

def test_performance():
    """Test recommendation performance"""
    print("\n" + "=" * 60)
    print("⚡ PERFORMANCE TESTING")
    print("=" * 60)
    
    recommender = ContentBasedRecommender(r"C:\Users\monst\Desktop\Movie-Netflix-Recommendation\data\netflix_titles_cleaned.csv")
    
    # Time the initialization
    print("🔄 Timing model initialization...")
    start_time = time.time()
    recommender.initialize_model()
    init_time = time.time() - start_time
    print(f"✅ Model initialized in {init_time:.2f} seconds")
    
    # Time recommendation generation
    print("\n🔄 Timing recommendation generation...")
    test_movie = "The Dark Knight"
    
    start_time = time.time()
    recommendations = recommender.recommend_content_based(test_movie, top_n=10)
    rec_time = time.time() - start_time
    
    print(f"✅ Generated {len(recommendations)} recommendations in {rec_time:.3f} seconds")
    
    # Show matrix information
    print(f"\n📊 TF-IDF Matrix Shape: {recommender.tfidf_matrix.shape}")
    print(f"📊 Similarity Matrix Shape: {recommender.cosine_sim.shape}")
    print(f"📊 Number of Movies: {len(recommender.df)}")

def test_similarity_analysis():
    """Test similarity analysis features"""
    print("\n" + "=" * 60)
    print("🔍 SIMILARITY ANALYSIS")
    print("=" * 60)
    
    recommender = ContentBasedRecommender(r"C:\Users\monst\Desktop\Movie-Netflix-Recommendation\data\netflix_titles_cleaned.csv")
    recommender.initialize_model()
    
    # Test different similarity thresholds
    test_movie = "The Dark Knight"
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    print(f"\n🎬 Testing similarity thresholds for: {test_movie}")
    print("-" * 50)
    
    for threshold in thresholds:
        recommendations = recommender.recommend_content_based(test_movie, top_n=10, min_similarity=threshold)
        
        if 'error' not in recommendations[0]:
            print(f"\n📈 Threshold {threshold}: {len(recommendations)} recommendations")
            if recommendations:
                avg_similarity = sum(rec['similarity_score'] for rec in recommendations) / len(recommendations)
                print(f"   Average similarity: {avg_similarity:.3f}")
                
                # Show top 3
                for i, rec in enumerate(recommendations[:3], 1):
                    print(f"   {i}. {rec['title']} (Score: {rec['similarity_score']})")
        else:
            print(f"📈 Threshold {threshold}: {recommendations[0]['error']}")

def main():
    """Run all tests"""
    print("🎬 NETFLIX CONTENT-BASED RECOMMENDATION SYSTEM")
    print("=" * 60)
    print("Testing enhanced content-based filtering capabilities...")
    
    try:
        # Run all tests
        test_basic_recommendations()
        test_enhanced_features()
        test_performance()
        test_similarity_analysis()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 