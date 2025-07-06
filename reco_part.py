import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
    except Exception as e:
        print(f"Warning: Could not download punkt: {e}")
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    try:
        nltk.download('stopwords', quiet=True)
    except Exception as e:
        print(f"Warning: Could not download stopwords: {e}")
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    try:
        nltk.download('wordnet', quiet=True)
    except Exception as e:
        print(f"Warning: Could not download wordnet: {e}")

class ContentBasedRecommender:
    def __init__(self, data_path):
        """
        Initialize the Content-Based Recommender
        
        Args:
            data_path (str): Path to the cleaned Netflix dataset
        """
        try:
            self.df = pd.read_csv(data_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {data_path}")
        except Exception as e:
            raise Exception(f"Error loading data: {e}")
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.indices = None
        self.feature_weights = {
            'title': 1.0,
            'director': 0.8,
            'cast': 0.6,
            'genres': 1.2,
            'description': 1.0,
            'country': 0.4
        }
        
    def preprocess_text(self, text):
        """
        Clean and preprocess text data
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def create_enhanced_features(self):
        """
        Create enhanced feature combinations with weighted importance
        """
        print("🔄 Creating enhanced features...")
        
        # Preprocess individual columns
        self.df['title_clean'] = self.df['title'].apply(self.preprocess_text)
        self.df['director_clean'] = self.df['director'].apply(self.preprocess_text)
        self.df['cast_clean'] = self.df['cast'].apply(self.preprocess_text)
        self.df['genres_clean'] = self.df['genres'].apply(self.preprocess_text)
        self.df['description_clean'] = self.df['description'].apply(self.preprocess_text)
        self.df['country_clean'] = self.df['country'].apply(self.preprocess_text)
        
        # Create weighted combined features by repeating text based on weights
        def repeat_text_by_weight(text, weight):
            """Repeat text based on weight to give it more importance"""
            if not isinstance(text, str) or not text:
                return ''
            multiplier = max(1, int(weight * 10))
            return ' '.join([text] * multiplier)

        # Apply weighting to each column
        self.df['title_weighted'] = self.df['title_clean'].apply(lambda x: repeat_text_by_weight(x, self.feature_weights['title']))
        self.df['director_weighted'] = self.df['director_clean'].apply(lambda x: repeat_text_by_weight(x, self.feature_weights['director']))
        self.df['cast_weighted'] = self.df['cast_clean'].apply(lambda x: repeat_text_by_weight(x, self.feature_weights['cast']))
        self.df['genres_weighted'] = self.df['genres_clean'].apply(lambda x: repeat_text_by_weight(x, self.feature_weights['genres']))
        self.df['description_weighted'] = self.df['description_clean'].apply(lambda x: repeat_text_by_weight(x, self.feature_weights['description']))
        self.df['country_weighted'] = self.df['country_clean'].apply(lambda x: repeat_text_by_weight(x, self.feature_weights['country']))

        # Concatenate all weighted features
        self.df['combined_weighted'] = (
            self.df['title_weighted'] + ' ' +
            self.df['director_weighted'] + ' ' +
            self.df['cast_weighted'] + ' ' +
            self.df['genres_weighted'] + ' ' +
            self.df['description_weighted'] + ' ' +
            self.df['country_weighted']
        )
        
        # Create genre-specific features
        self.df['genre_list'] = self.df['genres'].str.split(', ')
        
        print("✅ Enhanced features created successfully!")
    
    def build_tfidf_matrix(self, max_features=5000, ngram_range=(1, 2)):
        """
        Build TF-IDF matrix with enhanced parameters
        
        Args:
            max_features (int): Maximum number of features
            ngram_range (tuple): Range of n-grams to consider
        """
        print("🔄 Building TF-IDF matrix...")
        
        # Initialize TF-IDF vectorizer with enhanced parameters
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
        
        # Fit and transform the combined text
        self.tfidf_matrix = self.tfidf.fit_transform(self.df['combined_weighted'])
        
        print(f"✅ TF-IDF matrix built with {self.tfidf_matrix.shape[1]} features!")
    
    def compute_similarity_matrix(self):
        """
        Compute cosine similarity matrix
        """
        print("🔄 Computing similarity matrix...")
        
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        # Create index mapping
        self.indices = pd.Series(self.df.index, index=self.df['title'].str.lower()).drop_duplicates()
        
        print("✅ Similarity matrix computed successfully!")
    
    def get_movie_info(self, title):
        """
        Get detailed information about a movie
        
        Args:
            title (str): Movie title
            
        Returns:
            dict: Movie information
        """
        title_lower = title.lower()
        if title_lower not in self.indices:
            return None
        
        idx = self.indices[title_lower]
        movie = self.df.iloc[idx]
        
        return {
            'title': movie['title'],
            'director': movie['director'],
            'cast': movie['cast'],
            'genres': movie['genres'],
            'country': movie['country'],
            'release_year': movie['release_year'],
            'rating': movie['rating'],
            'duration': movie['duration'],
            'description': movie['description']
        }
    
    def recommend_content_based(self, title, top_n=5, min_similarity=0.1):
        """
        Get content-based recommendations
        
        Args:
            title (str): Movie title
            top_n (int): Number of recommendations
            min_similarity (float): Minimum similarity threshold
            
        Returns:
            list: List of recommended movies with details
        """
        title_lower = title.lower()
        
        if title_lower not in self.indices:
            return [{"error": "❌ Movie not found in the dataset."}]
        
        idx = self.indices[title_lower]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        
        # Filter by minimum similarity and sort
        sim_scores = [(i, score) for i, score in sim_scores if score >= min_similarity]
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
        
        if not sim_scores:
            return [{"error": "❌ No similar movies found."}]
        
        movie_indices = [i[0] for i in sim_scores]
        recommendations = []
        
        for i, movie_idx in enumerate(movie_indices):
            movie = self.df.iloc[movie_idx]
            score = sim_scores[i][1]
            
            recommendation = {
                'rank': i + 1,
                'title': movie['title'],
                'genres': movie['genres'],
                'director': movie['director'],
                'cast': movie['cast'],
                'country': movie['country'],
                'release_year': movie['release_year'],
                'rating': movie['rating'],
                'duration': movie['duration'],
                'description': movie['description'][:200] + '...' if len(movie['description']) > 200 else movie['description'],
                'similarity_score': round(score, 3)
            }
            recommendations.append(recommendation)
        
        return recommendations
    
    def get_similar_movies_by_genre(self, genre, top_n=10):
        """
        Get movies similar by genre
        
        Args:
            genre (str): Genre to search for
            top_n (int): Number of recommendations
            
        Returns:
            list: List of movies in the genre
        """
        genre_lower = genre.lower()
        genre_movies = self.df[self.df['genres'].str.lower().str.contains(genre_lower, na=False)]
        
        if genre_movies.empty:
            return [{"error": f"❌ No movies found for genre: {genre}"}]
        
        # Sort by release year (newer first) and return top_n
        genre_movies = genre_movies.sort_values('release_year', ascending=False).head(top_n)
        
        recommendations = []
        for _, movie in genre_movies.iterrows():
            recommendation = {
                'title': movie['title'],
                'genres': movie['genres'],
                'director': movie['director'],
                'release_year': movie['release_year'],
                'rating': movie['rating'],
                'description': movie['description'][:150] + '...' if len(movie['description']) > 150 else movie['description']
            }
            recommendations.append(recommendation)
        
        return recommendations
    
    def get_movies_by_director(self, director, top_n=10):
        """
        Get movies by the same director
        
        Args:
            director (str): Director name
            top_n (int): Number of recommendations
            
        Returns:
            list: List of movies by the director
        """
        director_lower = director.lower()
        director_movies = self.df[self.df['director'].str.lower().str.contains(director_lower, na=False)]
        
        if director_movies.empty:
            return [{"error": f"❌ No movies found for director: {director}"}]
        
        # Sort by release year (newer first) and return top_n
        director_movies = director_movies.sort_values('release_year', ascending=False).head(top_n)
        
        recommendations = []
        for _, movie in director_movies.iterrows():
            recommendation = {
                'title': movie['title'],
                'genres': movie['genres'],
                'release_year': movie['release_year'],
                'rating': movie['rating'],
                'description': movie['description'][:150] + '...' if len(movie['description']) > 150 else movie['description']
            }
            recommendations.append(recommendation)
        
        return recommendations
    
    def initialize_model(self):
        """
        Initialize the complete recommendation model
        """
        print("🚀 Initializing Content-Based Recommendation Model...")
        self.create_enhanced_features()
        self.build_tfidf_matrix()
        self.compute_similarity_matrix()
        print("✅ Model initialized successfully!")

# Initialize the recommender
recommender = ContentBasedRecommender("data/netflix_titles_cleaned.csv")

# Function to get recommendations (for backward compatibility)
def recommend_content_based(title, top_n=5):
    """
    Legacy function for backward compatibility
    """
    if not hasattr(recommender, 'cosine_sim'):
        recommender.initialize_model()
    
    recommendations = recommender.recommend_content_based(title, top_n)
    
    # Convert to old format for compatibility
    if recommendations and 'error' in recommendations[0]:
        return [recommendations[0]['error']]
    
    results = []
    for rec in recommendations:
        summary = f"""🎬 **{rec['title']}** (Rank: #{rec['rank']})
🧾 *Genres:* {rec['genres']}
🎭 *Director:* {rec['director']}
🌍 *Country:* {rec['country']}
📅 *Year:* {rec['release_year']}
📊 *Rating:* {rec['rating']}
⏱️ *Duration:* {rec['duration']}
📃 *Description:* {rec['description']}
📈 *Similarity Score:* `{rec['similarity_score']}`"""
        results.append(summary)
    
    return results

def evaluate_model(y_true, y_pred):
    """
    Evaluate model performance
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, mae