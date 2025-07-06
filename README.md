# 🎬 Netflix Movie Recommendation System

**Advanced Content-Based Filtering with Machine Learning**

A sophisticated movie recommendation system that uses content-based filtering to suggest similar movies based on various features like genres, directors, cast, descriptions, and more.

## 🚀 Features

### **Enhanced Content-Based Filtering**
- **TF-IDF Vectorization**: Advanced text processing with n-grams and optimized parameters
- **Weighted Feature Importance**: Different weights for title, director, cast, genres, description, and country
- **Cosine Similarity**: Precise similarity calculations with configurable thresholds
- **Multi-Modal Recommendations**: Content-based, genre-based, and director-based filtering

### **Advanced Text Processing**
- **NLTK Integration**: Professional text preprocessing with lemmatization
- **Stop Word Removal**: Enhanced text cleaning for better feature extraction
- **Special Character Handling**: Robust text normalization
- **Missing Data Handling**: Intelligent handling of incomplete information

### **User Interface**
- **Streamlit Web App**: Beautiful, responsive web interface
- **Real-time Recommendations**: Instant movie suggestions
- **Performance Metrics**: Response time and similarity score tracking
- **Multiple Recommendation Types**: Content-based, genre-based, and director-based
- **Interactive Controls**: Adjustable similarity thresholds and recommendation counts

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Movie-Netflix-Recommendation
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 📊 How It Works

### **1. Data Preprocessing**
```python
# Enhanced feature creation with weighted importance
feature_weights = {
    'title': 1.0,        # High importance for movie titles
    'director': 0.8,     # Important for director-based similarity
    'cast': 0.6,         # Moderate importance for cast
    'genres': 1.2,       # Very important for genre matching
    'description': 1.0,  # Important for content similarity
    'country': 0.4       # Lower importance for country
}
```

### **2. TF-IDF Vectorization**
```python
# Advanced TF-IDF with optimized parameters
tfidf = TfidfVectorizer(
    max_features=5000,      # Limit features for efficiency
    stop_words='english',   # Remove common words
    ngram_range=(1, 2),     # Use unigrams and bigrams
    min_df=2,              # Minimum document frequency
    max_df=0.95,           # Maximum document frequency
    sublinear_tf=True      # Apply sublinear scaling
)
```

### **3. Similarity Calculation**
```python
# Cosine similarity with configurable thresholds
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
recommendations = recommend_content_based(movie, top_n=5, min_similarity=0.2)
```

## 🎯 Usage Examples

### **Content-Based Recommendations**
```python
from reco_part import ContentBasedRecommender

# Initialize the recommender
recommender = ContentBasedRecommender("data/netflix_titles_cleaned.csv")
recommender.initialize_model()

# Get recommendations
recommendations = recommender.recommend_content_based("The Dark Knight", top_n=5)
```

### **Genre-Based Recommendations**
```python
# Get movies by genre
action_movies = recommender.get_similar_movies_by_genre("Action", top_n=10)
```

### **Director-Based Recommendations**
```python
# Get movies by director
nolan_movies = recommender.get_movies_by_director("Christopher Nolan", top_n=10)
```

## 📈 Performance Features

### **Model Statistics**
- **Total Movies**: ~8,000+ Netflix titles
- **TF-IDF Features**: 5,000+ unique features
- **Response Time**: <1 second for recommendations
- **Similarity Accuracy**: Configurable thresholds (0.1-0.9)

### **Advanced Metrics**
- **Similarity Scores**: Precise cosine similarity calculations
- **Performance Tracking**: Response time monitoring
- **Quality Metrics**: Average similarity scores
- **Error Handling**: Robust error management

## 🔧 Configuration

### **Similarity Thresholds**
- **0.1-0.3**: Broad recommendations (more diverse)
- **0.3-0.6**: Balanced recommendations (recommended)
- **0.6-0.9**: Strict recommendations (very similar)

### **Feature Weights**
Adjust the importance of different features:
```python
feature_weights = {
    'title': 1.0,        # Movie title importance
    'director': 0.8,     # Director importance
    'cast': 0.6,         # Cast importance
    'genres': 1.2,       # Genre importance (highest)
    'description': 1.0,  # Description importance
    'country': 0.4       # Country importance (lowest)
}
```

## 🧪 Testing

Run the comprehensive test suite:
```bash
python test_content_based.py
```

The test suite includes:
- **Basic Recommendations**: Test core functionality
- **Enhanced Features**: Test advanced capabilities
- **Performance Testing**: Measure response times
- **Similarity Analysis**: Test different thresholds

## 📊 Model Architecture

```
Input Data → Text Preprocessing → TF-IDF Vectorization → Cosine Similarity → Recommendations
     ↓              ↓                    ↓                    ↓              ↓
Movie Info → NLTK Processing → Feature Matrix → Similarity Matrix → Ranked Results
```

## 🎨 Web Interface Features

### **Sidebar Controls**
- **Recommendation Type**: Choose between content-based, genre-based, or director-based
- **Movie Selection**: Search and select from 8,000+ movies
- **Parameters**: Adjust number of recommendations and similarity thresholds
- **Model Info**: View dataset statistics

### **Main Interface**
- **Recommendation Cards**: Beautiful display of movie recommendations
- **Performance Metrics**: Real-time response time and similarity scores
- **Movie Information**: Detailed movie details and descriptions
- **Statistics Dashboard**: Model performance and dataset overview

## 🔮 Future Enhancements

### **Planned Features**
- **Collaborative Filtering**: User-based recommendations
- **Deep Learning**: Neural network-based similarity
- **Image Analysis**: Movie poster-based recommendations
- **Sentiment Analysis**: Review-based recommendations
- **A/B Testing**: Recommendation quality evaluation

### **Advanced ML Models**
- **BERT Embeddings**: Advanced text understanding
- **Graph Neural Networks**: Movie relationship modeling
- **Autoencoders**: Dimensionality reduction
- **Ensemble Methods**: Combined recommendation approaches

## 📝 API Reference

### **ContentBasedRecommender Class**

#### **Methods**
- `initialize_model()`: Initialize the complete recommendation system
- `recommend_content_based(title, top_n, min_similarity)`: Get content-based recommendations
- `get_similar_movies_by_genre(genre, top_n)`: Get genre-based recommendations
- `get_movies_by_director(director, top_n)`: Get director-based recommendations
- `get_movie_info(title)`: Get detailed movie information

#### **Parameters**
- `title` (str): Movie title for recommendations
- `top_n` (int): Number of recommendations (default: 5)
- `min_similarity` (float): Minimum similarity threshold (default: 0.1)
- `genre` (str): Genre for filtering
- `director` (str): Director for filtering

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new features
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Hitesh Sharma** - Netflix Movie Recommendation System

---

**🎬 Enjoy discovering your next favorite movie!**