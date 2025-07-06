import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Load your cleaned movie dataset
df = pd.read_csv(r"C:\Users\monst\Desktop\project\Movie-Netflix-Recommendation\data\netflix_titles_cleaned.csv")

# Fill missing descriptions
#  done
# Create a combined field
# done

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined'])

# Cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Index mapping: movie title to index
indices = pd.Series(df.index, index=df['title'].str.lower()).drop_duplicates()

def recommend_content_based(title, top_n=5):
    title = title.lower()

    if title not in indices:
        return ["❌ Movie not found in the dataset."]

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
    movie_indices = [i[0] for i in sim_scores]

    recommendations = df.iloc[movie_indices][['title', 'genres', 'description']]
    results = []
    for i, row in enumerate(recommendations.itertuples()):
        score = sim_scores[i][1]  # Cosine similarity score
        summary = f"""🎬 **{row.title}**\n  
🧾 *Genres:* {row.genres} \n 
📃 *Description:* {row.description[:250]}...\n  
📈 *Similarity Score:* `{score:.2f}`"""

    for _, row in recommendations.iterrows():
        summary = f"""🎬 **{row['title']}**  🧾 *Genres:* {row['genres']}📃 *Description:* {row['description'][:250]}..."""
        results.append(summary)

    return results


def evaluate_dl_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, mae