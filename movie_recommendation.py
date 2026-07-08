import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("movies.csv")

print("========== DATASET ==========")
print(movies.head())

# Check missing values
print("\nMissing Values:")
print(movies.isnull().sum())

# Fill missing values
movies["genres"] = movies["genres"].fillna("")

# Convert text into numbers
vectorizer = TfidfVectorizer(stop_words="english")
feature_vectors = vectorizer.fit_transform(movies["genres"])

# Calculate similarity
similarity = cosine_similarity(feature_vectors)

print("\nSimilarity Matrix Shape:", similarity.shape)

# User input
movie_name = input("\nEnter your favourite movie: ")

# Find matching movie
movie_list = movies["title"].tolist()

matched_movie = None

for movie in movie_list:
    if movie.lower() == movie_name.lower():
        matched_movie = movie
        break

if matched_movie is None:
    print("\nMovie not found in dataset.")
else:
    index = movies[movies.title == matched_movie].index[0]

    similarity_scores = list(enumerate(similarity[index]))

    sorted_movies = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    print("\nTop 5 Recommended Movies:\n")

    count = 1

    for movie in sorted_movies[1:6]:
        movie_index = movie[0]
        print(count, ".", movies.iloc[movie_index]["title"])
        count += 1