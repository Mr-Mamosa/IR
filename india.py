from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Set up the corpus (Knowledge Source)
corpus = [
    "India has the second-largest population in the world.",
    "It is surrounded by oceans from three sides which are Bay Of Bengal in the east, the Arabian Sea in the west and Indian oceans in the south.",
    "Tiger is the national animal of India.",
    "Peacock is the national bird of India.",
    "Mango is the national fruit of India."
]

query = "Which is the national bird of India?"

# 2. Initialize TfidfVectorizer
# We use 'english' stop_words to ignore common terms like 'the', 'is', etc.
vectorizer = TfidfVectorizer(stop_words='english')

# 3. Fit and Transform the Corpus and Query
# We combine them to ensure the vector space (vocabulary) is consistent
tfidf_matrix = vectorizer.fit_transform(corpus)
query_vector = vectorizer.transform([query])

# 4. Calculate Cosine Similarity
# This measures the angle between the query vector and each document vector
similarity_scores = cosine_similarity(query_vector, tfidf_matrix)

# 5. Find and display the best answer
# argmax() finds the index of the highest similarity score
most_similar_index = similarity_scores.argmax()
answer = corpus[most_similar_index]

print(f"Query: {query}")
print(f"Answer: {answer}")
print(f"Similarity Score: {round(similarity_scores[0][most_similar_index], 4)}")
