# pip install scikit-learn nltk numpy
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import nltk
from nltk.corpus import stopwords
import numpy as np
from numpy.linalg import norm

# 1. Setup Data
corpus = [
    "The sun is the star at the center of the solar system.",
    "She wore a beautiful dress to the party last night.",
    "The book on the table caught my attention immediately."
]
query = ["solar system"]

# 2. Preprocessing Setup
nltk.download('stopwords', quiet=True)
stop_words_list = stopwords.words('english')

# 3. Initialize Vectorizer and Transformer
# We use CountVectorizer to get term frequencies (TF)
vectorizer = CountVectorizer(stop_words=stop_words_list)
transformer = TfidfTransformer()

# 4. Fit and Transform the Corpus
# Step A: Get word counts (TF)
counts = vectorizer.fit_transform(corpus).toarray()
# Step B: Apply TF-IDF weighting (IDF is calculated based on these 3 docs)
tfidf_matrix = transformer.fit_transform(counts).toarray()

# 5. Transform the Query
# Use the same vocabulary and IDF weights from the corpus
query_counts = vectorizer.transform(query).toarray()
query_tfidf = transformer.transform(query_counts).toarray()

# 6. Similarity Calculation (Cosine Similarity)
def cosine_similarity(a, b):
    # Handle cases where a vector might be all zeros to avoid division by zero
    if norm(a) == 0 or norm(b) == 0:
        return 0.0
    return round(np.inner(a, b) / (norm(a) * norm(b)), 3)

print("--- TF-IDF Matrix (Documents) ---")
print(tfidf_matrix)
print("\n--- Vocabulary ---")
print(vectorizer.get_feature_names_out())

print("\n--- Cosine Similarity (Query vs Documents) ---")
for i, doc_vector in enumerate(tfidf_matrix):
    score = cosine_similarity(doc_vector, query_tfidf[0])
    print(f"Document {i+1} Similarity: {score}")


# -----------

