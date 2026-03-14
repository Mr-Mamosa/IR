from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# 1. Define the Documents
documents = [
    "Machine learning is the study of computer algorithms that improve through experience.",
    "Deep learning is a subset of machine learning.",
    "Natural language processing is a field of artificial intelligence.",
    "Computer vision is a field of study that enables computers to interpret the visual world.",
    "Reinforcement learning is a machine learning algorithm.",
    "Information retrieval is the process of obtaining information from a collection.",
    "Text mining is the process of deriving high-quality information from text.",
    "Data clustering is the task of dividing a set of objects into groups.",
    "Hierarchical clustering builds a tree of clusters.",
    "K-means clustering is a method of vector quantization."
]

# 2. Vectorize the Text (TF-IDF)
# Converting text to numerical format while ignoring common English stop words
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

# 3. Apply K-Means Algorithm
# We will choose k=3 clusters for this specific set
k = 3
model = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1, random_state=42)
model.fit(X)

# 4. Analyze Results
print("--- Document Clusters ---")
clusters = model.labels_ 
for i, cluster_id in enumerate(clusters):
    print(f"Cluster {cluster_id}: {documents[i][:50]}...")
