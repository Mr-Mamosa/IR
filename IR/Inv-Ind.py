import nltk
from nltk.corpus import stopwords

# 1. Setup Data and Preprocessing
document1 = "best of luck tycs students for your practical examination."
document2 = "tycs students please carry your journal at the time of practical examination."

nltk.download('stopwords', quiet=True)
stop = set(stopwords.words('english'))

# Convert to lowercase tokens [cite: 13, 66]
tokens1 = set(document1.lower().replace('.', '').split())
tokens2 = set(document2.lower().replace('.', '').split())

# 2. Build Inverted Index [cite: 15, 63, 67]
# Only index words that aren't in the stopwords list [cite: 18, 19]
inverted_index = {}
all_words = (tokens1 | tokens2) - stop

for word in all_words:
    docs = []
    if word in tokens1: docs.append("Document 1") 
    if word in tokens2: docs.append("Document 2") 
    inverted_index[word] = set(docs) 

# 3. Simple Document Retrieval (AND Logic) [cite: 75, 89]
query = ["tycs", "journal"]
# Get document sets for each query term [cite: 78]
res1 = inverted_index.get(query[0], set())
res2 = inverted_index.get(query[1], set())

# Find common documents using intersection [cite: 80, 91]
final_result = res1.intersection(res2)

# 4. Display Output
print("Inverted Index:", inverted_index)
print(f"\nSearch for {query}:", list(final_result))
