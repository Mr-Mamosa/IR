import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import heapq

def extractive_summary(text, num_sentences=3):
    # Download necessary resources from NLTK [cite: 502, 503, 504]
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)

    stop_words = set(stopwords.words('english'))
    word_frequencies = {}

    # Step 1: Calculate Word Frequencies [cite: 506]
    for word in word_tokenize(text):
        if word.lower() not in stop_words:
            if word not in word_frequencies:
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    # Step 2: Normalize Frequencies [cite: 511, 513]
    maximum_frequency = max(word_frequencies.values())
    for word in word_frequencies:
        word_frequencies[word] = (word_frequencies[word] / maximum_frequency)

    # Step 3: Score Sentences [cite: 515, 516]
    sentence_scores = {}
    for sent in sent_tokenize(text):
        for word in word_tokenize(sent.lower()):
            if word in word_frequencies:
                if sent not in sentence_scores:
                    sentence_scores[sent] = word_frequencies[word]
                else:
                    sentence_scores[sent] += word_frequencies[word]

    # Step 4: Select Top Sentences
    summary_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(summary_sentences)
    return summary

# Example Text (Source: Avengers script example from journal) [cite: 531]
text_content = """
The Avengers assembled once again when a powerful alien force threatened Earth.
Iron Man used his advanced technology to analyze the enemy's weaknesses,
while Captain America led the team with courage and strategy.
Thor unleashed the power of lightning to defend civilians,
and Hulk smashed through enemy lines with unstoppable strength.
Black Widow and Hawkeye provided tactical support, ensuring the team stayed coordinated.
Despite overwhelming odds, the Avengers worked together, proving that unity and sacrifice are the true keys to victory.
In the end, Earth was saved, but the heroes knew that new threats would always rise.
"""

result = extractive_summary(text_content, num_sentences=2)
print("--- SUMMARY ---")
print(result)
