from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt

# 1. Binary Ground Truth (1 = Relevant, 0 = Not Relevant)
# Based on your journal example [cite: 206]
y_true = [0, 1, 1, 0, 1, 1]

# 2. Model's estimation scores (probability of relevance)
y_scores = [0.1, 0.4, 0.35, 0.8, 0.65, 0.9]

# 3. Calculate Average Precision (AP) t
average_precision = average_precision_score(y_true, y_scores)

print(f'Average precision-recall score: {average_precision:.4f}')
