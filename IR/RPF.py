def calculate_metrics(tp, fp, fn):
    # Precision: Accuracy of positive predictions
    precision = tp / (tp + fp)

    # Recall: Ability to find all relevant instances
    recall = tp / (tp + fn)

    # F-measure: Harmonic mean of Precision and Recall
    f_measure = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f_measure

# Given Values
true_positive = 20 
false_positive = 10 
false_negative = 30 

# Perform Calculation
p, r, f = calculate_metrics(true_positive, false_positive, false_negative)

# Display Results [cite: 183, 184]
print(f"True Positive: {true_positive}")
print(f"False Positive: {false_positive}")
print(f"False Negative: {false_negative}\n")

print(f"Precision: {round(p, 4)}")
print(f"Recall: {round(r, 4)}")
print(f"F-measure: {round(f, 4)}")
