from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

# Load the Iris dataset and convert it to a DataFrame
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add the target as a new column
data['target'] = iris.target

# Convert the target to a binary classification (e.g., "Yes" for class 1, "No" for others)
data['target'] = data['target'].apply(lambda x: 'Yes' if x == 1 else 'No')

# Display the dataset
print("Sample of the dataset:")
print(data.head())

# Initialize the most specific hypothesis (S) and most general hypothesis (G)
def initialize_hypotheses(num_attributes):
    S = ['∅'] * num_attributes   # Most specific hypothesis
    G = [['?'] * num_attributes]  # Most general hypothesis
    return S, G

# Update the specific hypothesis (S) to be consistent with a positive example
def update_specific_hypothesis(S, example):
    for i, value in enumerate(example):
        if S[i] == '∅':
            S[i] = value
        elif S[i] != value:
            S[i] = '?'
    return S

# Update the general hypotheses (G) to exclude a negative example
def update_general_hypotheses(G, example, S):
    G = [g for g in G if all(g[i] == '?' or g[i] == example.iloc[i] for i in range(len(example)))]
    
    # Add new hypotheses if S is not consistent with the negative example
    new_general_hypotheses = []
    for g in G:
        for i in range(len(g)):
            if g[i] == '?':
                hypothesis = g[:]
                hypothesis[i] = S[i]
                if hypothesis not in new_general_hypotheses:
                    new_general_hypotheses.append(hypothesis)
    return new_general_hypotheses or G  # Ensure G does not become empty

# Candidate-Elimination algorithm
def candidate_elimination_algorithm(data):
    num_attributes = len(data.columns) - 1
    S, G = initialize_hypotheses(num_attributes)
    
    for _, row in data.iterrows():
        example, target = row.iloc[:-1], row.iloc[-1]  # Use iloc for positional indexing
        
        if target == 'Yes':  # Positive example
            # Update S
            S = update_specific_hypothesis(S, example)
            # Remove hypotheses from G that are inconsistent with S
            G = [g for g in G if all(g[i] == '?' or g[i] == S[i] for i in range(len(S)))]
            
        else:  # Negative example
            # Update G
            G = update_general_hypotheses(G, example, S)
            
    return S, G

# Run the Candidate-Elimination algorithm on the Iris dataset
S, G = candidate_elimination_algorithm(data)
print("Final Specific Hypothesis (S):", S)
print("Final General Hypotheses (G):", G)
