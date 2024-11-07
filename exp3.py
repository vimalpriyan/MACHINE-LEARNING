import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from collections import Counter
import math

# Load the Iris dataset and prepare it for binary classification
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target

# Convert the target to binary classification (class 1 = "Yes", others = "No")
data['target'] = data['target'].apply(lambda x: 'Yes' if x == 1 else 'No')

# Display a sample of the dataset
print("Sample of the dataset:")
print(data.head())

# Function to calculate entropy
def entropy(labels):
    label_counts = Counter(labels)
    total_count = len(labels)
    return -sum((count/total_count) * math.log2(count/total_count) for count in label_counts.values())

# Function to calculate information gain
def information_gain(data, split_attribute, target_attribute="target"):
    total_entropy = entropy(data[target_attribute])
    values = data[split_attribute].unique()
    
    weighted_entropy = sum(
        (len(subset) / len(data)) * entropy(subset[target_attribute])
        for value in values
        for subset in [data[data[split_attribute] == value]]
    )
    return total_entropy - weighted_entropy

# Function to find the best attribute to split on
def find_best_attribute(data, attributes, target_attribute="target"):
    return max(attributes, key=lambda attr: information_gain(data, attr, target_attribute))

# Recursive function to build the decision tree
def id3(data, attributes, target_attribute="target"):
    labels = data[target_attribute]
    
    # Base cases
    if len(labels.unique()) == 1:
        return labels.iloc[0]  # Return the single class label
    if not attributes:
        return labels.mode()[0]  # Return the most common label if no attributes left

    # Choose the best attribute
    best_attr = find_best_attribute(data, attributes, target_attribute)
    tree = {best_attr: {}}

    # Recursively create branches for each value of the best attribute
    for value in data[best_attr].unique():
        subset = data[data[best_attr] == value]
        subtree = id3(
            subset,
            [attr for attr in attributes if attr != best_attr],
            target_attribute
        )
        tree[best_attr][value] = subtree

    return tree

# Build the decision tree using ID3 algorithm
attributes = list(data.columns[:-1])
decision_tree = id3(data, attributes)
print("Decision Tree:", decision_tree)

# Function to classify a new sample with the decision tree
def classify(tree, sample):
    if not isinstance(tree, dict):
        return tree  # Return the leaf value (class label)

    attribute = next(iter(tree))
    attribute_value = sample[attribute]
    
    if attribute_value in tree[attribute]:
        return classify(tree[attribute][attribute_value], sample)
    else:
        return None  # If the value wasn't seen in training, return None or a default

# Classify a new sample
new_sample = pd.Series({
    'sepal length (cm)': 5.1,
    'sepal width (cm)': 2.5,
    'petal length (cm)': 3.0,
    'petal width (cm)': 1.1
})
classification = classify(decision_tree, new_sample)
print("Classification of the new sample:", classification)
