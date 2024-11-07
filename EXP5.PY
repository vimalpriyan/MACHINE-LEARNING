import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter

# Load the Iris dataset
iris = load_iris()
data = iris.data
target = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# Function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# K-NN function
def knn(X_train, y_train, test_point, k=3):
    # Calculate distances from the test point to all training points
    distances = [(euclidean_distance(test_point, x), y) for x, y in zip(X_train, y_train)]
    # Sort distances and get the labels of the k closest neighbors
    k_nearest_neighbors = sorted(distances, key=lambda x: x[0])[:k]
    k_nearest_labels = [label for _, label in k_nearest_neighbors]
    # Return the most common label among the neighbors
    most_common_label = Counter(k_nearest_labels).most_common(1)[0][0]
    return most_common_label

# Testing the K-NN function on the test set
def predict(X_train, y_train, X_test, k):
    predictions = [knn(X_train, y_train, test_point, k) for test_point in X_test]
    return predictions

# Making predictions and calculating accuracy
k = 3  # Number of neighbors
y_pred = predict(X_train, y_train, X_test, k)
accuracy = np.mean(y_pred == y_test) * 100

print(f"Predicted labels: {y_pred}")
print(f"Actual labels: {y_test}")
print(f"Accuracy: {accuracy:.2f}%")
