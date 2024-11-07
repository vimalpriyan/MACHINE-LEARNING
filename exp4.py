import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer

# Load and prepare the Iris dataset for binary classification
iris = load_iris()
data = iris.data
target = (iris.target == 1).astype(int)  # Binary classification (target is 1 or not)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# Standardize features for better training performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Neural Network parameters
input_neurons = X_train.shape[1]   # Number of input features
hidden_neurons = 5                 # Number of neurons in the hidden layer
output_neurons = 1                 # Single output for binary classification
learning_rate = 0.1                # Learning rate for weight updates
epochs = 1000                      # Number of iterations for training

# Initialize weights and biases
np.random.seed(42)
weights_input_hidden = np.random.uniform(-1, 1, (input_neurons, hidden_neurons))
weights_hidden_output = np.random.uniform(-1, 1, (hidden_neurons, output_neurons))
bias_hidden = np.zeros((1, hidden_neurons))
bias_output = np.zeros((1, output_neurons))

# Activation function (Sigmoid) and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Training the network using backpropagation
for epoch in range(epochs):
    # Forward propagation
    hidden_input = np.dot(X_train, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)
    
    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    final_output = sigmoid(final_input)
    
    # Calculate the error
    error = y_train.reshape(-1, 1) - final_output
    if epoch % 100 == 0:
        print(f"Epoch {epoch} - Error: {np.mean(np.abs(error))}")
    
    # Backward propagation
    d_output = error * sigmoid_derivative(final_output)
    
    error_hidden_layer = d_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_output)
    
    # Update weights and biases
    weights_hidden_output += hidden_output.T.dot(d_output) * learning_rate
    bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    weights_input_hidden += X_train.T.dot(d_hidden_layer) * learning_rate
    bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

# Testing the model
hidden_layer_activation = sigmoid(np.dot(X_test, weights_input_hidden) + bias_hidden)
output_layer_activation = sigmoid(np.dot(hidden_layer_activation, weights_hidden_output) + bias_output)
predictions = (output_layer_activation > 0.5).astype(int)

# Calculate accuracy
accuracy = np.mean(predictions.flatten() == y_test) * 100
print(f"Testing Accuracy: {accuracy:.2f}%")
