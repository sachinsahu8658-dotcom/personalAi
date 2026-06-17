import numpy as np

class PersonalAI:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize random weights and biases between layers
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        # Forward pass: pass inputs through the hidden layer to the output
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def train(self, X, y, epochs=1000, lr=0.1):
        # Backpropagation loop to help the AI learn from its mistakes
        for epoch in range(epochs):
            # 1. Forward propagation
            output = self.forward(X)
            
            # 2. Calculate the error (Loss)
            error = y - output
            
            # 3. Backpropagation calculation
            d_output = error * self.sigmoid_derivative(output)
            error_hidden = d_output.dot(self.W2.T)
            d_hidden = error_hidden * self.sigmoid_derivative(self.a1)
            
            # 4. Update the weights and biases (The actual learning)
            self.W2 += self.a1.T.dot(d_output) * lr
            self.b2 += np.sum(d_output, axis=0, keepdims=True) * lr
            self.W1 += X.T.dot(d_hidden) * lr
            self.b1 += np.sum(d_hidden, axis=0, keepdims=True) * lr
