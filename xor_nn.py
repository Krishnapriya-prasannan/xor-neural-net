import numpy as np

# Inputs
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Outputs
y = np.array([[0],
              [1],
              [1],
              [0]])

np.random.seed(42)

W1 = np.random.randn(2, 4)
b1 = np.zeros((1, 4))

W2 = np.random.randn(4, 1)
b2 = np.zeros((1, 1))

# Activations
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    sx = sigmoid(x)
    return sx * (1 - sx)

lr = 0.5  
epochs = 20000

for epoch in range(epochs):
    z1 = np.dot(X, W1) + b1
    a1 = tanh(z1)

    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    loss = np.mean((y - a2) ** 2)

    d_a2 = 2 * (a2 - y)
    d_z2 = d_a2 * sigmoid_derivative(z2)
    d_W2 = np.dot(a1.T, d_z2)
    d_b2 = np.sum(d_z2, axis=0, keepdims=True)

    d_a1 = np.dot(d_z2, W2.T)
    d_z1 = d_a1 * tanh_derivative(z1)
    d_W1 = np.dot(X.T, d_z1)
    d_b1 = np.sum(d_z1, axis=0, keepdims=True)

    W2 -= lr * d_W2
    b2 -= lr * d_b2
    W1 -= lr * d_W1
    b1 -= lr * d_b1

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

print("\nFinal predictions:")
z1 = np.dot(X, W1) + b1
a1 = tanh(z1)
z2 = np.dot(a1, W2) + b2
a2 = sigmoid(z2)
print(np.round(a2))  
