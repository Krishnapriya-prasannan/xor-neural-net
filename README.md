
# Day 5 - AI for Logic Learning (Solving XOR with a Neural Network)

This is part of my **#100DaysOfAI** challenge.  
On **Day 5**, I built a simple neural network from scratch using **NumPy** to solve the classic **XOR logic gate problem**, which cannot be solved using linear models.

---

## Goal  
To implement a minimal neural network using Python and NumPy that can learn the XOR pattern through forward propagation, backpropagation, and training over time.

---

## Technologies Used  

| Tool   | Purpose                           |
|--------|-----------------------------------|
| Python | Main programming language         |
| NumPy  | Numerical operations & matrix math|
| VS Code | Code editor                      |

---

##  How It Works

- Defined the XOR inputs and expected outputs.
- Built a 2-layer neural network:
  - **Input layer**: 2 neurons  
  - **Hidden layer**: 4 neurons  
  - **Output layer**: 1 neuron  
- Used `tanh` as the activation function.
- Trained the model using **manual gradient descent** and **mean squared error (MSE)**.
- After training, the network correctly predicts the XOR output.

---

##  Highlights

- No ML libraries were used — only **NumPy** for math operations.
- Implemented both **forward** and **backward** propagation manually.
- Demonstrated how a neural network can solve **non-linear problems** like XOR.

---

##  What I Learned

- How to build and train a neural network from scratch  
- Importance of hidden layers for non-linear problems  
- How forward pass and backpropagation work  
- How to apply activation functions and calculate gradients manually  

---

