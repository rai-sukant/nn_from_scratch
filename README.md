# **MNIST Digit Classification using a Custom Neural Network**

## **Overview**
This project implements a basic digit classification model using a custom-built neural network from scratch in Python. The model is trained on the **MNIST** dataset, which consists of handwritten digits, to classify them accurately.

---

## **Dataset**

The dataset used in this project is the **MNIST** dataset, which contains 70,000 grayscale images of handwritten digits (0-9). The dataset is split into:
- **Training set**: 60,000 images
- **Test set**: 10,000 images

### **Features of the Dataset**:
- **Image Dimensions**: 28 x 28 pixels
- **Label**: Each image is associated with a digit label (0-9)

---

## **Objective of the Model**
The primary objective of this model is to classify handwritten digits with high accuracy by training a neural network using the MNIST dataset. The key goals include:
- **Understanding the fundamentals of neural networks**: Implementing the architecture, forward propagation, and backpropagation from scratch.
- **Achieving good accuracy**: Targeting a classification accuracy above 80% on the test dataset.

---

## **Model Architecture**

The model follows a simple feedforward architecture consisting of:
1. **Input Layer**: 784 neurons (28 x 28 pixels flattened).
2. **Hidden Layer**: 10 neurons (using ReLU activation).
3. **Output Layer**: 10 neurons (using softmax activation for classification).

### **Activation Functions**:
- **ReLU (Rectified Linear Unit)**: Used for hidden layers to introduce non-linearity.
- **Softmax**: Used for the output layer to generate probabilities for each digit class.

---

## **Training Strategy**

### **Loss Function**:
- **Cross-Entropy Loss**: Measures the performance of the classification model whose output is a probability value between 0 and 1.

### **Optimizer**:
- **Gradient Descent**: Used for optimizing the weights and biases of the neural network.

### **Training Procedure**:
1. **Initialization**: Randomly initialize weights and biases.
2. **Forward Propagation**: Compute the predictions using the current weights and biases.
3. **Backward Propagation**: Calculate gradients and update parameters using gradient descent.

---

## **Evaluation Metrics**

The model's performance is evaluated based on:
- **Accuracy**: The ratio of correctly predicted instances to the total instances.
- **Visual Inspection**: Displaying some test images with their predicted labels to verify model performance.

---

## **Project Structure**


