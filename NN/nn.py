import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('./_data/train.csv')
data = np.array(data)

m, n = data.shape

np.random.shuffle(data)

# Transpose data
data = data

# Separando os labels e as features
Y = data[:, 0]
X = data[:, 1:n + 1]

#/255 pois os valores estao entre 0 e 255 e precisamos de 0 e 1

treinamento_num = 5000
test_num = 10000

x_test = X[treinamento_num:test_num, :].T
x_test = x_test / 255
y_test = Y[treinamento_num:test_num].T

x_train = X[0:treinamento_num, :].T
x_train = x_train / 255
y_train = Y[0:treinamento_num].T

print(x_train)
print(y_train)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

def inicializar_valores_wb():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5

    return W1, b1, W2, b2

def ReLU(x):
    return x * (x > 0)

def dReLU(x):
    return 1. * (x >= 0)
#softmax = e^x / sum(e^x)
def softmax(x):
    A = np.exp(x) / sum(np.exp(x))    
    return A
def forward_propagation(x, W1, b1, W2, b2):
    Z1 = W1.dot(x) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)

    loss = np.sum(-np.log(A2[y_train])) / len(y_train)
    return Z1, A1, Z2, A2, loss

"""
Converts a 1D array of integers representing labels into a one-hot encoded matrix.
example: [0, 2, 1] -> [[1, 0, 0], [0, 0, 1], [0, 1, 0]]
Parameters:
    Y (numpy.ndarray): A 1D array of integers representing the labels.
Returns:
    numpy.ndarray: A one-hot encoded matrix where each row corresponds to a label in Y.
"""
def convert_number_to_matrix_one(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y
def backward_propagation(Z1, A1, Z2, A2, W1, W2, x, y):
    one_hot_y = convert_number_to_matrix_one(y)
    dZ2 = A2 - one_hot_y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * dReLU(Z1)
    dW1 = 1 / m * dZ1.dot(x.T)
    db1 = 1 / m * np.sum(dZ1)

    return dW1, db1, dW2, db2

def gradient_descent(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    return W1, b1, W2, b2

def get_accuracy(predictions, y):
    return (np.sum(np.argmax(predictions, 0) == y) / y.size)
def rodar_epoca(x, y, learning_rate, iterations):
    W1, b1, W2, b2 = inicializar_valores_wb()
    for i in range(iterations):
        Z1, A1, Z2, A2, loss = forward_propagation(x, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = backward_propagation(Z1, A1, Z2, A2, W1, W2, x, y)

        W1, b1, W2, b2 = gradient_descent(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
        if i % 10 == 0:
            print("Iteração: ", i)
            print("Acurácia: ", get_accuracy(A2, y))
            print("Loss: ", loss)
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

epoch_qtd = 5000

def predict(x, W1, b1, W2, b2):
    _, _, _, A2 = forward_propagation(x, W1, b1, W2, b2)
    predictions = get_predictions(A2)
    return predictions

def predict_values(index, W1, b1, W2, b2):
    image = x_test[:, index, None]
    prediction = predict(image, W1, b1, W2, b2)
    label = y_train[index]
    print("Predição: ", prediction, "Valor Real: ", label)


W1, b1, W2, b2 = rodar_epoca(x_train, y_train, 0.1, epoch_qtd)

# print("Predições: ", predict_values(2, W1, b1, W2, b2))

dev_predictions = predict(x_test, W1, b1, W2, b2)
print("dev_predictions:", dev_predictions)
print("dev_predictions y:", y_test)

def get_accuracy_dev(predictions, y):
    predictions_correct = 0

    for i in range(len(predictions)):
        if predictions[i] == y[i]:
            predictions_correct += 1

    return predictions_correct / len(predictions)

print("Acurácia DEV: ", get_accuracy_dev(dev_predictions, y_test))