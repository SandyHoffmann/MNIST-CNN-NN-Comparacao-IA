import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from layers import Convolution, MaxPooling, Softmax

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

print(X.shape, Y.shape)



x_test = X[treinamento_num:test_num, :].reshape(len(X[treinamento_num:test_num, :]), 28, 28)
x_test = x_test / 255
y_test = Y[treinamento_num:test_num].T

x_train = X[0:treinamento_num, :].reshape(len(X[0:treinamento_num, :]), 28, 28)
x_train = x_train / 255
y_train = Y[0:treinamento_num].T

# 28x28x1 -> 26x26x8
conv = Convolution.Convolution(8, 3)

# 26x26x8 -> 13x13x8
pool = MaxPooling.MaxPooling(2)

# 13x13x8 -> 10
softmax = Softmax.Softmax(13 * 13 * 8, 10)

show_sample_forward = False
show_sample_backward = True

def plot(A, tamanho=False):
    if tamanho:
        for i in range(tamanho):
            plt.subplot(1, tamanho, i + 1)
            plt.imshow(A[i, :, :], cmap='gray')
    else:
        for i in range(A.shape[2]):
            plt.subplot(1, A.shape[2], i + 1)
            plt.imshow(A[:, :, i], cmap='gray')

    plt.show()
def forward_propagation(x, y):
    global show_sample_forward

    A = conv.forward(x)

    if show_sample_forward:
        # mostrando layers convolucionais
        plot(A)


    # retornando apenas os pixeis mais altos, diminuindo resolucao
    A = pool.forward(A)

    if show_sample_forward:
        # mostrando layers max pooling
        plot(A)


    # softmax vai dar as probabilidades dentro de um vetor de 10 elementos
    A = softmax.forward(A)
    
    #cross entropy (-ln(A[y]))
    loss = -np.log(A[y])
    acc = 1 if np.argmax(A) == y else 0
    show_sample_forward = False

    return A, loss, acc

def treinamento(x_train, y_train, lr=.005):
    global show_sample_backward
    out, loss, acc = forward_propagation(x_train, y_train)

    grad = np.zeros(10)
    # derivada cross entropy
    # σL / σout = { 0 se i != y, -1/out[i] se i = y }
    grad[y_train] = 0
    if out[y_train] != 0:
        grad[y_train] = -1 / out[y_train]

    
    grad = softmax.backward(grad, lr)
    if show_sample_backward:
        plot(grad)

    grad = pool.backward(grad)

    if show_sample_backward:
        plot(grad)

    grad = conv.backward(grad, lr)

    if show_sample_backward:
        plot(grad, 8)

    show_sample_backward = False
    return loss, acc

epoch_qtd = 100

for epoch in range(1):
    print('--- Epoch %d ---' % (epoch + 1))

    loss = 0
    num_correct = 0
    for i, (im, label) in enumerate(zip(x_train, y_train)):
        if i > 0 and  i % epoch_qtd == epoch_qtd - 1:
            print('[Step {:5d}] Past {:d} steps: Average Loss {:.3f} | Accuracy: {:2.2f}%'
                   .format(i + 1, epoch_qtd, loss / i, num_correct))
            
            loss = 0
            num_correct = 0

        l, acc = treinamento(im, label)
        loss += l
        num_correct += acc

for epoch in range(1):
    print('--- Epoch %d ---' % (epoch + 1))

    num_errors = 0
    num_correct = 0
    print(x_test.shape, y_test.shape)
    for i, (im, label) in enumerate(zip(x_test, y_test)):
        
        if i > 0 and  i % epoch_qtd == epoch_qtd - 1:
            print('[Step {:5d}] Past {:d} steps: Average Loss {:.3f} | Accuracy: {:2.2f}%'
                   .format(i + 1, epoch_qtd, num_errors / i, num_correct / i))

        out, l, acc = forward_propagation(im, label)
        
        num_correct += acc
        num_errors += 1 - acc