import numpy as np
class Softmax:

    # num_outputs = 10, pois mnist tem 10 classes de digitos
    def __init__(self, num_inputs, num_outputs):
        self.input = None
        self.input_dim = None
        self.valores = None
        # inicializando pesos aleatorios
        self.weights = np.random.randn(num_inputs, num_outputs) / num_inputs
        self.bias = np.zeros(num_outputs)

    def forward(self, input):
        self.input = input
        self.input_dim = input.shape
        self.input = self.input.flatten()
        # A = X * W + b
        self.valores = np.dot(self.input, self.weights) + self.bias
        # exponencial, ou seja, e^x (retornando array de probabilidades)
        exp = np.exp(self.valores)
        # para normalizar dividindo pelo somatorio do array (conforme softmax)
        return exp / np.sum(exp, axis=0)
        
    def backward(self, dZ, learning_rate):
        output = None

        for i, g in enumerate(dZ):
            if g  == 0:
                continue

            #etk = np.exp(self.valores) = np.dot(self.input, self.weights) + self.bias
            total_erros = np.exp(self.valores)

            # S = np.sum(total_erros)
            soma_total_erros = np.sum(total_erros)

            #etc = total_erros[i]
            # -etc * etk / (S ** 2)
            media_geral_erros = -total_erros[i] * total_erros / (soma_total_erros ** 2)

            # -etk * (S - etc) / (S ** 2)
            media_geral_erros[i] = total_erros[i] * (soma_total_erros - total_erros[i]) / (soma_total_erros ** 2)

            # gradient * media_geral_erros
            gradient_loss = g * media_geral_erros

            output = self.weights @ gradient_loss

            self.weights -= learning_rate * self.input[np.newaxis].T @ gradient_loss[np.newaxis]

            self.bias -= learning_rate * gradient_loss
        return output.reshape(self.input_dim)
    