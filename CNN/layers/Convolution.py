import numpy as np

class Convolution:
    def __init__(self, qtd_filtros, tamanho_filtro):
        self.input = None
        self.qtd_filtros = qtd_filtros # 8 filtros
        self.tamanho_filtro = tamanho_filtro # 3x3
        # https://www.quora.com/What-is-an-intuitive-explanation-of-the-Xavier-Initialization-for-Deep-Neural-Networks
        self.filtros = np.random.randn(qtd_filtros, tamanho_filtro, tamanho_filtro) / (tamanho_filtro * tamanho_filtro) # 8x3x3, divide por 9 para ter uma media dos valores e nao variar muito (Xavier Initialization)

    def iterar_elementos(self, imagem):
        # itera cada regiao, de tamanho 3x3 e retorna para as operacoes de foward e backward
        for i in range(imagem.shape[0] - (self.tamanho_filtro - 1)):

            for j in range(imagem.shape[1] - (self.tamanho_filtro - 1)):

                yield imagem[i: i + self.tamanho_filtro, j: j + self.tamanho_filtro], i, j
    
    def forward(self, input):
        self.input = input
        output = np.zeros((input.shape[0] - (self.tamanho_filtro - 1), input.shape[1] - (self.tamanho_filtro - 1), self.qtd_filtros))

        for k, i, j in self.iterar_elementos(input):
            # k: imagem, i: linha, j: coluna
            # output é a imagem com o filtro aplicado, somando o valor da imagem com o filtro ao output final
            output[i, j] = np.sum(k * self.filtros, axis=(1, 2))

        return output
    
    def backward(self, dZ, learning_rate):
        dz_filtros = np.zeros(self.filtros.shape)

        for k, i, j in self.iterar_elementos(self.input):

            for filtro in range(self.qtd_filtros):
                dz_filtros[filtro] += dZ[i, j, filtro] * k

        self.filtros -= learning_rate * dz_filtros
        return self.filtros