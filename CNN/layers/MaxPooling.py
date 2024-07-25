import numpy as np

class MaxPooling:
    def __init__(self, tamanho_filtro):
        self.input = None
        self.tamanho_filtro = tamanho_filtro

    # iterando elementos 2 x 2
    def iterar_elementos(self, imagem):

        for i in range(imagem.shape[0] // 2):

            for j in range(imagem.shape[1] // 2):

                yield imagem[i * 2: i * 2 + self.tamanho_filtro, j * 2: j * 2 + self.tamanho_filtro], i, j
    
    def forward(self, input):
        self.input = input

        output = np.zeros((input.shape[0] // 2, input.shape[1] // 2, input.shape[2]))

        for k, i, j in self.iterar_elementos(input):
            # de um array tridimensional (2, 2, 8) ele compara o elemento de cada coluna com cada linha e retorna o maximo
            # axis 0 = linha, axis 1 = coluna
            output[i, j] = np.amax(k, axis=(0, 1))
            # output[i, j].shape = (8,)

        return output
    
    def backward(self, dZ):

        dz_input = np.zeros(self.input.shape)

        for k, i, j in self.iterar_elementos(self.input):

            altura, larguta, qtd = k.shape
            maximo = np.amax(k, axis=(0, 1))

            for nova_altura in range(altura):
                for nova_largura in range(larguta):
                    for nova_qtd in range(qtd):
                        if k[nova_altura, nova_largura, nova_qtd] == maximo[nova_qtd]:
                            dz_input[i * 2 + nova_altura, j * 2 + nova_largura, nova_qtd] = dZ[i, j, nova_qtd]

        return dz_input