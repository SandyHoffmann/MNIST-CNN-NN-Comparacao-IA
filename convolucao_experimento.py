import skimage as skimage
import numpy as np
from matplotlib import pyplot as plt

img = skimage.data.chelsea()
img = skimage.color.rgb2gray(img)
# Preparando filtros para imagem

# 2 filtros 3x3
l1_filter = np.zeros((2, 3, 3))

# primeiro filtro, valores para detectar bordas verticais
l1_filter[0, :, :] = np.array([[[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]])
# segundo filtro, valores para detectar bordas horizontais
l1_filter[1, :, :] = np.array([[[-1, -1, -1], [0, 0, 0], [1, 1, 1]]])


def convolucao(img, filter):
    # considerando que o filtro e a imagem tem o mesmo tamanho

    # img = 300 x 451

    # 298 filtros de 449 x 2
    feature_map = np.zeros((img.shape[0] - filter.shape[1] + 1, img.shape[1] - filter.shape[1] + 1, filter.shape[0]))
    # print(feature_map.shape)

    for f_filtro in range(filter.shape[0]):
        filtro = filter[f_filtro, :]
        if len(filtro.shape) > 2:
            convolucoes = conv_2d_(img[:, :], filtro[:, :])
            for i in range(convolucoes.shape[-1]):
                print(filtro)
                convolucoes = convolucoes + conv_2d_(img[:, :], filtro[:, i])
        else:
            convolucoes = conv_2d_(img[:, :], filtro[:, :])
        
        feature_map[:, :, f_filtro] = convolucoes
    print(feature_map.shape)
    return feature_map 

def conv_2d_(image, filter):
    tamanho_filtro = filter.shape[1]
    # mantem tamanho da imagem
    resultado = np.zeros((image.shape))

    linha_inicial = int(tamanho_filtro / 2)
    linha_final = img.shape[0] - int(tamanho_filtro / 2)

    coluna_inicial = int(tamanho_filtro / 2)
    coluna_final = img.shape[1] - int(tamanho_filtro / 2)

    print(f'linha_inicial: {linha_inicial}, linha_final: {linha_final}, coluna_inicial: {coluna_inicial}, coluna_final: {coluna_final}')

    primeiro_valor = False
    for r in range(linha_inicial, linha_final):
        for c in range(coluna_inicial, coluna_final):
            # Pegando parte da imagem para o aplicar o filtro (3x3)
            regiao_atual = img[r - int(tamanho_filtro / 2): r + int(tamanho_filtro / 2) + 1,
                              c - int(tamanho_filtro / 2): c + int(tamanho_filtro / 2) + 1]
            # Aplicando o filtro
            resultado_atual = regiao_atual * filter

            # Soma dos valores
            conv_sum = np.sum(resultado_atual)  
            if not primeiro_valor:
                print(conv_sum)
                primeiro_valor = True
            # Saida da convolução
            resultado[r, c] = conv_sum

    # resultado final
    resultado_final = resultado[int(tamanho_filtro / 2): resultado.shape[0] - int(tamanho_filtro / 2),
                          int(tamanho_filtro / 2): resultado.shape[1] - int(tamanho_filtro / 2)]
   
    print(f'Resultado final: {resultado_final.shape}')
    print("------------------------------------")

    return resultado_final

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')

nova_imagem = convolucao(img, l1_filter)
plt.subplot(1, 3, 2)

plt.imshow(nova_imagem[:, :, 0], cmap='gray')
plt.subplot(1, 3, 3)

plt.imshow(nova_imagem[:, :, 1], cmap='gray')
plt.show()

print(img.shape)