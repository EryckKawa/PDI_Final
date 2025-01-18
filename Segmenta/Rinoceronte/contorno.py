import cv2
import numpy as np
from scipy.ndimage import label

# Carregar a imagem
imagem = cv2.imread('Segmenta/Rinoceronte/rinoceronte.png')

# Converter para níveis de cinza
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Aplicar limiarização (método de Otsu)
_, mascara = cv2.threshold(imagem_cinza, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Definir a distância mínima para agrupamento (em pixels)
distancia_minima = 10  # Ajuste conforme necessário

# Criar uma matriz de distância entre pixels
from scipy.ndimage import distance_transform_edt
distancia = distance_transform_edt(mascara == 0)  # Distância até o fundo (preto)

# Criar uma máscara para pixels próximos
mascara_proximos = (distancia <= distancia_minima).astype(np.uint8) * 255

# Encontrar componentes conectados na máscara de pixels próximos
num_labels, labels = cv2.connectedComponents(mascara_proximos)

# Encontrar o maior grupo de pixels conectados
maior_grupo = None
maior_area = 0

for label_num in range(1, num_labels):  # Ignorar o fundo (label 0)
    area = np.sum(labels == label_num)  # Calcular a área do grupo
    if area > maior_area:
        maior_area = area
        maior_grupo = label_num

# Criar uma máscara para o maior grupo de pixels conectados
mascara_maior_grupo = (labels == maior_grupo).astype(np.uint8) * 255

# Extrair o maior grupo de pixels da imagem original
objeto_maior_grupo = cv2.bitwise_and(imagem, imagem, mask=mascara_maior_grupo)

# Salvar o objeto do maior grupo
cv2.imwrite('objeto_maior_grupo.jpg', objeto_maior_grupo)