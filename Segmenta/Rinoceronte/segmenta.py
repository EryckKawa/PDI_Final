import cv2
import numpy as np

# Função para converter RGB para HSI
def rgb_to_hsi(imagem):
    imagem = imagem.astype(np.float32) / 255.0
    r, g, b = cv2.split(imagem)
    
    # Calcular o canal Hue (Matiz)
    numerador = 0.5 * ((r - g) + (r - b))
    denominador = np.sqrt((r - g)**2 + (r - b) * (g - b))
    theta = np.arccos(np.clip(numerador / (denominador + 1e-6), -1.0, 1.0))
    h = np.where(b <= g, theta, 2 * np.pi - theta)
    
    # Calcular o canal Saturation (Saturação)
    s = 1 - 3 * np.minimum(np.minimum(r, g), b) / (r + g + b + 1e-6)
    
    # Calcular o canal Intensity (Intensidade)
    i = (r + g + b) / 3.0
    
    # Combinar os canais
    hsi = cv2.merge((h, s, i))
    return hsi

# Carregar a imagem
imagem = cv2.imread('Segmenta/Rinoceronte/rinoceronte.jpg')

# Converter para HSI
hsi = rgb_to_hsi(imagem)
h, s, i = cv2.split(hsi)

# Normalizar o canal Hue para o intervalo [0, 255]
h_normalizado = cv2.normalize(h, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Aplicar limiarização no canal Hue
limiar_inferior = 20  # Ajuste conforme necessário
limiar_superior = 100  # Ajuste conforme necessário
mascara = cv2.inRange(h_normalizado, limiar_inferior, limiar_superior)

# Refinar a máscara com operações morfológicas
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)  # Fechamento para preencher buracos
mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)   # Abertura para remover ruídos

# Aplicar a máscara na imagem original
rinoceronte_segmentado = cv2.bitwise_and(imagem, imagem, mask=mascara)

# Exibir os resultados
cv2.imshow("Canal Hue Normalizado", h_normalizado)
cv2.imshow("Máscara", mascara)
cv2.imshow("Rinoceronte Segmentado", rinoceronte_segmentado)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Salvar o resultado
cv2.imwrite('rinoceronte_segmentado.jpg', rinoceronte_segmentado)