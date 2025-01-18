import cv2
import numpy as np

# Carregar a imagem
imagem = cv2.imread('Segmenta/Rinoceronte/rinoceronte.png')

# Converter para níveis de cinza
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Aplicar limiarização (método de Otsu)
_, mascara = cv2.threshold(imagem_cinza, 0, 255, cv2.THRESH_BINARY)

# Encontrar contornos na máscara binária
contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Encontrar o maior contorno (maior objeto)
maior_contorno = max(contornos, key=cv2.contourArea)

# Criar uma máscara para o maior objeto
mascara_maior_objeto = np.zeros_like(mascara)
cv2.drawContours(mascara_maior_objeto, [maior_contorno], -1, 255, thickness=cv2.FILLED)

# Extrair o maior objeto da imagem original
maior_objeto = cv2.bitwise_and(imagem, imagem, mask=mascara_maior_objeto)

# Criar uma imagem de fundo preto
fundo_preto = np.zeros_like(imagem)

# Copiar o maior objeto para o fundo preto
fundo_preto[mascara_maior_objeto == 255] = imagem[mascara_maior_objeto == 255]

# Exibir os resultados
cv2.imshow("Máscara Binária", mascara)
cv2.imshow("Máscara do Maior Objeto", mascara_maior_objeto)
cv2.imshow("Maior Objeto sobre Fundo Preto", fundo_preto)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Salvar o resultado
cv2.imwrite('maior_objeto_fundo_preto.jpg', fundo_preto)