import cv2
import numpy as np
from matplotlib import pyplot as plt

# Carregar a imagem
image_path = 'Segmenta/Outra/vaca.jpg'
image = cv2.imread(image_path)

# Converter para o espaço de cores HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Definir intervalos de HSV para verde (grama) e azul (céu)
lower_green = np.array([36, 25, 25])
upper_green = np.array([86, 255, 255])

lower_blue = np.array([85, 77, 1])
upper_blue = np.array([200, 255, 255])

# Criar máscaras para verde e azul
mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)

# Combinar máscaras
mask_combined = cv2.bitwise_or(mask_green, mask_blue)

# Inverter a máscara para manter a vaca
mask_inverted = cv2.bitwise_not(mask_combined)

# Aplicar a máscara invertida à imagem original
result = cv2.bitwise_and(image, image, mask=mask_inverted)

# Converter para escala de cinza
gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

# Aplicar limiarização para binarizar a imagem (sem fechamento prévio)
_, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

# Encontrar contornos na imagem binária
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Encontrar o contorno com a maior área (objeto mais denso)
max_contour = max(contours, key=cv2.contourArea)

# Criar uma máscara para o objeto mais denso
mask_object = np.zeros_like(binary)
cv2.drawContours(mask_object, [max_contour], -1, 255, thickness=cv2.FILLED)

# Extrair o objeto mais denso da imagem original
object_extracted = cv2.bitwise_and(result, result, mask=mask_object)

# Converter o objeto extraído para escala de cinza
object_gray = cv2.cvtColor(object_extracted, cv2.COLOR_BGR2GRAY)

# Binarizar o objeto extraído
_, object_binary = cv2.threshold(object_gray, 1, 255, cv2.THRESH_BINARY)

# Definir o kernel com formato elíptico e tamanho 7x7
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))

# Aplicar operação de fechamento na imagem binária com 2 iterações
object_binary_closed = cv2.morphologyEx(object_binary, cv2.MORPH_CLOSE, kernel, iterations=2)

# Salvar o objeto extraído binarizado
cv2.imwrite("objeto_extratido_binario.jpg", object_binary_closed)

# Converter para RGB para exibição com matplotlib
result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
object_extracted_rgb = cv2.cvtColor(object_extracted, cv2.COLOR_BGR2RGB)

# Exibir as imagens
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.title("Imagem Segmentada")
plt.imshow(result_rgb)

plt.subplot(1, 3, 2)
plt.title("Objeto Extraído")
plt.imshow(object_extracted_rgb)

plt.subplot(1, 3, 3)
plt.title("Objeto Binarizado com Fechamento")
plt.imshow(object_binary_closed, cmap='gray')

plt.show()