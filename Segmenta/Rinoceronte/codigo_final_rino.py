import cv2
import numpy as np

# Carregar a imagem
image = cv2.imread('Segmenta/Rinoceronte/rinoceronte.jpg')

# Converter a imagem para o espaço de cores HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Definir o intervalo de cores para o céu (tons de azul)
lower_blue = np.array([1, 30, 30])  # Valores mínimos de H, S, V
upper_blue = np.array([40, 255, 255])  # Valores máximos de H, S, V

# Definir o intervalo de cores para a grama (tons de verde)
lower_green = np.array([20, 0, 0])  # Valores mínimos de H, S, V
upper_green = np.array([105, 255, 255])  # Valores máximos de H, S, V

# Criar uma máscara para o céu
mask_sky = cv2.inRange(hsv_image, lower_blue, upper_blue)

# Criar uma máscara para a grama
mask_grass = cv2.inRange(hsv_image, lower_green, upper_green)

# Combinar as máscaras (céu + grama)
mask_combined = cv2.bitwise_or(mask_sky, mask_grass)

# Inverter a máscara combinada para manter tudo, exceto o céu e a grama
mask_inv = cv2.bitwise_not(mask_combined)

# Aplicar a máscara na imagem original
result = cv2.bitwise_and(image, image, mask=mask_inv)

# Converter a imagem resultante para nível de cinza
gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

# Aplicar um limiar (threshold) para binarizar a imagem
_, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

# Encontrar contornos na imagem binarizada
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Encontrar o contorno com a maior área (objeto com maior densidade de pixels)
largest_contour = max(contours, key=cv2.contourArea)

# Criar uma máscara para o maior contorno
mask_largest = np.zeros_like(gray)
cv2.drawContours(mask_largest, [largest_contour], -1, 255, thickness=cv2.FILLED)

# Aplicar a máscara do maior contorno na imagem original
final_result = cv2.bitwise_and(image, result, mask=mask_largest)

# Binarizar o final_result
# Converter para nível de cinza
final_gray = cv2.cvtColor(final_result, cv2.COLOR_BGR2GRAY)

# Aplicar um limiar (threshold) para binarizar a imagem
_, final_binary = cv2.threshold(final_gray, 1, 255, cv2.THRESH_BINARY)

# Escolher o elemento estruturante
kernel_type = cv2.MORPH_CROSS  # Pode ser MORPH_RECT, MORPH_ELLIPSE ou MORPH_CROSS
kernel_size = (7, 7)  # Tamanho do kernel
kernel = cv2.getStructuringElement(kernel_type, kernel_size)

# Aplicar fechamento na imagem binarizada com múltiplas iterações
iterations = 2  # Número de iterações
closed = cv2.morphologyEx(final_binary, cv2.MORPH_CLOSE, kernel, iterations=iterations)

# Salvar ou mostrar os resultados
cv2.imwrite('imagem_sem_ceu_e_grama.jpg', result)
cv2.imwrite('imagem_objeto_maior_densidade.jpg', final_result)
cv2.imwrite('imagem_objeto_binarizada.jpg', final_binary)
cv2.imwrite('imagem_objeto_fechamento.jpg', closed)

