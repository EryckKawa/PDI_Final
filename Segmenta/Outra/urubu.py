import cv2
import numpy as np

# Carregar a imagem
imagem = cv2.imread('Segmenta/Outra/pica.jpg')

# Converter a imagem para o espaço de cores HSV
hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

# Definir os intervalos de cor para o verde no espaço HSV
verde_baixo = np.array([40, 50, 60])  # Valores mínimos de H, S, V
verde_alto = np.array([100, 255, 255])  # Valores máximos de H, S, V

# Criar uma máscara para o fundo verde
mascara_verde = cv2.inRange(hsv, verde_baixo, verde_alto)

# Inverter a máscara para obter o urubu
mascara_urubu = cv2.bitwise_not(mascara_verde)

# Aplicar a máscara invertida na imagem original para segmentar o urubu
urubu_segmentado = cv2.bitwise_and(imagem, imagem, mask=mascara_urubu)

# Definir os intervalos de cor para o tom de madeira no espaço HSV
madeira_baixo = np.array([0, 20, 0])  # Valores mínimos de H, S, V para marrom
madeira_alto = np.array([10, 10, 10])  # Valores máximos de H, S, V para marrom

# Criar uma máscara para o tom de madeira
mascara_madeira = cv2.inRange(hsv, madeira_baixo, madeira_alto)

# Inverter a máscara de madeira para remover esses tons
mascara_sem_madeira = cv2.bitwise_not(mascara_madeira)

# Aplicar a máscara sem madeira na imagem segmentada
urubu_sem_madeira = cv2.bitwise_and(urubu_segmentado, urubu_segmentado, mask=mascara_sem_madeira)

# Salvar a imagem segmentada sem o tom de madeira
cv2.imwrite('urubu_sem_madeira.jpg', urubu_sem_madeira)

# Esperar até que uma tecla seja pressionada e fechar as janelas
cv2.waitKey(0)
cv2.destroyAllWindows()