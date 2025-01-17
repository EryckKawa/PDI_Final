import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

# Carregar a imagem
imagem = cv2.imread('codigo/mascara.png')

# Converter para escala de cinza
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Aplicar binarização com o método de Otsu
_, mascara_binaria = cv2.threshold(imagem_cinza, 0, 255, cv2.THRESH_BINARY)

# Aplicar uma operação de dilatação para aumentar as áreas brancas
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # Aumente o tamanho do kernel
mascara_dilatada = cv2.dilate(mascara_binaria, kernel, iterations=1)

# Preencher buracos usando scipy.ndimage.binary_fill_holes
mascara_binaria_bool = mascara_dilatada.astype(bool)
mascara_preenchida = binary_fill_holes(mascara_binaria_bool)

# Converter de volta para o formato uint8 (0 e 255)
mascara_preenchida = mascara_preenchida.astype(np.uint8) * 255

# Exibir as máscaras
cv2.imwrite('mascara_preenchida.png', mascara_preenchida)
cv2.waitKey(0)
cv2.destroyAllWindows()