import cv2
import numpy as np

# Carregar a imagem
imagem = cv2.imread('codigo/mascara.png')

# Converter para escala de cinza
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)


# Aplicar binarização com o método de Otsu
_, mascara_binaria = cv2.threshold(imagem_cinza, 0, 255, cv2.THRESH_BINARY)

# Remover pontos pretos com operações morfológicas (fechamento)
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
#mascara_limpa = cv2.morphologyEx(mascara_binaria, cv2.MORPH_CLOSE, kernel)

# Exibir a máscara final
cv2.imshow("Máscara Binarizada", mascara_binaria)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Salvar a máscara
cv2.imwrite('mascara_binaria_limpa.png', mascara_binaria)
