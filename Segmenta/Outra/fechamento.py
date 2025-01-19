import cv2
import numpy as np

# Load the image
image = cv2.imread('Segmenta/Outra/flamingo.jfif')

# Convert to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define blue color range
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([140, 255, 255])

# Create a mask for blue tones
mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)

# Invert the mask to exclude blue pixels
mask_not_blue = cv2.bitwise_not(mask_blue)

# Apply the mask to the original image to exclude blue pixels
filtered_image = cv2.bitwise_and(image, image, mask=mask_not_blue)

# Convert to grayscale
gray = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)

# Apply thresholding
_, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour
largest_contour = max(contours, key=cv2.contourArea)

# Create a mask for the largest contour
mask_largest = np.zeros_like(gray)
cv2.drawContours(mask_largest, [largest_contour], -1, 255, thickness=cv2.FILLED)

# Apply the largest contour mask to the original image
final_result = np.zeros_like(image)
final_result[mask_largest != 0] = image[mask_largest != 0]

# Optional: Apply morphological operations to clean up
kernel = np.ones((5,5), np.uint8)
mask_largest_clean = cv2.morphologyEx(mask_largest, cv2.MORPH_CLOSE, kernel)
final_result_clean = cv2.bitwise_and(image, image, mask=mask_largest_clean)

# Save or display the results
cv2.imwrite('segmented_image.jpg', final_result_clean)
cv2.imshow('Segmented Image', final_result_clean)
cv2.waitKey(0)
cv2.destroyAllWindows()