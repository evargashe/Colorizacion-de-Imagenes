import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray  # Para convertir la imagen a escala de grises
from skimage.io import imread       # Para cargar la imagen
from ultralytics import YOLO


# Ruta de la imagen de entrada
image_path = 'example/000000144984.jpg'

# Leer y convertir la imagen a escala de grises
original_image = imread(image_path)
gray_image = rgb2gray(original_image)  # Convertir a escala de grises

# Mostrar la imagen original en escala de grises
plt.figure(figsize=(6, 6))
plt.imshow(gray_image, cmap='gray')
plt.title("Imagen Original en Escala de Grises")
plt.axis('off')
plt.show()
