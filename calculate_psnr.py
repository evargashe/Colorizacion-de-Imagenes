import os
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.transform import resize
import numpy as np


# Directorios de las imágenes
ground_truth_dir = 'example'
output_dir = 'results'

# Listar las imágenes en el directorio de ground truth (asumiendo que ambas carpetas tienen las mismas imágenes)
ground_truth_files = os.listdir(ground_truth_dir)
output_files = os.listdir(output_dir)

# Asegurarse de que las imágenes en ambas carpetas coincidan
ground_truth_files = sorted(ground_truth_files)
output_files = sorted(output_files)

# Inicializar variables para el cálculo del PSNR promedio
psnr_values = []

# Iterar sobre las imágenes y calcular el PSNR para cada par de imágenes
for gt_file, output_file in zip(ground_truth_files, output_files):
    # Leer las imágenes
    ground_truth_image = imread(os.path.join(ground_truth_dir, gt_file))
    output_image = imread(os.path.join(output_dir, output_file))
    
    # Verificar que las imágenes tengan las mismas dimensiones
    if ground_truth_image.shape[:2] != output_image.shape[:2]:
        output_image = resize(output_image, ground_truth_image.shape[:2], 
                              mode='reflect', anti_aliasing=True)
        output_image = (output_image * 255).astype('uint8')  # Convertir a uint8

    # Calcular PSNR para esta imagen
    psnr_value = psnr(ground_truth_image, output_image, data_range=255)
    psnr_values.append(psnr_value)

    # Mostrar PSNR por imagen
    print(f"PSNR para {gt_file}: {psnr_value:.2f} dB")

# Calcular el PSNR promedio
average_psnr = np.mean(psnr_values)
num_images = len(psnr_values)  # Contar cuántas imágenes se procesaron

# Mostrar el PSNR promedio y cuántas imágenes se utilizaron
print(f"\nPSNR Promedio: {average_psnr:.2f} dB")
print(f"Cantidad de imágenes utilizadas: {num_images}")