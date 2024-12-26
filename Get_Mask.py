import numpy as np
import matplotlib.pyplot as plt

# Cargar el archivo .npz
archivo = "./example_bbox/000000145020.npz"
data = np.load(archivo)

# Mostrar las claves disponibles
print("Claves disponibles:", data.files)

# Acceder a las máscaras bajo la clave correcta
mascaras = data['masks']  # Cambia 'masks' si se usa otra clave para las máscaras

# Mostrar el número total de máscaras
num_mascaras = len(mascaras)
print(f"Número total de máscaras: {num_mascaras}")

# Si hay múltiples máscaras, iterar y mostrarlas
for i, mascara in enumerate(mascaras):
    plt.figure()
    plt.imshow(mascara, cmap='gray')
    plt.title(f"Máscara {i + 1}")
    plt.axis('off')
    plt.show()