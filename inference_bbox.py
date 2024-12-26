from os.path import join, isfile, isdir
from os import listdir
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from argparse import ArgumentParser

import numpy as np
import cv2
from tqdm import tqdm

# Importar la librería de Ultralytics YOLOv8
from ultralytics import YOLO

parser = ArgumentParser()
parser.add_argument("--test_img_dir", type=str, default='example', help='testing images folder')
parser.add_argument('--filter_no_obj', action='store_true')
args = parser.parse_args()

# Configuración de directorios
input_dir = args.test_img_dir
image_list = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]
output_npz_dir = "{0}_bbox".format(input_dir)
if os.path.isdir(output_npz_dir) is False:
    print('Create path: {0}'.format(output_npz_dir))
    os.makedirs(output_npz_dir)

# Cargar el modelo YOLOv8 preentrenado con segmentación
model = YOLO('yolov8l-seg.pt') # Puedes cambiar por otros pesos como 'yolov8m-seg.pt' o 'yolov8l-seg.pt'

for image_path in tqdm(image_list):
    print(f"Processing image: {image_path}")

    # Leer la imagen
    img_path = join(input_dir, image_path)
    img = cv2.imread(img_path)

    # Convertir la imagen a espacio LAB para emular el preprocesamiento anterior
    lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    l_stack = np.stack([l_channel, l_channel, l_channel], axis=2)

    # Realizar inferencia con YOLOv8
    results = model.predict(img, conf=0.5, iou=0.5)


    # Inicializar listas para las predicciones
    pred_bboxes = []
    pred_scores = []
    pred_masks = []  # Para almacenar las máscaras

    for result in results:
        boxes = result.boxes.data.cpu().numpy()  # Coordenadas de los bounding boxes
        scores = result.boxes.conf.cpu().numpy()  # Confianza asociada a cada caja
        masks = result.masks.data.cpu().numpy() if result.masks is not None else []  # Máscaras

        if boxes is not None:
            for i in range(len(boxes)):
                pred_bboxes.append(boxes[i])
                pred_scores.append(scores[i])
                if len(masks) > 0:
                    pred_masks.append(masks[i])  # Guardar las máscaras

    # Validar si no se detectaron objetos y se requiere eliminar imágenes
    if args.filter_no_obj and len(pred_bboxes) == 0:
        print(f'delete {image_path}')
        os.remove(img_path)
        continue

    # Guardar los resultados en un archivo npz
    save_path = join(output_npz_dir, image_path.split('.')[0])
    np.savez(save_path, bbox=np.array(pred_bboxes), scores=np.array(pred_scores), masks=np.array(pred_masks))

