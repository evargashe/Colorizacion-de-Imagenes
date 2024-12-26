import numpy as np
from PIL import Image
from skimage import color
import torch

def gen_gray_color_pil(color_img_path):
    '''
    return: RGB and GRAY pillow image object
    '''
    rgb_img = Image.open(color_img_path)
    if len(np.asarray(rgb_img).shape) == 2:
        rgb_img = np.stack([np.asarray(rgb_img), np.asarray(rgb_img), np.asarray(rgb_img)], 2)
        rgb_img = Image.fromarray(rgb_img)
    gray_img = np.round(color.rgb2gray(np.asarray(rgb_img)) * 255.0).astype(np.uint8)
    gray_img = np.stack([gray_img, gray_img, gray_img], -1)
    gray_img = Image.fromarray(gray_img)
    return rgb_img, gray_img

def read_to_pil(img_path):
    '''
    return: pillow image object HxWx3
    '''
    out_img = Image.open(img_path)
    if len(np.asarray(out_img).shape) == 2:
        out_img = np.stack([np.asarray(out_img), np.asarray(out_img), np.asarray(out_img)], 2)
        out_img = Image.fromarray(out_img)
    return out_img

def gen_maskrcnn_bbox_fromPred(pred_data_path, box_num_upbound=-1):
    '''
    ## Arguments:
    - pred_data_path: Detectron2 predict results
    - box_num_upbound: object bounding boxes number. Default: -1 means use all the instances.
    '''
    pred_data = np.load(pred_data_path)
    assert 'bbox' in pred_data, "No 'bbox' key found in the prediction data."
    assert 'scores' in pred_data, "No 'scores' key found in the prediction data."

    # Convert to integer arrays
    pred_bbox = pred_data['bbox']
    
    # Validación del formato
    if pred_bbox.shape[1] >= 4:  # Asegurar que tiene al menos 4 columnas
        pred_bbox = pred_bbox[:, :4].astype(np.int32)  # Tomar solo las primeras 4 columnas
    else:
        print(f"Invalid bbox format in {pred_data_path}, skipping.")
        return []

    # Limitar el número de cajas si es necesario
    if box_num_upbound > 0 and len(pred_bbox) > box_num_upbound:
        pred_scores = pred_data['scores']
        top_indices = np.argsort(pred_scores)[-box_num_upbound:]  # Tomar las cajas con mejores puntuaciones
        pred_bbox = pred_bbox[top_indices]
    
    # Imprimir resultados para depuración
    # print(f"Total bounding boxes: {len(pred_bbox)}")
    
    return pred_bbox




def get_box_info(pred_bbox, original_shape, final_size):
    assert len(pred_bbox) == 4
    resize_startx = int(pred_bbox[0] / original_shape[0] * final_size)
    resize_starty = int(pred_bbox[1] / original_shape[1] * final_size)
    resize_endx = int(pred_bbox[2] / original_shape[0] * final_size)
    resize_endy = int(pred_bbox[3] / original_shape[1] * final_size)
    rh = resize_endx - resize_startx
    rw = resize_endy - resize_starty
    if rh < 1:
        if final_size - resize_endx > 1:
            resize_endx += 1
        else:
            resize_startx -= 1
        rh = 1
    if rw < 1:
        if final_size - resize_endy > 1:
            resize_endy += 1
        else:
            resize_starty -= 1
        rw = 1
    L_pad = resize_startx
    R_pad = final_size - resize_endx
    T_pad = resize_starty
    B_pad = final_size - resize_endy
    return [L_pad, R_pad, T_pad, B_pad, rh, rw]