#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
preprocess_img.py
-----------------
Módulo para preprocesar imágenes de rayos X.
Recibe el array proveniente de read_img.py y devuelve
un tensor listo para el modelo (1, 512, 512, 1).
"""

import cv2
import numpy as np


def preprocess(image: np.ndarray, target_size: tuple = (512, 512)) -> np.ndarray:
    # Resize
    resized = cv2.resize(image, target_size)

    # Escala de grises (si no lo está ya)
    if len(resized.shape) == 3 and resized.shape[-1] == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized

    # CLAHE (contraste adaptativo)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)

    # Normalización [0,1]
    normalized = enhanced / 255.0

    # Expandir dimensiones -> (1, H, W, 1)
    preprocessed = np.expand_dims(normalized, axis=-1)
    preprocessed = np.expand_dims(preprocessed, axis=0)

    return preprocessed


def main(array: np.ndarray):
    """Función principal para ser usada por integrator"""
    try:
        processed = preprocess(array)
        print(f"[OK] Imagen preprocesada con shape: {processed.shape}")
        return processed
    except Exception as e:
        print(f"Error en preprocesamiento: {e}")
        return None


if __name__ == "__main__":
    print("[INFO] Este módulo no funciona solo. Debes pasarle un array desde read_img.")
