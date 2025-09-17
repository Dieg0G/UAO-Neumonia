#!/usr/bin/env python
# -*- coding: utf-8 -*-

    """
    Preprocesa una imagen para que pueda ser usada por el modelo de predicción.

    Pasos:
        1. Resize a target_size.
        2. Conversión a escala de grises.
        3. Aplicación de CLAHE (ecualización adaptativa).
        4. Normalización (0 a 1).
        5. Expansión de dimensiones -> batch tensor (1, H, W, 1).

    Args:
        image (np.ndarray): Imagen de entrada (array NumPy).
        target_size (tuple): Tamaño de salida (ancho, alto). Default: (512, 512).

    Returns:
        np.ndarray: Imagen preprocesada lista para el modelo con shape (1, H, W, 1).
    """

import cv2
import numpy as np


def preprocess(image: np.ndarray, target_size: tuple = (512, 512)) -> np.ndarray:
    # Resize
    resized = cv2.resize(image, target_size)

    # Escala de grises
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # CLAHE (ecualización adaptativa)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)

    # Normalización [0,1]
    normalized = enhanced / 255.0

    # Expandir dimensiones -> (1, H, W, 1)
    preprocessed = np.expand_dims(normalized, axis=-1)
    preprocessed = np.expand_dims(preprocessed, axis=0)

    return preprocessed


if __name__ == "__main__":
    # Test rápido (con imagen dummy)
    dummy = np.zeros((600, 600, 3), dtype=np.uint8)
    processed = preprocess(dummy)
    print("Shape procesada:", processed.shape)