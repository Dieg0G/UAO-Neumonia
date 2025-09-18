#!/usr/bin/env python
# -*- coding: utf-8 -*-

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