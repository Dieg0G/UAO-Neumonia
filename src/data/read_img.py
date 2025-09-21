#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
read_img.py
-----------
Soporta lectura de imágenes en formato:
- DICOM (.dcm)
- Imágenes estándar (.jpg, .jpeg, .png)

Entrega:
1. np.ndarray -> para preprocesamiento posterior.
2. PIL.Image -> para visualización en la interfaz gráfica.
"""

import os
import numpy as np
import cv2
import pydicom
from PIL import Image


def read_img(path: str):
    """
    Lee imagen desde path en formato DICOM o estándar.
    Retorna:
        img_array (np.ndarray): arreglo procesable
        img_show (PIL.Image): imagen lista para interfaz
    """
    ext = os.path.splitext(path)[-1].lower()

    if ext == ".dcm":
        # ---- DICOM ----
        dicom_img = pydicom.dcmread(path)
        img_array = dicom_img.pixel_array
        img_show = Image.fromarray(img_array)

    elif ext in [".jpg", ".jpeg", ".png"]:
        # ---- JPEG/PNG ----
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"No se pudo leer la imagen: {path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # asegurar RGB
        img_array = img_rgb
        img_show = Image.fromarray(img_rgb)

    else:
        raise ValueError(f"Formato no soportado: {ext}")

    return img_array, img_show


def main(path="data/raw/bateria.jpeg"):
    """Función principal para ser usada por integrator"""
    try:
        array, img = read_img(path)
        print(f"[OK] Imagen cargada con forma: {array.shape}")
        img.show()  # Solo si quieres mostrar la imagen aquí
        return array
    except Exception as e:
        print(f"[ERROR] {e}")
        return None


if __name__ == "__main__":
    main()

