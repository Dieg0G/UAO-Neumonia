"""
read_img.py
-----------
Módulo para la lectura de imágenes en formato DICOM.
Entrega la imagen en dos formatos:
1. PIL.Image -> para visualización en la interfaz gráfica.
2. np.ndarray -> para el preprocesamiento posterior.
"""

import pydicom
import numpy as np
from PIL import Image


def read_dicom(path: str):
    dicom_img = pydicom.dcmread(path)              # Leer archivo DICOM
    img_array = dicom_img.pixel_array              # Convertir a arreglo NumPy
    img_show = Image.fromarray(img_array)          # Imagen lista para interfaz

    return img_array, img_show


if __name__ == "__main__":
    # Prueba de lectura rápida
    try:
        array, img = read_dicom("data/raw/viral.dcm")
        print(f"[OK] Imagen cargada con forma: {array.shape}")
        img.show()  # Muestra la imagen en una ventana
    except Exception as e:
        print(f"Error cargando DICOM: {e}")