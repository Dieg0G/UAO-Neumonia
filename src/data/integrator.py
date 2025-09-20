#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script integrador para ejecutar todo el flujo:
1. Leer imagen DICOM
2. Preprocesar imagen
3. Cargar modelo
4. Generar Grad-CAM
5. Mostrar y guardar resultado
"""

import os
import matplotlib.pyplot as plt
import cv2

from src.data.read_img import read_dicom
from src.data.preprocess_img import preprocess
from src.data.load_model import model_fun
from src.data.grad_cam import grad_cam


def main():
    # 1. Ruta de la imagen de prueba
    dicom_path = "data/raw/sample.dcm"

    # 2. Leer imagen DICOM
    array, _ = read_dicom(dicom_path)
    print(f"[OK] Imagen cargada con shape: {array.shape}")

    # 3. Preprocesar imagen
    processed = preprocess(array)
    print(f"[OK] Imagen preprocesada con shape: {processed.shape}")

    # 4. Cargar modelo
    model = model_fun()
    print("[OK] Modelo cargado")

    # 5. Generar Grad-CAM
    gradcam_img = grad_cam(processed, array, model)
    print("[OK] Grad-CAM generado")

    # 6. Mostrar resultado
    plt.imshow(gradcam_img)
    plt.axis("off")
    plt.title("Resultado Grad-CAM")
    plt.show()

    # 7. Guardar en disco
    os.makedirs("data/processed", exist_ok=True)
    out_path = "data/processed/gradcam_result.jpg"
    cv2.imwrite(out_path, gradcam_img[:, :, ::-1])  # convertir a BGR para OpenCV
    print(f"[OK] Resultado guardado en: {out_path}")


if __name__ == "__main__":
    main()
