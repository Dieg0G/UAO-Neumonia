#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
integrator.py
-------------
Pipeline completo:
1. Leer imagen
2. Preprocesar
3. Cargar modelo
4. Generar Grad-CAM anotado
5. Retornar resultado (para GUI o pruebas)
"""

import os
import cv2
import matplotlib.pyplot as plt

from src.data.read_img import read_img
from src.data.preprocess_img import preprocess
from src.data.load_model import model_fun
from src.data.grad_cam import main as gradcam_main   # 🔥 usar main(), no grad_cam()


def run_pipeline(path: str, save: bool = True):
    """
    Ejecuta el pipeline completo y devuelve el resultado.

    Args:
        path (str): ruta de la imagen a procesar
        save (bool): si True, guarda el resultado en disco

    Returns:
        (np.ndarray, str): imagen RGB anotada + etiqueta de clase
    """
    # 1. Leer imagen
    array, _ = read_img(path)
    print(f"[OK] Imagen cargada con shape: {array.shape}")

    # 2. Preprocesar
    processed = preprocess(array)
    print(f"[OK] Imagen preprocesada con shape: {processed.shape}")

    # 3. Cargar modelo
    model = model_fun()
    print("[OK] Modelo cargado")

    # 4. Grad-CAM anotado
    gradcam_img, pred_class = gradcam_main(processed, array, model)
    print(f"[OK] Grad-CAM generado - Clase predicha: {pred_class}")

    # 5. Guardar si corresponde
    if save:
        os.makedirs("data/processed", exist_ok=True)
        out_path = "data/processed/gradcam_result.jpg"
        cv2.imwrite(out_path, gradcam_img[:, :, ::-1])  # convertir a BGR
        print(f"[OK] Resultado guardado en: {out_path}")

    return gradcam_img, pred_class


if __name__ == "__main__":
    # Test rápido
    test_path = "data/raw/sample.dcm"
    result_img, result_class = run_pipeline(test_path)
    plt.imshow(result_img)
    plt.title(f"Predicción: {result_class}")
    plt.axis("off")
    plt.show()
