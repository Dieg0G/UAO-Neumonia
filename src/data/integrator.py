"""
integrator.py
-------------
Módulo integrador del flujo:
1. Lectura de imagen DICOM.
2. Preprocesamiento.
3. Carga del modelo.
4. Predicción.
5. Grad-CAM (mapa de calor).
"""

import numpy as np
from src.data.read_img import read_dicom
from src.data.preprocess_img import preprocess
from src.data.load_model import model_fun
from src.data.grad_cam import grad_cam


def predict_pipeline(path: str, layer_name: str = "conv10_thisone"):
    """
    Ejecuta el flujo completo sobre una imagen.

    Args:
        path (str): Ruta al archivo DICOM.
        layer_name (str): Nombre de la capa conv para Grad-CAM.

    Returns:
        tuple: (label, probabilidad, heatmap)
    """
    # 1. Lectura
    img_array, img_pil = read_dicom(path)

    # 2. Preprocesamiento
    preprocessed = preprocess(img_array)

    # 3. Modelo
    model = model_fun()

    # 4. Predicción
    preds = model.predict(preprocessed)
    class_idx = np.argmax(preds[0])
    proba = float(np.max(preds[0]) * 100)

    labels = ["bacteriana", "normal", "viral"]
    label = labels[class_idx]

    # 5. Grad-CAM
    heatmap = grad_cam(model, preprocessed, layer_name)

    return label, proba, heatmap, img_pil


if __name__ == "__main__":
    try:
        resultado, prob, heatmap, img = predict_pipeline("data/raw/sample.dcm")
        print(f"[OK] Predicción: {resultado} ({prob:.2f}%)")
    except Exception as e:
        print(f"[ERROR] {e}")

