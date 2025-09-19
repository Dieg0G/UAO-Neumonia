import numpy as np
from src.data.preprocess_img import preprocess
from src.data.load_model import model_fun
from src.data.grad_cam import grad_cam

def predict(array):
    """
    Ejecuta el flujo completo sobre una imagen RGB/Grayscale.
    Args:
        array (np.ndarray): Imagen de entrada (RGB o escala de grises)
    Returns:
        tuple: (label, proba, heatmap)
    """
    # 1. Preprocesar imagen
    batch_img = preprocess(array)
    # 2. Cargar modelo y predecir
    model = model_fun()
    preds = model.predict(batch_img)
    class_idx = int(np.argmax(preds[0]))
    proba = float(np.max(preds[0]) * 100)
    labels = ["bacteriana", "normal", "viral"]
    label = labels[class_idx]
    # 3. Grad-CAM
    heatmap = grad_cam(array)
    return label, proba, heatmap