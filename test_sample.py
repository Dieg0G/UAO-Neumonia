import pytest
import numpy as np
import os
import cv2
import tensorflow as tf
import tempfile
from PIL import Image
from unittest.mock import Mock, patch

from src.data.preprocess_img import preprocess
from src.data.load_model import model_fun
from src.data.grad_cam import grad_cam
from src.data.predict import predict


# =============================
# TEST 1: Preprocesamiento de imagen
# =============================
def test_preprocess_returns_correct_shape():
    """
    Verifica que preprocess() devuelve una imagen en formato batch (1, 512, 512, 1)
    """
    # Crear imagen dummy RGB
    dummy_img = np.random.randint(0, 255, (600, 600, 3), dtype=np.uint8)

    # Procesar
    processed = preprocess(dummy_img)

    # Validaciones
    assert isinstance(processed, np.ndarray), "Debe devolver un array NumPy"
    assert processed.shape == (1, 512, 512, 1), f"Shape esperado (1,512,512,1), recibido {processed.shape}"
    assert processed.dtype == np.float32, "Debe estar en float32"
    assert np.min(processed) >= 0 and np.max(processed) <= 1, "Valores deben estar normalizados entre 0 y 1"

    print("✅ preprocess(): Formato correcto y normalización válida")

    # =============================
# TEST 2: Carga del modelo
# =============================
def test_model_fun_loads_model():
    """
    Verifica que model_fun() carga correctamente el archivo .h5
    """
    # Asegurar que el modelo existe
    model_path = "data/models/conv_MLP_84.h5"
    if not os.path.exists(model_path):
        pytest.skip(f"Modelo no encontrado: {model_path}. Descárgalo desde: https://drive.google.com/file/d/1wQmz9qYkKjVdZt6J7eXpNcF8yRbLxTqP/view?usp=sharing")

    model = model_fun()

    # Validaciones
    assert model is not None, "El modelo no debe ser None"
    assert hasattr(model, 'predict'), "El modelo debe tener método predict"
    assert model.input_shape == (None, 512, 512, 1), f"Input shape incorrecto: {model.input_shape}"

    print("✅ model_fun(): Modelo cargado correctamente")

def test_preprocess_returns_correct_shape():
	"""
	Verifica que preprocess() devuelve una imagen en formato batch (1, 512, 512, 1)
	"""
	# Crear imagen dummy RGB
	dummy_img = np.random.randint(0, 255, (600, 600, 3), dtype=np.uint8)

	# Procesar
	processed = preprocess(dummy_img)

	# Validaciones
	assert isinstance(processed, (np.ndarray, tf.Tensor)), "Debe devolver un array NumPy o TensorFlow"
	assert processed.shape == (1, 512, 512, 1), f"Shape incorrecto: {processed.shape}"
     
     # =============================
# TEST 3: Grad-CAM sin errores
# =============================
def test_grad_cam_returns_valid_heatmap(): 
    """
    Verifica que grad_cam() genera un heatmap válido sin errores
    """
    # Saltar si el modelo no está disponible
    model_path = "data/models/conv_MLP_84.h5"
    if not os.path.exists(model_path):
        pytest.skip("Modelo no encontrado. No se puede probar grad_cam()")

    # Crear imagen dummy RGB
    dummy_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

    # Generar heatmap
    heatmap = grad_cam(dummy_img)

    # Validaciones
    assert heatmap.shape == (512, 512, 3), f"Heatmap debe ser (512,512,3), obtenido {heatmap.shape}"
    assert heatmap.dtype == np.uint8, "Heatmap debe ser uint8"
    assert np.min(heatmap) >= 0 and np.max(heatmap) <= 255, "Valores deben estar en [0,255]"

    print("✅ grad_cam(): Heatmap generado correctamente")

    # =============================
# TEST 4: Función predict completa (integración)
# =============================
def test_predict_returns_correct_format():
    """
    Prueba de integración: predict() debe retornar (label, proba, heatmap)
    """
    # Saltar si el modelo no está disponible
    model_path = "data/models/conv_MLP_84.h5"
    if not os.path.exists(model_path):
        pytest.skip("Modelo no encontrado. No se puede probar predict()")

    # Crear imagen dummy RGB
    dummy_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

    # Ejecutar predict
    label, proba, heatmap = predict(dummy_img)

    # Validaciones
    assert isinstance(label, str), "Label debe ser string"
    assert label in ["bacteriana", "normal", "viral"], f"Label inválido: {label}"

    assert isinstance(proba, float), "Probabilidad debe ser float"
    assert 0 <= proba <= 100, f"Probabilidad fuera de rango [0,100]: {proba}"

    assert isinstance(heatmap, np.ndarray), "Heatmap debe ser numpy array"
    assert heatmap.shape == (512, 512, 3), f"Heatmap shape incorrecto: {heatmap.shape}"

    print("✅ predict(): Retorna formato correcto (label, proba, heatmap)")
