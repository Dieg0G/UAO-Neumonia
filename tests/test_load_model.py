# tests/test_load_model_real.py
import os
import pytest
import tensorflow as tf
from src.data.load_model import model_fun

def test_model_fun_loads_real_model():
    """
    Test real: verifica que model_fun carga el modelo desde la ruta definida
    y devuelve un tf.keras.Model sin lanzar errores.
    """
    model_path = r"E:\MODULO01IA\UAO-Neumonia\src\models\conv_MLP_84.h5"

    # Verificar que el archivo existe
    assert os.path.exists(model_path), f"Archivo de modelo no encontrado: {model_path}"

    # Intentar cargar el modelo
    model = model_fun()
    
    # Verificar que sea un tf.keras.Model
    assert isinstance(model, tf.keras.Model), "model_fun no devolvió un objeto tf.keras.Model"
