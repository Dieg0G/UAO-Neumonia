# tests/test_read_img_real_default.py
import pytest
from src.data.read_img import read_img
from PIL import Image
import numpy as np

def test_read_default_image():
    """Prueba que read_img lea correctamente la imagen usada en main()"""
    test_path = "data/raw/bateria.jpeg"  # ruta real que usa tu script
    
    img_array, img_show = read_img(test_path)
    
    # Verificaciones básicas
    assert isinstance(img_array, np.ndarray), "img_array no es un np.ndarray"
    assert isinstance(img_show, Image.Image), "img_show no es un PIL.Image"
    assert img_array.shape[0] > 0 and img_array.shape[1] > 0, "La imagen tiene dimensiones inválidas"
