# tests/test_grad_cam.py
import pytest
import numpy as np
import cv2
import os
from src.data.grad_cam import main
from src.data.load_model import model_fun
from src.data.read_img import read_img

@pytest.mark.parametrize("img_path", ["data/raw/bateria.jpeg"])
def test_gradcam_image_generated(tmp_path, img_path):
    # 1) Leer imagen real
    img_array, _ = read_img(img_path)

    # 2) Redimensionar a (512,512) y convertir a escala de grises
    img_resized = cv2.resize(img_array, (512, 512))
    if img_resized.ndim == 3 and img_resized.shape[2] == 3:
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_resized

    processed = np.expand_dims(img_gray / 255.0, axis=(0, -1))  # (1,512,512,1)

    # 3) Cargar modelo real
    model = model_fun()

    # 4) Ruta temporal para guardar resultado
    save_path = tmp_path / "gradcam_result.jpg"

    # 5) Ejecutar Grad-CAM
    result_img = main(
        processed=processed,
        original=img_gray,
        model=model,
        save_path=str(save_path)
    )

    # 6) Comprobar que se generó la imagen y existe el archivo
    assert isinstance(result_img, np.ndarray)
    assert result_img.shape[-1] == 3  # RGB
    assert os.path.exists(save_path)
