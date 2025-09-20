import numpy as np
from src.data.read_img import read_dicom
from src.data.preprocess_img import preprocess
from src.data.load_model import model_fun
from src.data.grad_cam import grad_cam

def test_grad_cam():
    array, _ = read_dicom("data/raw/sample.dcm")
    processed = preprocess(array)
    model = model_fun()
    
    heatmap_img = grad_cam(processed, array, model)
    
    assert isinstance(heatmap_img, np.ndarray), "Grad-CAM no devolvió un np.ndarray"
    assert heatmap_img.shape[2] == 3, "Grad-CAM no devolvió una imagen RGB"