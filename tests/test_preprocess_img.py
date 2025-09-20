import numpy as np
from src.data.preprocess_img import preprocess
from src.data.read_img import read_dicom

def test_preprocess():
    array, _ = read_dicom("data/raw/sample.dcm")
    processed = preprocess(array)
    
    assert processed.shape == (1, 512, 512, 1), "El preprocesamiento no generó el shape esperado"
    assert processed.max() <= 1.0 and processed.min() >= 0.0, "La imagen no está normalizada"
