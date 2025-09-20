import os
from src.data.read_img import read_dicom

def test_read_dicom():
    dicom_path = "data/raw/sample.dcm"
    assert os.path.exists(dicom_path), "El archivo DICOM no existe"
    
    array, img = read_dicom(dicom_path)
    assert array is not None and array.shape[0] > 0, "El array no fue cargado correctamente"
    assert img is not None, "La imagen PIL no fue creada"
