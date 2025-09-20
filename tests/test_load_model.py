from src.data.load_model import model_fun

def test_model_load():
    model = model_fun()
    assert model is not None, "El modelo no fue cargado"
    assert hasattr(model, "predict"), "El modelo no tiene método predict"