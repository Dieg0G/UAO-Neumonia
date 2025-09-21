# tests/test_preprocess_img.py
import numpy as np
import pytest
from src.data.preprocess_img import preprocess

def test_preprocess_grayscale():
    img = np.random.randint(0, 256, (600, 400), dtype=np.uint8)  # grayscale
    processed = preprocess(img)
    assert isinstance(processed, np.ndarray)
    assert processed.shape == (1, 512, 512, 1)
    assert processed.max() <= 1.0 and processed.min() >= 0.0

def test_preprocess_rgb():
    img = np.random.randint(0, 256, (600, 400, 3), dtype=np.uint8)  # RGB
    processed = preprocess(img)
    assert isinstance(processed, np.ndarray)
    assert processed.shape == (1, 512, 512, 1)
    assert processed.max() <= 1.0 and processed.min() >= 0.0

def test_invalid_input():
    img = "not an image"
    with pytest.raises(Exception):
        preprocess(img)
