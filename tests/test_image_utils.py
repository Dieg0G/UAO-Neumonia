"""
Pruebas unitarias para el sistema de detección de neumonía
Autor: [Tu nombre]
Proyecto: Detección de Neumonía con IA
"""

import pytest
import numpy as np
import os
from unittest.mock import Mock, patch
from PIL import Image
import tempfile

# Importar las funciones a probar
from src.image_utils import (
    smart_load_image, 
    validate_image_file,
    get_image_info
)


class TestImageUtils:
    """Pruebas para las utilidades de procesamiento de imágenes"""
    
    def test_smart_load_image_jpg_real(self):
        """
        TEST 1: Probar que smart_load_image carga correctamente imágenes JPG reales
        Usa una de las imágenes que ya tienes en tu proyecto
        """
        # Usar una imagen real de tu proyecto
        image_path = "data/jpg/normal/NORMAL2-IM-1144-0001.jpeg"
        
        # Verificar que el archivo existe (skip si no existe)
        if not os.path.exists(image_path):
            pytest.skip(f"Imagen de prueba no encontrada: {image_path}")
        
        # Probar la función
        loaded_img = smart_load_image(image_path, as_gray=True)
        
        # Verificaciones
        assert loaded_img is not None, "La imagen no debería ser None"
        assert isinstance(loaded_img, Image.Image), "Debería retornar un objeto PIL.Image"
        assert loaded_img.mode == 'L', "Debería estar en escala de grises (modo L)"
        
        # Verificar que tiene dimensiones válidas
        width, height = loaded_img.size
        assert width > 0 and height > 0, "La imagen debe tener dimensiones válidas"
        
        print(f"✅ Imagen cargada correctamente: {loaded_img.size}, modo: {loaded_img.mode}")
    
    def test_smart_load_image_file_not_found(self):
        """
        TEST 2: Verificar que se maneja correctamente cuando un archivo no existe
        """
        # Probar con archivo inexistente
        with pytest.raises(FileNotFoundError):
            smart_load_image("archivo_que_no_existe.jpg")
        
        print("✅ Error FileNotFoundError manejado correctamente")
    
    def test_validate_image_file_real_image(self):
        """
        TEST 3: Probar validación con imágenes reales del proyecto
        """
        # Probar con imagen real que sabemos que existe
        test_images = [
            "data/jpg/normal/NORMAL2-IM-1144-0001.jpeg",
            "data/jpg/bacteria/person1710_bacteria_4526.jpeg", 
            "data/jpg/virus/person1497_virus_2607.jpeg"
        ]
        
        for image_path in test_images:
            if os.path.exists(image_path):
                result = validate_image_file(image_path)
                assert result == True, f"Imagen válida debería pasar validación: {image_path}"
                print(f"✅ Imagen válida: {image_path}")
                break
        else:
            pytest.skip("No se encontraron imágenes de prueba en el proyecto")
    
    def test_validate_image_file_invalid_extension(self):
        """
        TEST 4: Probar que rechaza archivos con extensiones no válidas
        """
        # Crear archivo temporal con extensión inválida
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp_file:
            tmp_file.write(b"Este no es un archivo de imagen")
            tmp_file.flush()
            
            # Probar validación
            result = validate_image_file(tmp_file.name)
            assert result == False, "Archivo .txt no debería ser válido como imagen"
            
            # Limpiar
            os.unlink(tmp_file.name)
        
        print("✅ Archivo con extensión inválida rechazado correctamente")
    
    def test_get_image_info_real_image(self):
        """
        TEST 5: Probar que get_image_info retorna información válida
        """
        # Usar imagen real del proyecto
        image_path = "data/jpg/normal/NORMAL2-IM-1144-0001.jpeg"
        
        if not os.path.exists(image_path):
            pytest.skip(f"Imagen de prueba no encontrada: {image_path}")
        
        # Probar la función
        info = get_image_info(image_path)
        
        # Verificaciones
        assert 'path' in info, "Info debe contener 'path'"
        assert 'size' in info, "Info debe contener 'size'"
        assert 'mode' in info, "Info debe contener 'mode'"
        assert 'file_size_mb' in info, "Info debe contener 'file_size_mb'"
        
        # Verificar tipos de datos
        assert isinstance(info['size'], tuple), "Size debe ser una tupla"
        assert len(info['size']) == 2, "Size debe tener 2 elementos (width, height)"
        assert isinstance(info['file_size_mb'], (int, float)), "file_size_mb debe ser numérico"
        assert info['file_size_mb'] > 0, "Tamaño de archivo debe ser positivo"
        
        print(f"✅ Info de imagen: {info}")


class TestPneumoniaDetectorIntegration:
    """Pruebas de integración para el detector de neumonía"""
    
    @pytest.fixture
    def mock_model(self):
        """Fixture: Modelo mock para pruebas"""
        model = Mock()
        model.input_shape = (None, 512, 512, 1)
        # Simulamos una predicción realista: normal con alta confianza
        model.predict.return_value = np.array([[0.05, 0.90, 0.05]])  
        return model
    
    def test_model_prediction_format(self, mock_model):
        """
        TEST 6: Verificar que el formato de predicción del modelo es correcto
        """
        # Crear imagen de prueba
        test_image = Image.new('L', (100, 100), color=128)
        
        # Importar función de preprocesamiento
        from src.image_utils import preprocess_for_model_with_model
        
        # Preprocesar imagen
        batch = preprocess_for_model_with_model(test_image, mock_model)
        
        # Verificar formato del batch
        assert isinstance(batch, np.ndarray), "Batch debe ser numpy array"
        assert batch.shape == (1, 512, 512, 1), "Batch debe tener shape correcto"
        assert np.all(batch >= 0) and np.all(batch <= 1), "Valores deben estar normalizados [0,1]"
        
        # Probar predicción
        prediction = mock_model.predict(batch)
        
        # Verificar formato de predicción
        assert isinstance(prediction, np.ndarray), "Predicción debe ser numpy array"
        assert prediction.shape == (1, 3), "Debe predecir 3 clases"
        
        print(f"✅ Batch shape: {batch.shape}, Predicción shape: {prediction.shape}")
        print(f"✅ Predicción: {prediction[0]}")


# Funciones de utilidad para las pruebas
def test_project_structure():
    """
    TEST 7: Verificar que la estructura del proyecto es correcta
    """
    required_dirs = ['src', 'data', 'modelo', 'tests']
    required_files = ['src/image_utils.py', 'proyecto-neumonia.py']
    
    # Verificar directorios
    for dir_name in required_dirs:
        assert os.path.exists(dir_name), f"Directorio requerido no existe: {dir_name}"
    
    # Verificar archivos importantes
    for file_name in required_files:
        assert os.path.exists(file_name), f"Archivo requerido no existe: {file_name}"
    
    print("✅ Estructura del proyecto correcta")


if __name__ == "__main__":
    print("🧪 Ejecutando pruebas unitarias...")
    print("Para ejecutar las pruebas completas, usa: pytest tests/ -v")