"""
Utilidades para procesamiento de imágenes médicas
Soporta formatos JPG, PNG y DICOM
Autor: Carlos Monsalve
Proyecto: Detección de Neumonía con IA
"""

import os
import pydicom
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import img_to_array


def load_image(path, as_gray=True):
    """
    Función auxiliar para cargar imágenes JPG/PNG regulares
    
    Args:
        path (str): Ruta al archivo de imagen
        as_gray (bool): Si convertir a escala de grises
    
    Returns:
        PIL.Image: Imagen cargada
    """
    pil_img = Image.open(path)
    
    if as_gray and pil_img.mode != 'L':
        pil_img = pil_img.convert('L')
    
    return pil_img


def smart_load_image(path, as_gray=True):
    """
    Carga automáticamente una imagen JPG/PNG o DICOM (.dcm).
    Retorna un objeto PIL.Image.
    
    Args:
        path (str): Ruta al archivo de imagen
        as_gray (bool): Si convertir a escala de grises
    
    Returns:
        PIL.Image: Imagen cargada y procesada
    
    Raises:
        ValueError: Si el formato de archivo no es soportado
        FileNotFoundError: Si el archivo no existe
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró la imagen: {path}")
    
    ext = os.path.splitext(path)[-1].lower()

    if ext in [".jpg", ".jpeg", ".png"]:
        return load_image(path, as_gray=as_gray)

    elif ext == ".dcm":
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)

        # Normalizar entre 0-255
        arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255
        arr = arr.astype(np.uint8)

        if as_gray:
            return Image.fromarray(arr, mode="L")
        else:
            return Image.fromarray(np.stack([arr] * 3, axis=-1), mode="RGB")

    else:
        raise ValueError(f"Formato de archivo no soportado: {ext}")


def preprocess_for_model_with_model(pil_img, model):
    """
    Preprocesa una imagen PIL para el modelo específico
    
    Args:
        pil_img (PIL.Image): Imagen PIL
        model: Modelo de TensorFlow/Keras
    
    Returns:
        np.array: Batch preparado para predicción con shape (1, height, width, channels)
    """
    # Obtener el tamaño de entrada del modelo
    input_shape = model.input_shape
    target_size = (input_shape[1], input_shape[2])  # (height, width)
    
    # Redimensionar imagen
    img_resized = pil_img.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convertir a array numpy
    img_array = img_to_array(img_resized)
    
    # Normalizar píxeles (0-1)
    img_array = img_array / 255.0
    
    # Agregar dimensión de batch
    batch = np.expand_dims(img_array, axis=0)
    
    return batch


def display_image(pil_img, title="Imagen"):
    """
    Muestra una imagen PIL usando matplotlib
    
    Args:
        pil_img (PIL.Image): Imagen a mostrar
        title (str): Título de la imagen
    """
    plt.figure(figsize=(8, 6))
    
    if pil_img.mode == 'L':
        # Imagen en escala de grises
        plt.imshow(pil_img, cmap='gray')
    else:
        # Imagen en color
        plt.imshow(pil_img)
    
    plt.title(title)
    plt.axis('off')
    plt.show()


def validate_image_file(file_path):
    """
    Valida que un archivo sea una imagen válida
    Útil para pruebas unitarias y validación de entrada
    
    Args:
        file_path (str): Ruta al archivo
    
    Returns:
        bool: True si es válido, False en caso contrario
    """
    if not os.path.exists(file_path):
        return False
    
    valid_extensions = ['.jpg', '.jpeg', '.png', '.dcm']
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension not in valid_extensions:
        return False
    
    try:
        # Intentar cargar la imagen
        smart_load_image(file_path)
        return True
    except Exception as e:
        print(f"Error validando imagen {file_path}: {e}")
        return False


def get_image_info(image_path):
    """
    Obtiene información básica de una imagen
    Útil para debugging y desarrollo
    
    Args:
        image_path (str): Ruta a la imagen
    
    Returns:
        dict: Diccionario con información de la imagen o error
    """
    try:
        pil_img = smart_load_image(image_path)
        
        info = {
            'path': image_path,
            'size': pil_img.size,  # (width, height)
            'mode': pil_img.mode,
            'format': getattr(pil_img, 'format', 'Unknown'),
            'file_size_mb': round(os.path.getsize(image_path) / (1024*1024), 2)
        }
        
        return info
    
    except Exception as e:
        return {'error': str(e)}


def batch_process_images(image_paths, model, as_gray=True):
    """
    Procesa múltiples imágenes para el modelo (función adicional útil)
    
    Args:
        image_paths (list): Lista de rutas de imágenes
        model: Modelo de TensorFlow/Keras
        as_gray (bool): Si procesar en escala de grises
    
    Returns:
        np.array: Batch de imágenes preprocesadas
    """
    batches = []
    
    for path in image_paths:
        try:
            pil_img = smart_load_image(path, as_gray=as_gray)
            batch = preprocess_for_model_with_model(pil_img, model)
            batches.append(batch[0])  # Remover dimensión de batch individual
        except Exception as e:
            print(f"Error procesando {path}: {e}")
            continue
    
    if batches:
        return np.array(batches)
    else:
        return np.array([])


# Función de utilidad para testing
def create_test_image(size=(100, 100), color=128, mode='L', save_path=None):
    """
    Crea una imagen de prueba para testing
    
    Args:
        size (tuple): Tamaño de la imagen (width, height)
        color (int): Color de la imagen (0-255 para escala de grises)
        mode (str): Modo de la imagen ('L' para grises, 'RGB' para color)
        save_path (str): Ruta donde guardar la imagen (opcional)
    
    Returns:
        PIL.Image: Imagen creada
    """
    if mode == 'L':
        img = Image.new('L', size, color=color)
    else:
        img = Image.new('RGB', size, color=(color, color, color))
    
    if save_path:
        img.save(save_path)
    
    return img


# Configuración para logging (opcional)
def setup_logging():
    """Configura logging básico para el módulo"""
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    return logger


if __name__ == "__main__":
    # Código de prueba rápida
    print("🔧 Módulo image_utils cargado correctamente")
    print("📋 Funciones disponibles:")
    print("  • smart_load_image()")
    print("  • preprocess_for_model_with_model()")
    print("  • display_image()")
    print("  • validate_image_file()")
    print("  • get_image_info()")
    print("  • batch_process_images()")
    print("  • create_test_image()")