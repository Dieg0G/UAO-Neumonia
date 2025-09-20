import os
import tensorflow as tf
from tensorflow import keras
import h5py

def handle_custom_loss(y_true, y_pred):
    """funcion personalizada para la función de pérdida."""
    return keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False, reduction='mean')

def model_fun():
    """
    Carga el modelo de detección de neumonía
    
    Returns:
        tf.keras.Model: Modelo cargado y listo para usar
    """
    model_path = r"E:\MODULO01IA\UAO-Neumonia\data\models\conv_MLP_84.h5"

    # Verificar que el archivo existe
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"archivo no encontrado: {model_path}")

    try:
        # Intentar cargar con objetos personalizados
        custom_objects = {
            'loss': handle_custom_loss,
        }
        model = keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    except Exception as e:
        print(f"error de carga personalizada: {e}")
        model = keras.models.load_model(model_path, compile=False)
        # Compilar el modelo después de cargarlo
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

def handle_custom_loss(y_true, y_pred):
    """funcion personalizada para la función de pérdida."""
    return keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False, reduction='mean')

if __name__ == "__main__":
    model = model_fun()
    print("Model loaded successfully:", model)