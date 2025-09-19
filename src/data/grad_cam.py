if __name__ == "__main__":
    from src.data.load_model import model_fun
    model = model_fun()
    print("\nMODEL SUMMARY:")
    model.summary()
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Módulo para generar mapas de calor Grad-CAM sobre predicciones del modelo de neumonía.
"""

import numpy as np
import cv2
import tensorflow as tf
from src.data.load_model import model_fun
from src.data.preprocess_img import preprocess


def grad_cam(array):
    """
    Genera un mapa de calor Grad-CAM superpuesto sobre la imagen original.
    Usa exclusivamente tensores de TensorFlow para el cálculo de gradientes.
    No usa operaciones NumPy entre la entrada y el gradiente.
    """
    # 1. Preprocesar la imagen → devuelve numpy.ndarray (1, 512, 512, 1)
    img_np = preprocess(array)  # Shape: (1, 512, 512, 1), dtype=float32
    img = tf.convert_to_tensor(img_np, dtype=tf.float32)

    # 2. Cargar modelo
    model = model_fun()
    # 3. Obtener capa convolucional final
    last_conv_layer = model.get_layer("conv10_thisone")

    # 4. Crear modelo intermedio que retorne activación convolucional + predicción
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )

    # 5. Calcular gradientes con GradientTape
    with tf.GradientTape() as tape:
        # Watch the input image for gradients
        tape.watch(img)
        conv_output, predictions = grad_model(img)
        # Select the predicted class index
        pred_index = tf.argmax(predictions[0], axis=-1)
        # Use tf.gather to select the correct class channel
        class_channel = tf.gather(predictions[0][0], pred_index)
        # Compute gradients of the class channel w.r.t. conv_output
        grads = tape.gradient(class_channel, conv_output)
    # Debugging output
    print('DEBUG: grads type:', type(grads))
    print('DEBUG: grads value:', grads)
    if grads is None:
        raise ValueError("No se pudieron calcular los gradientes. Verifica que 'conv10_thisone' sea diferenciable.")

    # 6. Promediar gradientes por canal
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 7. Aplicar pesos a las activaciones
    conv_output = conv_output[0]  # Remove batch: (H, W, F)
    weighted_conv = conv_output * pooled_grads  # Broadcasting

    # 8. Generar heatmap
    heatmap = tf.reduce_sum(weighted_conv, axis=-1)
    heatmap = tf.maximum(heatmap, 0)  # ReLU
    heatmap = heatmap / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())  # Normalizar

    # 9. Convertir a NumPy para OpenCV
    heatmap = heatmap.numpy()

    # 10. Redimensionar al tamaño original de la imagen (512x512)
    heatmap_resized = cv2.resize(heatmap, (512, 512))

    # 11. Convertir a color
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # 12. Preparar imagen original
    img2 = cv2.resize(array, (512, 512))
    if len(img2.shape) == 2 or img2.shape[-1] == 1:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    img2 = img2.astype(np.uint8)

    # 13. Superponer
    alpha = 0.7
    superimposed_img = cv2.addWeighted(heatmap_colored, alpha, img2, 1 - alpha, 0)

    # 14. Convertir BGR → RGB para Tkinter
    return superimposed_img[:, :, ::-1]