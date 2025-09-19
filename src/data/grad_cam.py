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


def grad_cam(array: np.ndarray) -> np.ndarray:
    """
    Genera un mapa de calor Grad-CAM superpuesto sobre la imagen original.

    Args:
        array (np.ndarray): Imagen original en formato RGB o grayscale (H, W, C), valores 0-255.

    Returns:
        np.ndarray: Imagen con heatmap superpuesto en formato RGB (H, W, 3), dtype uint8.
    """
    # 1. Preprocesar la imagen usando el módulo externo
    img = preprocess(array)  # shape: (1, 512, 512, 1)

    # 2. Cargar el modelo entrenado
    model = model_fun()

    # 3. Obtener la capa convolucional final
    last_conv_layer = model.get_layer("conv10_thisone")

    # 4. Crear un modelo que devuelva la salida de la capa convolucional y las predicciones
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )

    # 5. Calcular gradientes usando GradientTape 
    with tf.GradientTape() as tape:
        tape.watch(img)  # Necesario para rastrear gradientes respecto a la entrada
        conv_output, predictions = grad_model(img, training=False)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # 6. Calcular gradientes de la clase predicha respecto a la capa convolucional
    grads = tape.gradient(class_channel, conv_output)
    if grads is None:
        raise ValueError("No se pudieron calcular los gradientes. Verifica capa 'conv10_thisone'")

    # 7. Promediar gradientes por canal (Global Average Pooling)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 8. Aplicar pesos de importancia a las características de la capa convolucional
    conv_output = conv_output[0]  # Quitar batch: (512, 512, F)
    pooled_grads = tf.reshape(pooled_grads, (1, 1, -1))  # Shape: (1, 1, F)
    weighted_conv = tf.multiply(conv_output, pooled_grads)

    # 9. Sumar canales para obtener heatmap
    heatmap = tf.reduce_sum(weighted_conv, axis=-1)
    heatmap = tf.maximum(heatmap, 0)  # ReLU
    heatmap = heatmap / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())  # Normalizar
    heatmap = heatmap.numpy()  # Convertir a numpy para OpenCV

    # 10. Redimensionar al tamaño original de la imagen (512x512)
    heatmap_resized = cv2.resize(heatmap, (512, 512))

    # 11. Convertir a color con colormap
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # 12. Preparar la imagen original para superposición
    img2 = cv2.resize(array, (512, 512))
    if len(img2.shape) == 2 or img2.shape[-1] == 1:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    img2 = img2.astype(np.uint8)

    # 13. Superponer heatmap con transparencia
    alpha = 0.7
    superimposed_img = cv2.addWeighted(heatmap_colored, alpha, img2, 1 - alpha, 0)

    # 14. Convertir de BGR a RGB para Tkinter
    return superimposed_img[:, :, ::-1]
