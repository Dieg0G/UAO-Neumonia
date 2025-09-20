#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
grad_cam.py
-----------
Función para generar Grad-CAM a partir de:
  - `processed` : salida de preprocess_img (1, H, W, C)
  - `original`  : imagen original (H_orig, W_orig) o (H_orig, W_orig, 3)
  - `model`     : tf.keras.Model ya cargado (no cargar dentro de este módulo)

Devuelve:
  - RGB uint8 (H_out, W_out, 3) con el heatmap superpuesto.
"""

from typing import Optional
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import backend as K


def grad_cam(processed: np.ndarray,
             original: np.ndarray,
             model: tf.keras.Model,
             layer_name: str = "conv10_thisone",
             out_size: tuple = (512, 512),
             alpha: float = 0.7) -> np.ndarray:
    """
    Genera Grad-CAM dado processed, original y model.

    Parameters
    ----------
    processed : np.ndarray
        Imagen preprocesada (1, H, W, C), dtype float32 o float64, valores en [0,1]
    original : np.ndarray
        Imagen original (H_orig, W_orig) o (H_orig, W_orig, 3), dtype uint8 o float
    model : tf.keras.Model
        Modelo ya cargado y listo para predecir.
    layer_name : str
        Nombre de la capa convolucional a usar.
    out_size : tuple
        Tamaño del heatmap final (width, height).
    alpha : float
        Transparencia para superponer heatmap sobre la original.

    Returns
    -------
    np.ndarray
        Imagen RGB uint8 (out_size) con heatmap superpuesto.
    """
    if processed is None:
        raise ValueError("`processed` no puede ser None.")
    if model is None:
        raise ValueError("`model` no puede ser None.")

    # 1) obtener la capa convolucional
    try:
        last_conv = model.get_layer(layer_name)
    except Exception as e:
        raise ValueError(f"No se encontró la capa '{layer_name}' en el modelo: {e}")

    # 2) construir modelo intermedio (activaciones conv + predicciones)
    grad_model = tf.keras.models.Model(inputs=model.input, outputs=[last_conv.output, model.output])

    # 3) calcular gradientes con GradientTape (todo en TF)
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(processed, training=False)

        # preds puede ser tensor o lista/tuple; normalizar a tensor
        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        preds = tf.convert_to_tensor(preds)

        # manejar shapes distintas:
        # - si preds tiene shape (num_classes,) -> vector de logits/probs
        # - si preds tiene shape (1, num_classes) -> batch de 1
        if preds.shape.rank == 1:
            # vector (num_classes,)
            pred_index = tf.argmax(preds, axis=-1)       # escalar tf.Tensor
            class_channel = preds[pred_index]            # escalar tf.Tensor
        else:
            # assumimos (batch, num_classes)
            pred_index = tf.argmax(preds, axis=1)[0]    # índice de la primera muestra
            class_channel = preds[:, pred_index]        # tensor shape (1,)

    if class_channel is None:
        raise RuntimeError("No se pudo obtener el canal de la clase para calcular gradientes.")

    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        raise RuntimeError("Gradientes = None. Verifica que la capa y el modelo sean compatibles para gradientes.")

    # 4) pooled_grads (importancia por canal) - aún en TF
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # shape (F,)

    # 5) obtener conv_outputs sin batch (h, w, F)
    conv_outputs = conv_outputs[0]  # tensor

    # 6) ponderar canales (operaciones TF, broadcasting funciona)
    weighted = conv_outputs * pooled_grads  # (h, w, F)

    # 7) heatmap: promedio sobre canales
    heatmap = tf.reduce_mean(weighted, axis=-1)  # (h, w)
    heatmap = tf.nn.relu(heatmap)                # ReLU
    denom = tf.reduce_max(heatmap) + K.epsilon()
    heatmap = heatmap / denom

    # 8) pasar a numpy (a partir de aquí usamos OpenCV)
    heatmap_np = heatmap.numpy()
    heatmap_resized = cv2.resize(heatmap_np, out_size)

    # 9) colormap
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)  # BGR

    # 10) preparar original para overlay
    orig = original.copy()
    # si original está en float 0..1 lo convertimos a 0..255
    if orig.dtype != np.uint8:
        orig = orig.astype(np.float32)
        if orig.max() <= 1.0:
            orig = (orig * 255.0)
        orig = np.clip(orig, 0, 255).astype(np.uint8)

    # convertir a BGR si es grayscale
    if orig.ndim == 2:
        orig_bgr = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
    elif orig.shape[-1] == 1:
        orig_bgr = cv2.cvtColor(orig[:, :, 0], cv2.COLOR_GRAY2BGR)
    else:
        orig_bgr = orig

    # redimensionar original al tamaño del heatmap
    orig_bgr = cv2.resize(orig_bgr, out_size)

    # 11) superponer
    superimposed = cv2.addWeighted(heatmap_colored, alpha, orig_bgr, 1 - alpha, 0)

    # devolver en RGB para la GUI (PIL/Tkinter esperan RGB)
    return superimposed[:, :, ::-1].astype(np.uint8)
