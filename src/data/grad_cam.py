#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
grad_cam.py
-----------
Genera Grad-CAM, predice la clase y etiqueta la imagen resultante.
Devuelve SOLO el array de la imagen RGB (H,W,3) listo para guardar/mostrar.
"""

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import warnings
from typing import Tuple


def _build_inputs_to_call(model, processed):
    """
    Construye el objeto que se pasará al modelo/grad_model:
    - si model.input_names existe y tiene 1 nombre -> devuelve {name: processed}
    - si processed ya es dict -> lo devuelve
    - else -> devuelve processed (fallback)
    """
    if isinstance(processed, dict):
        return processed
    input_names = getattr(model, "input_names", None)
    if input_names and len(input_names) == 1:
        return {input_names[0]: processed}
    return processed


def grad_cam(processed: np.ndarray,
             original: np.ndarray,
             model: tf.keras.Model,
             layer_name: str = "conv10_thisone",
             out_size: tuple = (512, 512),
             alpha: float = 0.7) -> np.ndarray:
    """
    Genera Grad-CAM dado processed, original y model.
    Devuelve RGB uint8 (out_size) con heatmap superpuesto.
    """
    if processed is None:
        raise ValueError("`processed` no puede ser None.")
    if model is None:
        raise ValueError("`model` no puede ser None.")

    # 1) obtener capa convolucional
    try:
        last_conv = model.get_layer(layer_name)
    except Exception as e:
        raise ValueError(f"No se encontró la capa '{layer_name}' en el modelo: {e}")

    # 2) modelo intermedio
    grad_model = tf.keras.models.Model(inputs=model.input,
                                       outputs=[last_conv.output, model.output])

    # 3) preparar inputs (dict o tensor)
    inputs_to_call = _build_inputs_to_call(model, processed)

    # 4) calcular gradientes (silenciando solo el warning puntual)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning,
                                message="The structure of `inputs` doesn't match the expected structure.")
        with tf.GradientTape() as tape:
            conv_outputs, preds = grad_model(inputs_to_call, training=False)

            if isinstance(preds, (list, tuple)):
                preds = preds[0]

            preds = tf.convert_to_tensor(preds)

            if preds.shape.rank == 1:
                pred_index = tf.argmax(preds, axis=-1)
                class_channel = preds[pred_index]
            else:
                pred_index = tf.argmax(preds, axis=1)[0]
                class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        raise RuntimeError("Gradientes = None. Verifica que la capa y el modelo sean compatibles.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    weighted = conv_outputs * pooled_grads
    heatmap = tf.reduce_mean(weighted, axis=-1)
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap / (tf.reduce_max(heatmap) + K.epsilon())

    # 5) heatmap → colormap
    heatmap_resized = cv2.resize(heatmap.numpy(), out_size)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)  # BGR

    # 6) original → BGR
    orig = original.copy()
    if orig.ndim == 2:
        orig_bgr = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
    elif orig.ndim == 3 and orig.shape[-1] == 1:
        orig_bgr = cv2.cvtColor(orig[:, :, 0], cv2.COLOR_GRAY2BGR)
    else:
        orig_bgr = orig
    orig_bgr = cv2.resize(orig_bgr, out_size)

    # 7) superponer y devolver en RGB
    superimposed_bgr = cv2.addWeighted(heatmap_colored, alpha, orig_bgr, 1 - alpha, 0)
    superimposed_rgb = superimposed_bgr[:, :, ::-1]
    return superimposed_rgb.astype(np.uint8)


def _draw_label_on_image_rgb(rgb_img: np.ndarray, text: str) -> np.ndarray:
    """
    Dibuja un rectángulo negro semitransparente y texto blanco en la esquina superior izquierda.
    Entrada: rgb_img (H,W,3) uint8
    Retorna imagen RGB anotada.
    """
    # Convert RGB -> BGR para usar cv2.draw
    bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    padding = 8
    x0, y0 = 10, 10
    x1 = x0 + tw + padding
    y1 = y0 + th + padding

    # rect background (opaco)
    cv2.rectangle(bgr, (x0 - 3, y0 - 3), (x1, y1), (0, 0, 0), -1)
    # text (blanco)
    text_org = (x0 + 2, y0 + th - 2)
    cv2.putText(bgr, text, text_org, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # volver a RGB
    annotated_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return annotated_rgb


def main(processed,
         original,
         model,
         layer_name: str = "conv10_thisone",
         out_size: tuple = (512, 512),
         alpha: float = 0.7,
         save_path: str = "data/processed/gradcam_result.jpg") -> np.ndarray:
    """
    Wrapper que:
     - obtiene la predicción (label, prob)
     - genera la imagen Grad-CAM
     - anota la imagen con label/prob
     - guarda y muestra la imagen
     - devuelve SOLO la imagen RGB (np.ndarray)
    """
    # 1) construir inputs para predecir (evita warnings)
    inputs_to_call = _build_inputs_to_call(model, processed)

    # 2) predecir (numpy array)
    try:
        preds = model.predict(inputs_to_call, verbose=0)
    except Exception:
        # fallback si algo raro ocurre
        preds = model.predict(processed, verbose=0)

    # normalizar preds a numpy
    preds_np = np.array(preds)
    class_idx = int(np.argmax(preds_np[0]))
    proba = float(np.max(preds_np[0]) * 100.0)
    labels = ["bacteriana", "normal", "viral"]
    label = labels[class_idx] if class_idx < len(labels) else f"class_{class_idx}"

    # 3) generar grad-cam (RGB)
    gradcam_img = grad_cam(processed, original, model, layer_name=layer_name, out_size=out_size, alpha=alpha)

    # 4) anotar la imagen con label+prob
    text = f"{label} ({proba:.2f}%)"
    annotated = _draw_label_on_image_rgb(gradcam_img, text)

    # 5) guardar (cv2 espera BGR)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    cv2.imwrite(save_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    print(f"[OK] Resultado guardado en: {save_path}")

    # 6) mostrar
    plt.imshow(annotated)
    plt.title(f"Predicción: {label} ({proba:.2f}%)")
    plt.axis("off")
    plt.show()

    # 7) devolver imagen anotada + etiqueta
    return annotated, label
