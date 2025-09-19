#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np


def preprocess(image: np.ndarray, target_size: tuple = (512, 512)) -> np.ndarray:
    import tensorflow as tf
    # Convert input to tensor if not already
    img = tf.convert_to_tensor(image, dtype=tf.float32)
    # Resize
    img = tf.image.resize(img, target_size)
    # If input is RGB, convert to grayscale
    if img.shape[-1] == 3:
        img = tf.image.rgb_to_grayscale(img)
    # Normalize [0,1]
    img = img / 255.0
    # Expand dims: (1, H, W, 1)
    img = tf.expand_dims(img, axis=0)
    return img


if __name__ == "__main__":
    # Test rápido (con imagen dummy)
    dummy = np.zeros((600, 600, 3), dtype=np.uint8)
    processed = preprocess(dummy)
    print("Shape procesada:", processed.shape)