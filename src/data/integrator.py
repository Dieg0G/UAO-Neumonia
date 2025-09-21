#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script integrador para ejecutar el flujo completo:
1. read_img
2. preprocess_img
3. load_model
4. grad_cam
"""

import os
import cv2

from src.data import read_img, preprocess_img, load_model, grad_cam


def main():
    # 1. Leer imagen
    array = read_img.main()

    # 2. Preprocesar
    processed = preprocess_img.main(array)

    # 3. Cargar modelo
    model = load_model.main()

    # 4. Grad-CAM
    gradcam_img = grad_cam.main(processed, array, model)

    # 5. Guardar en disco
    os.makedirs("data/processed", exist_ok=True)
    out_path = "data/processed/gradcam_result.jpg"
    cv2.imwrite(out_path, gradcam_img[:, :, ::-1])
    print(f"[OK] Resultado guardado en: {out_path}")


if __name__ == "__main__":
    main()
