import argparse 
import numpy as np
from tensorflow import keras
import tensorflow as tf
from PIL import Image
import os
import sys

# 👉 Agregamos la carpeta src al path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from image_utils import smart_load_image, preprocess_for_model_with_model

CLASS_NAMES = ["Normal", "Bacteria", "Virus"]

def load_model(path="modelo/conv_MLP_84.h5"):
    return keras.models.load_model(path, compile=False)


def make_gradcam_heatmap(img_array, model, class_index):
    last_conv_layer = model.get_layer("conv10_thisone")

    grad_model = tf.keras.models.Model(
        [model.inputs], [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)

        if isinstance(predictions, list):
            predictions = predictions[0]

        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)
    return heatmap


def predict_and_heatmap(model, img_path, out_path=None):
    img = smart_load_image(img_path)
    batch = preprocess_for_model_with_model(img, model)

    predictions = model(batch, training=False)

    if isinstance(predictions, list):
        predictions = predictions[0]

    probs = predictions.numpy().flatten()
    class_idx = int(np.argmax(probs))

    heatmap = make_gradcam_heatmap(batch, model, class_idx)

    if out_path:
        import matplotlib.pyplot as plt

        img_resized = img.resize((512, 512)).convert("RGB")
        heatmap_resized = Image.fromarray(np.uint8(255 * heatmap)).resize(
            img_resized.size, Image.BICUBIC
        )
        heatmap_resized = np.array(heatmap_resized)

        jet = plt.colormaps["jet"]
        jet_heatmap = jet(heatmap_resized / 255.0)[:, :, :3]
        jet_heatmap = Image.fromarray(np.uint8(jet_heatmap * 255)).resize(
            img_resized.size
        )

        superimposed_img = Image.blend(img_resized, jet_heatmap, alpha=0.4)
        superimposed_img.save(out_path)

    return {"class": class_idx, "probs": probs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True, help="ruta a la imagen")
    parser.add_argument(
        "--out", required=False, help="ruta para guardar la imagen con heatmap (png)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.img):
        raise FileNotFoundError(f"Imagen no encontrada: {args.img}")

    model = load_model()
    res = predict_and_heatmap(model, args.img, out_path=args.out)

    # 📊 Mostrar resultados completos
    print("\nResultados de la predicción:")
    for i, prob in enumerate(res["probs"]):
        print(f"  {CLASS_NAMES[i]:8s}: {prob:.4f}")

    print(f"\nPredicción final: {CLASS_NAMES[res['class']]} "
          f"({res['class']}) | Prob: {res['probs'][res['class']]:.4f}")

    if args.out:
        print(f"Heatmap guardado en: {args.out}")