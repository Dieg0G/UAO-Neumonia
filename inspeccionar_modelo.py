from tensorflow import keras

MODEL_PATH = "modelo/conv_MLP_84.h5"

print("🔍 Cargando modelo desde:", MODEL_PATH)
model = keras.models.load_model(MODEL_PATH, compile=False)

print("\n📋 Capas Conv2D en el modelo:")
for i, layer in enumerate(model.layers):
    if "conv" in layer.name.lower():
        try:
            shape = layer.output.shape
        except:
            shape = "?"
        print(f"{i:3d} | {layer.name:20s} | {shape}")
