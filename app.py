import streamlit as st
from integrator import load_model, predict_and_heatmap
from PIL import Image

st.title("Detector de Neumonía con Grad-CAM")

uploaded = st.file_uploader("Sube una radiografía (JPEG/PNG)", type=["jpeg","jpg","png"])
if uploaded:
    img = Image.open(uploaded).convert("L").resize((512,512))
    img.save("temp.png")
    model = load_model()
    res = predict_and_heatmap(model, "temp.png", out_path="out.png")
    output_path = "out.png"
    # Bloque de visualización en columnas
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(output_path, caption="Heatmap Grad-CAM", use_container_width=True)
    with col2:
        st.subheader(f"Predicción: {res['class']} | Confianza: {res['probs'][res['class']]:.2%}")
