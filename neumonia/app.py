#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
app.py
------
Interfaz gráfica con Tkinter para:
1. Seleccionar una imagen médica (.dcm, .jpg, .jpeg, .png).
2. Ejecutar el pipeline (integrator.py).
3. Mostrar el Grad-CAM resultante y la clase predicha.
"""

import os
import sys
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Agregar la raíz del proyecto al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.integrator import run_pipeline


class PneumoniaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Detector de Neumonía con Grad-CAM")
        self.root.geometry("900x750")
        self.root.configure(bg="#f4f4f4")

        # Botón para cargar imagen
        self.btn_load = tk.Button(
            root,
            text="📂 Seleccionar Imagen",
            command=self.load_image,
            font=("Arial", 14),
            bg="#007acc",
            fg="white",
            relief="flat",
            padx=20,
            pady=10
        )
        self.btn_load.pack(pady=20)

        # Label para mostrar la imagen
        self.img_label = tk.Label(root, bg="#f4f4f4")
        self.img_label.pack(pady=10)

        # Label para mostrar la clase predicha
        self.pred_label = tk.Label(
            root, text="", font=("Arial", 16, "bold"),
            bg="#f4f4f4", fg="#333"
        )
        self.pred_label.pack(pady=10)

    def load_image(self):
        filetypes = [
            ("DICOM files", "*.dcm"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
        ]
        path = filedialog.askopenfilename(
            title="Seleccionar imagen",
            initialdir="data/raw",
            filetypes=filetypes
        )
        if not path:
            return

        try:
            # Ejecutar pipeline (devuelve imagen anotada y clase)
            result_img, pred_class = run_pipeline(path, save=True)

            # Convertir imagen a formato PIL
            img_pil = Image.fromarray(result_img)
            img_pil = img_pil.resize((512, 512))
            img_tk = ImageTk.PhotoImage(img_pil)

            # Mostrar imagen en la GUI
            self.img_label.configure(image=img_tk)
            self.img_label.image = img_tk

            # Mostrar clase predicha
            self.pred_label.configure(text=f"Clase predicha: {pred_class}")

            messagebox.showinfo("Éxito", "Grad-CAM generado correctamente.")
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error:\n{e}")
            print(f"[ERROR] {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = PneumoniaApp(root)   # ✅ Ahora usamos la clase correcta
    root.mainloop()
