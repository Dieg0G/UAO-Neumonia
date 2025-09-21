import tkinter as tk
import tkcap
from fpdf import FPDF
import os

# 1. Crear GUI
root = tk.Tk()
root.geometry("400x300")
root.title("Mi Proyecto - GUI Principal")

label = tk.Label(root, text="Bienvenido a mi proyecto\nTkinter + tkcap + PDF", font=("Arial", 14))
label.pack(pady=80)

# Renderizar antes de capturar
root.update()

# 2. Capturar ventana con tkcap
cap = tkcap.CAP(root)
imagen_captura = "captura_gui.png"
cap.capture(imagen_captura)

print(f"✅ Captura guardada como {imagen_captura}")

# 3. Generar PDF
pdf = FPDF()
pdf.add_page()
pdf.set_font("Helvetica", "B", 16)
pdf.cell(0, 10, "Documentación del Proyecto", ln=True, align="C")

pdf.set_font("Helvetica", "", 12)
pdf.multi_cell(0, 10,
               "Este documento contiene la documentación del proyecto.\n"
               "Incluye una captura automática de la interfaz gráfica (Tkinter).")

if os.path.exists(imagen_captura):
    pdf.image(imagen_captura, x=15, y=60, w=180)

pdf.output("documentacion_proyecto.pdf")
print("✅ PDF generado: documentacion_proyecto.pdf")

# 4. Ejecutar GUI
root.mainloop()
