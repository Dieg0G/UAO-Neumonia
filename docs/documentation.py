from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak

import os

# Ruta de salida del PDF
pdf_path = r"E:\MODULO01IA\UAO-Neumonia\reports\Proyecto_IA_Neumonia.pdf"
doc = SimpleDocTemplate(pdf_path, pagesize=A4)
story = []

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='CenterTitle', alignment=1, fontSize=18, spaceAfter=20))

# ---- Portada ----
story.append(Paragraph("Proyecto IA: Detección de Neumonía", styles['CenterTitle']))
story.append(Paragraph("Autores: César Augusto Campos Rodríguez, Diego Fernando Guillen," \
" Carlos Javier Monsalve Ávila ", styles['Normal']))
story.append(Paragraph("Fecha: 2025-09-21", styles['Normal']))
story.append(PageBreak())

# ---- Sección: Documentación del Proyecto ----
story.append(Paragraph("1. Documentación del Proyecto", styles['Heading1']))
story.append(Paragraph(
    "Este proyecto tiene como objetivo re factorizar con alta cohesion baja dependencia un sistema de inteligencia artificial "
    "para la detección automatizada de neumonía en imágenes médicas. "
    "El sistema incluye preprocesamiento de imágenes, carga de modelos y generación de resultados mediante Grad-CAM.",
    styles['Normal']
))
story.append(Spacer(1, 12))

# ---- Sección: Metodología ----
story.append(Paragraph("2. Metodología", styles['Heading1']))
story.append(Paragraph(
    "Se siguió un enfoque de desarrollo modular con las siguientes fases:\n"
    "1. Lectura de imágenes y preprocesamiento.\n"
    "2. Entrenamiento y carga de modelo convolucional.\n"
    "3. Generación de mapas de atención (Grad-CAM) para interpretación.\n"
    "4. Validación mediante tests unitarios con pytest.\n"
    "5. Registro de resultados y generación de reportes gráficos.",
    styles['Normal']
))
story.append(Spacer(1, 12))

# ---- Sección: Flujo del Proyecto ----
story.append(Paragraph("3. Flujo del Proyecto", styles['Heading1']))
story.append(Paragraph(
    "El flujo general del proyecto es el siguiente:\n"
    "Imagen cruda -> Preprocesamiento -> Modelo -> Grad-CAM -> Resultados",
    styles['Normal']
))
story.append(Spacer(1, 12))

# ---- Sección: Imágenes de Pruebas ----
story.append(Paragraph("4. Imágenes de Pruebas", styles['Heading1']))

figuras_dir = r"E:\MODULO01IA\UAO-Neumonia\reports\figuras"
for img_file in os.listdir(figuras_dir):
    if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(figuras_dir, img_file)
        story.append(Paragraph(img_file, styles['Normal']))
        img = Image(img_path, width=400, height=300)  # ajustar tamaño
        story.append(img)
        story.append(Spacer(1, 12))

# ---- Sección: Resultados de Pytest ----
story.append(PageBreak())
story.append(Paragraph("5. Resultados de Pytest", styles['Heading1']))

# Puedes generar un txt de pytest y cargarlo en el PDF
pytest_output_file = r"E:\MODULO01IA\UAO-Neumonia\reports\pytest_results.txt"
if os.path.exists(pytest_output_file):
    with open(pytest_output_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            story.append(Paragraph(line.strip(), styles['Code']))
else:
    story.append(Paragraph("No se encontró el archivo de resultados de pytest.", styles['Normal']))

# ---- Generar PDF ----
doc.build(story)
print(f"PDF generado en {pdf_path}")
