¡Perfecto! Basado en toda la conversación anterior y en el contenido del archivo `README.md` que subiste, aquí tienes una **versión refinada, organizada y profesional del README.md del proyecto**, integrando todos los avances técnicos, correcciones de arquitectura y buenas prácticas discutidas (especialmente la separación de responsabilidades, uso de `tf.GradientTape`, estructura de paquetes, y eliminación de preprocesamiento redundante).

---

# 🏥 Sistema de Detección de Neumonía con Deep Learning y Grad-CAM

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.0-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-GNU%20GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

> **Aplicación de inteligencia artificial para el apoyo al diagnóstico médico de neumonía en radiografías de tórax**, con visualización explicativa mediante Grad-CAM.

---

## 📌 Descripción

Este sistema implementa un modelo de **Deep Learning basado en Redes Neuronales Convolucionales (CNN)** para clasificar imágenes radiográficas de tórax en tres categorías:

- **Neumonía Bacteriana**
- **Neumonía Viral**
- **Sin Neumonía**

Además, integra la técnica de **Grad-CAM (Gradient-weighted Class Activation Mapping)** para generar mapas de calor que resaltan las regiones de la imagen más relevantes para la decisión del modelo, mejorando la transparencia y confianza clínica en las predicciones.

La interfaz gráfica permite cargar imágenes en formatos DICOM, JPG y PNG, realizar predicciones en tiempo real y generar reportes PDF o CSV para documentación médica.

---

## 🧩 Arquitectura del Proyecto

```
proyecto-neumonia/
├── data/
│   ├── external/
│   │   └── models/
│   │       └── conv_MLP_84.h5           # Modelo entrenado
│   ├── processed/
│   └── raw/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_model.py                # Carga el modelo .h5
│   │   ├── preprocess_img.py            # Preprocesa imágenes (resize, CLAHE, normalización)
│   │   └── grad_cam.py                  # Genera mapa de calor Grad-CAM (sin preprocesar)
│   └── interface/
│       └── detector_neumonia.py         # Interfaz gráfica Tkinter (punto de entrada)
├── tests/
│   ├── test_historial.py
│   └── test_imports.py
├── requirements.txt
├── README.md
└── LICENSE
```

> ✅ **Estructura modular y conforme a PEP8**: Cada módulo tiene una única responsabilidad.  
> ✅ **Separación clara**: Preprocesamiento, inferencia y visualización están desacoplados.  
> ✅ **No se duplica lógica**: `grad_cam.py` recibe la imagen ya preprocesada por `preprocess_img.py`.

---

¡Perfecto! A continuación, te presento las **instrucciones actualizadas para usar `uv`** (el nuevo gestor de paquetes rápido y moderno de Python) en lugar de Anaconda/Miniconda, manteniendo toda la estructura, claridad y profesionalismo del README original.

---

### ✅ **Requisitos Previos**

- **Python 3.9** (instalado en el sistema)
- `uv` instalado (recomendado como reemplazo de `pip` + `venv`)
- CUDA (opcional — la aplicación funciona perfectamente en CPU)

> 💡 **¿Qué es `uv`?**  
> `uv` es un gestor de entornos y paquetes extremadamente rápido, escrito en Rust, compatible con `pip`, `pyproject.toml` y `requirements.txt`. Es hasta 10x más rápido que `pip` y se integra perfectamente con los flujos de trabajo modernos de Python.

---

### 📦 Pasos de Instalación

1. **Clonar el repositorio**:
   ```bash
   git clone https://github.com/Dieg0G/UAO-Neumonia.git
   cd UAO-Neumonia
   ```

2. **Instalar `uv`**:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   > ✅ O si usas Homebrew en macOS/Linux:  
   > ```bash
   > brew install uv
   > ```

   > 🖥️ En Windows, descarga el ejecutable desde: [https://github.com/astral-sh/uv/releases](https://github.com/astral-sh/uv/releases)

3. **Crear y activar el entorno virtual con `uv`**:
   ```bash
   uv venv .venv
   source .venv/bin/activate    # Linux/macOS
   # o
   .\.venv\Scripts\activate     # Windows
   ```
   > ⚠️ Esto crea un entorno virtual en `.venv/` dentro del proyecto — ¡no requiere `conda`!

4. **Instalar dependencias con `uv`**:
   ```bash
   uv pip install -r requirements.txt
   ```
   > ✅ `uv` es compatible con `requirements.txt` y lo instala mucho más rápido que `pip`.

5. **Verificar la estructura de archivos**:
   Asegúrate de que existan estos archivos vacíos (para que Python reconozca los paquetes):
   ```
   src/__init__.py
   src/data/__init__.py
   ```

   Si no existen, créalos:
   ```bash
   touch src/__init__.py
   touch src/data/__init__.py
   ```

   > En Windows (PowerShell):
   > ```powershell
   > New-Item -ItemType File -Path "src\__init__.py"
   > New-Item -ItemType File -Path "src\data\__init__.py"
   > ```

---

### ▶️ Ejecución de la Aplicación

### ✅ Opción Recomendada: Desde la raíz del proyecto

> 📌 **Importante**: Todos los comandos deben ejecutarse desde la carpeta raíz del proyecto (`UAO-Neumonia/`), **no desde `src/interface/`**.

```bash
cd UAO-Neumonia
source .venv/bin/activate    # Linux/macOS
# o
.\.venv\Scripts\activate     # Windows

python -m src.interface.detector_neumonia
```

> ✅ El entorno está activado, `uv` ya instaló todas las dependencias, y el módulo se ejecuta sin errores.






### 🐳 Docker 


```bash
sudo docker build -t neumonia-app .
xhost +local:docker
sudo docker run -it \
  --net=host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  neumonia-app
```


### 🛠️ Troubleshooting (actualizado para `uv`)

| Problema | Solución |
|--------|----------|
| `Command 'uv' not found` | Instala `uv` con `curl -LsSf https://astral.sh/uv/install.sh | sh` o descárgalo desde [GitHub Releases](https://github.com/astral-sh/uv/releases). |
| `ModuleNotFoundError: No module named 'src'` | ❗ Asegúrate de estar en la **raíz del proyecto** (`UAO-Neumonia/`) y que el entorno virtual esté activado (`source .venv/bin/activate`). |
| Error al cargar el modelo | Verifica que `conv_MLP_84.h5` esté en `data/external/models/` y que el nombre sea exacto. |
| `ANTIALIAS` deprecated | Ya está corregido en el código: se usa `Image.LANCZOS`. |
| Grad-CAM no muestra heatmap | Confirma que la capa `"conv10_thisone"` existe en tu modelo. Si usas otro modelo, actualiza el nombre en `src/data/grad_cam.py`. |



## 🖥️ Uso de la Interfaz Gráfica

1. **Ingrese la cédula del paciente** en el campo “Cédula Paciente”.
2. Haga clic en **“Cargar Imagen”** y seleccione un archivo en formato:
   - `.dcm` (DICOM)
   - `.jpg`, `.jpeg`, `.png`
3. Una vez cargada, haga clic en **“Predecir”**.
   - El sistema mostrará:
     - La imagen original en el panel izquierdo.
     - La imagen con mapa de calor Grad-CAM en el panel derecho.
     - La clase predicha: *Bacteriana*, *Viral* o *Normal*.
     - La probabilidad de predicción en %.
4. Para guardar el resultado:
   - **Guardar**: Almacena la cédula, clase y probabilidad en `historial.csv`.
   - **PDF**: Genera un reporte en formato PDF de la pantalla actual.
5. Para limpiar y comenzar de nuevo, use **“Borrar”**.

> 💡 **Nota**: Las imágenes se procesan automáticamente a 512×512 píxeles y en escala de grises. No es necesario preprocesarlas manualmente.

---

## 🧠 Técnica de Explicación: Grad-CAM

### ¿Qué hace Grad-CAM?
Grad-CAM identifica qué regiones de la imagen influyeron más en la decisión del modelo. Usa gradientes de la salida respecto a la última capa convolucional para generar un mapa de calor superpuesto sobre la imagen original.

### ✅ Implementación en este proyecto
- **Usa `tf.GradientTape()`** (TF 2.x moderno) → Reemplaza `K.gradients`
- **No realiza preprocesamiento interno** → Solo usa la imagen ya procesada por `preprocess_img.py`.
- **Inyección de dependencias**: El modelo y la capa convolucional (`conv10_thisone`) son inyectados dinámicamente.
- **Compatible con cualquier modelo TF/Keras** → Solo requiere nombre correcto de la capa final.

### 📁 Archivos clave relacionados:
| Archivo | Rol |
|--------|-----|
| `src/data/preprocess_img.py` | Preprocesa la imagen (resize, CLAHE, normalización, batch). |
| `src/data/load_model.py` | Carga el modelo entrenado (`conv_MLP_84.h5`). |
| `src/data/grad_cam.py` | Genera el heatmap usando gradientes y lo superpone a la imagen original. |



---

## 📚 Modelo CNN Utilizado

Basado en el trabajo de [Pasa et al., 2019](https://arxiv.org/abs/1904.08711):

- **5 bloques convolucionales** con conexiones residuales (skip connections).
- Filtros por bloque: 16 → 32 → 48 → 64 → 80.
- Max Pooling después de cada bloque.
- Capas densas finales: 1024 → 1024 → 3 (clases).
- Regularización: Dropout del 20% en capas 4, 5 y primera densa.
- **Capa de interés para Grad-CAM**: `conv10_thisone` (última convolución antes del clasificador).

> ✅ El modelo fue entrenado en un conjunto de radiografías de tórax públicas y alcanza altos niveles de precisión en clasificación binaria y ternaria.

---

## 📂 Formatos de Entrada Soportados

| Formato | Extensión | Notas |
|--------|----------|-------|
| DICOM | `.dcm` | Autoconvertido a RGB y normalizado |
| JPEG | `.jpg`, `.jpeg` | Convertido a escala de grises |
| PNG | `.png` | Convertido a escala de grises |

> ✅ **Resolución recomendada**: 512×512 píxeles (automáticamente ajustada).

---

## 🧪 Pruebas Unitarias

### Instalar dependencias de prueba
```bash
pip install pytest
```

### Ejecutar pruebas
```bash
pytest tests/
```

### Ejecutar pruebas individuales
```bash
pytest tests/test_historial.py -v
pytest tests/test_imports.py -v
```

> ✅ Las pruebas verifican:
> - Importación correcta de módulos
> - Funcionalidad del historial CSV
> - Formato de salida de funciones clave

---

## 🐳 Docker (Opcional)

### Construir la imagen
```bash
sudo docker build -t neumonia-app .
```

### Ejecutar el contenedor
```bash
# Dar acceso al display X11
xhost +local:docker

# Ejecutar el contenedor
sudo docker run -it \
  --net=host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  neumonia-app
```

### Usar imagen preconstruida desde Docker Hub
```bash
sudo docker pull gaanvalo/neumonia-app:v2
sudo docker run -it \
  --net=host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  gaanvalo/neumonia-app:v2
```

---

## 🛠️ Troubleshooting

| Problema | Solución |
|--------|----------|
| `ModuleNotFoundError: No module named 'src'` | ✅ **Ejecuta siempre desde la raíz del proyecto** (`proyecto-neumonia/`) con `python -m src.interface.detector_neumonia`. Asegúrate de que `src/__init__.py` y `src/data/__init__.py` existan. |
| Error con CUDA | ✅ La aplicación funciona perfectamente en CPU. No es necesario tener GPU. |
| Error al cargar el modelo | ✅ Verifica que `conv_MLP_84.h5` esté en `data/external/models/` y que el nombre coincida exactamente. |
| `IndentationError` o errores de importación | ✅ Usa un editor como VS Code y asegúrate de que no mezcles espacios y tabulaciones. Usa 4 espacios por indentación. |
| Grad-CAM no muestra heatmap | ✅ Verifica que la capa `"conv10_thisone"` exista en tu modelo. Si usas otro modelo, actualiza el nombre en `grad_cam.py`. |

---

## 📄 Estandarización y Buenas Prácticas

- ✅ **PEP8**: Todo el código sigue normas de estilo Python.
- ✅ **Docstrings**: Todas las funciones tienen documentación clara.
- ✅ **Desacoplamiento**: Módulos independientes, sin lógica duplicada.
- ✅ **Modernización**: Uso de `tf.GradientTape()` en lugar de `K.function`.
- ✅ **Extensibilidad**: Fácil reemplazo del modelo o capa de interés.


---

## 👥 Autores

- **Proyecto Original**:  
  Isabella Torres Revelo ([@isa-tr](https://github.com/isa-tr))  
  Nicolás Díaz Salazar ([@nicolasdiazsalazar](https://github.com/nicolasdiazsalazar))

- **Fork**:  
  Carlos Monsalve ([@](https://github.com/carlosmonsalve16))
  Cesar Campos  ([@](https://github.com/c3sarc4mpos))
  Diego Guillen ([@](https://github.com/Dieg0G))
