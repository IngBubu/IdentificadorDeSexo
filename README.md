# 🧑‍🤖 Identificador de Género

Este proyecto es un **clasificador de género** basado en imágenes. Utiliza un modelo de red neuronal convolucional (CNN) entrenado con **TensorFlow/Keras** para predecir si una imagen corresponde a un "Hombre" o "Mujer".

## 🌟 Características

- **Entrenamiento**: Entrena un modelo con tus propios datos organizados en carpetas.
- **Clasificación**: Clasifica imágenes individuales con el modelo entrenado.
- **Predicción en Tiempo Real**: Usa la cámara web para realizar predicciones en vivo.
- **Soporte para múltiples formatos**: Compatible con imágenes en **JPEG**, **PNG**, **BMP**, **GIF**, entre otros.
- **Fácil de Usar**: Scripts simples y personalizables para entrenar y probar el modelo.

## 📂 Estructura Del Proyecto 

![Captura de pantalla del proyecto](assets/EstructuraDelProyecto.png)

## 🛠️ Requisitos

1. **Python 3.9 o superior**.
2. Librerías necesarias:
   - TensorFlow
   - OpenCV
   - Numpy
   - Matplotlib

   Instálalas ejecutando:
   ```bash
   pip install tensorflow opencv-python numpy matplotlib
3. Asegúrate de organizar tus imágenes en estas carpetas:
    * dataset/hombres/: Imágenes de hombres.
    * dataset/mujeres/: Imágenes de mujeres.
4. Formatos de imagen soportados:

    * JPEG (.jpg, .jpeg)
    * PNG (.png)
    * BMP (.bmp)
    * GIF (.gif) (solo el primer cuadro si es animado)
    * TIFF (.tiff)

## 🚀 Cómo Usar

1️⃣ Entrenar el Modelo
Asegúrate de que las carpetas dataset/hombres y dataset/mujeres contengan suficientes imágenes para entrenar (recomendado: al menos 100 por clase).
1. Activa el entorno virtual:
    ```bash
    .\venv\Scripts\activate
2. Ejecuta el script de entrenamiento:
    ``bash
    python scripts/train.py
3. El modelo entrenado se guardará automáticamente en models/gender_classifier.h5.

2️⃣ Clasificar una Imagen
1. Ejecuta el script de predicción:
    python scripts/predict.py
2. Proporciona la ruta de la imagen a clasificar:
    Ingresa la ruta de la imagen que deseas clasificar: dataset/random/prueba.jpg
3. Verás la predicción en la terminal:
    ```bash
    Predicción: Mujer

3️⃣ Predicciones en Tiempo Real (Cámara)
1. Asegúrate de que tu cámara esté conectada.
2. Ejecuta el script de predicción con cámara
    ```bash
    python scripts/webcam_predict.py
3. Una ventana aparecerá mostrando el video en tiempo real y la predicción ("Hombre" o "Mujer") sobre cada cuadro.
4. Presiona q para salir.

4️⃣ Clasificar Varias Imágenes
Si tienes una carpeta con imágenes aleatorias:

1. Usa un script como batch_predict.py (si lo configuraste).
2. Ejecuta el script y proporciona la ruta de la carpeta:
    ```bash
    python scripts/batch_predict.py
3. Clasificará todas las imágenes en la carpeta y mostrará los resultados.

❗ Problemas Comunes
El modelo no se encuentra:

Asegúrate de que el archivo gender_classifier.h5 esté en la carpeta models/.
Predicciones incorrectas:

Aumenta la cantidad de imágenes en ambas clases para mejorar el entrenamiento.
Considera usar Transfer Learning si las predicciones siguen siendo poco precisas.
Error con la cámara:

Verifica que OpenCV esté correctamente instalado y que tu cámara esté configurada.

Convierte las imágenes a formatos como JPEG o PNG antes de usarlas.
📈 Mejoras Futuras
Transfer Learning: Usa modelos preentrenados como MobileNet o ResNet para mejorar la precisión con menos datos.
Aumento de Datos (Data Augmentation): Genera más datos de entrenamiento con técnicas de rotación, zoom, y espejado.
Interfaz Gráfica: Agrega un GUI para facilitar la interacción con el modelo.
Despliegue: Lleva el modelo a una aplicación web o móvil.
🧑‍💻 Contribuciones
Si deseas contribuir a este proyecto, eres bienvenido. Puedes agregar nuevas funcionalidades, mejorar el rendimiento o sugerir cambios.

📝 Licencia
Este proyecto es de uso libre para fines educativos y experimentales. Si decides usarlo en un proyecto mayor, por favor, da el crédito correspondiente.

¡Gracias por usar el Identificador de Género! 🎉

