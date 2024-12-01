from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Configuración
IMG_SIZE = 128  # Tamaño al que se redimensionará la imagen
MODEL_PATH = os.path.join(os.getcwd(), 'models', 'gender_classifier.h5')  # Ruta al modelo guardado

# Cargar el modelo entrenado
print(f"Cargando modelo desde: {MODEL_PATH}")
try:
    model = load_model(MODEL_PATH)
    print("Modelo cargado correctamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit(1)  # Salir del programa si el modelo no se carga correctamente

def classify_image(image_path):
    """
    Clasifica una imagen como hombre o mujer.
    
    Args:
        image_path (str): Ruta de la imagen a clasificar.

    Returns:
        str: "Hombre" o "Mujer" según la predicción del modelo.
    """
    try:
        # Cargar y preprocesar la imagen
        img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(img) / 255.0  # Normalizar la imagen
        img_array = np.expand_dims(img_array, axis=0)  # Expandir dimensiones para el modelo
        
        # Realizar predicción
        prediction = model.predict(img_array)[0][0]
        return "Hombre" if prediction < 0.5 else "Mujer"
    except Exception as e:
        return f"Error al procesar la imagen: {e}"

if __name__ == "__main__":
    # Solicitar la ruta de la imagen al usuario
    image_path = input("Ingresa la ruta de la imagen que deseas clasificar: ").strip()

    # Verificar si la imagen existe
    if not os.path.isfile(image_path):
        print("La ruta de la imagen no existe o no es válida. Por favor, verifica y vuelve a intentarlo.")
    else:
        # Clasificar la imagen
        result = classify_image(image_path)
        print(f"Predicción: {result}")
