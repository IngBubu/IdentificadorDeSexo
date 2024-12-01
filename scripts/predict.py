from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Configuración básica
IMG_SIZE = 128  # Tamaño al que se ajustará la imagen
MODEL_PATH = os.path.join(os.getcwd(), 'models', 'gender_classifier.h5')  # Ruta del modelo

# Carga del modelo entrenado
print(f"Cargando modelo desde: {MODEL_PATH}")
try:
    model = load_model(MODEL_PATH)
    print("Modelo cargado correctamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit(1)

# Función para clasificar una imagen
def classify_image(image_path):
    """
    Clasifica la imagen como 'Hombre' o 'Mujer'.
    """
    try:
        img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))  # Cargar la imagen
        img_array = img_to_array(img) / 255.0  # Normalizar los píxeles
        img_array = np.expand_dims(img_array, axis=0)  # Ajustar dimensiones para el modelo
        prediction = model.predict(img_array)[0][0]  # Realizar predicción
        return "Hombre" if prediction < 0.5 else "Mujer"
    except Exception as e:
        return f"Error al procesar la imagen: {e}"

# Punto de entrada principal
if __name__ == "__main__":
    # Solicitar ruta de la imagen
    image_path = input("Ingresa la ruta de la imagen que deseas clasificar: ").strip()

    # Verificar si la imagen existe
    if not os.path.isfile(image_path):
        print("La ruta de la imagen no existe o no es válida. Por favor, verifica y vuelve a intentarlo.")
    else:
        # Clasificar la imagen
        result = classify_image(image_path)
        print(f"Predicción: {result}")
