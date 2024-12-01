import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Configuración
IMG_SIZE = 128  # Tamaño esperado por el modelo
MODEL_PATH = 'models/gender_classifier.h5'  # Ruta al modelo guardado

# Cargar el modelo entrenado
print("Cargando modelo...")
model = load_model(MODEL_PATH)
print("Modelo cargado correctamente.")

# Función para procesar los cuadros de la cámara
def preprocess_frame(frame):
    """
    Preprocesa un cuadro capturado desde la cámara para ser clasificado por el modelo.

    Args:
        frame (numpy.ndarray): Cuadro capturado desde la cámara.

    Returns:
        numpy.ndarray: Imagen redimensionada y normalizada lista para el modelo.
    """
    frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))  # Redimensionar
    frame_normalized = frame_resized / 255.0  # Normalizar los píxeles
    frame_expanded = np.expand_dims(frame_normalized, axis=0)  # Expandir dimensiones
    return frame_expanded

# Captura de video desde la cámara
cap = cv2.VideoCapture(0)  # Usa la cámara predeterminada del sistema

print("Presiona 'q' para salir.")
while True:
    # Leer el cuadro actual de la cámara
    ret, frame = cap.read()
    if not ret:
        print("No se pudo acceder a la cámara.")
        break

    # Procesar el cuadro y realizar predicción
    input_data = preprocess_frame(frame)
    prediction = model.predict(input_data)[0][0]

    # Determinar la clase
    label = "Mujer" if prediction >= 0.5 else "Hombre"

    # Mostrar la predicción en el cuadro
    cv2.putText(frame, f"Prediccion: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Clasificador de Genero", frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()
