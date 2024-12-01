import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Configuración básica
IMG_SIZE = 128  # Tamaño de las imágenes
BATCH_SIZE = 32  # Tamaño del lote
EPOCHS = 10  # Número de épocas
DATASET_DIR = os.path.join(os.getcwd(), 'dataset')  # Ruta al dataset
MODEL_DIR = os.path.join(os.getcwd(), 'models')  # Carpeta para guardar el modelo

# Verificar que el dataset exista
if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(f"No se encontró la carpeta: {DATASET_DIR}")

# Crear la carpeta para guardar el modelo si no existe
os.makedirs(MODEL_DIR, exist_ok=True)

# Generadores de datos (entrenamiento y validación)
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalización de píxeles
    rotation_range=30,  # Rotación aleatoria
    width_shift_range=0.2,  # Desplazamiento horizontal
    height_shift_range=0.2,  # Desplazamiento vertical
    shear_range=0.2,  # Transformación de corte
    zoom_range=0.2,  # Zoom aleatorio
    horizontal_flip=True,  # Volteo horizontal
    validation_split=0.2  # División para validación (20%)
)

# Generador para datos de entrenamiento
train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

# Generador para datos de validación
validation_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# Construcción del modelo
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),  # Primera capa convolucional
    MaxPooling2D((2, 2)),  # Primera capa de agrupamiento
    Conv2D(64, (3, 3), activation='relu'),  # Segunda capa convolucional
    MaxPooling2D((2, 2)),  # Segunda capa de agrupamiento
    Conv2D(128, (3, 3), activation='relu'),  # Tercera capa convolucional
    MaxPooling2D((2, 2)),  # Tercera capa de agrupamiento
    Flatten(),  # Aplanar datos para entrada a capa densa
    Dense(128, activation='relu'),  # Capa completamente conectada
    Dropout(0.5),  # Regularización
    Dense(1, activation='sigmoid')  # Capa de salida (clasificación binaria)
])

# Compilación del modelo
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Entrenamiento del modelo
print("Iniciando entrenamiento...")
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS
)

# Guardar el modelo entrenado
model_path = os.path.join(MODEL_DIR, 'gender_classifier.h5')
model.save(model_path)
print(f"Modelo guardado en: {model_path}")

# Visualización de los resultados del entrenamiento
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

# Precisión
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Precisión (Entrenamiento)')
plt.plot(history.history['val_accuracy'], label='Precisión (Validación)')
plt.title('Precisión del modelo durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()

# Pérdida
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Error (Entrenamiento)')
plt.plot(history.history['val_loss'], label='Error (Validación)')
plt.title('Error del modelo durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

plt.tight_layout()
plt.show()
