import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Configuración
IMG_SIZE = 128  # Tamaño al que se redimensionarán las imágenes
BATCH_SIZE = 32  # Número de imágenes procesadas a la vez
EPOCHS = 10  # Número de épocas de entrenamiento
DATASET_DIR = os.path.join(os.getcwd(), 'dataset')  # Ruta al dataset
MODEL_DIR = os.path.join(os.getcwd(), 'models')  # Carpeta donde se guardará el modelo

# Verificar si la carpeta dataset existe
if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(f"La carpeta dataset no existe en: {DATASET_DIR}")

# Crear directorio para guardar el modelo si no existe
os.makedirs(MODEL_DIR, exist_ok=True)

# Generadores de datos con aumento de datos
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalización de imágenes
    rotation_range=30,  # Rotación aleatoria de las imágenes
    width_shift_range=0.2,  # Desplazamiento horizontal aleatorio
    height_shift_range=0.2,  # Desplazamiento vertical aleatorio
    shear_range=0.2,  # Transformaciones de corte
    zoom_range=0.2,  # Zoom aleatorio
    horizontal_flip=True,  # Inversión horizontal
    validation_split=0.2  # Separar 20% de los datos para validación
)

# Generador para el conjunto de entrenamiento
train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True  # Mezclar las imágenes en cada época
)

# Generador para el conjunto de validación
validation_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False  # No mezclar para validación
)

# Definición del modelo CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Clasificación binaria
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

# Graficar los resultados (opcional)
import matplotlib.pyplot as plt

# Visualización de los resultados del entrenamiento
plt.figure(figsize=(12, 4))

# Precisión
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Precisión (Entrenamiento)')
plt.plot(history.history['val_accuracy'], label='Precisión (Validación)')
plt.title('Precisión del modelo durante el entrenamiento')
plt.xlabel('Número de iteraciones (Épocas)')
plt.ylabel('Proporción de predicciones correctas (Precisión)')
plt.legend()

# Pérdida
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Error (Entrenamiento)')
plt.plot(history.history['val_loss'], label='Error (Validación)')
plt.title('Error del modelo durante el entrenamiento')
plt.xlabel('Número de iteraciones (Épocas)')
plt.ylabel('Diferencia entre predicción y valor real (Pérdida)')
plt.legend()

plt.tight_layout()
plt.show()