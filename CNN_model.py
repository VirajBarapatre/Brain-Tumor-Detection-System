import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Define dataset path
dataset_path = 'Brain_Tumor_Data_Preprocessed'
img_size = (150, 150)
batch_size = 32

# Image Data Generator for Augmentation
datagen = ImageDataGenerator(rescale=1.0/255)

# Load Training and Testing Data
train_generator = datagen.flow_from_directory(
    f'{dataset_path}/train',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary')

test_generator = datagen.flow_from_directory(
    f'{dataset_path}/test',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary')

# CNN Model Building
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Compile the Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model Summary
model.summary()

# Model Training
history = model.fit(train_generator, epochs=10, validation_data=test_generator)

# Save Model
model.save('Brain_Tumor_Model.h5')
print("✅ Model Training Completed and Saved Successfully!")

# Visualization
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
