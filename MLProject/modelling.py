import os
import numpy as np
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras import layers, models

NUM_CLASSES = 7

# Gunakan data dummy
print("Menggunakan data dummy...")
X_train = np.random.rand(70, 64, 64, 3).astype('float32')
X_val = np.random.rand(30, 64, 64, 3).astype('float32')
y_train = np.random.randint(0, NUM_CLASSES, 70)
y_val = np.random.randint(0, NUM_CLASSES, 30)

def build_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

mlflow.tensorflow.autolog()

model = build_model(X_train.shape[1:])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=3, batch_size=16)
print("Training selesai!")
