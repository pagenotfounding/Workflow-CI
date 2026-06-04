import os
import numpy as np
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load data dari folder relatif
DATASET_PATH = 'wadai_preprocessing'
X_train = np.load(os.path.join(DATASET_PATH, 'X_train.npy'))
X_val = np.load(os.path.join(DATASET_PATH, 'X_val.npy'))
y_train = np.load(os.path.join(DATASET_PATH, 'y_train.npy'))
y_val = np.load(os.path.join(DATASET_PATH, 'y_val.npy'))

NUM_CLASSES = 7

def build_model():
    model = models.Sequential([
        layers.Input(shape=(256,256,3)),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

mlflow.set_experiment("wadai-classification")
mlflow.tensorflow.autolog()

with mlflow.start_run():
    model = build_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=32)
    print("Training selesai!")
