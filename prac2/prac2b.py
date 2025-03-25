import os
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


training = r"C:\Users\DELL\Documents\mlmed\practice2\training_set_pixel_size_and_HC.csv"
testing = r"C:\Users\DELL\Documents\mlmed\practice2\test_set_pixel_size.csv"
img_train = r"C:\Users\DELL\Documents\mlmed\practice2\training_set"
img_test = r"C:\Users\DELL\Documents\mlmed\practice2\test_set"


def load_image(file_path, img_size=(224, 224)):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, img_size)
    img = img / 255.0  
    return img

def load_dataset(csv_path, img_folder, is_train=True):
    df = pd.read_csv(csv_path)
    data, labels = [], []
    img_size = (224, 224)
    for _, row in df.iterrows():
        file_path = os.path.join(img_folder, row['filename'])
        img = load_image(file_path, img_size)
        data.append(img)
        if is_train:
            labels.append(row['head circumference (mm)']) 
    
    data = np.array(data).reshape(-1, img_size[0], img_size[1], 1)
    labels = np.array(labels) if is_train else None
    return data, labels

X_train, y_train = load_dataset(training, img_train, is_train=True)
X_test, _ = load_dataset(testing, img_test, is_train=False)


model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1)  
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mae', metrics=['mae'])

es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_split=0.2, epochs=30, batch_size=16, callbacks=[es])

y_pred = model.predict(X_test)
print("Predicted head circumference values:", y_pred)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()