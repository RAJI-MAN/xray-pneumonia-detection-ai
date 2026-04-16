from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# -----------------------------
# PATHS
# -----------------------------
base_dir = os.path.join("data", "chest_xray")
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

IMG_SIZE = 224   # 🔥 better for MobileNet
BATCH_SIZE = 32

# -----------------------------
# DATA GENERATORS (IMPORTANT FIX)
# -----------------------------
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=10,
    zoom_range=0.1
)

val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_data = val_gen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False   # 🔥 VERY IMPORTANT
)

# -----------------------------
# MODEL (TRANSFER LEARNING + FINE-TUNING)
# -----------------------------
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

# 🔥 fine-tune last 20 layers
for layer in base_model.layers[-5:]:
    layer.trainable = True

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(64, activation='relu')(x)
output = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs=base_model.input, outputs=output)

# -----------------------------
# COMPILE
# -----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# TRAIN
# -----------------------------
class_weight = {0: 1.5, 1: 1.0}

model.fit(
    train_data,
    validation_data=val_data,
    epochs=5,
    class_weight=class_weight
)

# -----------------------------
# EVALUATE
# -----------------------------
loss, acc = model.evaluate(test_data)
print("Test Accuracy:", acc)

# -----------------------------
# SINGLE IMAGE TEST
# -----------------------------
img_path = "data/chest_xray/test/NORMAL/IM-0001-0001.jpeg"

img = cv2.imread(img_path)

if img is None:
    print("Error loading image")
else:
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    print("Prediction value:", prediction[0][0])

    if prediction[0][0] > 0.8:
        print("PNEUMONIA detected")
    else:
        print("NORMAL")

# -----------------------------
# FULL DATASET PREDICTION (ONLY ONCE)
# -----------------------------
Y_pred = model.predict(test_data)
y_pred = (Y_pred > 0.8).astype(int)
y_true = test_data.classes

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_true, y_pred))

# -----------------------------
# FAILURE ANALYSIS
# -----------------------------
file_paths = test_data.filepaths

false_negatives = []
false_positives = []

for i in range(len(y_true)):
    if y_true[i] == 1 and y_pred[i] == 0:
        false_negatives.append(file_paths[i])
    elif y_true[i] == 0 and y_pred[i] == 1:
        false_positives.append(file_paths[i])

print("False Negatives:", len(false_negatives))
print("False Positives:", len(false_positives))

# -----------------------------
# SHOW ERRORS
# -----------------------------
print("\nShowing False Negatives:")
for i in range(min(5, len(false_negatives))):
    img = plt.imread(false_negatives[i])
    plt.imshow(img, cmap='gray')
    plt.title("FALSE NEGATIVE (Missed Pneumonia)")
    plt.axis('off')
    plt.show()

print("\nShowing False Positives:")
for i in range(min(5, len(false_positives))):
    img = plt.imread(false_positives[i])
    plt.imshow(img, cmap='gray')
    plt.title("FALSE POSITIVE (Wrong Detection)")
    plt.axis('off')
    plt.show()
    print("Test Accuracy:", acc)


model.save("code/model.h5")
print("Model saved successfully")
