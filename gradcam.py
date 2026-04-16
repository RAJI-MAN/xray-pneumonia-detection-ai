import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

print("Starting Grad-CAM...")

# -----------------------------
# 1. LOAD MODEL
# -----------------------------
model = load_model("code/model.h5", compile=False)
print("Model loaded")

# -----------------------------
# 2. LOAD IMAGE
# -----------------------------
img_path = "data/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg"  # change if needed
img = cv2.imread(img_path)

if img is None:
    print("Error loading image")
    exit()

print("Image loaded")

# Resize to 224 (IMPORTANT for MobileNetV2)
img = cv2.resize(img, (224, 224))
img = img / 255.0
img_array = np.expand_dims(img, axis=0)

# -----------------------------
# 3. FIND LAST CONV LAYER
# -----------------------------
last_conv_layer_name = None

for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer_name = layer.name
        break

print("Using layer:", last_conv_layer_name)

last_conv_layer = model.get_layer(last_conv_layer_name)

# -----------------------------
# 4. CREATE GRAD MODEL
# -----------------------------
grad_model = tf.keras.models.Model(
    inputs=model.input,
    outputs=[last_conv_layer.output, model.output]
)

# -----------------------------
# 5. COMPUTE GRAD-CAM
# -----------------------------
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_array)
    loss = predictions[:, 0]

grads = tape.gradient(loss, conv_outputs)

pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

conv_outputs = conv_outputs[0]

heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
heatmap = tf.squeeze(heatmap)

# Normalize heatmap
heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

# -----------------------------
# 6. DISPLAY
# -----------------------------
# Load original image again
original = cv2.imread(img_path)
original = cv2.resize(original, (224, 224))

# Resize heatmap
heatmap = cv2.resize(heatmap, (224, 224))
heatmap = np.uint8(255 * heatmap)

# Apply color map
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# Overlay
superimposed = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

# Show results
plt.figure(figsize=(10,4))

plt.subplot(1,3,1)
plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(heatmap)
plt.title("Heatmap")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
plt.title("Grad-CAM")
plt.axis("off")

plt.show()

print("Finished")