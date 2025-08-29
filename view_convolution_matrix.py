# view_convolution_matrix.py

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Step 1: Load the trained model
print("[INFO] Loading trained mask detection model...")
model = load_model("mask_detector.model")  # Make sure this file exists

# Step 2: List all convolutional layers
print("\n[INFO] Listing convolutional layers...")
conv_layers = []
for i, layer in enumerate(model.layers):
    if 'conv' in layer.name.lower():
        conv_layers.append(layer)
        print(f"{i}. Layer Name: {layer.name} | Type: {layer.__class__.__name__}")

if not conv_layers:
    print("[ERROR] No convolutional layers found.")
    exit()

# Step 3: Search for the first layer with valid weights
found = False
for layer in conv_layers:
    weights = layer.get_weights()
    if len(weights) == 2:
        filters, biases = weights
        found = True
        break
    elif len(weights) == 1:
        filters = weights[0]
        biases = None
        found = True
        break

if not found:
    print("[ERROR] No valid convolutional layer with weights found.")
    exit()

# Step 4: Display filter information
print(f"\n[INFO] Inspecting Layer: {layer.name}")
print(f"Filter shape: {filters.shape}")  # e.g., (3, 3, 3, 32)
if biases is not None:
    print(f"Bias shape: {biases.shape}")

# Step 5: Save filters to a file (optional)
np.save("convolution_filters.npy", filters)
print("[INFO] Saved filters to convolution_filters.npy")

# Step 6: Visualize first few filters from first input channel (e.g., Red)
print("\n[INFO] Visualizing and saving first 6 filters (channel 0)...")
n_filters = min(6, filters.shape[3])  # total filters in output
for i in range(n_filters):
    f = filters[:, :, 0, i]  # Visualize filter from first input channel
    f_min, f_max = f.min(), f.max()
    f = (f - f_min) / (f_max - f_min + 1e-8)  # Normalize

    plt.figure(figsize=(4, 4))
    plt.imshow(f, cmap='viridis')
    plt.title(f"{layer.name} - Filter #{i}", fontsize=14)
    plt.axis('off')
    plt.text(0.5, -0.1, f"Filter #{i}", fontsize=12, ha='center', transform=plt.gca().transAxes)

    # Save each filter as image
    plt.savefig(f"filter_{i}.png")
    plt.show()

print("✅ All filters visualized and saved successfully!")
