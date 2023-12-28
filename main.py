# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import cv2
from scipy.ndimage import uniform_filter
from keras.layers import Input, Concatenate, Conv2D, Conv2DTranspose, LeakyReLU
from keras.models import Model
import time
from sklearn.metrics import mean_squared_error

# Get Folder Names
dataset_path = "complete_ms_data/"
groundtruth_images = []
image_size = 512
patch_size = 64
stride = 32

# Get a list of subfolders in the dataset path
subfolders = sorted([f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))])

print(subfolders)

# Import All Images
for subfolder in subfolders:
    subfolder_path = os.path.join(dataset_path, subfolder)
    print(subfolder_path)

    hyperspectral_image = np.empty((image_size, image_size, 31), dtype=np.uint8)
    i = 0
    for filename in sorted(os.listdir(subfolder_path)):
        if filename.endswith(".png"):
            image_file = filename
            image_path = os.path.join(subfolder_path, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            hyperspectral_image[:, :, i] = image
            i = i + 1

    for y in range(0, image_size - patch_size + 1, stride):
        for x in range(0, image_size - patch_size + 1, stride):
            patch = hyperspectral_image[y:y+patch_size, x:x+patch_size, :]
            groundtruth_images.append(patch)

groundtruth_arrays = np.array(groundtruth_images)
print("Ground Truth Hyperspectral Images Shape:", groundtruth_arrays.shape)

# Obtain A Low Resolution HSI Image of Shape (8,8,31)
filter_size = (8, 8)
lowres_hsi_images = np.zeros((groundtruth_arrays.shape[0], 8, 8, 31))
for i in range(groundtruth_arrays.shape[0]):
    for j in range(groundtruth_arrays.shape[3]):
        band = groundtruth_arrays[i, :, :, j]
        lowres_band = uniform_filter(band, size=filter_size)
        lowres_hsi_images[i, :, :, j] = lowres_band[::8, ::8]

plt.imshow(lowres_hsi_images[1][:, :, 0], cmap='gray')
print("Low-Resolution HSI Images Shape:", lowres_hsi_images.shape)

# Obtain A High Resolution RGB Image of Shape (64,64,1)
highres_rgb_images = np.zeros((groundtruth_arrays.shape[0], 64, 64, 1))
for i in range(groundtruth_arrays.shape[0]):
    highres_rgb_images[i, :, :, 0] = np.mean(groundtruth_arrays[i], axis=-1)[:64, :64]

plt.imshow(highres_rgb_images[1][:, :, 0], cmap='gray')
print("High-Resolution Ground Truth Images Shape:", highres_rgb_images.shape)

# Split Images To Form A Test And Train Set
train_ratio = 0.8
num_samples = groundtruth_arrays.shape[0]
num_train_samples = int(train_ratio * num_samples)
train_lowres_hsi = lowres_hsi_images[:num_train_samples]
train_highres_rgb = highres_rgb_images[:num_train_samples]
test_lowres_hsi = lowres_hsi_images[num_train_samples:]
test_highres_rgb = highres_rgb_images[num_train_samples:]

train_lowres_hsi = train_lowres_hsi / 255.0
train_highres_rgb = train_highres_rgb / 255.0
test_lowres_hsi = test_lowres_hsi / 255.0
test_highres_rgb = test_highres_rgb / 255.0

def create_super_resolution_model(input_shape_hr, input_shape_lr, output_shape):
    hr_input = Input(shape=input_shape_hr)
    lr_input = Input(shape=input_shape_lr)

    upsampled_lr = Conv2DTranspose(1, (8, 8), strides=(8, 8), padding='same')(lr_input)
    concat = Concatenate()([hr_input, upsampled_lr])

    x = Conv2D(64, (3, 3), strides=1, padding='same')(concat)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(128, (3, 3), strides=1, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(256, (3, 3), strides=1, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2DTranspose(128, (3, 3), strides=1, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2DTranspose(64, (3, 3), strides=1, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    output = Conv2D(1, (3, 3), strides=1, padding='same', activation='linear')(x)

    model = Model(inputs=[hr_input, lr_input], outputs=output)
    return model

input_shape_hr = (64, 64, 1)
input_shape_lr = (8, 8, 31)
output_shape = (64, 64, 1)

model = create_super_resolution_model(input_shape_hr, input_shape_lr, output_shape)

print(model.summary())
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model_1_visualization.png', show_shapes=True, show_layer_names=True)

model.compile(optimizer='adam', loss='mse')

model.fit([train_highres_rgb, train_lowres_hsi], train_highres_rgb,
          validation_data=([test_highres_rgb, test_lowres_hsi], test_highres_rgb),
          epochs=10, batch_size=32)

model.save('model.h5')

from keras.models import load_model
loaded_model = load_model('model.h5')
num_visualize = 6
visualize_indices = np.random.choice(len(subfolders), min(num_visualize, len(subfolders)), replace=False)

for i in visualize_indices:
    input_lowres = test_lowres_hsi[i:i+1]
    input_highres = test_highres_rgb[i:i+1]
    start_time = time.time()

    predicted_highres = model.predict([input_highres, input_lowres])

    end_time = time.time()
    elapsed_time = end_time - start_time

    rmse = np.sqrt(mean_squared_error(input_highres.flatten(), predicted_highres.flatten()))

    predicted_highres_512 = cv2.resize(predicted_highres[0][:, :, 0], (512, 512))

    # Get the folder name for the current sample
    folder_name = subfolders[i]

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(input_lowres[0][:, :, 0], cmap='gray')
    plt.title("Low Resolution HSI")

    plt.subplot(1, 3, 2)
    plt.imshow(input_highres[0][:, :, 0], cmap='gray')
    plt.title("High Resolution Ground Truth")

    plt.subplot(1, 3, 3)
    plt.imshow(predicted_highres_512, cmap='gray')
    plt.title(f"{folder_name} - Pred (RMSE: {rmse:.4f}, Time: {elapsed_time:.4f} seconds)")

    plt.show()
