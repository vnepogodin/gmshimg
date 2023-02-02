#!/bin/python

import numpy as np
import matplotlib.pyplot as plt
import random
import cv2

# Define the number of images to be generated for each class
N_IMAGES_PER_CLASS = 4000
# Define the size of the images
IMAGE_SIZE = (64, 64)
# Define the number of classes
N_CLASSES = 3
# Define the classes
CLASSES = ['circle', 'rectangle', 'triangle']
# Define the label map
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}

# Define the function to draw a random circle
def draw_circle(image_size):
    center = (random.randint(0, image_size[0]), random.randint(0, image_size[1]))
    radius = random.randint(10, 30)
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    thickness = random.randint(1, 5)
    image = np.zeros(image_size + (3,), dtype=np.uint8)
    image = cv2.circle(image, center, radius, color, thickness)
    return image

# Define the function to draw a random rectangle
def draw_rectangle(image_size):
    top_left = (random.randint(0, image_size[0]), random.randint(0, image_size[1]))
    bottom_right = (random.randint(0, image_size[0]), random.randint(0, image_size[1]))
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    thickness = random.randint(1, 5)
    image = np.zeros(image_size + (3,), dtype=np.uint8)
    image = cv2.rectangle(image, top_left, bottom_right, color, thickness)
    return image

# Define the function to draw a random triangle
def draw_triangle(image_size):
    pt1 = (random.randint(0, image_size[0]), random.randint(0, image_size[1]))
    pt2 = (random.randint(0, image_size[0]), random.randint(0, image_size[1]))
    pt3 = (random.randint(0, image_size[0]), random.randint(0, image_size[1]))
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    thickness = random.randint(1, 5)
    image = np.zeros(image_size + (3,), dtype=np.uint8)
    image = cv2.polylines(image, [np.array([pt1, pt2, pt3])], True, color, thickness)
    return image

# Generate dataset
data = []
labels = []

# Generate the images and labels for each class
for shape, label in LABEL_MAP.items():
    for i in range(N_IMAGES_PER_CLASS):
        if shape == 'circle':
            image = draw_circle(IMAGE_SIZE)
        elif shape == 'rectangle':
            image = draw_rectangle(IMAGE_SIZE)
        elif shape == 'triangle':
            image = draw_triangle(IMAGE_SIZE)

        # Appened to the numpy arrays
        data.append(image)
        labels.append(label)

data = np.array(data, dtype=np.uint8)
labels = np.array(labels, dtype=np.int32)

# Save the dataset
np.save("train-data.npy", data)
np.save("train-labels.npy", labels)
