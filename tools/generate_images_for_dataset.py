#!/bin/python

import numpy as np
import matplotlib.pyplot as plt
import random
import os
import cv2

# Define the number of images to be generated for each class
N_IMAGES_PER_CLASS = 10000
# Define the size of the images
IMAGE_SIZE = (64, 64)
# Define the number of classes
N_CLASSES = 3
# Define the classes
CLASSES = ['circle', 'rectangle', 'triangle']
# Define the label map
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}

# Define the dataset path
dataset_path = 'geometric_shapes_images'

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

# Create directories
for class_title in CLASSES:
    class_path = f"{dataset_path}/{class_title}/"
    if not os.path.isdir(class_path):
        os.makedirs(class_path)

# Generate the images and labels for each class
for class_title in CLASSES:
    for i in range(N_IMAGES_PER_CLASS):
        if class_title == 'circle':
            image = draw_circle(IMAGE_SIZE)
        elif class_title == 'rectangle':
            image = draw_rectangle(IMAGE_SIZE)
        elif class_title == 'triangle':
            image = draw_triangle(IMAGE_SIZE)

        # Save the image to disk
        cv2.imwrite(f"{dataset_path}/{class_title}/{class_title}_{i}.jpg", image.astype(np.uint8))

# Load the dataset
#images = np.load('train-data.npy', allow_pickle=True)
#labels = np.load('train-labels.npy', allow_pickle=True)

# Plot some of the images and labels
#for i in range(10):
#    plt.imshow(images[i])
#    plt.title(CLASSES[labels[i]])
#    plt.savefig(f"{labels[i]}-{i}.png")
#    #plt.show()
