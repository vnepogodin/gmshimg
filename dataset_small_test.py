#!/usr/bin/python

import numpy as np
from PIL import Image, ImageOps
from tensorflow import keras
import tensorflow as tf

class_names = ['circle', 'rectangle', 'triangle']

print('---- TF config ----')
print('GPU: ', tf.config.list_physical_devices('GPU'))
print('-------------------')

def predict(model, image_path):
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open(image_path).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    return prediction

def display_prediction(predictions):
    for i, class_prob in enumerate(predictions[0]):
        print(f"Class {i} probability: {class_prob * 100:.2f}%")

    class_index = np.argmax(predictions)
    confidence_score = predictions[0][class_index] * 100

    # Print prediction and confidence score
    print(f"\nPredicted class: {class_names[class_index]}")
    print(f"Confidence Score:: {confidence_score:.2f}%")


# Load the trained model
model = keras.models.load_model('gmshimg_small.h5')

# Print model summary
model.summary()

# Make predictions
display_prediction(predict(model, 'predict_images/triangle_1669.jpg'))
display_prediction(predict(model, 'predict_images/rectangle_2424.jpg'))
display_prediction(predict(model, 'predict_images/circle_3828.jpg'))
display_prediction(predict(model, 'predict_images/photo_rectangle.jpg'))
display_prediction(predict(model, 'predict_images/photo_triangle.jpg'))
