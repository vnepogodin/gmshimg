#!/usr/bin/python

import numpy as np
import PIL
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow import keras

import my_dataset

class_names = ['circle', 'rectangle', 'triangle']
IMAGE_SIZE = (64, 64)


print('---- TF config ----')
print('GPU: ', tf.config.list_physical_devices('GPU'))
print('-------------------')


# Pre-process the data
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, (64, 64, 3))
    return image, label

def predict(model, image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(64, 64))
    x = tf.keras.utils.img_to_array(img)
    x = tf.reshape(x, (64, 64, 3))
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    return preds

def display_prediction(predictions):
    for i, class_prob in enumerate(predictions[0]):
        print(f"Class {i} probability: {class_prob * 100:.2f}%")

    class_index = np.argmax(predictions)
    prediction_percentage = predictions[0][class_index] * 100
    print(f"\nPredicted class: {class_names[class_index]}")
    print(f"Prediction percentage: {prediction_percentage:.2f}%")


# Load the trained model
model = keras.models.load_model('gmshimg.h5')

# Print model summary
model.summary()

test_ds, metadata = tfds.load('my_dataset', split='train[:20%]', with_info=True, shuffle_files=True, as_supervised=True)

# Build your train pipeline
test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)
print(f'test image count {len(test_ds)}')

test_ds = test_ds.map(preprocess).batch(32)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_ds)
print('Test accuracy: {:5.2f}%'.format(100 * test_accuracy))

# Make predictions
display_prediction(predict(model, 'predict_images/triangle_1669.jpg'))
display_prediction(predict(model, 'predict_images/rectangle_2424.jpg'))
display_prediction(predict(model, 'predict_images/circle_3828.jpg'))

# Display the percentage of predictions for each class
#print(predictions[0])
