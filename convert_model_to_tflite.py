#!/usr/bin/python

import tensorflow as tf
from tensorflow import keras

print('---- TF config ----')
print('GPU: ', tf.config.list_physical_devices('GPU'))
print('-------------------')

# Load the trained model
model = keras.models.load_model('gmshimg.h5')

# Print model summary
model.summary()

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('gmshimg.tflite', 'wb') as f:
  f.write(tflite_model)
