#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import my_dataset

CLASSES = ['circle', 'rectangle', 'triangle']
IMAGE_SIZE = (64, 64)

print('---- TF config ----')
print('GPU: ', tf.config.list_physical_devices('GPU'))
print('-------------------')

"""
ds, info = tfds.load('my_dataset', with_info=True, as_supervised=True)
train_ds = ds['train']
tfds.as_dataframe(train_ds.cache().take(4), info)
"""

# Pre-process the data
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, (64, 64, 3))
    return image, label
"""
train_ds = train_ds.map(preprocess).batch(32)

# Define the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))

# Compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(train_ds, batch_size=32, epochs=5)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(val_ds)
print("Test accuracy:", test_accuracy)
exit(0)
"""

train_ds, val_ds = tfds.load('my_dataset', split=['train[80%:]', 'train[:20%]'], shuffle_files=True, as_supervised=True)

# Build your input pipeline
train_ds = train_ds.cache().shuffle(1024).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
print(f'train image count {len(train_ds)}')
print(f'test image count {len(val_ds)}')

train_ds = train_ds.map(preprocess).batch(32)
val_ds = val_ds.map(preprocess).batch(32)

#plt.figure(figsize=(10, 10))
#for train_data in ds:
#  image_batch, label_batch = train_data["image"], train_data["label"]
#  print(image_batch.shape)
#  print(label_batch.shape)
#  break

  #print(label)
  #plt.imshow(image[0].numpy().astype("uint8"))
  #plt.title(CLASSES[label])
  #plt.axis("off")
  #print(image, label)

#print(ds)
#df = tfds.as_dataframe(ds.take(10), info)
#print(df)
#print(info.features)

#normalization_layer = layers.Rescaling(1./255)

#normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
#image_batch, labels_batch = next(iter(normalized_ds))
#first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
#print(np.min(first_image), np.max(first_image))

num_classes = len(CLASSES)

"""
model = Sequential([
  layers.Rescaling(1./255, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

# Build the model
model = Sequential([
  layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
  layers.MaxPooling2D((2, 2)),
  layers.Conv2D(64, (3, 3), activation='relu'),
  layers.MaxPooling2D((2, 2)),
  layers.Conv2D(128, (3, 3), activation='relu'),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes, activation='softmax')
])
"""

data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(64,
                                  64,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

model = Sequential([
  data_augmentation,
  layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
  layers.MaxPooling2D((2, 2)),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(64, activation='relu'),
  layers.Dense(10, activation='softmax')
])

#model.build()

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#model.compile(optimizer='adam',
#              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              #loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#              metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
epochs = 20
history = model.fit(
  train_ds,
  validation_data=val_ds,
  batch_size=32,
  epochs=epochs
)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(val_ds)
print('Test accuracy:', test_accuracy)

# Save the model on the test data
model.save('gmshimg.h5')

"""
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
"""

