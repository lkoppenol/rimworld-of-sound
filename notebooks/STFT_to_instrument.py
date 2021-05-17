import numpy as np
import os
import tensorflow as tf
import pathlib
import sys
import keras

batch_size = 16
img_height = 126
img_width = 1025
nr_of_labels = 11

data_dir = r"D:\Projects\nsynth-data\data\stft\train"
train_data_dir = pathlib.Path(data_dir)
test_data_dir = r"D:\Projects\nsynth-data\data\stft\test"
test_data_dir = pathlib.Path(test_data_dir)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_data_dir,
    label_mode = 'categorical',
    image_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode = 'grayscale'
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_data_dir,
    label_mode = 'categorical',
    image_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode = 'grayscale'
)

model = keras.Sequential([
    keras.layers.Input(shape=(126, 1025, 1)),
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(3, 3)),
    keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(3, 3)),
	keras.layers.Dropout(0.1),
    keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(3, 3)),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(nr_of_labels, activation='softmax'),
])

opt = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics='categorical_accuracy')

model.summary()
history = model.fit(x=train_ds, epochs=10, batch_size=batch_size, validation_data=test_ds, verbose=1)