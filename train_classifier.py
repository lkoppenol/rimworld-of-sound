import os
from tensorflow import keras
import tensorflow as tf
from rimworld.utils import *
from dotenv import load_dotenv

load_dotenv()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DEBUG = True

ROOT_FOLDER = os.getenv('ROOT_FOLDER')
if DEBUG:
    print(ROOT_FOLDER)
LABEL = 'instrument_and_pitch_single_label'
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-5
modelsavename = LABEL + '_classifier_' + str(int(time.time()))
if DEBUG:
    print(modelsavename)

print('Dont forget to chance activation and metrics check comments for hints')

"""
MULTILABEL

metric = tf.keras.metrics.Precision(thresholds=None, top_k=None, class_id=None, name=None, dtype=None)
adam = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=metric)


SINGLE LABEL ONEHOT ENCODE
adam = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics='categorical_accuracy')
"""

model = keras.Sequential([
    keras.layers.Input(shape=(126, 1025, 1)),
    keras.layers.Conv2D(8, kernel_size=(5, 10), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(3, 10)),
    keras.layers.Conv2D(16, kernel_size=(5, 10), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(3, 10)),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(label_shapes[LABEL], activation='softmax'),
])

model.summary()

train_folder = os.path.join(ROOT_FOLDER, 'train')
train_dataset = get_image_dataset(train_folder, LABEL, label_shapes[LABEL], BATCH_SIZE)
valid_folder = os.path.join(ROOT_FOLDER, 'valid')
valid_dataset = get_image_dataset(valid_folder, LABEL, label_shapes[LABEL], BATCH_SIZE)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="checkpoints/" + modelsavename,
    save_weights_only=True,
    monitor='val_loss',
    mode='max',
    save_best_only=True)

adam = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics='categorical_accuracy')
model.fit(train_dataset, epochs=EPOCHS, validation_data=valid_dataset,
          callbacks=[model_checkpoint_callback, early_stop])

model.save(modelsavename)
