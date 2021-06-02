import os
from tensorflow import keras
from rimworld.utils import *
from dotenv import load_dotenv
import time

load_dotenv()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DEBUG = False

ROOT_FOLDER = os.getenv('ROOT_FOLDER')
if DEBUG:
    print(ROOT_FOLDER)

LABEL = "organ_pitcher"

BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 1e-5
modelsavename = LABEL + '_classifier_' + str(int(time.time()))
if DEBUG:
    print(modelsavename)

model = keras.Sequential([
    keras.layers.Input(shape=(126, 1025, 1)),
    keras.layers.Conv2D(8, kernel_size=(3, 3), activation='swish', padding="SAME"),
    keras.layers.Conv2D(8, kernel_size=(3, 3), activation='swish', padding="SAME"),
    keras.layers.MaxPooling2D(pool_size=(3, 10)),
    keras.layers.Conv2D(16, kernel_size=(3, 3), activation='swish', padding="SAME"),
    keras.layers.Conv2D(16, kernel_size=(3, 3), activation='swish', padding="SAME"),
    keras.layers.MaxPooling2D(pool_size=(3, 10)),
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation='swish', padding="SAME"),
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation='swish', padding="SAME"),
    keras.layers.Flatten(),
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