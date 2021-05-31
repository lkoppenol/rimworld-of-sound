import tensorflow as tf
from keras import layers
from loguru import logger

def _one_hot(tensor: tf.Tensor, size) -> tf.Tensor:
    """
    One hot encode a tensor and return it as 1D tensor
    :param tensor:
    :param size: number of unique values in tensor
    :return:
    """
    hot_tensor = tf.one_hot(tensor, size)
    shaped_tensor = tf.reshape(hot_tensor, (size,))
    return shaped_tensor

@tf.autograph.experimental.do_not_convert
def _parse_function(example_proto):
    # Schema
    features = {
        "pitch": tf.io.FixedLenFeature([1], dtype=tf.int64),
        "audio": tf.io.FixedLenFeature([64000], dtype=tf.float32),
        "velocity": tf.io.FixedLenFeature([1], dtype=tf.int64),
    }
    example = tf.io.parse_single_example(example_proto, features)
    data = example['audio']
    label_name = 'pitch'
    label_value_count = 128
    # label_name = 'velocity'
    # label_value_count = 4
    label = _one_hot(example[label_name], label_value_count)
    return data, label

def run():
    DATA_PATH = '../data/tfrecords/nsynth-test.tfrecord'
    batch_size = 32
    audio_length = 64_000
    parsed_dataset = tf.data \
        .TFRecordDataset(DATA_PATH) \
        .map(_parse_function) \
        .batch(batch_size)
    # print(parsed_dataset.shapes )
    model = tf.keras.Sequential([
        layers.Input(shape=(audio_length,), batch_size=batch_size),
        layers.Reshape(target_shape=(audio_length, 1)),
        layers.Conv1D(32, 10, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 10, activation='relu'),
        layers.MaxPooling1D(256),
        layers.Flatten(),
        layers.Dense(128, activation='softmax')
        layers.Activation
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.Accuracy()]
    )
    model.summary()
    model.fit(parsed_dataset, epochs=50)
if __name__ == "__main__":
    run()