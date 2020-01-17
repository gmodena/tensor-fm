import numpy as np
import tensorflow as tf

TF_DATASET_BATCH_SIZE = 5000

def to_tf_dataset(
        X, y, dtype=tf.float32, batch_size=TF_DATASET_BATCH_SIZE, shuffle_buffer_size=None
):
    X = np.array(X)
    y = np.array(y)

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            " The number of training "
            "examples is not the same as the number of "
            "labels"
        )
    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(X, dtype=dtype), tf.cast(y, dtype=dtype))
    ).batch(batch_size)

    if shuffle_buffer_size:
        dataset = dataset.shuffle(shuffle_buffer_size)

    return dataset


def to_tf_tensor(X, dtype=tf.float32):
    X = np.array(X)
    return tf.cast(X, dtype=dtype)

