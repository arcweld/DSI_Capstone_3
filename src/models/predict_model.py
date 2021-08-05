import sys
import os

import tensorflow as tf

def restore_model(model_path='model/save_weights'):
    path = os.path.join(model_path)
    restored_model = tf.keras.models.load_model(path)
    return restored_model

if __name__ == "__main__":
    model = restore_model():
    model.summary()
