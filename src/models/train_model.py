import os
import datetime

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential

def define_model():
    model = Sequential()
    model.add(Conv2D(16, (3,3), activation='relu', input_shape=(28,28,4)))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model



def train_model(model, X, y, X_test, y_test, tb=True, cp=True):

    model.fit(X, y, batch_size=200, epochs=100, verbose=1, validation_data=(X_test, y_test), callbacks=[tbcallback, cp_callback])
    model.save_weights(os.path.join('..','..','models','model_weights'))

def define_tensorboard():
    log_dir = os.path.join('..',"logs/fit/",
                            datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tb_callback = TensorBoard(log_dir=log_dir,
                              histogram_freq=1,
                              write_graph=True)
    return tb_callback

def define_checkpoint():
    checkpoint_path = os.path.join('..','models','checkpoints.ckpt')
    cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                  save_weights_only=True,
                                  verbose=1)
    return cp_callback


if __name__ == "__main__":
    model = define_model()
    model.summary()
    define_tensorboard()
    define_checkpoint()
