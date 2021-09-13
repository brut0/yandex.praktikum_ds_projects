from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import numpy as np


def load_train(path):
    train_datagen = ImageDataGenerator(rescale=1/255.,
                                      horizontal_flip=True,
                                      vertical_flip=True)
 
    train_datagen_flow = train_datagen.flow_from_directory(
      path,
      target_size=(150, 150),
      batch_size=16,
      class_mode='sparse',
      subset='training',
      seed=12345)

    return train_datagen_flow


def create_model(input_shape):
    n_cat = 12

    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(1, 1), padding='same',
                     activation="relu", input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3),
                     activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3),
                     activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3),
                     activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
 
    model.add(Flatten())
    model.add(Dense(units=86, activation='relu'))
    model.add(Dense(units=24,  activation='relu'))
    model.add(Dense(units=n_cat,  activation='softmax'))

    optimizer = Adam(lr=0.001) 
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['acc'])

    return model

def train_model(model, train_data, test_data, batch_size=None, epochs=11,
            steps_per_epoch=None, validation_steps=None):

    model.fit(train_data, 
              validation_data=test_data,
              batch_size=batch_size, epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2)

    return model 

