import keras
from keras.layers import Dense, Flatten
import os

def build_dqn_model():
    model = keras.models.Sequential()
    model.add(Flatten(input_shape=((88, 80, 1))))
    model.add(Dense(256, activation='relu', kernel_initializer='glorot_normal')) # 512
    model.add(Dense(9, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.summary()
    if os.path.isfile('Pacman_agent'):
        model.load_weights('Pacman_agent')
    return model