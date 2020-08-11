import keras
from keras.layers import Dense, Flatten
from os.path import isfile

def build_dqn_model():
    model = keras.models.Sequential()
    model.add(Flatten(input_shape=((80, 80, 1))))
    model.add(Dense(512, activation='relu', kernel_initializer='glorot_normal'))
    model.add(Dense(6, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.summary()
    if isfile('Pong_agent'):
        model.load_weights('Pong_agent')
    return model