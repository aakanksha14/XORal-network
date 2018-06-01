import tensorflow as tf
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils

attributes = [ [0, 0], [0, 1], [1,0], [1, 1] ]

labels = [ [1, 0], [0, 1], [0, 1], [1, 0] ]

#numpy conversion

data = np.array(attributes, 'int64')
target = np.array(labels, 'int64')

#model
model = Sequential()
model.add(Dense(3 , input_shape=(len(attributes[0]),))) 
model.add(Activation('sigmoid')) 
model.add(Dense(len(labels[0])))
model.add(Activation('relu'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

#training
score = model.fit(data, target, epochs=100, verbose=0)
print(score.history)

