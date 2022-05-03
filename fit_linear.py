# Inspiration: Keith Galli's Intro to Neural Nets
# https://www.youtube.com/watch?v=aBIGJeHRZLQ

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

train_df = pd.read_csv('./datasets/linear/train.csv')
np.random.shuffle(train_df.values)

print(train_df.head())

model = keras.Sequential([
	keras.layers.Dense(4, input_shape=(2,), activation='relu'),
	keras.layers.Dense(2, activation='sigmoid')])

model.compile(optimizer='adam', 
	          loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	          metrics=['accuracy'])

X = np.column_stack((train_df.x0.values, train_df.x1.values))

model.fit(X, train_df.label.values, batch_size=4, epochs=5)

test_df = pd.read_csv('./datasets/linear/test.csv')
test_x = np.column_stack((test_df.x0.values, test_df.x1.values))

print("EVALUATION")
model.evaluate(test_x, test_df.label.values)