import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # to ignore warning errors
import timeit
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

m = 200
N = 3
w = np.random.randn(m, m).astype('float32')
b = np.random.randn(m, ).astype('float32')

model = Sequential()
model.add(Dense(m, input_shape = (N, m, m, N), activation='linear'))
model.layers[0].set_weights([w.T, b])
model.compile(loss='mse', optimizer='adam')

# state = np.array([-23.5, 17.8])
state = np.random.randn(N, m, m).astype('float32')

def predict_very_slow():
    return model.predict(state[np.newaxis])[0]

def predict_slow():
    ws = model.layers[0].get_weights()
    return np.matmul(ws[0].T, state) + ws[1]

def predict_fast():
    return np.matmul(w, state) + b

print(
     timeit.timeit(predict_very_slow, number=100),
     timeit.timeit(predict_slow, number=100),
     timeit.timeit(predict_fast, number=100)
     )
