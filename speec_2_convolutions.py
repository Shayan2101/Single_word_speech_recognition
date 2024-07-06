import tensorflow as tf
from keras._tf_keras.keras.utils import to_categorical
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LSTM
from keras._tf_keras.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from preproces import *

# Save data to array file first
max_len =11
buckets =20
save_data_to_array(max_len=11, n_mfcc=20)
labels=["bed", "bird", "cat" , "dog","down"]
# Loading train set and test set
X_train, X_test, y_train, y_test = get_train_test()
# Feature dimension
channels = 1
epochs = 50
batch_size = 100

num_classes = 5

X_train = X_train.reshape(X_train.shape[0], buckets, max_len, channels)
X_test = X_test.reshape(X_test.shape[0], buckets, max_len, channels)
print(X_train[100])

plt.imshow(X_train[50, :, :, 0])
plt.show()
print(y_train[50])

y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)
X_train = X_train.reshape(X_train.shape[0], buckets, max_len)
X_test = X_test.reshape(X_test.shape[0], buckets, max_len)

model = Sequential()
model.add(Flatten(input_shape=(buckets, max_len)))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss="categorical_crossentropy",
                optimizer="adam",
                metrics=['accuracy'])


model = Sequential()
model.add(Conv2D(32,
    (3, 3),
    input_shape = (buckets, max_len, channels),
    activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32,
    (3, 3),
    # input_shape = (buckets, max_len, channels),
    activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation="softmax"))
model.compile(loss="categorical_crossentropy",
                optimizer="adam",
                metrics=["accuracy"])

tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)

model.fit(X_train, y_train_hot, epochs=epochs, validation_data=(X_test, y_test_hot), callbacks=[tensorboard_callback])
score = model.evaluate(X_test, y_test_hot, verbose="auto")
# Print test accuracy
print('\n', 'Test accuracy:', score[1])




'''
tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)

model.fit(X_train, y_train_hot, epochs=epochs, validation_data=(X_test, y_test_hot), callbacks=[tensorboard_callback])

#LSTM

model = Sequential()
model.add(LSTM(16, input_shape=(buckets, max_len, channels), activation="sigmoid"))
model.add(Dense(1, activation='sigmoid'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss="categorical_crossentropy",
                optimizer="adam",
                metrics=['accuracy'])

tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)

model.fit(X_train, y_train_hot, epochs=epochs, validation_data=(X_test, y_test_hot), callbacks=[tensorboard_callback])


score = model.evaluate(X_test, y_test_hot, verbose=0)
# Print test accuracy
print('\n', 'Test accuracy:', score[1])

'''



'''
y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)
X_train = X_train.reshape(X_train.shape[0], buckets, max_len)
X_test = X_test.reshape(X_test.shape[0], buckets, max_len)

model = Sequential()
model.add(Flatten(input_shape=(buckets, max_len)))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss="categorical_crossentropy",
                optimizer="adam",
                metrics=['accuracy'])

model.fit(X_train, y_train_hot, epochs=epochs, validation_data=(X_test, y_test_hot), callbacks=[WandbCallback(data_type="image", labels=labels)])
# build model
model = Sequential()
model.add(LSTM(16, input_shape=(buckets, max_len, channels), activation="sigmoid"))
model.add(Dense(1, activation='sigmoid'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss="categorical_crossentropy",
                optimizer="adam",
                metrics=['accuracy'])

model.fit(X_train, y_train_hot, epochs=epochs, validation_data=(X_test, y_test_hot), callbacks=[WandbCallback(data_type="image", labels=labels)])
'''

