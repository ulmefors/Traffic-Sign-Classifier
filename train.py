import os
import json
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Lambda, Dropout, MaxPooling2D
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.callbacks import TensorBoard
import config
import helper
import time
import datetime


# Overwrite model
overwrite_model = True

# Dataset ('test', 'train', 'valid')
dataset = 'train'

# Load images and labels
X, y = helper.load_data(dataset)

# Select subsample for faster debugging
sample_fraction = 1
num_samples = X.shape[0]
sample_size = round(sample_fraction * num_samples)
X = X[:sample_size]
y = y[:sample_size]

# Split into training and validation
test_fraction = 0.20
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0, test_size=test_fraction)

# Hyperparams
shape = X.shape[1:]
num_classes = config.__num_classes__
learning_rate = 0.001
batch_size = 512
epochs = 10

# Class number to classification columns (categorical to dummy variables)
y_train = np_utils.to_categorical(y_train, num_classes)
y_val = np_utils.to_categorical(y_val, num_classes)

# Model of Convolutional Neural Network
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=shape, output_shape=shape))
model.add(Conv2D(3, (1, 1), activation='relu'))
model.add(Conv2D(16, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Print model summary
model.summary()

# Compile model
model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

# Start training train
start_time = time.time()

# Configure Tensorboard log
log_dir = os.path.join(config.__tensorboard_log_dir,
                       datetime.datetime.fromtimestamp(start_time).strftime('%y-%m-%d_%H-%M'))
tbCallBack = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)

# Train model
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                    verbose=2, validation_data=(X_val, y_val), callbacks=[tbCallBack])

# Training duration
training_time = time.time() - start_time

# Print metrics of validation set
print('')
print('*** Training Complete ***')
print('Elapsed time: %.1f seconds' % training_time)
scores = model.evaluate(X_val, y_val, verbose=0)
names = model.metrics_names
print('')
print('*** Metrics ***')
for name, score in zip(names, scores):
    print('%s: \t%.4f' % (name, score))

# Overwrite saved model
if overwrite_model:
    model.save_weights(config.__model_weights__, overwrite=True)
    with open(config.__model_file__, 'w') as outfile:
        json.dump(model.to_json(), outfile)
    plot_model(model, config.__model_diagram__, show_shapes=True)
    print('')
    print('*** Model Saved ***')
