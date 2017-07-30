import json
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Lambda, Dropout
from keras.optimizers import Adam
from keras.utils.visualize_util import plot
import config
import helper


# Dataset ('test', 'train', 'valid')
dataset = 'train'

# Load images and labels
X, y = helper.load_data(dataset)

# Select subsample for faster debugging
sample_fraction = 1
nb_samples = X.shape[0]
sample_size = round(sample_fraction * nb_samples)
X = X[:sample_size]
y = y[:sample_size]

# Split into training and validation
test_fraction = 0.20
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0, test_size=test_fraction)

# Hyperparams
shape = X.shape[1:]
nb_classes = config.__nb_classes__
batch_size = 32
nb_epoch = 10

# Class number to classification columns (categorical to dummy variables)
y_train = np_utils.to_categorical(y_train, nb_classes)
y_val = np_utils.to_categorical(y_val, nb_classes)

# Model of Convolutional Neural Network
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=shape, output_shape=shape))
model.add(Conv2D(3, 1, 1, activation='elu'))
model.add(Conv2D(16, 5, 5, subsample=(2, 2), activation='elu'))
model.add(Conv2D(24, 3, 3, activation='elu'))
model.add(Conv2D(32, 3, 3, activation='elu'))
model.add(Conv2D(48, 3, 3, activation='elu'))
model.add(Conv2D(64, 3, 3, activation='elu'))
model.add(Conv2D(128, 3, 3, activation='elu'))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(128, activation='elu'))
model.add(Dropout(0.3))
model.add(Dense(nb_classes, activation='softmax'))

model.summary()

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=2, validation_data=(X_val, y_val))

# Print metrics of validation set
score = model.evaluate(X_val, y_val, verbose=0)
names = model.metrics_names
for i in range(len(score)):
    print('%s: %.3f' % (names[i], score[i]))

# Save model
model.save_weights(config.__model_weights__, overwrite=True)
with open(config.__model_file__, 'w') as outfile:
    json.dump(model.to_json(), outfile)
plot(model, config.__model_diagram__, show_shapes=True)
