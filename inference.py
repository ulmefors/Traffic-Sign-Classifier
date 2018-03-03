import json
import keras
from keras.models import model_from_json
from keras.utils import np_utils
from keras.optimizers import Adam
import config
import helper

# Dataset ('test', 'train', 'valid')
dataset = 'test'

# Load images and labels
X, y = helper.load_data(dataset)

# Load model
model_file = config.__model_file__
with open(model_file, 'r') as jfile:
    if keras.__version__ >= '1.2.0':
        model = model_from_json(json.loads(jfile.read()))
    else:
        model = model_from_json(jfile.read())

# Compile model and load weights
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.load_weights(config.__model_weights__)

# Evaluate model performace
print('Evaluating performance on %d samples' % X.shape[0])
y_cat = np_utils.to_categorical(y, config.__num_classes__)
scores = model.evaluate(X, y_cat, verbose=0)
names = model.metrics_names
for name, score in zip(names, scores):
    print('%s: \t%.4f' % (name, score))
