import pandas as pd
import glob
import cv2
import json
import numpy as np
import keras
from keras.models import model_from_json
from keras.utils import np_utils
from keras.optimizers import Adam
import config
import helper


# Convert pickled data to human readable images
image_files = 'images/test/sign*.png'
if len(glob.glob(image_files)) == 0:
    helper.extract_images()

# Load images and save in X numpy array
X = []
for file in glob.glob(image_files):
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    X.append(img)
X = np.array(X)

# Load labels
y = pd.read_csv('test_labels.csv', header=None).values
y_cat = np_utils.to_categorical(y, config.__nb_classes__)

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
score = model.evaluate(X, y_cat, verbose=0)
names = model.metrics_names
for i in range(len(score)):
    print('%s: %.3f' % (names[i], score[i]))
