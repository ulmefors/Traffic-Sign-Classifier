import pickle
import cv2
import os
import pandas as pd

# Load Data
with open('data/test.p', 'rb') as f:
    data = pickle.load(f)
X = data['features']
y = data['labels']

# Save to CSV. No label for columns or rows
y = pd.DataFrame(y)
y.to_csv('test_labels.csv', header=False, index=False)

# Create image directory
directory = 'images/test'
if not os.path.exists(directory):
    os.makedirs(directory)

# Load images and save as picture files
nb_images = X.shape[0]
for i in range(nb_images):
    pass
    file = directory + '/sign_%05d.png' % i
    img = X[i]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file, img)
