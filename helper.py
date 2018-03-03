import pickle
import cv2
import os
import pandas as pd
from urllib.request import urlretrieve
from tqdm import tqdm
import config
import zipfile
import glob
import numpy as np


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def load_data(dataset):
    # Convert pickled data to human readable images
    image_files = os.path.join(config.__images_dir__, dataset, 'sign*.png')
    if len(glob.glob(image_files)) == 0:
        extract_images()

    # Sort file names in alphabetical order to line up with labels
    files = glob.glob(image_files)
    files.sort()

    # Load images and save in X matrix. Convert to numpy array.
    X = []
    for file in files:
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        X.append(img)
    X = np.array(X)

    # Load labels
    labels_file = os.path.join(config.__labels_dir__, '%s.csv' % dataset)
    y = pd.read_csv(labels_file, header=None).values

    # Return images and labels
    return X, y


def maybe_download_traffic_signs():

    data_file = os.path.join(config.__data_dir__, config.__data_file__)

    if not os.path.exists(data_file):
        if not os.path.exists(config.__data_dir__):
            os.makedirs(config.__data_dir__)

        # Download Traffic Sign data
        print('Downloading Traffic Sign data...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                config.__data_url__,
                data_file,
                pbar.hook)

    # Extract
    print('Extracting Traffic Sign data...')
    zip_ref = zipfile.ZipFile(data_file, 'r')
    zip_ref.extractall(config.__data_dir__)
    zip_ref.close()


def extract_images():

    # Download data
    maybe_download_traffic_signs()

    for dataset in config.__datasets__:
        # Load Data
        with open(os.path.join(config.__data_dir__, '%s.p' % dataset), 'rb') as f:
            data = pickle.load(f)
        X = data['features']
        y = data['labels']

        # Save to CSV. No label for columns or rows
        y = pd.DataFrame(y)
        labels_dir = config.__labels_dir__
        if not os.path.exists(labels_dir):
            os.makedirs(labels_dir)
        labels_file = os.path.join(labels_dir, '%s.csv' % dataset)
        y.to_csv(labels_file, header=False, index=False)

        # Create image directory
        directory = os.path.join(config.__images_dir__, '%s' % dataset)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Load images and save as picture files
        num_images = X.shape[0]
        for i in range(num_images):
            file = directory + '/sign_%05d.png' % i
            img = X[i]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(file, img)
