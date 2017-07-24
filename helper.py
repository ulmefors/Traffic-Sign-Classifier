import pickle
import cv2
import os
import pandas as pd
from urllib.request import urlretrieve
from tqdm import tqdm
import config
import zipfile


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


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

    for dataset in ['train', 'valid', 'test']:
        # Load Data
        with open('data/%s.p' % dataset, 'rb') as f:
            data = pickle.load(f)
        X = data['features']
        y = data['labels']

        # Save to CSV. No label for columns or rows
        y = pd.DataFrame(y)
        y.to_csv('%s_labels.csv' % dataset, header=False, index=False)

        # Create image directory
        directory = 'images/%s' % dataset
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Load images and save as picture files
        nb_images = X.shape[0]
        for i in range(nb_images):
            file = directory + '/sign_%05d.png' % i
            img = X[i]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(file, img)
