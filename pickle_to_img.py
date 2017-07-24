import pickle
import cv2
import os
import pandas as pd


def main():
    datasets = ['train', 'test']

    for dataset in datasets:
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

if __name__ == "main":
    main()
