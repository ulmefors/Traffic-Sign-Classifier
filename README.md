# Traffic sign classifier

## Run

Inference using existing pre-trained model

    python inference.py

Train new model

    python train.py

## Prerequisites

Version numbers represent tested releases. Running Python 3.5 facilitates OpenCV installation.

- python 3.5.3
- pandas 0.20.3
- opencv 3.1.0
- keras 2.0.6
- tensorflow 1.2.1
- sklearn 0.18.2
- tqdm 4.15.0
- scipy 0.19.1
- numpy 1.13.1

### Conda installation Mac OS and Linux

    // Training and inference
    conda create -n traffic_signs python=3.5
    source activate traffic_signs
    pip install keras tensorflow sklearn pandas tqdm scipy numpy
    conda install -c menpo opencv

    // Visualization
    macOS: brew install graphviz
    Ubuntu: apt install graphviz
    pip install h5py graphviz pydot
    