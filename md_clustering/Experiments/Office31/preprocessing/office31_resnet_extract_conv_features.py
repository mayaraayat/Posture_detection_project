import os
import pickle
import argparse
import numpy as np
import tensorflow as tf

from tensorflow import keras

from PIL import Image
from sklearn.preprocessing import OneHotEncoder

import adapt

parser = argparse.ArgumentParser(description='Arguments for Wasserstein Dictionary Learning')
parser.add_argument('--batch_size',
                    type=int,
                    default=64)
parser.add_argument('--target',
                    type=str,
                    default='amazon')
parser.add_argument('--dataset_name',
                    type=str,
                    default='modern_office31')
args = parser.parse_args()


DATASET_NAME = args.dataset_name.lower()
BATCH_SIZE = args.batch_size
DOMAIN_NAMES = ['amazon', 'dslr', 'webcam']

base_path = os.path.abspath('/')
data_path = os.path.join(base_path, 'data', 'modern_office31')
model_path = os.path.join(base_path, 'data', 'pretrained')
folds_path = os.path.join(base_path, 'data', 'modern_office31', 'folds')

selected_sources = [d for d in DOMAIN_NAMES if d != args.target.lower()]
selected_target = [d for d in DOMAIN_NAMES if d == args.target.lower()][0]

print("data_path: {}, folds_path: {}".format(data_path, folds_path))
print("Sources: {}".format(selected_sources))
print("Target: {}".format(selected_target))

name2cat = {
    'headphones': 0,
    'bike': 1,
    'mouse': 2,
    'file_cabinet': 3,
    'bottle': 4,
    'desk_lamp': 5,
    'back_pack': 6,
    'desktop_computer': 7,
    'letter_tray': 8,
    'mug': 9,
    'bookcase':10,
    'projector':11,
    'pen':12,
    'laptop_computer':13,
    'speaker':14,
    'punchers':15,
    'calculator':16,
    'tape_dispenser':17,
    'phone':18,
    'ruler':19,
    'mobile_phone':20,
    'printer':21,
    'paper_notebook':22,
    'ring_binder':23,
    'scissors':24,
    'keyboard':25,
    'trash_can':26,
    'bike_helmet':27,
    'monitor':28,
    'desk_chair':29,
    'stapler': 30
}

domain_name2cat = {
    'amazon': 0, 'dslr': 1, 'webcam': 2
}


def get_Xy(domains, path_to_folder, folds_path, train=True, test=True, transform=None):
    X = []
    y = []
    d = []
    ind_valid = []

    for domain in domains:
        path = os.path.join(path_to_folder, domain, 'images')
        
        filenames = []

        with open(os.path.join(folds_path, '{}_train_filenames.txt'.format(domain)), 'r') as f:
            train_filenames = f.read().split('\n')[:-1]
            for _ in train_filenames:
                ind_valid.append(0)

        if train: filenames += train_filenames

        with open(os.path.join(folds_path, '{}_test_filenames.txt'.format(domain)), 'r') as f:
            test_filenames = f.read().split('\n')[:-1]
            for _ in test_filenames:
                ind_valid.append(1)

        if test: filenames += test_filenames

        for class_and_filename in filenames:
            c, filename = class_and_filename.split('/')
            file_path = os.path.join(path, c, filename)
            image = Image.open(file_path)
            image = image.resize((224, 224), Image.ANTIALIAS)
            x = keras.applications.resnet50.preprocess_input(np.array(image, dtype=int))
            X.append(x)
            y.append(name2cat[c])
            d.append(domain_name2cat[domain])
    return X, y, ind_valid, d

with tf.device('/cpu:0'):
    resnet50 = keras.applications.resnet50.ResNet50(include_top=False, input_shape=(224, 224, 3), pooling="avg")

    first_layer = resnet50.get_layer('conv5_block2_out')
    inputs = keras.layers.Input(first_layer.output_shape[1:])

    for layer in resnet50.layers[resnet50.layers.index(first_layer)+1:]:
        if layer.name == "conv5_block3_1_conv":
            x = layer(inputs)
        elif layer.name == "conv5_block3_add":
            x = layer([inputs, x])
        else:
            x = layer(x)

    first_blocks = keras.models.Model(resnet50.input, first_layer.output)
    last_block = keras.models.Model(inputs, x)
    last_block.save("./.tmp/resnet50_last_block.hdf5")

    one_hot = OneHotEncoder(sparse=False)
    dataset = {}

    for source in selected_sources:
        print("Preprocessing source {}".format(source))
        Xs, ys, valid_s, ds = get_Xy([source], train=True, test=True, path_to_folder=data_path, folds_path=folds_path)
        Xs, ys, ds = np.stack(Xs), np.array(ys), np.array(ds)
        Hs = first_blocks.predict(Xs).transpose([0, 3, 1, 2])
        Ys = one_hot.fit_transform(ys.reshape(-1, 1))
        del Xs
        print("Resulting array: {}".format(Hs.shape))

        dataset[source] = {"Features": Hs, "Labels": Ys}

    valid_s = np.array(valid_s)

    print("Preprocessing target domain data (train)")
    Xt, yt, valid_t, dt = get_Xy([selected_target], train=True, test=False, path_to_folder=data_path, folds_path=folds_path)
    Xt, yt, dt = np.stack(Xt), np.array(yt), np.array(dt)
    Ht = first_blocks.predict(Xt).transpose([0, 3, 1, 2])
    del Xt
    print("Resulting array: {}".format(Ht.shape))

    valid_t = np.array(valid_t)

    print("Preprocessing target domain data (test)")
    Xts, yts, _, dts = get_Xy([selected_target], train=False, test=True, path_to_folder=data_path, folds_path=folds_path)
    Xts, yts, dts = np.stack(Xts), np.array(yts), np.array(dts)
    Hts = first_blocks.predict(Xts).transpose([0, 3, 1, 2])
    del Xts
    print("Resulting array: {}".format(Hts.shape))

    
    Yt = one_hot.fit_transform(yt.reshape(-1, 1))
    Yts = one_hot.fit_transform(yts.reshape(-1, 1))

    del last_block, first_blocks

    dataset[selected_target] = {'Train': {'Features': Ht, 'Labels': Yt},
                                'Test': {'Features': Hts, 'Labels': Yts}}

    with open(os.path.abspath('./data/{}/features/cnn-feats-{}.pkl'.format(DATASET_NAME, selected_target)), 'wb') as f:
        pickle.dump(dataset, f)