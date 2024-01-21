import os
import sys

sys.path.append(os.path.abspath('/'))

import torch
import random
import pickle
import argparse
import numpy as np
import tensorflow as tf

from tqdm.auto import tqdm

from models import DeepNeuralNet
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
from dictionary_learning.utils import Office31Dataset

parser = argparse.ArgumentParser(description='Arguments for extracting features from the Office31 dataset using a pretrained ResNet50')
parser.add_argument('--dataset',
                    type=str,
                    default='office31')
args = parser.parse_args()

batch_size = 64
dataset_name = args.dataset.lower()
domains = ["amazon", "webcam", "dslr"]
device = torch.device('cuda')

for domain in domains:
    dataset = {}
    source_domains = [d for d in domains if d != domain]

    T = ResNet50_Weights.IMAGENET1K_V2.transforms()

    source_loaders = []
    for source_domain in source_domains:
        tr_dataset = Office31Dataset(root=os.path.abspath('./data/{}'.format(dataset_name)), domains=[source_domain], transform=T)
        tr_loader = torch.utils.data.DataLoader(tr_dataset, batch_size=batch_size, shuffle=False, num_workers=10)
        
        source_loaders.append(tr_loader)

    va_dataset = Office31Dataset(root=os.path.abspath('./data/{}'.format(dataset_name)), domains=[domain], transform=T, train=True, test=False)
    va_loader = torch.utils.data.DataLoader(va_dataset, batch_size=batch_size, shuffle=False, num_workers=10)

    ts_dataset = Office31Dataset(root=os.path.abspath('./data/{}'.format(dataset_name)), domains=[domain], transform=T, train=False, test=True)
    ts_loader = torch.utils.data.DataLoader(ts_dataset, batch_size=batch_size, shuffle=False, num_workers=10)

    # Create model
    resnet_model = resnet50(weights='IMAGENET1K_V2', progress=True)
    resnet_model.fc = torch.nn.Identity()

    task = torch.nn.Sequential(torch.nn.Linear(in_features=2048, out_features=31))

    ckpt_folder = os.path.abspath('./results/{}/logs/baseline/{}/version_0/checkpoints'.format(dataset_name, domain))
    ckpt_filename = os.listdir(ckpt_folder)[0]
    ckpt_path = os.path.join(ckpt_folder, ckpt_filename)
    model = DeepNeuralNet(encoder=resnet_model,
                        task=task,
                        learning_rate=0.0,
                        loss_fn=torch.nn.CrossEntropyLoss(),
                        l2_penalty=0.0).load_from_checkpoint(
                                ckpt_path,
                                encoder=resnet_model,
                                task=task,
                                learning_rate=0.0,
                                loss_fn=torch.nn.CrossEntropyLoss()
                            )
    # Gets encoder & task and set them to eval mode
    ϕ = model.encoder.eval().to(device)

    # Encodes source domain
    for src, loader in zip(source_domains, source_loaders):
        Xs, Ys = [], []
        for xs, ys in tqdm(loader):
            with torch.no_grad():
                hs = ϕ(xs.to(device)).to('cpu')

                Xs.append(hs)
                Ys.append(ys)
        Xs, Ys = torch.cat(Xs, dim=0), torch.cat(Ys, dim=0)
        print("Source {}, Xs.shape: {}, Ys.shape: {}".format(src, Xs.shape, Ys.shape))
        dataset[src] = {"Features": Xs, "Labels": Ys}
    
    # Encodes Target Domain
    Xt, Yt = [], []
    for xt, yt in tqdm(va_loader):
        with torch.no_grad():
            ht = ϕ(xt.to(device)).to('cpu')

            Xt.append(ht)
            Yt.append(yt)
    Xt, Yt = torch.cat(Xt, dim=0), torch.cat(Yt, dim=0)
    print("Target (tr) {}, Xt.shape: {}, Yt.shape: {}".format(domain, Xt.shape, Yt.shape))

    # Encodes Target Domain
    Xts, Yts = [], []
    for xts, yts in tqdm(ts_loader):
        with torch.no_grad():
            hts = ϕ(xts.to(device)).to('cpu')

            Xts.append(hts)
            Yts.append(yts)
    Xts, Yts = torch.cat(Xts, dim=0), torch.cat(Yts, dim=0)
    print("Target (ts) {}, Xt.shape: {}, Yt.shape: {}".format(domain, Xts.shape, Yts.shape))

    dataset[domain] = {"Train": {"Features": Xt, "Labels": Yt},
                       "Test": {"Features": Xts, "Labels": Yts}}
    

    with open(os.path.abspath("./data/{}_pyl_resnet_features_{}.pickle".format(dataset_name, domain)), 'wb') as f:
        pickle.dump(dataset, f)