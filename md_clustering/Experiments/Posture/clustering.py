
# Import the necessary functions

from dictionary_learning.DaDiL_clustering import *
from md_clustering.utils.clustering_utils import clusters
import warnings
import sys
import os
import pickle
import random
import json
sys.path.append('../../')
warnings.filterwarnings('ignore')

with open ('Results/features_dic.pkl','rb') as file:
    dic = pickle.load(file)

with open ('Results/labels_dic.pkl','rb') as file:
    lab = pickle.load(file)  
# Import features , labels and atoms

data_list = list(dic.values())
features = []
for array in data_list:
    tensor = torch.from_numpy(array)
    features.append(tensor)
Y1 = np.load(
    'Results/KMeans/MappedLabels_Domain1.npy', allow_pickle=True)
Y2 = np.load(
    'Results/KMeans/MappedLabels_Domain2.npy', allow_pickle=True)
Y3 = np.load(
    'Results/KMeans/MappedLabels_Domain3.npy', allow_pickle=True)

# Load XP and YP NumPy files
XP = []
YP = []
for i in range(len(features)):
    x_file_path = os.path.join(
        'Results/Atoms', f'xatom_{i}.npy')
    y_file_path = os.path.join(
        'Results/Atoms', f'yatom_{i}.npy')

    # Load XP
    loaded_x = np.load(x_file_path)
    XP.append(torch.tensor(loaded_x))

    # Load YP
    loaded_y = np.load(y_file_path)
    YP.append(torch.tensor(loaded_y))
def Dadclustering(features,Y1,Y2,Y3,XP,YP):


    # Define hyperparameters
    n_classes = 7
    n_samples = 3000
    batch_size = 128
    n_components = 3
    n_datasets = 3
    reg = 0.0
    reg_labels = 0.0
    num_iter_max = 80

    Xs = features

    if torch.is_tensor(Y1):
        Ys = [torch.nn.functional.one_hot(Y1.long(), num_classes=n_classes).float(),
              torch.nn.functional.one_hot(torch.from_numpy(Y2).long(), num_classes=n_classes).float(),
              torch.nn.functional.one_hot(torch.from_numpy(Y3).long(), num_classes=n_classes).float()]
    else:

        Ys=[torch.nn.functional.one_hot(torch.from_numpy(Y1).long(), num_classes=n_classes).float(),torch.nn.functional.one_hot(torch.from_numpy(Y2).long(), num_classes=n_classes).float(),torch.nn.functional.one_hot(torch.from_numpy(Y3).long(), num_classes=n_classes).float()]

    # Perform the DaDiL clustering
    cluster_labels,XAtom,YAtom = dadil_clustering(
        Xs, Ys, XP, YP, n_samples, n_components, reg, reg_labels, batch_size, n_classes, num_iter_max)

    print(cluster_labels[0].shape)
    print(cluster_labels[1].shape)

    domain_1 = clusters(features[0], cluster_labels[0], n_classes)
    domain_2 = clusters(features[1], cluster_labels[1], n_classes)
    domain_3 = clusters(features[2], cluster_labels[2], n_classes)

    # Cluster data for each domain

    domain_1.cluster_data()
    domain_2.cluster_data()
    domain_3.cluster_data()

    print("longeur de domain 2",len(domain_2.cluster_tensors))

    np.save('Results/DaDiL/MappedLabels_Domain1.npy',
            cluster_labels[0])
    mapped_labels_domain_2 = domain_2.clusters_mapping(
        domain_1.cluster_tensors)
    np.save('Results/DaDiL/MappedLabels_Domain2.npy',
            mapped_labels_domain_2)

    # Map cluster labels for Domain 3 using Domain 1's cluster tensors

    mapped_labels_domain_3 = domain_3.clusters_mapping(domain_1.cluster_tensors)
    np.save('Results/DaDiL/MappedLabels_Domain3.npy',mapped_labels_domain_3)
    print(len(mapped_labels_domain_2))

    #print(len(mapped_labels_domain_3))

    return(cluster_labels[0],mapped_labels_domain_2,mapped_labels_domain_3,XAtom,YAtom)


if __name__ == "__main__":
    Dadclustering()
