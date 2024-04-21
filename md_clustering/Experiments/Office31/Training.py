import torch
import numpy as np
import warnings
import os
import sys
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, fowlkes_mallows_score
import matplotlib.pyplot as plt
from split_data import *
from KMeans_baseline import *
from initialize_atoms import initialize_atoms
from clustering import dadclustering
sys.path.append('../../')
from dictionary_learning.weighted_barycenters import compute_barycenters

warnings.filterwarnings('ignore')

def Train(xmax,features, labels,n_samples, reg, reg_labels, batch_size, n_classes, num_iter_max):



#    features = [torch.tensor(X_train), features[0], features[1]]
 #   labels = [y_train, labels[0], labels[1]]




    Ys = []
    [*mapped_labels_domain] = KMeans_baseline(features)
    for Y in mapped_labels_domain:
        Ys.append(torch.nn.functional.one_hot(torch.tensor(Y).long(), num_classes=n_classes).float())



    ari_history = []

    for i in range(len(labels)):
        true_labels = labels[i]
        mapped_labels = mapped_labels_domain[i]
        # Calculate evaluation metrics
        ari = adjusted_rand_score(true_labels, mapped_labels)


        ari_history.append(ari)


    global_ari = []


    global_ari.append(ari_history)

    print("global_ari history : ", global_ari)


    x=0
    while x<xmax:

        ari_history = []


        n_classes = 31
        n_samples = 3000
        batch_size_atoms = 128
        ϵ = 0.01
        η_A = 0.0
        lr = 1e-1
        num_iter_max_atoms = 20
        num_iter_dil_atoms = 50

        XP,YP=initialize_atoms(features,mapped_labels_domain,n_classes,n_samples,batch_size_atoms,ϵ,η_A,lr,num_iter_max_atoms,num_iter_dil_atoms)



        mapped_labels_domain,XAtom,YAtom=dadclustering(features, Ys, XP, YP, n_samples, reg, reg_labels, batch_size, n_classes, num_iter_max)




        for i in range(len(labels)):
            true_labels = labels[i]
            mapped_labels=mapped_labels_domain[i]
            # print("len(true labels : " , len(true_labels))
            # print("len(mapped_labels  : " , len(mapped_labels))

        # Calculate evaluation metrics
            ari = adjusted_rand_score(true_labels, mapped_labels)



            ari_history.append(ari)

        print("ari history : ",ari_history)

        global_ari.append(ari_history)


        print("global_ari history : ", global_ari)
        x+=1



    np.save('Results/DaDiL/global_ari.npy',
            global_ari)


    num_domains = len(global_ari[0])  # Number of domains
    epoch_count = len(global_ari)  # Number of epochs

    for i in range(num_domains):
        plt.plot([ari[i] for ari in global_ari], label=f'Domain {i + 1}')

    # plt.xlabel('Epoch')
    # plt.ylabel('ARI score')
    # plt.title('Evolution of Clustering Performance Over Epochs')
    # plt.legend()
    # plt.show()

    return (XAtom, YAtom)







if __name__ == "__main__":
    main(2)
