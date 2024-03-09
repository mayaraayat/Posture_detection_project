import torch
import numpy as np
import warnings
import os
import sys
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, fowlkes_mallows_score
import matplotlib.pyplot as plt

from KMeans_baseline import *
from initialize_atoms import initialize_atoms
from clustering import Dadclustering
sys.path.append('../../')
from dictionary_learning.weighted_barycenters import compute_barycenters

warnings.filterwarnings('ignore')

def extract_pressure_matrices(json_data):
    pressure_matrices = []
    for entry in json_data['pressureData']:
        pressure_matrix = entry["pressureMatrix"]
        pressure_matrices.append({"pressureMatrix": pressure_matrix})
    return pressure_matrices

def extract_features_from_pressure_matrices(pressure_matrices):
    flattened_data = [np.array(item["pressureMatrix"]).flatten() for item in pressure_matrices]
    return np.concatenate(flattened_data, axis=0)
dic = {}
lab = {}
for i in range(6,9):
    folder_path = f'data/Subject_{i}/Posture_data'

    # Get all files in the folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    labels = [int(f.split('.')[0][-1])for f in files]
    l = []

    for f in files:
        file_path = os.path.join(folder_path, f)
        # Check if the file is a JSON file
        if file_path.endswith('.json'):
            with open(file_path, 'r') as file:
                try:
                    data = json.load(file) 
                    pressure_matrices = extract_pressure_matrices(data)
                    features = extract_features_from_pressure_matrices(pressure_matrices)
                    l.append(features)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in file {file_path}: {e}")
        else:
            print(f"Skipping non-JSON file: {file_path}")
     
    num = list(range(7))
    random.shuffle(num)
    
    dic[i] = [l[j] for j in num]
    lab[i] = [labels[j] for j in num]

def main(xmax):



    features=list(dic.values())

    labels = list(lab.values())

    '''
    data_list = features.tolist()
    alldata = []
    for array in data_list:
        tensor = torch.from_numpy(array)

        alldata.append(tensor)

    alllabels = [labels[len(alldata[1]):3608], labels[:len(alldata[0])],
                 labels[len(alldata[0]):(len(alldata[0]) + 793)]]
    ylabels1 = []

    # Convert each one-hot encoded array to labels and append them to the list
    for array in alllabels:
        labels = np.argmax(array, axis=1)
        ylabels1.append(labels)
    alllabels = ylabels1
    alldata = [alldata[-1], alldata[0], alldata[1]]

    features=alldata
    labels=alllabels

    print(len(alldata[0]),len(alldata[1]),len(alldata[2]))
    print(len(alllabels[0]),len(alllabels[1]),len(alllabels[2]))'''

    Y1,Y2,Y3=KMeans_baseline(features, labels)
    print(len(Y1),len(Y2),len(Y3))
    mapped_labels_domain=[Y1,Y2,Y3]
    ari_history = []
    nmi_history = []
    fmi_history = []
    for i in range(len(labels)):
        true_labels = labels[i]
        mapped_labels = mapped_labels_domain[i]
        # Calculate evaluation metrics
        ari = adjusted_rand_score(true_labels, mapped_labels)

        nmi = normalized_mutual_info_score(true_labels, mapped_labels)

        fmi = fowlkes_mallows_score(true_labels, mapped_labels)


        ari_history.append(ari)
        nmi_history.append(nmi)
        fmi_history.append(fmi)

    global_ari = []
    global_nmi = []
    global_fmi = []

    global_ari.append(ari_history)

    print("global_ari history : ", global_ari)
    global_nmi.append(nmi_history)
    global_fmi.append(fmi_history)

    x=0
    while x<xmax:

        ari_history = []
        nmi_history = []
        fmi_history = []



        n_classes = 7
        n_samples = 3000
        batch_size = 128
        ϵ = 0.01
        η_A = 0.0
        lr = 1e-1
        num_iter_max = 20
        num_iter_dil = 50

        XP,YP=initialize_atoms(features,Y1,Y2,Y3,n_classes,n_samples,batch_size,ϵ,η_A,lr,num_iter_max,num_iter_dil)

        print(len(XP[0]),len(XP[1]),len(XP[2]))
        print(len(YP[0]),len(YP[1]),len(YP[2]))

        mapped_labels_domain_1,mapped_labels_domain_2,mapped_labels_domain_3,XAtom,YAtom=Dadclustering(features,Y1,Y2,Y3,XP,YP)
        mapped_labels_domain =[mapped_labels_domain_1,mapped_labels_domain_2,mapped_labels_domain_3]
        Y1, Y2, Y3=mapped_labels_domain_1,mapped_labels_domain_2,mapped_labels_domain_3


        for i in range(len(labels)):

            true_labels = labels[i]
            mapped_labels=mapped_labels_domain[i]
        # Calculate evaluation metrics
            ari = adjusted_rand_score(true_labels, mapped_labels)

            nmi = normalized_mutual_info_score(true_labels, mapped_labels)

            fmi = fowlkes_mallows_score(true_labels,mapped_labels)

            ari_history.append(ari)
            nmi_history.append(nmi)
            fmi_history.append(fmi)
        print("ari history : ",ari_history)

        global_ari.append(ari_history)
        global_nmi.append(nmi_history)
        global_fmi.append(fmi_history)

        print("global_ari history : ", global_ari)
        x+=1



    np.save('Results/DaDiL/global_ari.npy',
            global_ari)

    plt.plot(global_ari)

    # Add labels and title

    plt.title('ARI')

    # Show the plot
    plt.show()

    np.save('Results/DaDiL/nmi_history.npy',
            global_nmi)
    np.save('Results/DaDiL/fmi_history.npy',
            global_fmi)

    return (XAtom, YAtom)


if __name__ == "__main__":
    main(2)
