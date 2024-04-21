import ot
import sys
from split_data import split_data
import torch
import pickle
import warnings
import numpy as np
import pytorch_lightning as pl
from Training import Train
from dictionary_learning.utilss import BalancedBatchSamplerDA
from dictionary_learning.utilss import DictionaryDADataset

from dictionary_learning.barycenters import wasserstein_barycenter,wasserstein_barycenter_with_cost
from dictionary_learning.barycentric_regression import WassersteinBarycentricRegression



sys.path.append('../../')
warnings.filterwarnings('ignore')
from sklearn.metrics import adjusted_rand_score

def main(features,labels,domain,train_ratio,n_epochs):
    if type(features) != list:
        data_list = features.tolist()
    else : 
        data_list = features
    alldata = []
    for array in data_list:
        tensor = torch.from_numpy(array)

        alldata.append(tensor)


    all_data = [torch.from_numpy(feature) for feature in features]

    # Calculate the number of domains
    num_domains = len(all_data)

    # Calculate the length of each domain's data
    domain_sizes = [len(data) for data in all_data]

    # Calculate the starting index of each domain's labels
    start_indices = [sum(domain_sizes[:i]) for i in range(num_domains)]

    # Calculate the ending index of each domain's labels
    end_indices = [start_indices[i] + domain_sizes[i] for i in range(num_domains)]

    # Slice the labels based on the start and end indices of each domain
    all_labels = [labels[start_indices[i]:end_indices[i]] for i in range(num_domains)]

    y_labels = [np.argmax(label, axis=1) for label in all_labels]

    # print(y_labels[0].shape,y_labels[1].shape,y_labels[2].shape)
    # print(all_data[0].shape,all_data[1].shape,all_data[2].shape)

    num_domains = len(all_data)

    test_size=1-train_ratio




    for i in range(num_domains):
        if domain == f'Domain_{i + 1}':
            if test_size >= 1:

                X_test=all_data[i]
                del all_data[i]
                y_test=y_labels[i]
                del y_labels[i]
            else:
                X_train, X_test, y_train, y_test = split_data(domain, features, labels, test_size=test_size)
                del all_data[i]
                del y_labels[i]
                all_data.append(torch.tensor((X_train)))

                y_labels.append(torch.tensor(y_train))
            break
    else:
        raise ValueError(f"Invalid domain '{domain}'")
    X_train, y_train = all_data, y_labels
    # print(X_train[0].shape)
    # print(X_train[1].shape)
    # print(X_train[2].shape)

    Yt=y_test
    Xt=torch.tensor(X_test)





    XAtom, yAtom, ari_kmeans= Train(n_epochs,X_train, y_train,3000, 0, 0.0, 128, 7,20)

    YAtom=[yAtom[i].argmax(dim=1) for i in range(len(yAtom))]

    train_dataset = DictionaryDADataset(XAtom, yAtom, Xt, Yt)
    num_iter_dil=100
    n_classes=7
    batch_size=260
    n_samples=2000
    batches_per_it=n_samples // batch_size
    S = BalancedBatchSamplerDA([ysk for ysk in YAtom],
                               n_target=len(Xt),
                               n_classes=n_classes,
                               batch_size=batch_size,
                               n_batches=batches_per_it)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=S)


    ϵ=0.0

    # Creates dictionary

    wbr = WassersteinBarycentricRegression(XAtom,
                                           yAtom,
                                           n_distributions=1,
                                           learning_rate=1e-2,
                                           sampling_with_replacement=True,
                                           weight_initialization='uniform')

    # Creates trainer object
    trainer = pl.Trainer(max_epochs=num_iter_dil, accelerator='cuda', logger=False, enable_checkpointing=False)
    trainer.fit(wbr, train_loader)



    # Get dictionary weights
    weights = wbr.A.detach().squeeze()
    print(weights.squeeze())
    # Reconstruct samples

    Xr= wasserstein_barycenter(XP=XAtom, YP=yAtom, n_samples=7, ϵ=0.0, α=weights.squeeze(),
                                    β=None, num_iter_max=10, verbose=True, propagate_labels=False,
                                    penalize_labels=False)
    new_clusters = []





    np.save('Results/DaDiL/TargetBarycenters.npy',
            Xr)







    #Method with EUCLIDIAN
    Cℓ = torch.cdist(Xr.cpu(), torch.tensor(Xt).float(), p=2) ** 2
    clusterss = Cℓ.argmin(dim=0)



    #Method with OT
    π = ot.emd([], [], np.array(Cℓ))
    new_clusters = torch.argmax(torch.transpose(torch.tensor(π), 0, 1), dim=1)







    ari = adjusted_rand_score(y_test, clusterss)



    print('test: ',ari)
    print('kmeans: ', ari_kmeans)






if __name__ == "__main__":

    r"""Features must be an array or list of n_domains arrays"""
    with open ('Results/features_dic_6_10.pkl','rb') as file:
        dic = pickle.load(file)
    features = list(dic.values())
    r"""Labels must be an array of shape (TotalnumberFeatures,num_classes)"""

    with open ('Results/labels_dic_6_10.pkl','rb') as file:
        lab = pickle.load(file)
    labels_list = list(lab.values())
    labels = np.concatenate(labels_list, axis = 0 )
    main(features,labels,"Domain_5",0.0,2)






