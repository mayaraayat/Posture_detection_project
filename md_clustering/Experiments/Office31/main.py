import ot
import sys
from split_data import split_data
import torch
import warnings
import numpy as np
import pytorch_lightning as pl
from Training import Train
from dictionary_learning.utilss import BalancedBatchSamplerDA
from dictionary_learning.utilss import DictionaryDADataset

from dictionary_learning.barycenters import wasserstein_barycenter,wasserstein_barycenter_with_cost
from dictionary_learning.barycentric_regression import WassersteinBarycentricRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from matplotlib.colors import ListedColormap

sys.path.append('../../')
warnings.filterwarnings('ignore')
from sklearn.metrics import adjusted_rand_score

def main(features,labels,domain,train_ratio,n_epochs):

    data_list = features.tolist()
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





    XAtom, yAtom = Train(n_epochs,X_train, y_train,3000, 0, 0, 128, 31,30)

    YAtom=[yAtom[i].argmax(dim=1) for i in range(len(yAtom))]

    train_dataset = DictionaryDADataset(XAtom, yAtom, Xt, Yt)
    num_iter_dil=100
    n_classes=31
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
    trainer = pl.Trainer(max_epochs=num_iter_dil, accelerator='cpu', logger=False, enable_checkpointing=False)
    trainer.fit(wbr, train_loader)



    # Get dictionary weights
    weights = wbr.A.detach().squeeze()
    print(weights.squeeze())
    # Reconstruct samples

    Xr= wasserstein_barycenter(XP=XAtom, YP=yAtom, n_samples=31, ϵ=0.0, α=weights.squeeze(),
                                    β=None, num_iter_max=10, verbose=True, propagate_labels=False,
                                    penalize_labels=False)
    new_clusters = []





    np.save('Results/DaDiL/TargetBarycenters.npy',
            Xr)







    #Method with EUCLIDIAN
    Cℓ = torch.cdist(Xr.cpu(), torch.tensor(Xt).float(), p=2) ** 2
    clusterss = Cℓ.argmin(dim=0)



    #Method with OT
    # π = ot.emd([], [], np.array(Cℓ))
    #  new_clusters = torch.argmax(torch.transpose(torch.tensor(π), 0, 2), dim=1)


    # Final_idea

    C_k = torch.cdist(torch.tensor(Xt), Xr, p=2) ** 2
    cost_matrix = np.linalg.norm(Xr[:, np.newaxis, :] - torch.tensor(Xt), axis=2)

    π_k = ot.emd([], [], np.array(cost_matrix))

    centroid = torch.mm(torch.tensor(π_k), torch.tensor(Xt))
    centroid_sums = centroid.sum(axis=1)

    # Normalize each centroid
    centroid = centroid * centroid_sums[:, np.newaxis]


    new_clusters = []
    Cℓ = torch.cdist(centroid.cpu(), torch.tensor(Xt).float(), p=2) ** 2
    clusters_ℓ = Cℓ.argmin(dim=0)
    unique_labels, counts = np.unique(clusters_ℓ, return_counts=True)




    ari = adjusted_rand_score(y_test, clusters_ℓ)


    print(ari)
    return(ari)




    #
    # # Concatenate Xt and Xr
    # combined_data = np.concatenate((Xt, centroid), axis=0)
    # print(Xt.shape)
    # print(centroid.shape)
    # # Initialize t-SNE for combined data
    # tsneBR = TSNE(n_components=2, random_state=42)
    #
    # # Apply t-SNE to reduce combined data to 2D
    # data_tsne_combined = tsneBR.fit_transform(combined_data)
    #
    # # Split the transformed data back into Xt and Xr
    # data_tsne_Xt = data_tsne_combined[:len(Xt)]
    # data_tsne_Xr = data_tsne_combined[len(Xt):]
    #
    # # Calculate transparency values for Xr
    # transparency = 1 - π_k[0] * 350
    # transparency = np.clip(transparency, 0, 1)
    #
    # # Create a colormap
    # cmap = plt.get_cmap('coolwarm')
    #
    # # Plot the t-SNE transformed Xr points with transparency
    # plt.figure(figsize=(8, 6))
    # plt.scatter(data_tsne_Xr[:, 0], data_tsne_Xr[:, 1],cmap=cmap, s=40)
    #
    # # Plot the t-SNE transformed Xt points and highlight the first point
    # plt.scatter(data_tsne_Xt[:, 0], data_tsne_Xt[:, 1], color='#B0E0E6', alpha=0.6, s=40)
    # plt.scatter(data_tsne_Xt[0, 0], data_tsne_Xt[0, 1], color='red', alpha=1, s=100)  # Highlight the first point
    #
    # plt.title('t-SNE Visualization of Clustering dslr')
    # plt.legend()
    # plt.show()

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, clusters_ℓ)

    # Plot confusion matrix as heatmap
    plt.figure(figsize=(8, 6))
    #sns.heatmap(cm, annot=True, fmt='d', cmap='blues')
    # Define custom colors for the palette (green and red)
    colors = ["#ffd700", "#800020", "#006400"]  # Gold, Burgundy, Dark Green

    # Create a custom colormap using ListedColormap
    custom_cmap = ListedColormap(colors)
    #sns.heatmap(cm, annot=True, cmap=custom_cmap, fmt='d')
    sns.heatmap(cm, annot=True, fmt='d', cmap='blues')


    # plt.title('Confusion Matrix')
    # plt.xlabel('Predicted Label')
    # plt.ylabel('True Label')
    # plt.show()





if __name__ == "__main__":
    train_aris=[]
    #for i in [32,64,128,256,612]:
    r"""Features must be an array of n_domains arrays"""
    features = np.load('Data/resnet50-all--modern_office31.npy', allow_pickle=True)

    r"""Labels must be an array of shape (TotalnumberFeatures,num_classes)"""

    labels = np.load('Data/labels-resnet50-all--modern_office31.npy', allow_pickle=True)
    iteration_aris=[]
    for j in range(5):

        ari=main(features,labels,"Domain_2",0,1)
        iteration_aris.append(ari)
        print("iteration_aris",iteration_aris)
    print("Average ARI Performance ",np.mean(iteration_aris))






