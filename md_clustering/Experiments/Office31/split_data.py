
from sklearn.model_selection import train_test_split
from KMeans_baseline import *

# Suppress warnings
warnings.filterwarnings('ignore')

def split_data(domain,features,labels,test_size):
    # Convert features to tensors
    # Convert features to tensors
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

    # Reorder data and labels according to the domain
    for i in range(num_domains):
        if domain == f'Domain_{i+1}':
            features = all_data[i]
            labels = y_labels[i]
            break
    else:
        raise ValueError(f"Invalid domain '{domain}'")

    # Print lengths of features and labels for debugging
    print('len features to split:', len(features))
    print('len labels to split:', len(labels))

    # Convert torch tensor to numpy array
    features_np = features.numpy()
    # Convert list to numpy array
    labels_np = np.array(labels)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(features_np, labels_np, test_size=test_size, stratify=labels_np, random_state=42)

    return X_train, X_test, y_train, y_test



if __name__ == "__main__":
    # Command line argument parsing
    features = np.load('Data/resnet50-all--modern_office31.npy', allow_pickle=True)
    labels = np.load('Data/labels-resnet50-all--modern_office31.npy', allow_pickle=True)

    # Split data based on the provided domain;è
    X_train, X_test, y_train, y_test = split_data('Domain_1',features,labels,test_size=0.7)
