import numpy as np
import torch
from torch.utils.data import Dataset, Subset

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label, center = self.dataset[self.idxs[item]]
        return image, label

def iid_partition_isic2019(dataset, num_users):
    """
    I.I.D partitioning of data over users
    """
    num_items = len(dataset) // num_users
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = list(np.random.choice(all_idxs, num_items, replace=True))
        # all_idxs = all_idxs - dict_users[i]
    return dict_users

def non_iid_partition_isic2019(dataset, num_users, num_shards, num_imgs):
    """
    non-I.I.D parititioning of data over users
    """
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = np.array(dataset.labels)

    # sort labels
    idxs_labels = np.vstack((idxs, labels[:23200]))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

def dirichlet_partition_isic2019(dataset, num_users, alpha):
    """
    Non-I.I.D partitioning of data over users using Dirichlet distribution
    """
    min_size = 0
    K = len(set(dataset.labels))  # Number of classes
    N = len(dataset)  # Number of samples
    
    while min_size < 10:
        idx_batch = [[] for _ in range(num_users)]
        for k in range(K):
            idx_k = np.where(np.array(dataset.labels) == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_users))
            proportions = np.array([p * (len(idx_j) < N / num_users) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    dict_users = {i: idx_batch[i] for i in range(num_users)}
    return dict_users

def create_isic2019_partition(dataset, num_users, partition_type='iid', alpha=0.5, num_shards=200):
    """
    Create a partition of the ISIC2019 dataset based on specified type
    """
    if partition_type == 'iid':
        return iid_partition_isic2019(dataset, num_users)
    elif partition_type == 'non_iid':
        num_imgs = len(dataset) // num_shards
        return non_iid_partition_isic2019(dataset, num_users, num_shards, num_imgs)
    elif partition_type == 'dirichlet':
        return dirichlet_partition_isic2019(dataset, num_users, alpha)
    else:
        raise ValueError("Invalid partition type. Choose 'iid', 'non_iid', or 'dirichlet'.")

def get_user_data_isic2019(dataset, dict_users, user_id):
    """
    Retrieve data for a specific user
    """
    user_indices = dict_users[user_id]
    return Subset(dataset, user_indices)

def compute_class_distribution(dataset, indices):
    """
    Compute the distribution of classes for a subset of data
    """
    labels = [dataset.labels[i] for i in indices]
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    data ={f"{unique_labels[i]}" : int(counts[i]) for i in range(len(unique_labels)) }
    return data

def compute_center_distribution(dataset, indices):
    """
    Compute the distribution of centers for a subset of data
    """
    centers = [dataset.centers[i] for i in indices]
    unique_centers, counts = np.unique(centers, return_counts=True)
    return dict(zip(unique_centers, counts))