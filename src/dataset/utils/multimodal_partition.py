import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Any
from collections import Counter

def iid_partition(dataset, num_users):
    """
    I.I.D partitioning of data over users
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def dirichlet_non_iid_partition(dataset, num_users, alpha):
    """
    Non-I.I.D partitioning of data over users using Dirichlet distribution
    """
    labels = dataset.labels if hasattr(dataset, 'labels') else np.array([y for _, y in dataset])
    num_classes = len(np.unique(labels))
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    for k in range(num_classes):
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet(np.repeat(alpha, num_users))
        proportions = np.array([p * (len(idx_j) < len(dataset) / num_users) for p, idx_j in zip(proportions, dict_users.values())])
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        dict_users = {i: np.concatenate((dict_users[i], idx_k[proportions[i-1] if i > 0 else 0:proportions[i]])) for i in range(num_users)}

    return dict_users

def pathological_non_iid_partition(dataset, num_users, num_shards):
    """
    Non-I.I.D parititioning of data over users
    """
    labels = dataset.labels if hasattr(dataset, 'labels') else np.array([y for _, y in dataset])
    num_items = int(len(dataset)/num_shards)
    dict_users, all_idxs = {i: np.array([], dtype='int64') for i in range(num_users)}, [i for i in range(len(dataset))]
    idxs_labels = np.vstack((all_idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    for i in range(num_users):
        rand_set = set(np.random.choice(num_shards, 2, replace=False))
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_items:(rand+1)*num_items]), axis=0)

    return dict_users

def split_data(config, dataset, num_users):
    """
    Split dataset among users
    """
    if config['partition'] == 'iid':
        dict_users = iid_partition(dataset, num_users)
    elif config['partition'] == 'non-iid-dirichlet':
        dict_users = dirichlet_non_iid_partition(dataset, num_users, alpha=config['dirichlet_alpha'])
    elif config['partition'] == 'non-iid-pathological':
        dict_users = pathological_non_iid_partition(dataset, num_users, num_shards=200)
    else:
        raise ValueError("Unknown partition method")
    
    return dict_users