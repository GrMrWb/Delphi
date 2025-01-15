import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Any
from collections import Counter

class MultiModalDatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        return self.dataset[self.idxs[item]]


def create_multi_modal_dataset(dataset, users, adversarial_clients, train_dict_users, valid_dict_users_iid, valid_dict_users_niid,
                               train_batch_size, modalities, labels, config, **kwargs):
    users_training_data, users_validation_data_iid, users_validation_data_niid = [], [], []

    for idx in range(users):
        train_idx = list(train_dict_users[idx])
        valid_idx_iid = list(valid_dict_users_iid[idx])
        
        train_dataset = MultiModalDatasetSplit(dataset, train_idx)
        valid_dataset_iid = MultiModalDatasetSplit(dataset, valid_idx_iid)

        training_data = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        validation_data_iid = DataLoader(valid_dataset_iid, batch_size=train_batch_size, shuffle=False)

        # Calculate unique labels and counts for training data
        train_labels = labels[train_idx]
        train_unique_labels, train_counts = np.unique(train_labels, return_counts=True)
        train_unique_counter = dict(zip(train_unique_labels, train_counts))

        # Calculate unique labels and counts for IID validation data
        valid_labels_iid = labels[valid_idx_iid]
        valid_unique_labels_iid, valid_counts_iid = np.unique(valid_labels_iid, return_counts=True)
        valid_unique_counter_iid = dict(zip(valid_unique_labels_iid, valid_counts_iid))

        users_training_data.append({
            'dataloader': training_data,
            'unique_labels': train_unique_labels,
            'unique_counter': train_unique_counter,
        })

        users_validation_data_iid.append({
            'dataloader': validation_data_iid,
            'unique_labels': valid_unique_labels_iid,
            'unique_counter': valid_unique_counter_iid,
        })

        if valid_dict_users_niid:
            valid_idx_niid = list(valid_dict_users_niid[idx])
            valid_dataset_niid = MultiModalDatasetSplit(dataset, valid_idx_niid)
            validation_data_niid = DataLoader(valid_dataset_niid, batch_size=train_batch_size, shuffle=False)

            valid_labels_niid = labels[valid_idx_niid]
            valid_unique_labels_niid, valid_counts_niid = np.unique(valid_labels_niid, return_counts=True)
            valid_unique_counter_niid = dict(zip(valid_unique_labels_niid, valid_counts_niid))

            users_validation_data_niid.append({
                'dataloader': validation_data_niid,
                'unique_labels': valid_unique_labels_niid,
                'unique_counter': valid_unique_counter_niid,
            })

    if not valid_dict_users_niid:
        return users_training_data, users_validation_data_iid
    else:
        return users_training_data, users_validation_data_iid, users_validation_data_niid