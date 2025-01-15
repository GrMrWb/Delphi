import numpy as np
import torch, copy
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Dataset
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        
        return image, label

def create_dataset(users, adversarial_clients, train_dict_users, valid_dict_users_iid, valid_dict_users_niid, train_batch_size, training_set, testing_set, config, val_size=0.25, **kwargs):
    if "validation_set" in kwargs:
        validation_set = kwargs["validation_set"]
    
    users_training_data, users_validation_data_iid = [], []
    
    if bool(valid_dict_users_niid):
        users_validation_data_niid = []
    
    if config["attack"]["available_data"]["type"] != "random":
        lengths = sorted(set(len(value) for value in train_dict_users.values()))
        if config["attack"]["available_data"]["type"] == "max":
            lengths = lengths[-len(adversarial_clients):]  
        else:
            lengths[:len(adversarial_clients)]
    
        keys = [key for key, value in train_dict_users.items() if len(value) in lengths]
        
        counter = 0
        for client in adversarial_clients:
            temp = train_dict_users[keys[counter]]
            train_dict_users[keys[counter]] = train_dict_users[client]
            train_dict_users[client] = temp
            
            if bool(valid_dict_users_niid):
                temp = valid_dict_users_niid[keys[counter]]
                valid_dict_users_niid[keys[counter]] = valid_dict_users_niid[client]
                valid_dict_users_niid[client] = temp
                
            counter += 1

    for idx in range(0,users):
        train_idx = list(train_dict_users[idx])
        
        training_data = DataLoader(DatasetSplit(training_set, train_idx), batch_size=train_batch_size, shuffle=True)
        
        training_unique_counter, validation_unique_counter_iid = {}, {}
        
        for tr_idx, (data, target) in enumerate(training_data):
            tmp, counter = torch.unique(target, return_counts=True)
            if tr_idx == 0:
                training_unique_labels = tmp
                for temp, count in zip(tmp, counter):
                    training_unique_counter[f"{temp}"] = int(count.numpy())
            else:
                for temp, count in zip(tmp, counter):
                    training_unique_labels = torch.cat([training_unique_labels, torch.tensor([temp])]) if not temp in training_unique_labels else training_unique_labels
                    training_unique_counter[f"{temp}"] = int(count.numpy()) if not f"{temp}" in training_unique_counter else training_unique_counter[f"{temp}"] + int(count.numpy())

        valid_idx_iid = list(valid_dict_users_iid[idx])
        validation_data_iid = DataLoader(DatasetSplit(testing_set, valid_idx_iid), batch_size=train_batch_size, shuffle=True)
        
        for val_idx, (data, target) in enumerate(validation_data_iid):
            tmp, counter = torch.unique(target, return_counts=True)
            
            if val_idx == 0:
                validation_unique_labels_iid = tmp
                for temp, count in zip(tmp, counter):
                    validation_unique_counter_iid[f"{temp}"] = int(count.numpy())
            else:
                for temp in tmp:
                    validation_unique_labels_iid = torch.cat([validation_unique_labels_iid, torch.tensor([temp])]) if not temp in validation_unique_labels_iid else validation_unique_labels_iid
                    validation_unique_counter_iid[f"{temp}"] = int(count.numpy())  if not f"{temp}" in validation_unique_counter_iid else validation_unique_counter_iid[f"{temp}"] + int(count.numpy())
        
        if bool(valid_dict_users_niid):
            validation_unique_counter_niid = {}
            valid_idx_niid = list(valid_dict_users_niid[idx])
            validation_data_niid = DataLoader(DatasetSplit(testing_set, valid_idx_niid), batch_size=train_batch_size, shuffle=True)
            
            for val_idx, (data, target) in enumerate(validation_data_niid):
                tmp, counter = torch.unique(target, return_counts=True)
                
                if val_idx == 0:
                    validation_unique_labels_niid = tmp
                    for temp, count in zip(tmp, counter):
                        validation_unique_counter_niid[f"{temp}"] = int(count.numpy())
                else:
                    for temp in tmp:
                        validation_unique_labels_niid = torch.cat([validation_unique_labels_niid, torch.tensor([temp])]) if not temp in validation_unique_labels_niid else validation_unique_labels_niid
                        validation_unique_counter_niid[f"{temp}"] = int(count.numpy())  if not f"{temp}" in validation_unique_counter_niid else validation_unique_counter_niid[f"{temp}"] + int(count.numpy())
            
            users_validation_data_niid.append({
                'dataloader' : validation_data_niid,
                'unique_labels': validation_unique_labels_niid,
                'unique_counter': validation_unique_counter_niid,
            })
        
        users_training_data.append({
            'dataloader' : training_data,
            'unique_labels': training_unique_labels,
            'unique_counter': training_unique_counter,
        })
        
        users_validation_data_iid.append({
            'dataloader' : validation_data_iid,
            'unique_labels': validation_unique_labels_iid,
            'unique_counter': validation_unique_counter_iid,
        })
    
    if not bool(valid_dict_users_niid):
        return users_training_data, users_validation_data_iid
    else:
        return users_training_data, users_validation_data_iid, users_validation_data_niid