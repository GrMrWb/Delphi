import sys
import os
sys.path.insert(0, os.getcwd())

import time
import torch
import numpy as np
import os
import random
import config
from tqdm import tqdm 
import torchvision.transforms as transforms
from os import path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from src.dataset.utils import DatasetSplit, pathological_non_iid_partition_DomainNet

config = config.server_configuration

class DomainNet(Dataset):
    def __init__(self, data_paths, data_labels, transforms, domain_name):
        super(DomainNet, self).__init__()
        self.data_paths = data_paths
        self.targets = data_labels
        self.transforms = transforms
        self.domain_name = [domain_name]

    def __getitem__(self, index):
        img = Image.open(self.data_paths[index])
        if not img.mode == "RGB":
            img = img.convert("RGB")
        label = self.targets[index]
        img = self.transforms(img)

        return img, label

    def __len__(self):
        return len(self.data_paths)

def read_DomainNet_data(dataset_path, domain_name, split="train"):
    data_paths = []
    data_labels = []
    split_file = path.join(dataset_path, "splits", "{}_{}.txt".format(domain_name, split))
    with open(split_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            data_path, label = line.split(' ')
            data_path = path.join(dataset_path, data_path)
            label = int(label)
            data_paths.append(data_path)
            data_labels.append(label)
    return data_paths, data_labels

def get_DomainNet_Dataloader(val_size=0.15, dataset_path="dataset/DomainNet/rawdata/", domain_name="real"):
    if not os.path.exists(dataset_path):
        download_DomainNet()
    
    train_data_paths, train_data_labels = read_DomainNet_data(dataset_path, domain_name, split="train")
    test_data_paths, test_data_labels = read_DomainNet_data(dataset_path, domain_name, split="test")
    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.75, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    transforms_test = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    
    split = int(np.floor(val_size * len(train_data_paths)))
    train_data_paths_trimmed, valid_data_paths = train_data_paths[split:], train_data_paths[:split]
    train_data_labels_trimmed, valid_data_labels = train_data_labels[split:], train_data_labels[:split]

    train_dataset = DomainNet(train_data_paths_trimmed, train_data_labels_trimmed, transforms_train, domain_name)
    valid_dataset = DomainNet(valid_data_paths, valid_data_labels, transforms_train, domain_name)
    test_dataset = DomainNet(test_data_paths, test_data_labels, transforms_test, domain_name)
    
    return train_dataset, valid_dataset, test_dataset

def download_DomainNet(dir_path="dataset/DomainNet/", data_path = "dataset/DomainNet/"):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    root = data_path +"rawdata"
    
    domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
    urls = [
        'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip', 
        'http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip', 
        'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip', 
        'http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip', 
        'http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip', 
        'http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip', 
    ]
    http_head = 'http://csr.bu.edu/ftp/visda/2019/multi-source/'
    # Get DomainNet data
    if not os.path.exists(root):
        os.makedirs(root)
        for d, u in zip(domains, urls):
            os.system(f'wget {u} -P {root}')
            os.system(f'unzip {root}/{d}.zip -d {root}')
            os.system(f'wget {http_head}domainnet/txt/{d}_train.txt -P {root}/splits')
            os.system(f'wget {http_head}domainnet/txt/{d}_test.txt -P {root}/splits')

def split_dataset_in_domains(domain, distribution, dataset_path="dataset/DomainNet/rawdata/", val_size=0.15):
    train_data_paths, train_data_labels = read_DomainNet_data(dataset_path, domain, split="train")
    start_row = 0
    train_users_paths, train_users_labels, val_users_paths, val_users_labels = [], [], [], []
    
    for user in distribution:
        end_row = int(start_row + np.floor(user*len(train_data_paths)))
        
        # train_amount = np.floor((end_row - start_row)*(1-val_size))
        val_amount = int(np.floor((end_row - start_row)*(val_size)))
        
        if not end_row == 0:
            train_users_paths.append(train_data_paths[start_row:end_row-val_amount])
            train_users_labels.append(train_data_labels[start_row:end_row-val_amount])
            val_users_paths.append(train_data_paths[(end_row-val_amount)+1:end_row])
            val_users_labels.append(train_data_labels[(end_row-val_amount)+1:end_row])
            
            start_row = end_row
        else:
            train_users_paths.append([])
            train_users_labels.append([])
            val_users_paths.append([])
            val_users_labels.append([])
        
    return train_users_paths, train_users_labels, val_users_paths, val_users_labels

def get_DomainNet_Dataloader_testset(domains, dataset_path="dataset/DomainNet/rawdata/"):
    for idx, domain_name in enumerate(domains):
        test_data_paths, test_data_labels = read_DomainNet_data(dataset_path, domain_name, split="test")

        transforms_test = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

        test_dataset = DomainNet(test_data_paths, test_data_labels, transforms_test, domain_name)
        
        if idx==0:
            testset=test_dataset
        else:
            testset.data_paths = testset.data_paths + test_dataset.data_paths
            testset.targets = testset.targets + test_dataset.targets
            testset.domain_name = testset.domain_name + test_dataset.domain_name
    
    return test_dataset
    
def feed_server_with_data_DomainNet(val_size, train_batch_size, config, adversarial_clients):
    lean = config["configuration"]
    learning_algorithm = "learning_config"
    users = config[learning_algorithm]["K"]
    
    domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
    shuffled_list = sorted(domains, key=lambda x: random.random())
    
    users_training_data, users_validation_data, unique_labels = [], [], []
    testset = None
    
    # New Code for Distribution
    distribution = pathological_non_iid_partition_DomainNet(num_users=users, num_domains=len(domains)).transpose()
    domains_sets_train_paths = []
    domains_sets_train_labels = []
    domains_sets_valid_paths = []
    domains_sets_valid_labels= []
    
    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.75, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    
    # Split Dataset into Domain
    for idx, domain in enumerate(domains):
        train_users_paths, train_users_labels, val_users_paths, val_users_labels = split_dataset_in_domains(domain, distribution[idx], val_size=val_size)

        domains_sets_train_paths.append(train_users_paths)
        domains_sets_train_labels.append(train_users_labels)
        domains_sets_valid_paths.append(val_users_paths)
        domains_sets_valid_labels.append(val_users_labels)
        
    domains_sets_train_paths = np.array(domains_sets_train_paths)
    domains_sets_train_labels = np.array(domains_sets_train_labels)
    domains_sets_valid_paths = np.array(domains_sets_valid_paths)
    domains_sets_valid_labels = np.array(domains_sets_valid_labels)
    
    # Build the dataset
    for idx in range(0, users):
        train_paths = domains_sets_train_paths[:,idx]
        train_labels = domains_sets_train_labels[:,idx]
        valid_paths = domains_sets_valid_paths[:,idx]
        valid_labels= domains_sets_valid_labels[:,idx]
        index = 0
        for idx_p in range(len(train_paths)):
            
            if len(train_paths[idx_p]) !=0:
                train_dataset = DomainNet(train_paths[idx_p], train_labels[idx_p], transforms_train, domains[idx_p])
                valid_dataset = DomainNet(valid_paths[idx_p], valid_labels[idx_p], transforms_train, domains[idx_p])
                
                training_unique_counters, validation_unique_counters = {}, {}
                
                if index==0:
                    train_set = train_dataset
                    valid_set = valid_dataset
                    index+=1
                else:
                    train_set.data_paths = train_set.data_paths + train_dataset.data_paths
                    train_set.targets = train_set.targets + train_dataset.targets
                    train_set.domain_name = train_set.domain_name + train_dataset.domain_name
                    
                    valid_set.data_paths = valid_set.data_paths + valid_dataset.data_paths
                    valid_set.targets = valid_set.targets + valid_dataset.targets
                    valid_set.domain_name = valid_set.domain_name + valid_dataset.domain_name
                                 
        train_set_dl = DataLoader(train_set, batch_size=train_batch_size, shuffle=True)      
        for tr_idx, (data, target) in enumerate(train_set_dl):
            tmp, counter = torch.unique(target, return_counts=True)
    
            if tr_idx == 0:
                training_unique_labels = tmp
                for temp, count in zip(tmp, counter):
                    training_unique_counters[f"{temp}"] = int(count.numpy())
            else:
                for temp in tmp:
                    training_unique_labels = torch.cat([training_unique_labels, torch.tensor([temp])]) if not temp in training_unique_labels else training_unique_labels
                    training_unique_counters[f"{temp}"] = int(count.numpy())  if not f"{temp}" in training_unique_counters else training_unique_counters[f"{temp}"] + int(count.numpy())
            
            print(f"\rTrainSet{tr_idx}/{len(train_set_dl)} User{idx}", end="\r")
            
        valid_set_dl = DataLoader(valid_set, batch_size=train_batch_size, shuffle=True)     
        for val_idx, (data, target) in enumerate(valid_set_dl):
            tmp, counter = torch.unique(target, return_counts=True)
    
            if val_idx == 0:
                validation_unique_labels = tmp
                for temp, count in zip(tmp, counter):
                    validation_unique_counters[f"{temp}"] = int(count.numpy())
            else:
                for temp in tmp:
                    validation_unique_labels = torch.cat([validation_unique_labels, torch.tensor([temp])]) if not temp in validation_unique_labels else validation_unique_labels
                    validation_unique_counters[f"{temp}"] = int(count.numpy())  if not f"{temp}" in validation_unique_counters else validation_unique_counters[f"{temp}"] + int(count.numpy())
            
            print(f"\rValidSet{val_idx}/{len(valid_set_dl)} User{idx}", end="\r")
        
        users_training_data.append({
            'dataloader' : train_set_dl,
            'unique_labels': training_unique_labels,
            'unique_counter': training_unique_counters,
        })
        users_validation_data.append({
            'dataloader' : valid_set_dl,
            'unique_labels': validation_unique_labels,
            'unique_counter': validation_unique_counters,
        })

    testset = get_DomainNet_Dataloader_testset(domains)
    
    del train_set_dl, valid_set_dl, valid_dataset, train_dataset
    
    return users_training_data, users_validation_data, testset, unique_labels

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.getcwd())
    download_DomainNet()