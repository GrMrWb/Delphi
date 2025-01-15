import numpy as np
import torch, copy
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Dataset
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

def split_data(config, trainset, val_size):
    selection = config["collection"]["selection"]
    learning_algorithm = f"learning_config"
    users = config[learning_algorithm]["K"]
    unique_labels = []
    
    if config["collection"]["datasets"][selection]["iid"]:
        train_dict_users, _, _ = iid_partition(trainset, users, val_size)
    else:
        if config["collection"]["non_iid"]["type"] == "dirichlet":
            train_dict_users, _ = dirichlet_non_iid_partition(trainset, users, config["collection"]["non_iid"]["alpha"], config["collection"]["datasets"][selection]["classes"], val_size)
        elif config["collection"]["non_iid"]["type"] == "pathological":
            train_dict_users, _, unique_labels = pathological_non_iid_partition(trainset, users, val_size=val_size)
        elif config["collection"]["non_iid"]["type"] == "imbalanced_per_class":
            train_dict_users, _, unique_labels = imbalanced_partition_per_class(trainset, users, config["collection"]["datasets"][selection]["classes"], config["collection"]["non_iid"]["imbalanced_ratio"])    
        elif config["collection"]["non_iid"]["type"] == "imbalanced":
            train_dict_users, _, unique_labels = imbalanced_partition(trainset, users, config["collection"]["datasets"][selection]["classes"], config["collection"]["non_iid"]["imbalanced_ratio"])    
    
    return train_dict_users, unique_labels

def split_data_test(config, testset, val_size, unique_labels):
    selection = config["collection"]["selection"]
    learning_algorithm = f"learning_config"
    users = config[learning_algorithm]["K"]
    
    val_size = 0.5
    
    train_set_length = int((1-val_size)*len(testset))
    valid_set_length = int((val_size)*len(testset))
    
    sets = torch.utils.data.random_split(testset, [train_set_length, valid_set_length])

    test_dict_users_iid, _, _ = iid_partition(sets[1], users)
    
    if config["collection"]["datasets"][selection]["iid"]:
        test_dict_users_niid = {}
    else:
        if config["collection"]["non_iid"]["type"] == "dirichlet":
            test_dict_users_niid, _ = dirichlet_non_iid_partition(sets[1], users, config["collection"]["non_iid"]["alpha"], config["collection"]["datasets"][selection]["classes"], val_size)
        elif config["collection"]["non_iid"]["type"] == "pathological":
            test_dict_users_niid, _, unique_labels_niid = pathological_non_iid_partition(sets[1], users, val_size=val_size, val_labels=unique_labels)
        elif config["collection"]["non_iid"]["type"] == "imbalanced_per_class":
            test_dict_users_niid, _, unique_labels = imbalanced_partition_per_class(sets[1], users, config["collection"]["datasets"][selection]["classes"], config["collection"]["non_iid"]["imbalanced_ratio"])    
        elif config["collection"]["non_iid"]["type"] == "imbalanced":
            test_dict_users_niid, _, unique_labels = imbalanced_partition(sets[1], users, config["collection"]["datasets"][selection]["classes"], config["collection"]["non_iid"]["imbalanced_ratio"], val_labels=unique_labels)    
        
    return test_dict_users_iid, test_dict_users_niid, unique_labels

def pathological_non_iid_partition(dataset, num_users, val_size=0.25, val_labels=None):
    """
    Sample non-I.I.D client data from any dataset
    :param dataset:
    :param num_users:
    :return:
    """
    
    try:
        labels = dataset.train_labels.numpy()
        num_class_per_client = int(len(np.unique(labels))/num_users)
        if num_class_per_client < 10:
            num_shards, num_imgs = 10, int(len(labels)/10)
        else:
            num_shards, num_imgs = num_class_per_client, int(len(labels)/num_class_per_client)
        idxs = np.arange(num_shards*num_imgs)
    except:
        if hasattr(dataset, "dataset"):
            labels = dataset.dataset.targets
        else:
            labels = dataset.targets
        num_class_per_client = int(len(np.unique(labels))/num_users)
        if num_class_per_client < 10:
            num_shards, num_imgs = 10, int(len(labels)/10)
        else:
            num_shards, num_imgs = num_class_per_client, int(len(labels)/num_class_per_client)

        idxs = np.arange(num_shards*num_imgs)
    idx_shard = [i for i in range(0,num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    unique_labels = {i: np.array([], dtype='int64') for i in range(num_users)}

    try:
        idxs_labels = np.vstack((idxs, labels))
    except ValueError:
        idxs = np.arange(len(labels))
        idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    num_imgs = 5000 if num_imgs > 5000 else num_imgs
    
    # divide and assign
    if  val_labels==None:
        for i in range(num_users):
            try:
                rand_set = set(np.random.choice(idx_shard, 2, replace=True))
            except:
                rand_set = set([np.random.randint(num_class_per_client), np.random.randint(num_class_per_client)])
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
                unique_labels[i] = np.concatenate((unique_labels[i], np.unique(idxs_labels[1][rand*num_imgs:(rand+1)*num_imgs])), axis=0)
    else:
        i=0
        for values in val_labels.values():
            for value in values:
                dict_users[i] = np.concatenate((dict_users[i], idxs[value*num_imgs:(value+1)*num_imgs]), axis=0)
                unique_labels[i] = np.concatenate((unique_labels[i], np.unique(idxs_labels[1][value*num_imgs:(value+1)*num_imgs])), axis=0)
            i+=1
        i-=1
    return dict_users, len(dict_users[i]), unique_labels

def pathological_non_iid_partition_DomainNet(num_users:int, num_domains:int = 6)->np.ndarray:
    # ONLY FOR 6 users
    s = np.zeros((num_domains, num_users))
    distro = [0 for _ in range(num_users)]
    i = np.random.randint(1,num_users)
    j = abs(i-3)
    distro[i] = abs(num_users/num_domains -1)
    distro[j] = 1-(distro[i])
    distro = [float(x) / np.sum(distro) for x in distro]
    for i in range(num_domains):
        distro.append(distro.pop(0))
        s[i] = distro
    print(s)
    return s.transpose()

def iid_partition(dataset, num_users, val_size=0.25):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
        
    num_items = int((len(dataset))/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    
    return dict_users, "" ,  num_items

def dirichlet_non_iid_partition(dataset, num_users, alpha, num_classes, val_size=0.25):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    
    idxs = [[] for i in range(num_classes)]
    
    for i in range(len(dataset)):
        idxs[dataset[i][1]].append(i)
    
    alpha = [alpha for i in range(num_classes)]
    s = np.random.dirichlet((alpha),num_users).transpose()
    
    train_dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    
    for i in range(num_users):
        data = (s[:,i]*(len(dataset)/num_users)).astype(int)
        data[num_classes-1] = len(dataset)/num_users-sum(data[0:(num_classes-1)])
        for j in range(num_classes):
            if len(idxs[j])>data[j]: 
                each_digit = np.random.choice(idxs[j], data[j], replace=False)
                train_dict_users[i] = np.concatenate((train_dict_users[i], each_digit), axis=0)               
                idxs[j] = list(set(idxs[j]) - set(each_digit))
            else:
                train_dict_users[i] = np.concatenate((train_dict_users[i], idxs[j]), axis=0)

    return train_dict_users, len(train_dict_users[i])

def imbalanced_partition_per_class(dataset, num_users, num_classes, alpha=0.25):
    """
    alpha controls the poisson distribution and how many indexes you will get
    """
    
    idxs = [[] for i in range(num_classes)]
    train_dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    
    for i in range(len(dataset)):
        idxs[dataset[i][1]].append(i)
    
    
    s = np.zeros((num_users, num_classes)).transpose()
    distro = np.random.poisson(alpha*num_classes, num_users) # Distribution for data
    distro = [float(x) / np.sum(distro) for x in distro] # Normalise Poisson Distribution
    
    # Populate the array s with the distribution for each user per digit
    for y in range(s.shape[0]):
        distro.append(distro.pop(0))
        s[y] = distro
    
    # Normalise the distribution for each user
    s = s.transpose()
    for y in range(s.shape[0]):
        s[y] = s[y] / np.sum(s[y])
    
    s = s.transpose()   
    for i in range(num_users):
        data = (s[:,i]*(len(dataset)/num_users)).astype(int)
        data[num_classes-1] = len(dataset)/num_users-sum(data[0:(num_classes-1)])
        for j in range(num_classes):
            if len(idxs[j])>data[j]: 
                each_digit = np.random.choice(idxs[j], data[j], replace=False)
                train_dict_users[i] = np.concatenate((train_dict_users[i], each_digit), axis=0)               
                idxs[j] = list(set(idxs[j]) - set(each_digit))
            else:
                train_dict_users[i] = np.concatenate((train_dict_users[i], idxs[j]), axis=0)
    
    return train_dict_users, "" ,  ""

def imbalanced_partition(dataset, num_users, num_classes, alpha=0.25, **kwargs):
    """
    alpha controls the poisson distribution and how many indexes you will get
    """    
    

    distro = np.random.poisson(alpha*num_classes, num_users) # Distribution for data
    distro = [float(x) / np.sum(distro) for x in distro] # Normalise Poisson Distribution
    distro = [1] if np.isnan(distro)[0] else distro
    
    while any(num == 0 for num in distro):
        distro = np.random.poisson(alpha*num_classes, num_users) # Distribution for data
        distro = [float(x) / np.sum(distro) for x in distro] # Normalise Poisson Distribution
        distro = [1] if np.isnan(distro)[0] else distro
        
    num_items = [int(x*len(dataset)) for x in distro]
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items[i], replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    
    return dict_users, "" ,  ""
    

def mnist_imblanced(dataset, num_users,alpha,beta):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    
    idxs = [[] for i in range(10)]
    for i in range(len(dataset)):
        idxs[dataset[i][1]].append(i)
    #print(len(idxs[1]) ) 
    beta = [beta for i in range(100)]
    data=np.random.dirichlet(beta,size=1)*len(dataset)
    datasize = (data[0]).astype(int)
    datasize[num_users-1] = len(dataset)-sum(datasize[0:num_users-1])
    
    alpha = [alpha for i in range(10)]
    s = np.random.dirichlet((alpha),num_users).transpose()
    
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
   # dict_users = [[] for i in range(num_users)]
    #dict_users = {}
    for i in range(num_users):
        
        number_digit = (s[:,i]*(datasize[i])).astype(int)
        number_digit[9] = datasize[i]-sum(number_digit[0:9])
        for j in range(10):
            if len(idxs[j])>number_digit[j]: 
                if i<=98:  
                    each_digit = np.random.choice(idxs[j], number_digit[j], replace=False)
                    dict_users[i] = np.concatenate((dict_users[i], each_digit), axis=0)               
                    idxs[j] = list(set(idxs[j]) - set(each_digit))
                else:
                    dict_users[i] = np.concatenate((dict_users[i], idxs[j]), axis=0) 
            else:
                dict_users[i] = np.concatenate((dict_users[i], idxs[j]), axis=0) 
    
    return dict_users,datasize