from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Dataset
from src.dataset.source.domainnet import get_DomainNet_Dataloader
from src.dataset import source
import logging
import random

logger = logging.getLogger(__name__)

from src.dataset.utils import *

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        
        return image, label

class DataSources():
    def __init__(self, config, data_iq=False, **kwargs):
        self.selection = config["collection"]["selection"]
        
        self.transform = None
        
        self.trainset = None
        self.testset = None

        self.train_batch_size = config["collection"]["datasets"][self.selection]["training_batch_size"]
        self.test_batch_size = config["collection"]["datasets"][self.selection]["testing_batch_size"]
        self.val_size = config["collection"]["datasets"][self.selection]["val_size"]

        self.adversarial_clients = kwargs["adversarial_clients"] if "adversarial_clients" in kwargs else []
        
        self.users_training_data = []

        self.dict_users = []
        
        self.unique_labels = None
        
        self.data_iq = data_iq
        
        if config["collection"]["datasets"][self.selection]["iid"] or self.selection in ["DomainNet", "ISIC2019"]:
            self.users_validation_data = []
            self.users_training_data, self.users_validation_data, self.testset, self.unique_labels = getattr(source, f"feed_server_with_data_{self.selection}")(self.val_size, self.train_batch_size, config, self.adversarial_clients)
        else:
            self.users_validation_data_iid = []
            self.users_validation_data_niid = []
            self.users_training_data, self.users_validation_data_iid, self.users_validation_data_niid, self.testset, self.unique_labels = getattr(source, f"feed_server_with_data_{self.selection}")(self.val_size, self.train_batch_size, config, self.adversarial_clients)
        
        self.testset = self.create_dataloader(test=True)        

    def create_dataloader(self, train=False, test=False, idx=None):
        if test:
            data = DataLoader(self.testset, batch_size=self.test_batch_size , shuffle=True)
            return data
        elif train:
            data = DataLoader(DatasetSplit(self.trainset, self.dict_users[idx]), batch_size=self.train_batch_size, shuffle=True)
            return data

    def get_training_data(self):
        return self.users_training_data

    def get_validation_data(self):
        return self.users_validation_data
            