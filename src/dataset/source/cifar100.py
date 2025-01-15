from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Dataset
from src.dataset.utils import split_data, create_dataset, split_data_test

def feed_server_with_data_CIFAR100(val_size, train_batch_size, config, adversarial_clients, CML=False):
    lean = config["configuration"]
    learning_algorithm = "learning_config"
    users = config[learning_algorithm]["K"]

    
    if config["learning_config"]["global_model"]['type'][:4] == "MLP_":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:  
        transform_train = transforms.Compose([
            transforms.Resize((70, 70)),
            transforms.RandomCrop((64, 64)),
            transforms.ToTensor()
        ])
    
    trainset = datasets.CIFAR100(config["collection"]["path"]["training"], download=True, train=True, transform=transform_train)
    testset= datasets.CIFAR100(config["collection"]["path"]["testing"], download=True, train=False, transform=transform_train)
       
    if CML:
        return trainset, trainset
        
    train_dict_users, unique_labels = split_data(config, trainset, val_size)
    valid_dict_users_iid, valid_dict_users_niid, unique_labels = split_data_test(config, testset, val_size, unique_labels)
    
    if not bool(valid_dict_users_niid):
        users_training_data, users_validation_data_iid = create_dataset(users, adversarial_clients, train_dict_users, valid_dict_users_iid, valid_dict_users_niid, train_batch_size, trainset, testset, config)
    
        return users_training_data, users_validation_data_iid, testset, unique_labels
    else:
        users_training_data, users_validation_data_iid, users_validation_data_niid = create_dataset(users, adversarial_clients, train_dict_users, valid_dict_users_iid, valid_dict_users_niid, train_batch_size, trainset, testset, config)
    
        return users_training_data, users_validation_data_iid, users_validation_data_niid, testset, unique_labels

