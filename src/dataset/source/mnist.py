from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Dataset
from src.dataset.utils import split_data, create_dataset

def feed_server_with_data_MNIST(val_size, train_batch_size, config, adversarial_clients, CML=False):
    lean = config["configuration"]
    learning_algorithm = "learning_config"
    users = config[learning_algorithm]["K"]
    
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,)),])
    trainset = datasets.MNIST(config["collection"]["path"]["training"], download=True, train=True, transform=transform)
    testset= datasets.MNIST(config["collection"]["path"]["testing"], download=True, train=False, transform=transform)

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