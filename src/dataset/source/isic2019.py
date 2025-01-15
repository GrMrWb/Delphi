import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from src.dataset.utils import create_isic2019_partition, get_user_data_isic2019, compute_class_distribution, compute_center_distribution

class ISIC2019Dataset(Dataset):
    def __init__(self, image_paths, centers, labels=None, transforms=None):
        self.image_paths = image_paths
        self.centers = centers
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        img = Image.open(img_path).convert('RGB')
        center = self.centers[index]

        if self.transforms:
            img = self.transforms(img)

        if self.labels is not None:
            label = self.labels[index]
            return img, label
        else:
            return img, center

    def __len__(self):
        return len(self.image_paths)

def read_ISIC2019_data(dataset_path, metadata_file, ground_truth_file=None):
    df = pd.read_csv(os.path.join(dataset_path, metadata_file))
    image_paths = [os.path.join(dataset_path, 'ISIC_2019_Training_Input', img_name + '.jpg') for img_name in df['image']]
    centers = df['dataset'].astype('category').cat.codes.values
    unique_centers = df['dataset'].unique()

    gt_df = pd.read_csv(os.path.join(dataset_path, ground_truth_file))
    class_columns = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
    
    # Convert one-hot encoded labels to class indices
    labels = np.argmax(gt_df[class_columns].values, axis=1)
    
    unique_diagnoses = np.array(class_columns)

    return image_paths, centers, labels, unique_diagnoses, unique_centers

def get_ISIC2019_Dataset(dataset_path, metadata_file, ground_truth_file=None):
    image_paths, centers, labels, unique_diagnoses, unique_centers = read_ISIC2019_data(dataset_path, metadata_file, ground_truth_file)

    transforms_data = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = ISIC2019Dataset(image_paths, centers, labels, transforms_data)
    return dataset, unique_diagnoses, unique_centers

def feed_server_with_data_ISIC2019(val_size, train_batch_size, config, adversarial_clients):
    dataset_path = "dataset/isic2019"
    metadata_file = "ISIC_2019_Training_Metadata.csv"
    ground_truth_file = "ISIC_2019_Training_GroundTruth.csv"  # Assuming this file exists

    full_dataset, unique_diagnoses, unique_centers = get_ISIC2019_Dataset(dataset_path, metadata_file, ground_truth_file)

    num_users = config["learning_config"]["K"]
    if config["collection"]["datasets"]["ISIC2019"]["iid"]:
        
        partition_type = "iid"
    else:
        partition_type = "non_iid"
        
    alpha = config["collection"]["datasets"]["ISIC2019"].get("alpha", 0.5)  # for Dirichlet distribution

    dict_users = create_isic2019_partition(full_dataset, num_users, partition_type, alpha)

    users_training_data = []
    users_validation_data = []

    for user_id in dict_users.keys():
        user_indices = dict_users[user_id]
        user_dataset = get_user_data_isic2019(full_dataset, dict_users, user_id)

        # Split user's data into train and validation
        train_size = int(0.8 * len(user_dataset))
        val_size = len(user_dataset) - train_size
        user_train_dataset, user_val_dataset = torch.utils.data.random_split(user_dataset, [train_size, val_size])

        train_loader = DataLoader(user_train_dataset, batch_size=train_batch_size, shuffle=True)
        val_loader = DataLoader(user_val_dataset, batch_size=train_batch_size, shuffle=False)

        train_class_dist = compute_class_distribution(full_dataset, user_train_dataset.indices)
        val_class_dist = compute_class_distribution(full_dataset, user_val_dataset.indices)

        train_center_dist = compute_center_distribution(full_dataset, user_train_dataset.indices)
        val_center_dist = compute_center_distribution(full_dataset, user_val_dataset.indices)

        users_training_data.append({
            'dataloader': train_loader,
            'unique_labels': list(train_class_dist.keys()),
            'unique_counter': train_class_dist,
            'center_distribution': train_center_dist
        })

        users_validation_data.append({
            'dataloader': val_loader,
            'unique_labels':list(val_class_dist.keys()),
            'unique_counter': val_class_dist,
            'center_distribution': val_center_dist
        })

    # Create a small test set from the full dataset
    test_size = int(0.1 * len(full_dataset))
    _, test_dataset = torch.utils.data.random_split(full_dataset, [len(full_dataset) - test_size, test_size])
    testset = DataLoader(test_dataset, batch_size=train_batch_size, shuffle=False)

    return users_training_data, users_validation_data, testset, unique_diagnoses