from src.dataset.source.cifar10 import feed_server_with_data_CIFAR10
from src.dataset.source.cifar100 import feed_server_with_data_CIFAR100
from src.dataset.source.mnist import feed_server_with_data_MNIST
from src.dataset.source.fmnist import feed_server_with_data_FashionMNIST
from src.dataset.source.domainnet import feed_server_with_data_DomainNet
from src.dataset.source.svhn import feed_server_with_data_SVHN
from src.dataset.source.isic2019 import feed_server_with_data_ISIC2019

__all__ = [
    "feed_server_with_data_CIFAR10",
    "feed_server_with_data_CIFAR100",
    "feed_server_with_data_MNIST",
    "feed_server_with_data_FashionMNIST",
    "feed_server_with_data_DomainNet",
    "feed_server_with_data_SVHN",
    "feed_server_with_data_ISIC2019",
]