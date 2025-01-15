from src.learning.utils.aggregation import FedAvg, Krum, KrumCuda, PFedAvg

from src.learning.utils.custom_optimiser import (
    pFedMeOptimizer, PTFedProxLoss, PerturbedGradientDescent,
    CustomCrossEntropy, custom_cross_entropy
)

from src.learning.utils.loss_function import (
    choose_loss_fn, choose_optimizer
)

from src.learning.utils.git import (
    commit_to_git, 
    commit_results
)

from src.learning.utils.gpu_utilities import get_gpu_temperature

# from src.learning.utils.sl_trainer import SplitTrainer

__all__ = [
    'pFedMeOptimizer',
    'PTFedProxLoss',
    "PerturbedGradientDescent",
    'FedAvg',
    'Krum',
    'KrumCuda',
    'PFedAvg',
    'commit_to_git',
    'commit_results',
    'CustomCrossEntropy',
    'custom_cross_entropy',
    'choose_optimizer',
    'choose_loss_fn', 
    'get_gpu_temperature',
    # 'SplitTrainer'
]