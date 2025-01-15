from botorch.utils.sampling import sample_simplex
from botorch import fit_gpytorch_model, fit_fully_bayesian_model_nuts
from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
# from botorch.optim import optimize_acqf, optimize_acqf_list
from botorch.models import SingleTaskGP, ModelListGP, KroneckerMultiTaskGP
from botorch.models.multitask import MultiTaskGP
from botorch.sampling import IIDNormalSampler, SobolQMCNormalSampler
from botorch.acquisition import ExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective, ConstrainedMCObjective
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement, 
    qNoisyExpectedImprovement 
    #, qExpectedHypervolumeImprovement, qNoisyExpectedHypervolumeImprovement
)
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood # , VariationalELBO
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)

from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP

from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)