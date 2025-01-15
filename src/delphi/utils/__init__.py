import matplotlib.pyplot as plt

from src.delphi.utils.activations import (
    capture_data
)

activations = [
    "capture_data"
]

from src.delphi.utils import bo_libraries

from src.delphi.utils.features import (
    find_the_best_features,
    find_the_best_features_FL,
    get_bounds
)

features = [
    "find_the_best_features",
    "find_the_best_features_FL",
    "get_bounds"
]

from src.delphi.utils.filtering import (
    get_the_best_samples,
    filter_data_for_BO,
    filter_output
)

filtering = [
    "get_the_best_samples",
    "filter_data_for_BO",
    "filter_output"
]

from src.delphi.utils.historical_data import (
    get_historical_data,
    get_historical_gradients,
    get_historical_distribution
    
)

historical_data = [
    "get_historical_data",
    "get_historical_gradients",
    "get_historical_distribution"
]

from src.delphi.utils.measurements import (
    confidence_level, 
    wasserstein_matrix,
    kl_divergence, 
    logits_score, 
    score_bo, 
    get_true_positives, 
    get_uncertainty,
    mi_BO
)

measurements = [
    "confidence_level",
    "wasserstein_matrix",
    "kl_divergence",
    "logits_score",
    "score_bo",
    "get_true_positives",
    "get_uncertainty",
    "mi_BO"
]

from src.delphi.utils.pdfs import (
    pdf_for_output,
    pdf_for_outputs
)

pdfs = [
    "pdf_for_output",
    "pdf_for_outputs"
    
]

from src.delphi.utils.similarity import (
    get_similarity_features,
    get_similarity_between_server
)

similarity = [
    "get_similarity_features",
    "get_similarity_between_server"
    
]

from src.delphi.utils.specialised_norms import (
    l2_norm,
    get_delta_weights
)

specialised_norms = [
    "l2_norm",
    "get_delta_weights"
]

from src.delphi.utils.weight_distribution import (
    get_weight_distribution
)

weight_distribution = [
    "get_weight_distribution"
]

from src.delphi.utils.weight_io import (
    read_weights,
    write_weights
)

weight_io = [
    "read_weights",
    "write_weights"
]

from src.delphi.utils.weight_modifier import (
    modify_the_weights_with_single_neuron
)

weight_modifier = [
    "modify_the_weights_with_single_neuron"
]

__all__ = [
    "bo_libraries",
]



__all__ = __all__ + activations + features + filtering + historical_data + measurements + pdfs + similarity + specialised_norms + weight_distribution + weight_io + weight_modifier

