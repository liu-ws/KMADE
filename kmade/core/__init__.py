from . import utils
from . import loss
from . import saving
from . import expr
from . import kmades
from . import kmafs
from . import kvaes

# Import core classes and functions directly into the kmade.core namespace
from .kmades import SGKMADE, MGKMADE, CSGKMADE, CMGKMADE
from .utils import (
    discrete_sample,
    gaussian_kernel_mmd,
    MMD,
    wasserstein_distances,
    pp_plot,
    corner_plot,
)
from .saving import save_model, load_model, save_expr, read_expr
from .expr import forward, sampler, sampler_kmaf, compute_u, likelihood, compute_kmaf
from .kmafs import SGKMAF, CSGKMAF, MGKMAF, CMGKMAF
from .kvaes import MGKVAE
from .loss import loss_function

# List the names of the imported objects in __all__ for wildcard imports
__all__ = [
    "utils",
    "loss",
    "saving",
    "expr",
    "kmades",
    "kmafs",
    "kvaes",
    # Classes from kmades.py
    "SGKMADE",
    "MGKMADE",
    "CSGKMADE",
    "CMGKMADE",
    # Functions from utils.py
    "discrete_sample",
    "gaussian_kernel_mmd",
    "MMD",
    "wasserstein_distances",
    "pp_plot",
    "corner_plot",
    # Functions from saving.py
    "save_model",
    "load_model",
    "save_expr",
    "read_expr",
    # Functions from expr.py
    "forward",
    "sampler",
    "sampler_kmaf",
    "compute_u",
    "likelihood",
    "compute_kmaf",
    # Classes from kmafs.py
    "SGKMAF",
    "CSGKMAF",
    "MGKMAF",
    "CMGKMAF",
    # Classes from kvaes.py
    "MGKVAE",
    # Functions from loss.py
    "loss_function",
]
