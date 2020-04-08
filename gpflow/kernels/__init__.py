from .base import Combination, Kernel, Product, Sum
from .convolutional import Convolutional
from .changepoints import ChangePoints
#from .my_changepoints import my_ChangePoints
#from .my2_changepoints import my2_ChangePoints
#from .my3_changepoints import my3_ChangePoints
#from .my4_changepoints import my4_ChangePoints
from .new_changepoints import new_ChangePoints
from .linears import Linear, Polynomial
from .misc import ArcCosine, Coregion
from .mo_kernels import (MultioutputKernel, SeparateIndependent, SharedIndependent,
                         IndependentLatent, LinearCoregionalization)
from .periodic import Periodic
from .statics import Constant, Static, White
from .stationaries import (SquaredExponential, Cosine, Exponential, Matern12,
                           Matern32, Matern52, RationalQuadratic, Stationary)

Bias = Constant
RBF = SquaredExponential
