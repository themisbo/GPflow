from collections.abc import Iterable
from typing import List, Optional, Union

import tensorflow as tf
import numpy as np

from ..base import Parameter
from ..utilities import positive
from .base import Combination, Kernel


class new_ChangePoints(Combination):
    r"""
    The ChangePoints kernel defines a fixed number of change-points along a 1d
    input space where different kernels govern different parts of the space.

    The kernel is by multiplication and addition of the base kernels with
    sigmoid functions (σ). A single change-point kernel is defined as:

        K₁(x, x') * (1 - σ(x)) * (1 - σ(x')) + K₂(x, x') * σ(x) * σ(x')

    where K₁ is deactivated around the change-point and K₂ is activated. The
    single change-point version can be found in \citet{lloyd2014}. Each sigmoid
    is a logistic function defined as:

        σ(x) = 1 / (1 + exp{-s(x - x₀)})

    parameterized by location "x₀" and steepness "s".

    @incollection{lloyd2014,
      author = {Lloyd, James Robert et al},
      title = {Automatic Construction and Natural-language Description of Nonparametric Regression Models},
      booktitle = {Proceedings of the Twenty-Eighth AAAI Conference on Artificial Intelligence},
      year = {2014},
      url = {http://dl.acm.org/citation.cfm?id=2893873.2894066},
    }
    """
    def __init__(self,
                 kernels: List[Kernel],
                 locations: List[float],
                 steepness: Union[float, List[float]] = 1.0,
                 name: Optional[str] = None):
        """
        :param kernels: list of kernels defining the different regimes
        :param locations: list of change-point locations in the 1d input space
        :param steepness: the steepness parameter(s) of the sigmoids, this can be
            common between them or decoupled
        """
        if len(kernels) != len(locations) + 1:
            raise ValueError("Number of kernels ({nk}) must be one more than the number of "
                             "changepoint locations ({nl})".format(nk=len(kernels), nl=len(locations)))

        if isinstance(steepness, Iterable) and len(steepness) != len(locations):
            raise ValueError("Dimension of steepness ({ns}) does not match number of changepoint "
                             "locations ({nl})".format(ns=len(steepness), nl=len(locations)))

        super().__init__(kernels, name=name)

        self.locations = Parameter(locations)
        self.steepness = Parameter(steepness, transform=positive())

    def _set_kernels(self, kernels: List[Kernel]):
        # it is not clear how to flatten out nested change-points
        self.kernels = list(kernels)

    def K(self, X: tf.Tensor, X2: Optional[tf.Tensor] = None) -> tf.Tensor:
        sig_X = self._sigmoids(X)  # N x 1 x Ncp
        sig_X2 = self._sigmoids(X2) if X2 is not None else sig_X

        # `starters` are the sigmoids going from 0 -> 1, whilst `stoppers` go 
        # from 1 -> 0, dimensions are N x N x Ncp
        starters = sig_X[:,:,None] * tf.transpose(sig_X2, perm=(1, 0, 2))
        starters = tf.reduce_sum(starters, 1)
        starters = tf.dtypes.cast(starters, X.dtype)
        stoppers = (1 - sig_X)[:,:,None] * tf.transpose((1 - sig_X2), perm=(1, 0, 2))
        stoppers = tf.reduce_sum(stoppers, 1)
        stoppers = tf.dtypes.cast(stoppers, X.dtype)
        # prepend `starters` with ones and append ones to `stoppers` since the
        # first kernel has no start and the last kernel has no end
        N = tf.shape(X)[0]
        if X2 is not None:
            N2 = tf.shape(X2)[0]
            ones = tf.ones((N, N2, 1), dtype=X.dtype)
        else:
            ones = tf.ones((N, N, 1), dtype=X.dtype)
        starters = tf.concat([ones,starters], axis=2)
        stoppers = tf.concat([stoppers,ones], axis=2)

        # now combine with the underlying kernels
        kernel_stack = tf.stack([k(X, X2) for k in self.kernels], axis=2)
        return tf.reduce_sum(kernel_stack * starters * stoppers, axis=2)

    def K_diag(self, X: tf.Tensor) -> tf.Tensor:
        N = tf.shape(X)[0]
        sig_X = self._sigmoids(X)

        starters = sig_X[:,:,None] * tf.transpose(sig_X, perm=(1, 0, 2))
        starters = tf.reduce_sum(starters, 1)
        starters = tf.dtypes.cast(starters, X.dtype)
        stoppers = (1 - sig_X)[:,:,None] * tf.transpose((1 - sig_X), perm=(1, 0, 2))
        stoppers = tf.reduce_sum(stoppers, 1)
        stoppers = tf.dtypes.cast(stoppers, X.dtype)

        N = tf.shape(X)[0]

        ones = tf.ones((N, N, 1), dtype=X.dtype)
        starters = tf.concat([ones,starters], axis=2)
        stoppers = tf.concat([stoppers,ones], axis=2)

        # now combine with the underlying kernels
        kernel_stack = tf.stack([k(X) for k in self.kernels], axis=2)
        test = tf.reduce_sum(kernel_stack * starters * stoppers, axis=2)
        return tf.linalg.diag_part(test)

    def _sigmoids(self, X: tf.Tensor) -> tf.Tensor:
        locations = tf.sort(self.locations)  # ensure locations are ordered
        locations = tf.reshape(locations, (1, 1, -1))
        steepness = tf.reshape(self.steepness, (1, 1, -1))
        return tf.sigmoid(steepness * (X[:, :, None] - locations))
