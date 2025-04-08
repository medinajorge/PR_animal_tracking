import torch
import types
try:
    import tensorflow as tf
    import tensorboard as tb
    tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
except:
    pass
def warn(*args, **kwargs):
    """
    Ignore warnings.
    """
    pass
import warnings
warnings.warn = warn

# configure network and trainer
from pytorch_forecasting.metrics import MultiLoss, QuantileLoss
from pytorch_forecasting.metrics.distributions import MQF2DistributionLoss
from pytorch_forecasting import TemporalFusionTransformer
try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger
    from lightning.pytorch import LightningModule, Callback
except:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning import LightningModule, Callback

import torch
import torch.distributions as dist
from pytorch_forecasting.metrics.base_metrics import MultivariateDistributionLoss
from sklearn.base import BaseEstimator
import torch.nn.functional as F
from typing import List
from copy import copy, deepcopy
from typing import Dict, List, Tuple, Union
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
from torchmetrics import Metric as LightningMetric

from pytorch_forecasting import Baseline
from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from pytorch_forecasting.metrics import MAE, MAPE, MASE, RMSE, SMAPE, MultiHorizonMetric, MultiLoss, QuantileLoss, DistributionLoss
from pytorch_forecasting.models.base_model import BaseModelWithCovariates, BaseModel
from pytorch_forecasting.models.nn import LSTM, MultiEmbedding
from pytorch_forecasting.models.temporal_fusion_transformer.sub_modules import (
    AddNorm,
    GateAddNorm,
    GatedLinearUnit,
    GatedResidualNetwork,
    InterpretableMultiHeadAttention,
    ScaledDotProductAttention,
    VariableSelectionNetwork,
)
from pytorch_forecasting.utils import create_mask, detach, integer_histogram, masked_op, padded_stack, to_list, unpack_sequence, unsqueeze_like, move_to_device
from pytorch_forecasting.models.base_model import _torch_cat_na, _concatenate_output
import os
import gc
import psutil
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import torch
import torch.nn as nn
from torch.nn.utils import rnn
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
import yaml

from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import EncoderNormalizer, GroupNormalizer, MultiNormalizer, NaNLabelEncoder
from pytorch_forecasting.metrics import (
    MAE,
    MASE,
    SMAPE,
    DistributionLoss,
    MultiHorizonMetric,
    MultiLoss,
    QuantileLoss,
    convert_torchmetric_to_pytorch_forecasting_metric,
)
from pytorch_forecasting.metrics.base_metrics import Metric
from pytorch_forecasting.models.nn.embeddings import MultiEmbedding
from pytorch_forecasting.utils import (
    create_mask,
    get_embedding_size,
    groupby_apply,
    to_list,
)

import logging
# from pytorch_lightning.tuner.tuning import Tuner
from optuna.integration import PyTorchLightningPruningCallback
import optuna.logging
import statsmodels.api as sm
from scipy import stats as ss

from phdu import savedata
from .custom_metrics import MAEMulti, RMSEMulti, beta_interval_1D, training_step_gpu
from . import params
from .preprocessing import load
from .dataset import ImputationDataset, ForecastingDataset


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


class BivariateGaussianMixture(dist.MixtureSameFamily):
    def __init__(self, num_mixtures, means, scale_trils, weights, beta=0.05, R=100):
        self.num_mixtures = num_mixtures
        self.R = 100
        self.beta = beta

        component_distribution = dist.MultivariateNormal(means, scale_tril=scale_trils)
        mixture_weights = dist.Categorical(weights)
        super().__init__(mixture_weights, component_distribution)
        if beta == 1:
            self.energy_score = self._energy_score_beta1
        elif beta == 2:
            self.energy_score = self._energy_score_beta2
        elif beta > 0 and beta < 2:
            self.energy_score = self._energy_score_beta
        else:
            raise ValueError("beta must be in (0, 2)")

    def _energy_score_beta2(self, y_actual):
        """
        Energy score for a bivariate Gaussian mixture distribution.

        The energy score is just the sum of squared errors
        """
        d = self.component_distribution
        w = self.mixture_distribution.probs
        es = (w * (d.mean - y_actual.unsqueeze(-2)).pow(2).sum(-1)).sum(-1)
        return es

    def _energy_score_beta1(self, y_actual):
        """
        Energy score for a bivariate Gaussian mixture distribution with beta = 1.
        """
        d = self.component_distribution
        w = self.mixture_distribution.probs
        x = d.sample((self.R,))
        norm_1 = torch.mean(torch.sqrt(torch.sum((x - y_actual.unsqueeze(-2).unsqueeze(0))**2, axis=-1)), axis=0)
        norm_2 = torch.sum(torch.sqrt(torch.sum((x[1:] - x[:-1])**2, axis=-1)), axis=0) / (2 * (self.R-1))
        es = (w * (norm_1 - norm_2)).sum(-1)
        return es

    def _energy_score_beta(self, y_actual):
        """
        Energy score for a bivariate Gaussian mixture distribution with beta in (0, 2)
        """
        d = self.component_distribution
        w = self.mixture_distribution.probs
        x = d.sample((self.R,))
        norm_1 =  torch.mean( (torch.sum((x - y_actual.unsqueeze(-2).unsqueeze(0))**2, axis=-1))**(self.beta/2),
                             axis=0)
        norm_2 = torch.sum( (torch.sum((x[1:] - x[:-1])**2, axis=-1))**(self.beta/2), axis=0) / (2 * (self.R-1))
        es = (w * (norm_1 - norm_2)).sum(-1)
        return es

class BivariateGaussianMixtureLoss(MultivariateDistributionLoss):
    """
    Loss function for bivariate Gaussian mixture distributions (NLL).
    The goal of this loss is to measure the fitness of the distribution to the target variable.

    Default quantile values for the marginal (1D) distributions:
        [0.0125, 0.9875] -> 97.5% prediction interval
        [0.025, 0.975] -> 95% prediction interval
        [0.125, 0.875] -> 75% prediction interval
        0.5 -> median prediction (for point estimate).
    """
    distribution_class = BivariateGaussianMixture

    def __init__(self,
                 name: str = None,
                 quantiles: List[float] = [0.0125, 0.025, 0.125, 0.5, 0.875, 0.975, 0.9875],
                 lambda_area: float = 5.,
                 reduction: str = "mean",
                 num_mixtures: int = 3,
                 epsilon: float = 0,
                 affine_transform: bool = True,
                 is_energy_score: bool = True,
                 beta: float = 1,
                 R: int = 100,
                 ):
        super().__init__(name=name, quantiles=quantiles, reduction=reduction)
        self.num_mixtures = num_mixtures
        self.distribution_arguments = list(range(6 * num_mixtures))
        self.epsilon = torch.tensor(epsilon)
        self.lambda_area = torch.tensor(lambda_area)
        self.affine_transform = affine_transform
        self.is_energy_score = is_energy_score
        self.beta = beta
        self.R = R
        if self.affine_transform is not None:
            if self.affine_transform:
                self.map_x_to_distribution = self._map_x_to_distribution_affine
                self.rescale_parameters = self._rescale_parameters_affine
            else:
                self.map_x_to_distribution = self._map_x_to_distribution_raw
                self.rescale_parameters = self._rescale_parameters_raw

    def to_quantiles(self, y_pred: torch.Tensor, quantiles: List[float] = None, n_samples: int = 100) -> torch.Tensor:
        """
        Convert network prediction into a quantile prediction. This is done by sampling from the distribution exclusively.

        Args:
            y_pred: prediction output of network
            quantiles (List[float], optional): quantiles for probability range. Defaults to quantiles as
                as defined in the class initialization.
            n_samples (int): number of samples to draw for quantiles. Defaults to 100.

        Returns:
            torch.Tensor: prediction quantiles (batch_size, max_prediction_length, len(quantiles), 2)
        """
        if quantiles is None:
            quantiles = self.quantiles
        distribution = self.map_x_to_distribution(y_pred)
        samples = distribution.sample((n_samples,)) # shape (n_samples, batch_size, max_prediction_length, 2)
        samples = samples.sort(dim=0).values
        quantiles = torch.quantile(samples, torch.tensor(quantiles, device=samples.device), dim=0) # shape (len(quantiles), batch_size, max_prediction_length, 2)
        quantiles = quantiles.permute(1, 2, 0, 3) # shape (batch_size, max_prediction_length, len(quantiles), 2)
        return quantiles

    def to_prediction(self, y_pred: torch.Tensor, n_samples: int = 100) -> torch.Tensor:
        """
        Convert network prediction into a point prediction.

        Args:
            y_pred: prediction output of network

        Returns:
            torch.Tensor: mean prediction
        """
        distribution = self.map_x_to_distribution(y_pred)
        try:
            return distribution.mean
        except NotImplementedError:
            return self.sample(y_pred, n_samples=n_samples).mean(-1)

    def update(self, y_pred, target):
        """
        Update method of metric that handles masking of values for joint target

        Do not override this method but :py:meth:`~loss` instead

        Args:
            y_pred (Dict[str, torch.Tensor]): network output
            target (Union[torch.Tensor, rnn.PackedSequence]): actual values

        Returns:
            torch.Tensor: loss as a single number for backpropagation
        """
        # unpack weight
        if isinstance(target, (list, tuple)) and not isinstance(target, rnn.PackedSequence):
            target, weight = target
        else:
            weight = None

        # unpack target
        if isinstance(target[0], rnn.PackedSequence):
            t1, lengths = unpack_sequence(target[0])
            t2, _ = unpack_sequence(target[1])
            target = torch.stack([t1, t2], dim=-1) # shape (batch_size, max_prediction_length, 2)
        else:
            if isinstance(target, (list, tuple)):
                target = torch.stack(target, dim=-1)
            lengths = torch.full((target.size(0),), fill_value=target.size(1), dtype=torch.long, device=target.device)

        losses = self.loss_raw(y_pred, target) # shape (batch_size, max_prediction_length)

        # weight samples
        if weight is not None:
            losses = losses * unsqueeze_like(weight, losses)
        self._update_losses_and_lengths(losses, lengths)

    def loss(self, y_pred: torch.Tensor, y_actual: torch.Tensor) -> torch.Tensor:
        """
        Calculate negative likelihood

        Args:
            y_pred: network output
            y_actual: actual values

        Returns:
            torch.Tensor: metric value on which backpropagation can be applied
        """
        return self.loss_raw(y_pred, y_actual).sum()

    def loss_raw(self, y_pred: torch.Tensor, y_actual: torch.Tensor) -> torch.Tensor:
        """
        Calculate negative likelihood without summing accross (instance, time).

        Args:
            y_pred: network output
            y_actual: actual values

        Returns:
            torch.Tensor: metric value on which backpropagation can be applied
        """
        distribution = self.map_x_to_distribution(y_pred)
        if self.is_energy_score:
            return distribution.energy_score(y_actual)
        else:
            loss_nll = -distribution.log_prob(y_actual)
            # area loss:
            if self.lambda_area > 0:
                *leading_dims, _ = y_pred.shape
                afine_scale = y_pred[..., 2:4].unsqueeze(-2).expand(*leading_dims, self.num_mixtures, -1)
                loss_area = distribution.mixture_distribution.probs * torch.clamp((torch.sqrt(distribution.component_distribution.variance) / afine_scale).sum(-1) - 1,
                                                       0)
                return loss_nll.unsqueeze(-1) + self.lambda_area*loss_area
            else:
                return loss_nll

    def _map_x_to_distribution_affine(self, x):
        """
        Map the output tensor to a mixture of bivariate Gaussian distributions.
        This is called after 'rescale_parameters'.
        An affine transformation is applied to the means and scale_trils, with loc and scale set by the target.

        Args:
            x (torch.Tensor): Output tensor from the model (after rescaling). Shape: (batch_size, max_prediction_length, num_mixtures*6 + 4)

        Returns:
            torch.distributions.Distribution: A mixture of bivariate Gaussian distributions.
        """
        *leading_dims, _ = x.shape
        end_means_idx = 4 + self.num_mixtures * 2
        end_scale_trils_idx = 4 + self.num_mixtures * 5

        affine_loc = x[..., :2].unsqueeze(-2).expand(*leading_dims, self.num_mixtures, -1)
        affine_scale = x[..., 2:4].unsqueeze(-2).expand(*leading_dims, self.num_mixtures, -1)

        means = affine_loc + affine_scale * x[..., 4:end_means_idx].reshape(*leading_dims, self.num_mixtures, 2)

        scale_trils_raw = x[..., end_means_idx:end_scale_trils_idx].reshape(*leading_dims, self.num_mixtures, 3)
        scale_trils = torch.zeros(*leading_dims, self.num_mixtures, 2, 2, device=x.device)
        scale_trils[..., 0, 0] = scale_trils_raw[..., 0] * affine_scale[..., 0]
        scale_trils[..., 1, 1] = scale_trils_raw[..., 1] * affine_scale[..., 1]
        scale_trils[..., 1, 0] = scale_trils_raw[..., 2] * affine_scale[..., 1]

        weights = x[..., end_scale_trils_idx:].reshape(*leading_dims, self.num_mixtures)
        distr = self.distribution_class(self.num_mixtures, means, scale_trils, weights, beta=self.beta, R=self.R)
        if self._transformation is None: # our case
            return distr
        else:
            return dist.TransformedDistribution(distr, [TorchNormalizer.get_transform(self._transformation)["inverse_torch"]])

    def _map_x_to_distribution_raw(self, x):
        """
        Map the output tensor to a mixture of bivariate Gaussian distributions.
        This is called after 'rescale_parameters'.

        Args:
            x (torch.Tensor): Output tensor from the model (after rescaling). Shape: (batch_size, max_prediction_length, num_mixtures*6)

        Returns:
            torch.distributions.Distribution: A mixture of bivariate Gaussian distributions.
        """
        *leading_dims, _ = x.shape
        end_means_idx = self.num_mixtures * 2
        end_scale_trils_idx = self.num_mixtures * 5

        means = x[..., :end_means_idx].reshape(*leading_dims, self.num_mixtures, 2)

        scale_trils_raw = x[..., end_means_idx:end_scale_trils_idx].reshape(*leading_dims, self.num_mixtures, 3)
        scale_trils = torch.zeros(*leading_dims, self.num_mixtures, 2, 2, device=x.device)
        scale_trils[..., 0, 0] = scale_trils_raw[..., 0]
        scale_trils[..., 1, 1] = scale_trils_raw[..., 1]
        scale_trils[..., 1, 0] = scale_trils_raw[..., 2]

        weights = x[..., end_scale_trils_idx:].reshape(*leading_dims, self.num_mixtures)

        distr = self.distribution_class(self.num_mixtures, means, scale_trils, weights, beta=self.beta, R=self.R)
        if self._transformation is None: # our case
            return distr
        else:
            return dist.TransformedDistribution(distr, [TorchNormalizer.get_transform(self._transformation)["inverse_torch"]])

    def _rescale_parameters_base(self, parameters: torch.Tensor, target_scale: torch.Tensor, encoder: BaseEstimator) -> torch.Tensor:
        """
        parameters: torch.tensor of shape (batch_size, max_prediction_length, num_mixtures*6)
        target_scale: List containing 2 torch.tensor of shape (batch_size, 2) # loc, scale
        encoder: List containing two EncoderNormalizer objects.

        Output (List): means (batch_size, max_prediction_length, num_mixtures*2)
                       scale_trils (batch_size, max_prediction_length, num_mixtures*3)
                       weights (batch_size, max_prediction_length, num_mixtures)
        """
        self._transformation = encoder.transformation
        if all(t is None for t in self._transformation):
            self._transformation = None

        batch_size, max_prediction_length, _ = parameters.size()

        # Splitting parameters into components
        means = parameters[:, :, :self.num_mixtures * 2]
        scale_trils = parameters[:, :, self.num_mixtures * 2:self.num_mixtures * 5]
        weights = parameters[:, :, self.num_mixtures * 5:]

        # Apply transformations
        scale_trils = scale_trils.reshape(batch_size, max_prediction_length, self.num_mixtures, 3)
        weights = F.softmax(weights)
        scale_trils[:, :, :, :2] = F.softplus(scale_trils[:, :, :, :2].clone())
        scale_trils[:, :, :, 2] = torch.tanh(scale_trils[:, :, :, 2].clone())
        # Reshape back to the original batch and sequence dimensions
        scale_trils = scale_trils.reshape(batch_size, max_prediction_length, self.num_mixtures * 3)
        return [means, scale_trils, weights]

    def _rescale_parameters_raw(self, parameters: torch.Tensor, target_scale: torch.Tensor, encoder: BaseEstimator,
                                x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        parameters: torch.tensor of shape (batch_size, max_prediction_length, num_mixtures*6)
        target_scale: List containing 2 torch.tensor of shape (batch_size, 2)
        encoder: List containing two EncoderNormalizer objects.
        x added for compatibility.

        Output shape: (batch_size, max_prediction_length, num_mixtures*6)
                      6 from: 2 means, 3 scale_trils, 1 weights
        """
        return torch.concat(self._rescale_parameters_base(parameters, target_scale, encoder), dim=-1)

    def _rescale_parameters_affine(self, parameters: torch.Tensor, target_scale: torch.Tensor, encoder: BaseEstimator,
                                x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        parameters: torch.tensor of shape (batch_size, max_prediction_length, num_mixtures*6)
        target_scale: List containing 2 torch.tensor of shape (batch_size, 2)
        encoder: List containing two EncoderNormalizer objects.

        Output shape: (batch_size, max_prediction_length, num_mixtures*6 + 4)
                      6 from: 2 means, 3 scale_trils, 1 weights
                      4 from: 2 target_loc, 2 target_scale
        """
        # target loc = last location, target_scale estimated from the observed trajectory.
        batch_size, max_prediction_length, _ = parameters.size()
        encoder_target = torch.stack(x['encoder_target'], axis=-1)
        target_means = encoder_target[:, -1].unsqueeze(1).expand(-1, max_prediction_length, -1) # (batch_size, max_prediction_length, 2)
        target_scales = (torch.stack([(encoder_target[:, i:] - encoder_target[:, :-i]).abs().mean(axis=1) for i in range(1, max_ediction_length+1)], axis=1)
                         + self.epsilon) # (batch_size, max_prediction_length, 2)
        return torch.concat([target_means, target_scales] + self._rescale_parameters_base(parameters, target_scale, encoder), dim=-1)

class BivariateGaussianQuantileLoss(BivariateGaussianMixtureLoss):
    """
    Loss function for bivariate Gaussian ellipsoidal quantile loss.
    The goal of this loss is to measure the fitness of ellipsoidal confidence region. Is an extension of the quantile loss to 2D for the Gaussian.

    Default quantile values for the marginal (1D) distributions:
        [0.0125, 0.9875] -> 97.5% prediction interval
        [0.025, 0.975] -> 95% prediction interval
        [0.125, 0.875] -> 75% prediction interval
        0.5 -> median prediction (for point estimate).
    """
    distribution_class = BivariateGaussianMixture

    # TODO: output = 5 * len(quantiles_2D), 1 Gaussian for each confidence interval.
    # TODO: maybe output = a,b, c1, c2, angle for each confidence interval. Would make more sense.

    def __init__(self,
                 name: str = None,
                 quantiles: List[float] = [0.0125, 0.025, 0.125, 0.5, 0.875, 0.975, 0.9875],
                 quantiles_2D: List[float] = [0.5, 0.9, 0.95],
                 reduction: str = "mean",
                 epsilon: float = 0,
                 affine_transform: bool = True,
                 ):
        super().__init__(name=name, quantiles=quantiles, reduction=reduction, epsilon=epsilon, num_mixtures=1)
        self.distribution_arguments = list(range(5)) # 2 means, 2 stds, corr.
        self.quantiles_2D = quantiles_2D
        self.critical_values = [torch.tensor(ss.chi2.ppf(q, df=2)) for q in quantiles_2D]
        self.affine_transform = affine_transform
        if self.affine_transform:
            self.map_x_to_distribution = self._map_x_to_distribution_affine
            self.rescale_parameters = self._rescale_parameters_affine
        else:
            self.map_x_to_distribution = self._map_x_to_distribution_raw
            self.rescale_parameters = self._rescale_parameters_raw

    def loss(self, y_pred: torch.Tensor, y_actual: torch.Tensor) -> torch.Tensor:
        """
        Calculate negative likelihood

        Args:
            y_pred: network output
            y_actual: actual values

        Returns:
            torch.Tensor: metric value on which backpropagation can be applied
        """
        return self.loss_raw(y_pred, y_actual).sum()

    def loss_raw(self, y_pred: torch.Tensor, y_actual: torch.Tensor) -> torch.Tensor:
        """
        Calculate negative likelihood without summing accross (instance, time).

        Args:
            y_pred: network output
            y_actual: actual values

        Returns:
            torch.Tensor: metric value on which backpropagation can be applied
        """
        distribution = self.map_x_to_distribution(y_pred)
        mean = distribution.mean
        cov = distribution.covariance_matrix

        # Transforming the data to the new coordinate system (rotated by the eigenvectors of the covariance matrix, w centered in the mean).
        T = torch.linalg.eigh(torch.inverse(cov)).eigenvectors
        eigenvalues = torch.linalg.eigvalsh(cov)
        T_inv = T.transpose(-2, -1)
        w = torch.matmul(T_inv, (y_actual - mean).unsqueeze(-1))
        w = w.squeeze(-1)

        losses = []
        for q, critical_value in zip(self.quantiles_2D, self.critical_values):
            semiaxes = torch.sqrt(eigenvalues*critical_value)
            # is_inside = torch.sign(self.ellipse_error_vec(w, semiaxes)) # < 0 inside the ellipse (border overestimation), > 0 outside the ellipse (border underestimation).
            # distance_to_closest = torch.sqrt(torch.square(w*(1-semiaxes/torch.linalg.vector_norm(w, dim=-1, keepdim=True))).sum(-1)) # distance to the closest point on the ellipse.
            # ellipse_loss = torch.max((q-1) * is_inside, q * is_inside) * distance_to_closest

            q_lower = (1 - q) / 2
            q_upper = 1 - q_lower

            error_upper_quantile = w - semiaxes # target - prediction
            loss_quantile_upper = torch.max((q_upper-1) * error_upper_quantile, q_upper * error_upper_quantile).sum(-1)

            error_lower_quantile = w + semiaxes # target - prediction
            loss_quantile_lower = torch.max((q_lower-1) * error_lower_quantile, q_lower * error_lower_quantile).sum(-1)

            losses.append(loss_quantile_upper + loss_quantile_lower)# + ellipse_loss)

        losses = 2 * torch.stack(losses, dim=-1) # (batch_size, max_prediction_length, quantiles_2D)

        return losses

    @staticmethod
    def _ellipse_error(w1, w2, a, b):
        """
        < 0 inside the ellipse (border overestimation).
        > 0 outside the ellipse (border underestimation).
        """
        return ((w1/a)** 2 + (w2/b)** 2) - 1

    @staticmethod
    def ellipse_error_vec(w, semiaxes):
        return torch.sum((w/semiaxes)**2, axis=-1) - 1

    def _map_x_to_distribution_affine(self, x):
        """
        Map the output tensor to a mixture of bivariate Gaussian distributions.
        This is called after 'rescale_parameters'.
        An affine transformation is applied to the means and scale_trils, with loc and scale set by the target.

        Args:
            x (torch.Tensor): Output tensor from the model (after rescaling). Shape: (batch_size, max_prediction_length, 9 params)

        Returns:
            torch.distributions.Distribution: A mixture of bivariate Gaussian distributions.
        """
        *leading_dims, _ = x.shape

        affine_loc = x[..., :2]
        affine_scale = x[..., 2:4]

        means = affine_loc + affine_scale * x[..., 4:6]

        scale_trils = torch.zeros(*leading_dims, 2, 2, device=x.device)
        scale_trils[..., 0, 0] = x[..., 6] * affine_scale[..., 0]
        scale_trils[..., 1, 1] = x[..., 7] * affine_scale[..., 1]
        scale_trils[..., 1, 0] = x[..., 8] * affine_scale[..., 1]

        distr = dist.MultivariateNormal(loc=means, scale_tril=scale_trils)
        if self._transformation is None: # our case
            return distr
        else:
            return dist.TransformedDistribution(distr, [TorchNormalizer.get_transform(self._transformation)["inverse_torch"]])

    def _map_x_to_distribution_raw(self, x):
        """
        Map the output tensor to a mixture of bivariate Gaussian distributions.
        This is called after 'rescale_parameters'.

        Args:
            x (torch.Tensor): Output tensor from the model (after rescaling). Shape: (batch_size, max_prediction_length, 5 params)

        Returns:
            torch.distributions.Distribution: A mixture of bivariate Gaussian distributions.
        """
        *leading_dims, _ = x.shape

        means = x[..., :2]

        scale_trils = torch.zeros(*leading_dims, 2, 2, device=x.device)
        scale_trils[..., 0, 0] = x[..., 2]
        scale_trils[..., 1, 1] = x[..., 3]
        scale_trils[..., 1, 0] = x[..., 4]

        distr = dist.MultivariateNormal(loc=means, scale_tril=scale_trils)
        if self._transformation is None: # our case
            return distr
        else:
            return dist.TransformedDistribution(distr, [TorchNormalizer.get_transform(self._transformation)["inverse_torch"]])

    def _rescale_parameters_base(self, parameters: torch.Tensor, target_scale: torch.Tensor, encoder: BaseEstimator) -> torch.Tensor:
        """
        parameters: torch.tensor of shape (batch_size, max_prediction_length, num_mixtures*6)
        target_scale: List containing 2 torch.tensor of shape (batch_size, 2) # loc, scale
        encoder: List containing two EncoderNormalizer objects.

        Output (List):  means (batch_size, max_prediction_length, 2)
                        scale_trils (batch_size, max_prediction_length, 3)
        """
        self._transformation = encoder.transformation
        if all(t is None for t in self._transformation):
            self._transformation = None

        batch_size, max_prediction_length, _ = parameters.size()

        # Splitting parameters into components
        means = parameters[:, :, :2]
        scale_trils = parameters[:, :, 2:]

        # Apply transformations
        scale_trils = scale_trils.reshape(batch_size, max_prediction_length, 3)
        scale_trils[:, :, :2] = F.softplus(scale_trils[:, :, :2].clone())
        scale_trils[:, :, 2] = torch.tanh(scale_trils[:, :, 2].clone())

        return [means, scale_trils]

    def _rescale_parameters_raw(self, parameters: torch.Tensor, target_scale: torch.Tensor, encoder: BaseEstimator,
                                x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        parameters: torch.tensor of shape (batch_size, max_prediction_length, num_mixtures*6)
        target_scale: List containing 2 torch.tensor of shape (batch_size, 2) # loc, scale
        encoder: List containing two EncoderNormalizer objects.

        Output shape: (batch_size, max_prediction_length, 5)
                      5 from: 2 means, 3 scale_trils
        """
        return torch.concat(self._rescale_parameters_base(parameters, target_scale, encoder), dim=-1)

    def _rescale_parameters_affine(self, parameters: torch.Tensor, target_scale: torch.Tensor, encoder: BaseEstimator,
                                   x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        parameters: torch.tensor of shape (batch_size, max_prediction_length, num_mixtures*6)
        target_scale: List containing 2 torch.tensor of shape (batch_size, 2) # loc, scale
        encoder: List containing two EncoderNormalizer objects.

        Output shape: (batch_size, max_prediction_length, 5 + 4)
                      5 from: 2 means, 3 scale_trils
                      4 from: 2 target_loc, 2 target_scale
        """
        batch_size, max_prediction_length, _ = parameters.size()

        # target loc = last location, target_scale estimated from the observed trajectory.
        encoder_target = torch.stack(x['encoder_target'], axis=-1)
        target_means = encoder_target[:, -1].unsqueeze(1).expand(-1, max_prediction_length, -1) # (batch_size, max_prediction_length, 2)
        target_scales = (torch.stack([(encoder_target[:, i:] - encoder_target[:, :-i]).abs().mean(axis=1) for i in range(1, max_prediction_length+1)], axis=1)
                         + self.epsilon) # (batch_size, max_prediction_length, 2)

        return torch.concat([target_means, target_scales] + self._rescale_parameters_base(parameters, target_scale, encoder), dim=-1)

class QuantileLossGCD(QuantileLoss):
    """
    Quantile loss for all quantiles, except for the mid point, which uses the great circle distance.

    The idea is that by using an analogous to RMSE for the mid point, the model learns the mean, instead of the median.
    """
    def __init__(self, quantiles: List[float], **kwargs):
        super().__init__(quantiles=quantiles, **kwargs)

    @staticmethod
    def mercator_inv(x, y, R=params.R_earth):
        """Returns latitude and longitude in radians"""
        lat = 2 * torch.arctan(torch.exp(y / R)) - torch.pi / 2
        lon = x / R
        return lat, lon

    @staticmethod
    def great_circle_distance(lat, lon, lat_f=None, lon_f=None, R=params.R_earth):
        if lat_f is None:
            lat_f, lat_0 = lat[1:], lat[:-1]
            lon_f, lon_0 = lon[1:], lon[:-1]
        else:
            lat_0, lon_0 = lat, lon
        sigma = 2*torch.arcsin(torch.sqrt(torch.sin(0.5*(lat_f-lat_0))**2 + torch.cos(lat_f)*torch.cos(lat_0)*torch.sin(0.5*(lon_f - lon_0))**2))
        return sigma * R # to km (if R in km)

    def gcd_loss(self, y_pred_midpoint: torch.Tensor, target: torch.Tensor, R=params.R_earth):
        lat_pred, lon_pred = self.mercator_inv(y_pred_midpoint[..., 0], y_pred_midpoint[..., 1], R=R)
        lat_target, lon_target = self.mercator_inv(target[..., 0], target[..., 1], R=R)
        return self.great_circle_distance(lat_pred, lon_pred, lat_target, lon_target, R=R)

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # calculate quantile loss
        losses = []
        midpoint = self.quantiles.index(0.5)
        for i, q in enumerate(self.quantiles):
            if i != midpoint:
                errors = target - y_pred[..., i]
                losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(-1))
            else:
                losses.append(self.gcd_loss(y_pred[..., i], target).unsqueeze(-1))
        losses = 2 * torch.cat(losses, dim=2)

        return losses

class QuantileLossRMSE(QuantileLoss):
    """
    Quantile loss that uses RMSE for the midpoint.
    """
    def __init__(self, quantiles: List[float], midpoint_weight=2, **kwargs):
        self.midpoint = quantiles.index(0.5)
        self.not_mid = [i for i in range(len(quantiles)) if i != self.midpoint]
        self.midpoint_weight = midpoint_weight
        super().__init__(quantiles=quantiles, **kwargs)
        self.rmse = RMSE()
        self.add_state("losses_mid", default=torch.tensor(0.0), dist_reduce_fx="cat")
        self.add_state("losses_q", default=torch.tensor(0.0), dist_reduce_fx="cat")

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # calculate quantile loss except for midpoint, which uses RMSE
        losses = []
        for i, q in enumerate(self.quantiles):
            if i != self.midpoint:
                errors = target - y_pred[..., i]
                losses.append(2 * torch.max((q - 1) * errors, q * errors).unsqueeze(-1))
            else:
                losses.append(self.midpoint_weight * self.rmse.loss(y_pred[..., i], target).unsqueeze(-1))
        losses = torch.cat(losses, dim=2)

        return losses

    def _update_losses_and_lengths(self, losses: torch.Tensor, lengths: torch.Tensor):
        losses = self.mask_losses(losses, lengths)
        if self.reduction == "none":
            losses_q = losses[..., self.not_mid]
            losses_mid = losses[..., self.midpoint]
            if self.losses_q.ndim == 0:
                self.losses_q = losses_q
                self.losses_mid = losses_mid
                self.lengths = lengths
            else:
                self.losses_q = torch.cat([self.losses_q, losses], dim=0)
                self.losses_mid = torch.cat([self.losses_mid, losses], dim=0)
                self.lengths = torch.cat([self.lengths, lengths], dim=0)
        else:
            losses_mid = losses[..., self.midpoint].sum()
            losses_q = losses[..., self.not_mid].sum()
            if not torch.isfinite(losses_q):
                losses = torch.tensor(1e9, device=losses.device)
                warnings.warn("Loss is not finite. Resetting it to 1e9")
            self.losses_mid = self.losses_mid + losses_mid
            self.losses_q = self.losses_q + losses_q
            self.lengths = self.lengths + lengths.sum()

    def compute(self):
        loss_q = self.losses_q / self.lengths
        loss_mid = self.losses_mid / self.lengths
        loss_mid = loss_mid.sqrt()
        loss = loss_q + loss_mid
        return loss

class QuantileLossZeroAttention(QuantileLoss):
    """
    Quantile loss that does not consider missing values in the prediction.
    """
    def __init__(self, quantiles: List[float], **kwargs):
        super().__init__(quantiles=quantiles, **kwargs)

    def update(self, y_pred, target, decoder_missing):
        """
        Update method of metric that handles masking of values.

        Do not override this method but :py:meth:`~loss` instead

        Args:
            y_pred (Dict[str, torch.Tensor]): network output
            target (Union[torch.Tensor, rnn.PackedSequence]): actual values

        Returns:
            torch.Tensor: loss as a single number for backpropagation
        """
        # unpack weight
        if isinstance(target, (list, tuple)) and not isinstance(target, rnn.PackedSequence):
            target, weight = target
        else:
            weight = None

        # unpack target
        if isinstance(target, rnn.PackedSequence):
            target, lengths = unpack_sequence(target)
        else:
            lengths = torch.full((target.size(0),), fill_value=target.size(1), dtype=torch.long, device=target.device)

        losses = self.loss(y_pred, target)
        # weight samples
        if weight is not None:
            losses = losses * unsqueeze_like(weight, losses)
        self._update_losses_and_lengths(losses, lengths, decoder_missing)

    def _update_losses_and_lengths(self, losses: torch.Tensor, lengths: torch.Tensor, decoder_missing: torch.Tensor = None):
        losses = self.mask_losses(losses, lengths)
        lengths = decoder_missing.shape[1] - decoder_missing.sum(dim=1) # number of non-missing values
        losses[decoder_missing] = 0.0
        if self.reduction == "none":
            if self.losses.ndim == 0:
                self.losses = losses
                self.lengths = lengths
            else:
                self.losses = torch.cat([self.losses, losses], dim=0)
                self.lengths = torch.cat([self.lengths, lengths], dim=0)
        else:
            losses = losses.sum()
            if not torch.isfinite(losses):
                losses = torch.tensor(1e9, device=losses.device)
                warnings.warn("Loss is not finite. Resetting it to 1e9")
            self.losses = self.losses + losses
            self.lengths = self.lengths + lengths.sum()


class CumQuantileLoss(MultiHorizonMetric):
    """
    Quantile loss, i.e. a quantile of ``q=0.5`` will give half of the mean absolute error as it is calcualted as

    Defined as ``max(q * (y-y_pred), (1-q) * (y_pred-y))``

    Calculates the cumulative of the target before computing the loss
    """

    def __init__(
        self,
        quantiles: List[float] = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98],
        **kwargs,
    ):
        """
        Quantile loss

        Args:
            quantiles: quantiles for metric
        """
        super().__init__(quantiles=quantiles, **kwargs)

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # target to cumsum
        target_cum = target.cumsum(dim=-1)
        # calculate quantile loss
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target_cum - y_pred[..., i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(-1))
        losses = 2 * torch.cat(losses, dim=2)

        return losses

    def to_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Convert network prediction into a point prediction.

        Args:
            y_pred: prediction output of network

        Returns:
            torch.Tensor: point prediction
        """
        if y_pred.ndim == 3:
            idx = self.quantiles.index(0.5)
            y_pred = y_pred[..., idx]
        return y_pred

    def to_quantiles(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Convert network prediction into a quantile prediction.

        Args:
            y_pred: prediction output of network

        Returns:
            torch.Tensor: prediction quantiles
        """
        return y_pred

class BaseModelJointTarget(BaseModel):
    """
    Adapted from pytorch_forecasting.models.base_model.BaseModel to handle joint target.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_dataset(cls, dataset: TimeSeriesDataSet, **kwargs) -> LightningModule:
        """
        Create model from dataset, i.e. save dataset parameters in model

        This function should be called as ``super().from_dataset()`` in a derived models that implement it

        Args:
            dataset (TimeSeriesDataSet): timeseries dataset

        Returns:
            BaseModel: Model that can be trained
        """
        if "output_transformer" not in kwargs:
            kwargs["output_transformer"] = dataset.target_normalizer
        net = cls(**kwargs)
        net.dataset_parameters = dataset.get_parameters()
        assert not isinstance(net.loss, MultiLoss), "MultiLoss not compatible with joint target"
        return net

    @staticmethod
    def deduce_default_output_parameters(
        dataset: TimeSeriesDataSet, kwargs: Dict[str, Any], default_loss: MultiHorizonMetric = None
    ) -> Dict[str, Any]:
        """
        Deduce default parameters for output for `from_dataset()` method.

        Determines ``output_size`` and ``loss`` parameters.

        Args:
            dataset (TimeSeriesDataSet): timeseries dataset
            kwargs (Dict[str, Any]): current hyperparameters
            default_loss (MultiHorizonMetric, optional): default loss function.
                Defaults to :py:class:`~pytorch_forecasting.metrics.MAE`.

        Returns:
            Dict[str, Any]: dictionary with ``output_size`` and ``loss``.
        """

        # infer output size
        def get_output_size(normalizer, loss):
            if isinstance(loss, QuantileLoss):
                return len(loss.quantiles)
            elif isinstance(normalizer, NaNLabelEncoder):
                return len(normalizer.classes_)
            elif isinstance(loss, DistributionLoss):
                return len(loss.distribution_arguments)
            else:
                return 1  # default to 1

        # handle multiple targets
        new_kwargs = {}
        n_targets = len(dataset.target_names)
        if default_loss is None:
            default_loss = MAE()
        loss = kwargs.get("loss", default_loss)
        if n_targets > 1:  # try to infer number of ouput sizes
            if isinstance(loss, MultiLoss):
                raise ValueError("MultiLoss not compatible with joint target")
            new_kwargs["output_size"] = get_output_size(dataset.target_normalizer, loss)
        else:
            raise ValueError("The number of targets must be greater than 1")
        return new_kwargs

    def log_metrics(
        self,
        x: Dict[str, torch.Tensor],
        y: torch.Tensor,
        out: Dict[str, torch.Tensor],
        prediction_kwargs: Dict[str, Any] = dict(use_metric=False)
    ) -> None:
        """
        Log metrics every training/validation step for multivariate target.
        If log type is DistributionLoss, then the distribution is passed to the logger.
        else the point prediction is passed to the logger.

        Args:
            x (Dict[str, torch.Tensor]): x as passed to the network by the dataloader
            y (torch.Tensor): y as passed to the loss function by the dataloader
            out (Dict[str, torch.Tensor]): output of the network
            prediction_kwargs (Dict[str, Any]): parameters for ``to_prediction()`` of the loss metric.
        """
        # logging losses - for each target
        if prediction_kwargs is None:
            prediction_kwargs = {}
        y_hat = out.prediction.detach()
        y_hat_point = self.to_prediction(out, **prediction_kwargs).detach()
        y_real = torch.stack((y[0][0], y[0][1]), dim=-1)

        for metric in self.logging_metrics:
            if isinstance(metric, MultivariateDistributionLoss):
                loss_value = metric(y_hat, y_real) # requires map_x_to_distribution within the loss
            else:
                loss_value = metric(y_hat_point, y_real)

            target_tag = ""

            self.log(
                f"{target_tag}{self.current_stage}_{metric.name}",
                loss_value,
                on_step=self.training,
                on_epoch=True,
                batch_size=len(x["decoder_target"]),
            )

    def predict(
        self,
        data: Union[DataLoader, pd.DataFrame, TimeSeriesDataSet],
        mode: Union[str, Tuple[str, str]] = "prediction",
        return_index: bool = False,
        return_decoder_lengths: bool = False,
        batch_size: int = 64,
        num_workers: int = 0,
        fast_dev_run: bool = False,
        show_progress_bar: bool = False,
        return_x: bool = False,
        return_y: bool = False,
        mode_kwargs: Dict[str, Any] = None,
        **kwargs,
    ):
        """
        Run inference / prediction.

        Args:
            dataloader: dataloader, dataframe or dataset
            mode: one of "prediction", "quantiles", or "raw", or tuple ``("raw", output_name)`` where output_name is
                a name in the dictionary returned by ``forward()``
            return_index: if to return the prediction index (in the same order as the output, i.e. the row of the
                dataframe corresponds to the first dimension of the output and the given time index is the time index
                of the first prediction)
            return_decoder_lengths: if to return decoder_lengths (in the same order as the output
            batch_size: batch size for dataloader - only used if data is not a dataloader is passed
            num_workers: number of workers for dataloader - only used if data is not a dataloader is passed
            fast_dev_run: if to only return results of first batch
            show_progress_bar: if to show progress bar. Defaults to False.
            return_x: if to return network inputs (in the same order as prediction output)
            mode_kwargs (Dict[str, Any]): keyword arguments for ``to_prediction()`` or ``to_quantiles()``
                for modes "prediction" and "quantiles"
            **kwargs: additional arguments to network's forward method

        Returns:
            output, x, index, decoder_lengths: some elements might not be present depending on what is configured
                to be returned
        """
        # convert to dataloader
        if isinstance(data, pd.DataFrame):
            data = TimeSeriesDataSet.from_parameters(self.dataset_parameters, data, predict=True)
        if isinstance(data, TimeSeriesDataSet):
            dataloader = data.to_dataloader(batch_size=batch_size, train=False, num_workers=num_workers)
        else:
            dataloader = data

        # mode kwargs default to None
        if mode_kwargs is None:
            mode_kwargs = {}

        # ensure passed dataloader is correct
        assert isinstance(dataloader.dataset, TimeSeriesDataSet), "dataset behind dataloader mut be TimeSeriesDataSet"

        # prepare model
        self.eval()  # no dropout, etc. no gradients
        # run predictions
        output = []
        decode_lenghts = []
        x_list = []
        y_list = []
        index = []
        progress_bar = tqdm(desc="Predict", unit=" batches", total=len(dataloader), disable=not show_progress_bar)
        with torch.no_grad():
            for x, (y, _) in dataloader: # _ = weight
                # move data to appropriate device
                data_device = x["encoder_cont"].device
                if data_device != self.device:
                    x = move_to_device(x, self.device)

                # make prediction
                out = self(x, **kwargs)  # raw output is dictionary

                lengths = x["decoder_lengths"]
                if return_decoder_lengths:
                    decode_lenghts.append(lengths)
                nan_mask = create_mask(lengths.max(), lengths)
                if isinstance(mode, (tuple, list)):
                    if mode[0] == "raw":
                        out = out[mode[1]]
                    else:
                        raise ValueError(
                            f"If a tuple is specified, the first element must be 'raw' - got {mode[0]} instead"
                        )
                elif mode == "prediction":
                    out = self.to_prediction(out, **mode_kwargs)
                    # mask non-predictions
                    if isinstance(out, (list, tuple)):
                        out = [
                            o.masked_fill(nan_mask, torch.tensor(float("nan"))) if o.dtype == torch.float else o
                            for o in out
                        ]
                    elif out.dtype == torch.float:  # only floats can be filled with nans
                        out = out.masked_fill(nan_mask.unsqueeze(-1), torch.tensor(float("nan")))
                elif mode == "quantiles":
                    out = self.to_quantiles(out, **mode_kwargs)
                    # mask non-predictions
                    if isinstance(out, (list, tuple)):
                        out = [
                            o.masked_fill(nan_mask.unsqueeze(-1), torch.tensor(float("nan")))
                            if o.dtype == torch.float
                            else o
                            for o in out
                        ]
                    elif out.dtype == torch.float:
                        out = out.masked_fill(nan_mask.unsqueeze(-1).unsqueeze(-1), torch.tensor(float("nan")))
                elif mode == "raw":
                    pass
                else:
                    raise ValueError(f"Unknown mode {mode} - see docs for valid arguments")

                out = move_to_device(out, device="cpu")

                output.append(out)
                if return_x:
                    x = move_to_device(x, "cpu")
                    x_list.append(x)
                if return_y:
                    y = torch.stack(y, dim=-1)
                    y = move_to_device(y, "cpu")
                    y_list.append(y)
                if return_index:
                    index.append(dataloader.dataset.x_to_index(x))
                progress_bar.update()
                if fast_dev_run:
                    break
        # concatenate output (of different batches)
        if isinstance(mode, (tuple, list)) or mode != "raw":
            if isinstance(output[0], (tuple, list)) and len(output[0]) > 0 and isinstance(output[0][0], torch.Tensor):
                output = [_torch_cat_na([out[idx] for out in output]) for idx in range(len(output[0]))]
            else:
                output = _torch_cat_na(output)
        elif mode == "raw":
            output = _concatenate_output(output)

        # generate output
        if return_x or return_y or return_index or return_decoder_lengths:
            output = [output]
        if return_x:
            output.append(_concatenate_output(x_list))
        if return_y:
            output.append(torch.cat(y_list, dim=0))
        if return_index:
            output.append(pd.concat(index, axis=0, ignore_index=True))
        if return_decoder_lengths:
            output.append(torch.cat(decode_lenghts, dim=0))
        return output

    def plot_prediction(
        self,
        x: Dict[str, torch.Tensor],
        out: Dict[str, torch.Tensor],
        idx: int = 0,
        add_loss_to_title: Union[Metric, torch.Tensor, bool] = False,
        show_future_observed: bool = True,
        ax=None,
        quantiles_kwargs: Dict[str, Any] = {},
        prediction_kwargs: Dict[str, Any] = {},
    ) -> plt.Figure:
        """
        Plot prediction of prediction vs actuals

        Args:
            x: network input
            out: network output
            idx: index of prediction to plot
            add_loss_to_title: if to add loss to title or loss function to calculate. Can be either metrics,
                bool indicating if to use loss metric or tensor which contains losses for all samples.
                Calcualted losses are determined without weights. Default to False.
            show_future_observed: if to show actuals for future. Defaults to True.
            ax: matplotlib axes to plot on
            quantiles_kwargs (Dict[str, Any]): parameters for ``to_quantiles()`` of the loss metric.
            prediction_kwargs (Dict[str, Any]): parameters for ``to_prediction()`` of the loss metric.

        Returns:
            matplotlib figure
        """
        # all true values for y of the first sample in batch
        encoder_targets = to_list(x["encoder_target"])
        decoder_targets = to_list(x["decoder_target"])

        y_raws = out["prediction"]  # raw predictions - used for calculating loss
        y_hats = self.to_prediction(out, **prediction_kwargs).permute(2, 0, 1) # target, batch, time
        y_quantiles = self.to_quantiles(out, **quantiles_kwargs).permute(3, 0, 1, 2) # target, batch, time, quantile

        # for each target, plot
        figs = []
        for y_raw, y_hat, y_quantile, encoder_target, decoder_target in zip(
            y_raws, y_hats, y_quantiles, encoder_targets, decoder_targets
        ):
            y_all = torch.cat([encoder_target[idx], decoder_target[idx]])
            max_encoder_length = x["encoder_lengths"].max()
            y = torch.cat(
                (
                    y_all[: x["encoder_lengths"][idx]],
                    y_all[max_encoder_length : (max_encoder_length + x["decoder_lengths"][idx])],
                ),
            )
            # move predictions to cpu
            y_hat = y_hat.detach().cpu()[idx, : x["decoder_lengths"][idx]]
            y_quantile = y_quantile.detach().cpu()[idx, : x["decoder_lengths"][idx]]
            y_raw = y_raw.detach().cpu()[idx, : x["decoder_lengths"][idx]]

            # move to cpu
            y = y.detach().cpu()
            # create figure
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.get_figure()
            n_pred = y_hat.shape[0]
            x_obs = np.arange(-(y.shape[0] - n_pred), 0)
            x_pred = np.arange(n_pred)
            prop_cycle = iter(plt.rcParams["axes.prop_cycle"])
            obs_color = next(prop_cycle)["color"]
            pred_color = next(prop_cycle)["color"]
            # plot observed history
            if len(x_obs) > 0:
                if len(x_obs) > 1:
                    plotter = ax.plot
                else:
                    plotter = ax.scatter
                plotter(x_obs, y[:-n_pred], label="observed", c=obs_color)
            if len(x_pred) > 1:
                plotter = ax.plot
            else:
                plotter = ax.scatter

            # plot observed prediction
            if show_future_observed:
                plotter(x_pred, y[-n_pred:], label=None, c=obs_color)

            # plot prediction
            plotter(x_pred, y_hat, label="predicted", c=pred_color)

            # plot predicted quantiles
            plotter(x_pred, y_quantile[:, y_quantile.shape[1] // 2], c=pred_color, alpha=0.15)
            for i in range(y_quantile.shape[1] // 2):
                if len(x_pred) > 1:
                    ax.fill_between(x_pred, y_quantile[:, i], y_quantile[:, -i - 1], alpha=0.15, fc=pred_color)
                else:
                    quantiles = torch.tensor([[y_quantile[0, i]], [y_quantile[0, -i - 1]]])
                    ax.errorbar(
                        x_pred,
                        y[[-n_pred]],
                        yerr=quantiles - y[-n_pred],
                        c=pred_color,
                        capsize=1.0,
                    )

            if add_loss_to_title is not False:
                if isinstance(add_loss_to_title, bool):
                    loss = self.loss
                elif isinstance(add_loss_to_title, torch.Tensor):
                    loss = add_loss_to_title.detach()[idx].item()
                elif isinstance(add_loss_to_title, Metric):
                    loss = add_loss_to_title
                else:
                    raise ValueError(f"add_loss_to_title '{add_loss_to_title}'' is unkown")
                if isinstance(loss, MASE):
                    loss_value = loss(y_raw[None], (y[-n_pred:][None], None), y[:n_pred][None])
                elif isinstance(loss, Metric):
                    try:
                        loss_value = loss(y_raw[None], (y[-n_pred:][None], None))
                    except Exception:
                        loss_value = "-"
                else:
                    loss_value = loss
                ax.set_title(f"Loss {loss_value}")
            ax.set_xlabel("Time index")
            fig.legend()
            figs.append(fig)

        # return multiple of target is a list, otherwise return single figure
        if isinstance(x["encoder_target"], (tuple, list)):
            return figs
        else:
            return fig

    def transform_output(
        self,
        prediction: Union[torch.Tensor, List[torch.Tensor]],
        target_scale: Union[torch.Tensor, List[torch.Tensor]],
        loss: Optional[Metric] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Extract prediction from network output and rescale it to real space / de-normalize it.

        Args:
            prediction (Union[torch.Tensor, List[torch.Tensor]]): normalized prediction
            target_scale (Union[torch.Tensor, List[torch.Tensor]]): scale to rescale prediction
            loss (Optional[Metric]): metric to use for transform

        Returns:
            torch.Tensor: rescaled prediction
        """
        if loss is None:
            loss = self.loss
        if isinstance(loss, MultiLoss):
            out = loss.rescale_parameters(
                prediction,
                target_scale=target_scale,
                encoder=self.output_transformer.normalizers,  # need to use normalizer per encoder
                **kwargs
            )
        else:
            out = loss.rescale_parameters(prediction, target_scale=target_scale, encoder=self.output_transformer, **kwargs)
        return out

class BaseModelWithCovariatesJointTarget(BaseModelJointTarget):
    """
    Model with additional methods using covariates.

    Assumes the following hyperparameters:

    Args:
        static_categoricals (List[str]): names of static categorical variables
        static_reals (List[str]): names of static continuous variables
        time_varying_categoricals_encoder (List[str]): names of categorical variables for encoder
        time_varying_categoricals_decoder (List[str]): names of categorical variables for decoder
        time_varying_reals_encoder (List[str]): names of continuous variables for encoder
        time_varying_reals_decoder (List[str]): names of continuous variables for decoder
        x_reals (List[str]): order of continuous variables in tensor passed to forward function
        x_categoricals (List[str]): order of categorical variables in tensor passed to forward function
        embedding_sizes (Dict[str, Tuple[int, int]]): dictionary mapping categorical variables to tuple of integers
            where the first integer denotes the number of categorical classes and the second the embedding size
        embedding_labels (Dict[str, List[str]]): dictionary mapping (string) indices to list of categorical labels
        embedding_paddings (List[str]): names of categorical variables for which label 0 is always mapped to an
             embedding vector filled with zeros
        categorical_groups (Dict[str, List[str]]): dictionary of categorical variables that are grouped together and
            can also take multiple values simultaneously (e.g. holiday during octoberfest). They should be implemented
            as bag of embeddings
    """

    @property
    def target_positions(self) -> torch.LongTensor:
        """
        Positions of target variable(s) in covariates.

        Returns:
            torch.LongTensor: tensor of positions.
        """
        # todo: expand for categorical targets
        if "target" in self.hparams:
            target = self.hparams.target
        else:
            target = self.dataset_parameters["target"]
        return torch.tensor(
            [self.hparams.x_reals.index(name) for name in to_list(target)],
            device=self.device,
            dtype=torch.long,
        )

    @property
    def reals(self) -> List[str]:
        """List of all continuous variables in model"""
        return list(
            dict.fromkeys(
                self.hparams.static_reals
                + self.hparams.time_varying_reals_encoder
                + self.hparams.time_varying_reals_decoder
            )
        )

    @property
    def categoricals(self) -> List[str]:
        """List of all categorical variables in model"""
        return list(
            dict.fromkeys(
                self.hparams.static_categoricals
                + self.hparams.time_varying_categoricals_encoder
                + self.hparams.time_varying_categoricals_decoder
            )
        )

    @property
    def static_variables(self) -> List[str]:
        """List of all static variables in model"""
        return self.hparams.static_categoricals + self.hparams.static_reals

    @property
    def encoder_variables(self) -> List[str]:
        """List of all encoder variables in model (excluding static variables)"""
        return self.hparams.time_varying_categoricals_encoder + self.hparams.time_varying_reals_encoder

    @property
    def decoder_variables(self) -> List[str]:
        """List of all decoder variables in model (excluding static variables)"""
        return self.hparams.time_varying_categoricals_decoder + self.hparams.time_varying_reals_decoder

    @property
    def categorical_groups_mapping(self) -> Dict[str, str]:
        """Mapping of categorical variables to categorical groups"""
        groups = {}
        for group_name, sublist in self.hparams.categorical_groups.items():
            groups.update({name: group_name for name in sublist})
        return groups


    @classmethod
    def from_dataset(
        cls,
        dataset: TimeSeriesDataSet,
        allowed_encoder_known_variable_names: List[str] = None,
        **kwargs,
    ) -> LightningModule:
        """
        Create model from dataset and set parameters related to covariates.

        Args:
            dataset: timeseries dataset
            allowed_encoder_known_variable_names: List of known variables that are allowed in encoder, defaults to all
            **kwargs: additional arguments such as hyperparameters for model (see ``__init__()``)

        Returns:
            LightningModule
        """
        # assert fixed encoder and decoder length for the moment
        if allowed_encoder_known_variable_names is None:
            allowed_encoder_known_variable_names = (
                dataset.time_varying_known_categoricals + dataset.time_varying_known_reals
            )

        # embeddings
        embedding_labels = {
            name: encoder.classes_
            for name, encoder in dataset.categorical_encoders.items()
            if name in dataset.categoricals
        }
        embedding_paddings = dataset.dropout_categoricals
        # determine embedding sizes based on heuristic
        embedding_sizes = {
            name: (len(encoder.classes_), get_embedding_size(len(encoder.classes_)))
            for name, encoder in dataset.categorical_encoders.items()
            if name in dataset.categoricals
        }
        embedding_sizes.update(kwargs.get("embedding_sizes", {}))
        kwargs.setdefault("embedding_sizes", embedding_sizes)

        new_kwargs = dict(
            static_categoricals=dataset.static_categoricals,
            time_varying_categoricals_encoder=[
                name for name in dataset.time_varying_known_categoricals if name in allowed_encoder_known_variable_names
            ]
            + dataset.time_varying_unknown_categoricals,
            time_varying_categoricals_decoder=dataset.time_varying_known_categoricals,
            static_reals=dataset.static_reals,
            time_varying_reals_encoder=[
                name for name in dataset.time_varying_known_reals if name in allowed_encoder_known_variable_names
            ]
            + dataset.time_varying_unknown_reals,
            time_varying_reals_decoder=dataset.time_varying_known_reals,
            x_reals=dataset.reals,
            x_categoricals=dataset.flat_categoricals,
            embedding_labels=embedding_labels,
            embedding_paddings=embedding_paddings,
            categorical_groups=dataset.variable_groups,
        )
        new_kwargs.update(kwargs)
        return super().from_dataset(dataset, **new_kwargs)



    def extract_features(
        self,
        x,
        embeddings: MultiEmbedding = None,
        period: str = "all",
    ) -> torch.Tensor:
        """
        Extract features

        Args:
            x (Dict[str, torch.Tensor]): input from the dataloader
            embeddings (MultiEmbedding): embeddings for categorical variables
            period (str, optional): One of "encoder", "decoder" or "all". Defaults to "all".

        Returns:
            torch.Tensor: tensor with selected variables
        """
        # select period
        if period == "encoder":
            x_cat = x["encoder_cat"]
            x_cont = x["encoder_cont"]
        elif period == "decoder":
            x_cat = x["decoder_cat"]
            x_cont = x["decoder_cont"]
        elif period == "all":
            x_cat = torch.cat([x["encoder_cat"], x["decoder_cat"]], dim=1)  # concatenate in time dimension
            x_cont = torch.cat([x["encoder_cont"], x["decoder_cont"]], dim=1)  # concatenate in time dimension
        else:
            raise ValueError(f"Unknown type: {type}")

        # create dictionary of encoded vectors
        input_vectors = embeddings(x_cat)
        input_vectors.update(
            {
                name: x_cont[..., idx].unsqueeze(-1)
                for idx, name in enumerate(self.hparams.x_reals)
                if name in self.reals
            }
        )
        return input_vectors

    def calculate_prediction_actual_by_variable(
        self,
        x: Dict[str, torch.Tensor],
        y_pred: torch.Tensor,
        normalize: bool = True,
        bins: int = 95,
        std: float = 2.0,
        log_scale: bool = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Calculate predictions and actuals by variable averaged by ``bins`` bins spanning from ``-std`` to ``+std``

        Args:
            x: input as ``forward()``
            y_pred: predictions obtained by ``self(x, **kwargs)``
            normalize: if to return normalized averages, i.e. mean or sum of ``y``
            bins: number of bins to calculate
            std: number of standard deviations for standard scaled continuous variables
            log_scale (str, optional): if to plot in log space. If None, determined based on skew of values.
                Defaults to None.

        Returns:
            dictionary that can be used to plot averages with :py:meth:`~plot_prediction_actual_by_variable`
        """
        support = {}  # histogram
        # averages
        averages_actual = {}
        averages_prediction = {}

        # mask values and transform to log space
        max_encoder_length = x["decoder_lengths"].max()
        mask = create_mask(max_encoder_length, x["decoder_lengths"], inverse=True)
        # select valid y values
        y_flat = x["decoder_target"][mask]
        y_pred_flat = y_pred[mask]

        # determine in which average in log-space to transform data
        if log_scale is None:
            skew = torch.mean(((y_flat - torch.mean(y_flat)) / torch.std(y_flat)) ** 3)
            log_scale = skew > 1.6

        if log_scale:
            y_flat = torch.log(y_flat + 1e-8)
            y_pred_flat = torch.log(y_pred_flat + 1e-8)

        # real bins
        positive_bins = (bins - 1) // 2

        # if to normalize
        if normalize:
            reduction = "mean"
        else:
            reduction = "sum"
        # continuous variables
        reals = x["decoder_cont"]
        for idx, name in enumerate(self.hparams.x_reals):
            averages_actual[name], support[name] = groupby_apply(
                (reals[..., idx][mask] * positive_bins / std).round().clamp(-positive_bins, positive_bins).long()
                + positive_bins,
                y_flat,
                bins=bins,
                reduction=reduction,
                return_histogram=True,
            )
            averages_prediction[name], _ = groupby_apply(
                (reals[..., idx][mask] * positive_bins / std).round().clamp(-positive_bins, positive_bins).long()
                + positive_bins,
                y_pred_flat,
                bins=bins,
                reduction=reduction,
                return_histogram=True,
            )

        # categorical_variables
        cats = x["decoder_cat"]
        for idx, name in enumerate(self.hparams.x_categoricals):  # todo: make it work for grouped categoricals
            reduction = "sum"
            name = self.categorical_groups_mapping.get(name, name)
            averages_actual_cat, support_cat = groupby_apply(
                cats[..., idx][mask],
                y_flat,
                bins=self.hparams.embedding_sizes[name][0],
                reduction=reduction,
                return_histogram=True,
            )
            averages_prediction_cat, _ = groupby_apply(
                cats[..., idx][mask],
                y_pred_flat,
                bins=self.hparams.embedding_sizes[name][0],
                reduction=reduction,
                return_histogram=True,
            )

            # add either to existing calculations or
            if name in averages_actual:
                averages_actual[name] += averages_actual_cat
                support[name] += support_cat
                averages_prediction[name] += averages_prediction_cat
            else:
                averages_actual[name] = averages_actual_cat
                support[name] = support_cat
                averages_prediction[name] = averages_prediction_cat

        if normalize:  # run reduction for categoricals
            for name in self.hparams.embedding_sizes.keys():
                averages_actual[name] /= support[name].clamp(min=1)
                averages_prediction[name] /= support[name].clamp(min=1)

        if log_scale:
            for name in support.keys():
                averages_actual[name] = torch.exp(averages_actual[name])
                averages_prediction[name] = torch.exp(averages_prediction[name])

        return {
            "support": support,
            "average": {"actual": averages_actual, "prediction": averages_prediction},
            "std": std,
        }

    def plot_prediction_actual_by_variable(
        self, data: Dict[str, Dict[str, torch.Tensor]], name: str = None, ax=None, log_scale: bool = None
    ) -> Union[Dict[str, plt.Figure], plt.Figure]:
        """
        Plot predicions and actual averages by variables

        Args:
            data (Dict[str, Dict[str, torch.Tensor]]): data obtained from
                :py:meth:`~calculate_prediction_actual_by_variable`
            name (str, optional): name of variable for which to plot actuals vs predictions. Defaults to None which
                means returning a dictionary of plots for all variables.
            log_scale (str, optional): if to plot in log space. If None, determined based on skew of values.
                Defaults to None.

        Raises:
            ValueError: if the variable name is unkown

        Returns:
            Union[Dict[str, plt.Figure], plt.Figure]: matplotlib figure
        """
        if name is None:  # run recursion for figures
            figs = {name: self.plot_prediction_actual_by_variable(data, name) for name in data["support"].keys()}
            return figs
        else:
            # create figure
            kwargs = {}
            # adjust figure size for figures with many labels
            if self.hparams.embedding_sizes.get(name, [1e9])[0] > 10:
                kwargs = dict(figsize=(10, 5))
            if ax is None:
                fig, ax = plt.subplots(**kwargs)
            else:
                fig = ax.get_figure()
            ax.set_title(f"{name} averages")
            ax.set_xlabel(name)
            ax.set_ylabel("Prediction")

            ax2 = ax.twinx()  # second axis for histogram
            ax2.set_ylabel("Frequency")

            # get values for average plot and histogram
            values_actual = data["average"]["actual"][name].cpu().numpy()
            values_prediction = data["average"]["prediction"][name].cpu().numpy()
            bins = values_actual.size
            support = data["support"][name].cpu().numpy()

            # only display values where samples were observed
            support_non_zero = support > 0
            support = support[support_non_zero]
            values_actual = values_actual[support_non_zero]
            values_prediction = values_prediction[support_non_zero]

            # determine if to display results in log space
            if log_scale is None:
                log_scale = scipy.stats.skew(values_actual) > 1.6

            if log_scale:
                ax.set_yscale("log")

            # plot averages
            if name in self.hparams.x_reals:
                # create x
                if name in to_list(self.dataset_parameters["target"]):
                    if isinstance(self.output_transformer, MultiNormalizer):
                        scaler = self.output_transformer.normalizers[self.dataset_parameters["target"].index(name)]
                    else:
                        scaler = self.output_transformer
                else:
                    scaler = self.dataset_parameters["scalers"][name]
                x = np.linspace(-data["std"], data["std"], bins)
                # reversing normalization for group normalizer is not possible without sample level information
                if not isinstance(scaler, (GroupNormalizer, EncoderNormalizer)):
                    x = scaler.inverse_transform(x.reshape(-1, 1)).reshape(-1)
                    ax.set_xlabel(f"Normalized {name}")

                if len(x) > 0:
                    x_step = x[1] - x[0]
                else:
                    x_step = 1
                x = x[support_non_zero]
                ax.plot(x, values_actual, label="Actual")
                ax.plot(x, values_prediction, label="Prediction")

            elif name in self.hparams.embedding_labels:
                # sort values from lowest to highest
                sorting = values_actual.argsort()
                labels = np.asarray(list(self.hparams.embedding_labels[name].keys()))[support_non_zero][sorting]
                values_actual = values_actual[sorting]
                values_prediction = values_prediction[sorting]
                support = support[sorting]
                # cut entries if there are too many categories to fit nicely on the plot
                maxsize = 50
                if values_actual.size > maxsize:
                    values_actual = np.concatenate([values_actual[: maxsize // 2], values_actual[-maxsize // 2 :]])
                    values_prediction = np.concatenate(
                        [values_prediction[: maxsize // 2], values_prediction[-maxsize // 2 :]]
                    )
                    labels = np.concatenate([labels[: maxsize // 2], labels[-maxsize // 2 :]])
                    support = np.concatenate([support[: maxsize // 2], support[-maxsize // 2 :]])
                # plot for each category
                x = np.arange(values_actual.size)
                x_step = 1
                ax.scatter(x, values_actual, label="Actual")
                ax.scatter(x, values_prediction, label="Prediction")
                # set labels at x axis
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=90)
            else:
                raise ValueError(f"Unknown name {name}")
            # plot support histogram
            if len(support) > 1 and np.median(support) < support.max() / 10:
                ax2.set_yscale("log")
            ax2.bar(x, support, width=x_step, linewidth=0, alpha=0.2, color="k")
            # adjust layout and legend
            fig.tight_layout()
            fig.legend()
            return fig

class MonotonicQuantiles(nn.Module):
    """
    Ensures monotonically increasing quantile output.

    For x[..., quantiles], takes the median as the prediction and,
    for q < me: negative cumsum of x[..., :me][..., ::-1]
        q > me: positive cumsum of x[..., me:]

    Softplus is used to ensure that the increments are positive.
    """
    def __init__(self, quantiles: List[float]):
        super(MonotonicQuantiles, self).__init__()
        self.quantiles = quantiles
        self.idx_median = quantiles.index(0.5)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        me = self.idx_median
        x_me = x[..., me].unsqueeze(-1)
        out = torch.empty_like(x)
        out[..., me] = x_me[..., 0]
        out[..., me+1:] = x_me + torch.cumsum(self.softplus(x[..., me+1:]), dim=-1)
        out[..., :me] = x_me - torch.flip(torch.cumsum(torch.flip(self.softplus(x[..., :me]), dims=[-1]), dim=-1), dims=[-1])
        return out


# class InterpretableMultiHeadAttention(nn.Module):
#     def __init__(self, n_head_time: int, n_head_cross: int, d_model: int, dropout: float = 0.0):
#         super(InterpretableMultiHeadAttention, self).__init__()

#         self.n_head_time = n_head_time
#         self.n_head_cross = n_head_cross
#         self.d_model = d_model
#         self.d_k = self.d_q = self.d_v = d_model // (n_head_time + n_head_cross)
#         self.dropout = nn.Dropout(p=dropout)

#         self.v_layer = nn.Linear(self.d_model, self.d_v * (n_head_time + n_head_cross))
#         self.q_layers_time = nn.ModuleList([nn.Linear(self.d_model, self.d_q) for _ in range(self.n_head_time)])
#         self.k_layers_time = nn.ModuleList([nn.Linear(self.d_model, self.d_k) for _ in range(self.n_head_time)])
#         self.q_layers_cross = nn.ModuleList([nn.Linear(self.d_model, self.d_q) for _ in range(self.n_head_cross)])
#         self.k_layers_cross = nn.ModuleList([nn.Linear(self.d_model, self.d_k) for _ in range(self.n_head_cross)])

#         self.attention = ScaledDotProductAttention()
#         self.w_h = nn.Linear(self.d_v * (n_head_time + n_head_cross), self.d_model, bias=False)

#         self.init_weights()

#     # init_weights method remains the same

#     def forward(self, q, k, v, mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
#         heads = []
#         attns = []
#         vs = self.v_layer(v).view(v.size(0), v.size(1), self.n_head_time + self.n_head_cross, self.d_v)

#         # Time-based attention heads
#         for i in range(self.n_head_time):
#             qs = self.q_layers_time[i](q)
#             ks = self.k_layers_time[i](k)
#             head, attn = self.attention(qs, ks, vs[:, :, i, :], mask)
#             head_dropout = self.dropout(head)
#             heads.append(head_dropout)
#             attns.append(attn)

#         # Cross-target attention heads
#         for i in range(self.n_head_cross):
#             qs = self.q_layers_cross[i](q)
#             ks = self.k_layers_cross[i](k)
#             head, attn = self.attention(qs, ks, vs[:, :, i + self.n_head_time, :], mask)
#             head_dropout = self.dropout(head)
#             heads.append(head_dropout)
#             attns.append(attn)

#         head = torch.cat(heads, dim=2)
#         attn = torch.cat(attns, dim=2)

#         outputs = self.w_h(head)
#         outputs = self.dropout(outputs)

#         # The query (`q`) and key (`k`) layers are separated into two groups:
#         # one for time-based attention and one for cross-target attention.
#         # The value (`v`) layers are shared across all heads but are split accordingly during the forward pass.
#         # Attention is computed separately for each head and then concatenated.
#         # The concatenated outputs are then passed through a final linear layer (`w_h`).

#         # The `forward` method handles both types of attention heads.
#         # The `vs` tensor is reshaped to separate the different attention heads.
#         # Time-based and cross-target attentions are then computed in their respective loops.
#         # The outputs from all heads are concatenated along the second dimension (representing different heads).
#         # Finally, a linear transformation is applied to the concatenated outputs.

#         return outputs, attn

class TemporalFusionTransformerDistribution(BaseModelWithCovariatesJointTarget):
    def __init__(
        self,
        loss_type: str = 'mixture',
        quantiles_2D: List[float] = [0.5, 0.9, 0.95],
        num_mixtures: int = 3,
        hidden_size: int = 16,
        lstm_layers: int = 1,
        loss: MultiHorizonMetric = None,
        epsilon: float = 1e-5,
        lambda_area: float = 5.,
        affine_transform: bool = True,
        is_energy_score: bool = False,
        beta: float = 1,
        R: int = 100,
        output_size: int = 18,
        dropout: float = 0.1,
        attention_head_size: int = 4,
        max_encoder_length: int = 10,
        static_categoricals: List[str] = [],
        static_reals: List[str] = [],
        time_varying_categoricals_encoder: List[str] = [],
        time_varying_categoricals_decoder: List[str] = [],
        categorical_groups: Dict[str, List[str]] = {},
        time_varying_reals_encoder: List[str] = [],
        time_varying_reals_decoder: List[str] = [],
        x_reals: List[str] = [],
        x_categoricals: List[str] = [],
        hidden_continuous_size: int = 8,
        hidden_continuous_sizes: Dict[str, int] = {},
        embedding_sizes: Dict[str, Tuple[int, int]] = {},
        embedding_paddings: List[str] = [],
        embedding_labels: Dict[str, np.ndarray] = {},
        learning_rate: float = 1e-3,
        log_interval: Union[int, float] = -1,
        log_val_interval: Union[int, float] = None,
        log_gradient_flow: bool = False,
        reduce_on_plateau_patience: int = 1000,
        monotone_constaints: Dict[str, int] = {},
        share_single_variable_networks: bool = False,
        causal_attention: bool = True,
        logging_metrics: nn.ModuleList = None,
        **kwargs,
    ):
        """
        Temporal Fusion Transformer for forecasting timeseries - use its :py:meth:`~from_dataset` method if possible.

        Implementation of the article
        `Temporal Fusion Transformers for Interpretable Multi-horizon Time Series
        Forecasting <https://arxiv.org/pdf/1912.09363.pdf>`_. The network outperforms DeepAR by Amazon by 36-69%
        in benchmarks.

        Enhancements compared to the original implementation (apart from capabilities added through base model
        such as monotone constraints):

        * static variables can be continuous
        * multiple categorical variables can be summarized with an EmbeddingBag
        * variable encoder and decoder length by sample
        * categorical embeddings are not transformed by variable selection network (because it is a redundant operation)
        * variable dimension in variable selection network are scaled up via linear interpolation to reduce
          number of parameters
        * non-linear variable processing in variable selection network can be shared among decoder and encoder
          (not shared by default)

        Tune its hyperparameters with
        :py:func:`~pytorch_forecasting.models.temporal_fusion_transformer.tuning.optimize_hyperparameters`.

        Args:

            hidden_size: hidden size of network which is its main hyperparameter and can range from 8 to 512
            lstm_layers: number of LSTM layers (2 is mostly optimal)
            dropout: dropout rate
            output_size: number of outputs (e.g. number of quantiles for QuantileLoss and one target or list
                of output sizes).
            loss: loss function taking prediction and targets
            epsilon: small number to avoid zero target scale
            attention_head_size: number of attention heads (4 is a good default)
            max_encoder_length: length to encode (can be far longer than the decoder length but does not have to be)
            static_categoricals: names of static categorical variables
            static_reals: names of static continuous variables
            time_varying_categoricals_encoder: names of categorical variables for encoder
            time_varying_categoricals_decoder: names of categorical variables for decoder
            time_varying_reals_encoder: names of continuous variables for encoder
            time_varying_reals_decoder: names of continuous variables for decoder
            categorical_groups: dictionary where values
                are list of categorical variables that are forming together a new categorical
                variable which is the key in the dictionary
            x_reals: order of continuous variables in tensor passed to forward function
            x_categoricals: order of categorical variables in tensor passed to forward function
            hidden_continuous_size: default for hidden size for processing continous variables (similar to categorical
                embedding size)
            hidden_continuous_sizes: dictionary mapping continuous input indices to sizes for variable selection
                (fallback to hidden_continuous_size if index is not in dictionary)
            embedding_sizes: dictionary mapping (string) indices to tuple of number of categorical classes and
                embedding size
            embedding_paddings: list of indices for embeddings which transform the zero's embedding to a zero vector
            embedding_labels: dictionary mapping (string) indices to list of categorical labels
            learning_rate: learning rate
            log_interval: log predictions every x batches, do not log if 0 or less, log interpretation if > 0. If < 1.0
                , will log multiple entries per batch. Defaults to -1.
            log_val_interval: frequency with which to log validation set metrics, defaults to log_interval
            log_gradient_flow: if to log gradient flow, this takes time and should be only done to diagnose training
                failures
            reduce_on_plateau_patience (int): patience after which learning rate is reduced by a factor of 10
            monotone_constaints (Dict[str, int]): dictionary of monotonicity constraints for continuous decoder
                variables mapping
                position (e.g. ``"0"`` for first position) to constraint (``-1`` for negative and ``+1`` for positive,
                larger numbers add more weight to the constraint vs. the loss but are usually not necessary).
                This constraint significantly slows down training. Defaults to {}.
            share_single_variable_networks (bool): if to share the single variable networks between the encoder and
                decoder. Defaults to False.
            causal_attention (bool): If to attend only at previous timesteps in the decoder or also include future
                predictions. Defaults to True.
            logging_metrics (nn.ModuleList[LightningMetric]): list of metrics that are logged during training.
                Defaults to nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE()]).
            **kwargs: additional arguments to :py:class:`~BaseModel`.
        """
        self.loss_type = loss_type
        self.quantiles_2D = quantiles_2D
        if loss_type == 'quantile':
            self.num_mixtures = 1
            self.output_size = 5
            if loss is None:
                loss = BivariateGaussianQuantileLoss(quantiles_2D=quantiles_2D)
        elif loss_type == 'mixture':
            self.num_mixtures = num_mixtures
            self.output_size = num_mixtures * 6
            self.epsilon = epsilon
            self.affine_transform = affine_transform
            if loss is None:
                loss = BivariateGaussianMixtureLoss(num_mixtures=num_mixtures, epsilon=epsilon, lambda_area=lambda_area, affine_transform=affine_transform,
                                                    is_energy_score=is_energy_score, beta=beta, R=R)
        else:
            raise ValueError(f"Unknown loss_type {loss_type}. Available: 'quantile', 'mixture'")

        if logging_metrics is None:
            # TODO: implement Euclidean distance
            # TODO: implement analogous of SMAPE for 2D
            # logging_metrics = nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE()])
            logging_metrics = nn.ModuleList([MAEMulti(), RMSEMulti()])
        self.save_hyperparameters()
        # store loss function separately as it is a module
        assert isinstance(loss, LightningMetric), "Loss has to be a PyTorch Lightning `Metric`"
        super().__init__(loss=loss, logging_metrics=logging_metrics, **kwargs)

        # processing inputs
        # embeddings
        self.input_embeddings = MultiEmbedding(
            embedding_sizes=self.hparams.embedding_sizes,
            categorical_groups=self.hparams.categorical_groups,
            embedding_paddings=self.hparams.embedding_paddings,
            x_categoricals=self.hparams.x_categoricals,
            max_embedding_size=self.hparams.hidden_size,
        )

        # continuous variable processing
        self.prescalers = nn.ModuleDict(
            {
                name: nn.Linear(1, self.hparams.hidden_continuous_sizes.get(name, self.hparams.hidden_continuous_size))
                for name in self.reals
            }
        )

        # variable selection
        # variable selection for static variables
        static_input_sizes = {
            name: self.input_embeddings.output_size[name] for name in self.hparams.static_categoricals
        }
        static_input_sizes.update(
            {
                name: self.hparams.hidden_continuous_sizes.get(name, self.hparams.hidden_continuous_size)
                for name in self.hparams.static_reals
            }
        )
        self.static_variable_selection = VariableSelectionNetwork(
            input_sizes=static_input_sizes,
            hidden_size=self.hparams.hidden_size,
            input_embedding_flags={name: True for name in self.hparams.static_categoricals},
            dropout=self.hparams.dropout,
            prescalers=self.prescalers,
        )

        # variable selection for encoder and decoder
        encoder_input_sizes = {
            name: self.input_embeddings.output_size[name] for name in self.hparams.time_varying_categoricals_encoder
        }
        encoder_input_sizes.update(
            {
                name: self.hparams.hidden_continuous_sizes.get(name, self.hparams.hidden_continuous_size)
                for name in self.hparams.time_varying_reals_encoder
            }
        )

        decoder_input_sizes = {
            name: self.input_embeddings.output_size[name] for name in self.hparams.time_varying_categoricals_decoder
        }
        decoder_input_sizes.update(
            {
                name: self.hparams.hidden_continuous_sizes.get(name, self.hparams.hidden_continuous_size)
                for name in self.hparams.time_varying_reals_decoder
            }
        )

        # create single variable grns that are shared across decoder and encoder
        if self.hparams.share_single_variable_networks:
            self.shared_single_variable_grns = nn.ModuleDict()
            for name, input_size in encoder_input_sizes.items():
                self.shared_single_variable_grns[name] = GatedResidualNetwork(
                    input_size,
                    min(input_size, self.hparams.hidden_size),
                    self.hparams.hidden_size,
                    self.hparams.dropout,
                )
            for name, input_size in decoder_input_sizes.items():
                if name not in self.shared_single_variable_grns:
                    self.shared_single_variable_grns[name] = GatedResidualNetwork(
                        input_size,
                        min(input_size, self.hparams.hidden_size),
                        self.hparams.hidden_size,
                        self.hparams.dropout,
                    )

        self.encoder_variable_selection = VariableSelectionNetwork(
            input_sizes=encoder_input_sizes,
            hidden_size=self.hparams.hidden_size,
            input_embedding_flags={name: True for name in self.hparams.time_varying_categoricals_encoder},
            dropout=self.hparams.dropout,
            context_size=self.hparams.hidden_size,
            prescalers=self.prescalers,
            single_variable_grns={}
            if not self.hparams.share_single_variable_networks
            else self.shared_single_variable_grns,
        )

        self.decoder_variable_selection = VariableSelectionNetwork(
            input_sizes=decoder_input_sizes,
            hidden_size=self.hparams.hidden_size,
            input_embedding_flags={name: True for name in self.hparams.time_varying_categoricals_decoder},
            dropout=self.hparams.dropout,
            context_size=self.hparams.hidden_size,
            prescalers=self.prescalers,
            single_variable_grns={}
            if not self.hparams.share_single_variable_networks
            else self.shared_single_variable_grns,
        )

        # static encoders
        # for variable selection
        self.static_context_variable_selection = GatedResidualNetwork(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            output_size=self.hparams.hidden_size,
            dropout=self.hparams.dropout,
        )

        # for hidden state of the lstm
        self.static_context_initial_hidden_lstm = GatedResidualNetwork(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            output_size=self.hparams.hidden_size,
            dropout=self.hparams.dropout,
        )

        # for cell state of the lstm
        self.static_context_initial_cell_lstm = GatedResidualNetwork(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            output_size=self.hparams.hidden_size,
            dropout=self.hparams.dropout,
        )

        # for post lstm static enrichment
        self.static_context_enrichment = GatedResidualNetwork(
            self.hparams.hidden_size, self.hparams.hidden_size, self.hparams.hidden_size, self.hparams.dropout
        )

        # lstm encoder (history) and decoder (future) for local processing
        self.lstm_encoder = LSTM(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.lstm_layers,
            dropout=self.hparams.dropout if self.hparams.lstm_layers > 1 else 0,
            batch_first=True,
        )

        self.lstm_decoder = LSTM(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.lstm_layers,
            dropout=self.hparams.dropout if self.hparams.lstm_layers > 1 else 0,
            batch_first=True,
        )

        # skip connection for lstm
        self.post_lstm_gate_encoder = GatedLinearUnit(self.hparams.hidden_size, dropout=self.hparams.dropout)
        self.post_lstm_gate_decoder = self.post_lstm_gate_encoder
        # self.post_lstm_gate_decoder = GatedLinearUnit(self.hparams.hidden_size, dropout=self.hparams.dropout)
        self.post_lstm_add_norm_encoder = AddNorm(self.hparams.hidden_size, trainable_add=False)
        # self.post_lstm_add_norm_decoder = AddNorm(self.hparams.hidden_size, trainable_add=True)
        self.post_lstm_add_norm_decoder = self.post_lstm_add_norm_encoder

        # static enrichment and processing past LSTM
        self.static_enrichment = GatedResidualNetwork(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            output_size=self.hparams.hidden_size,
            dropout=self.hparams.dropout,
            context_size=self.hparams.hidden_size,
        )

        # attention for long-range processing
        self.multihead_attn = InterpretableMultiHeadAttention(
            d_model=self.hparams.hidden_size, n_head=self.hparams.attention_head_size, dropout=self.hparams.dropout
        )
        self.post_attn_gate_norm = GateAddNorm(
            self.hparams.hidden_size, dropout=self.hparams.dropout, trainable_add=False
        )
        self.pos_wise_ff = GatedResidualNetwork(
            self.hparams.hidden_size, self.hparams.hidden_size, self.hparams.hidden_size, dropout=self.hparams.dropout
        )

        # output processing -> no dropout at this late stage
        self.pre_output_gate_norm = GateAddNorm(self.hparams.hidden_size, dropout=None, trainable_add=False)

        # if self.n_targets > 1 and not self.force_1D_output:  # if to run with multiple targets
        #     self.output_layer = nn.ModuleList(
        #         [nn.Linear(self.hparams.hidden_size, output_size) for output_size in self.hparams.output_size]
        #     )
        # else:
        if isinstance(self.hparams.output_size, list):
            self.hparams.output_size = self.hparams.output_size[0]
        self.output_layer = nn.Linear(self.hparams.hidden_size, self.hparams.output_size)


    @classmethod
    def from_dataset(
        cls,
        dataset: TimeSeriesDataSet,
        allowed_encoder_known_variable_names: List[str] = None,
        **kwargs,
    ):
        """
        Create model from dataset.

        Args:
            dataset: timeseries dataset
            allowed_encoder_known_variable_names: List of known variables that are allowed in encoder, defaults to all
            **kwargs: additional arguments such as hyperparameters for model (see ``__init__()``)

        Returns:
            TemporalFusionTransformer
        """
        # add maximum encoder length
        # update defaults
        new_kwargs = copy(kwargs)
        new_kwargs["max_encoder_length"] = dataset.max_encoder_length
        new_kwargs.update(cls.deduce_default_output_parameters(dataset, kwargs, BivariateGaussianMixtureLoss(num_mixtures=kwargs['num_mixtures'])))

        # create class and return
        return super().from_dataset(
            dataset, allowed_encoder_known_variable_names=allowed_encoder_known_variable_names, **new_kwargs
        )



    def expand_static_context(self, context, timesteps):
        """
        add time dimension to static context
        """
        return context[:, None].expand(-1, timesteps, -1)



    def get_attention_mask(self, encoder_lengths: torch.LongTensor, decoder_lengths: torch.LongTensor):
        """
        Returns causal mask to apply for self-attention layer.
        """
        decoder_length = decoder_lengths.max()
        if self.hparams.causal_attention:
            # indices to which is attended
            attend_step = torch.arange(decoder_length, device=self.device)
            # indices for which is predicted
            predict_step = torch.arange(0, decoder_length, device=self.device)[:, None]
            # do not attend to steps to self or after prediction
            decoder_mask = (attend_step >= predict_step).unsqueeze(0).expand(encoder_lengths.size(0), -1, -1)
        else:
            # there is value in attending to future forecasts if they are made with knowledge currently
            #   available
            #   one possibility is here to use a second attention layer for future attention (assuming different effects
            #   matter in the future than the past)
            #   or alternatively using the same layer but allowing forward attention - i.e. only
            #   masking out non-available data and self
            decoder_mask = create_mask(decoder_length, decoder_lengths).unsqueeze(1).expand(-1, decoder_length, -1)
        # do not attend to steps where data is padded
        encoder_mask = create_mask(encoder_lengths.max(), encoder_lengths).unsqueeze(1).expand(-1, decoder_length, -1)
        # combine masks along attended time - first encoder and then decoder
        mask = torch.cat(
            (
                encoder_mask,
                decoder_mask,
            ),
            dim=2,
        )
        return mask

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        input dimensions: n_samples x time x variables
        """
        encoder_lengths = x["encoder_lengths"]
        decoder_lengths = x["decoder_lengths"]
        x_cat = torch.cat([x["encoder_cat"], x["decoder_cat"]], dim=1)  # concatenate in time dimension
        x_cont = torch.cat([x["encoder_cont"], x["decoder_cont"]], dim=1)  # concatenate in time dimension
        timesteps = x_cont.size(1)  # encode + decode length
        max_encoder_length = int(encoder_lengths.max())
        input_vectors = self.input_embeddings(x_cat)
        input_vectors.update(
            {
                name: x_cont[..., idx].unsqueeze(-1)
                for idx, name in enumerate(self.hparams.x_reals)
                if name in self.reals
            }
        )

        # Embedding and variable selection
        if len(self.static_variables) > 0:
            # static embeddings will be constant over entire batch
            static_embedding = {name: input_vectors[name][:, 0] for name in self.static_variables}
            static_embedding, static_variable_selection = self.static_variable_selection(static_embedding)
        else:
            static_embedding = torch.zeros(
                (x_cont.size(0), self.hparams.hidden_size), dtype=self.dtype, device=self.device
            )
            static_variable_selection = torch.zeros((x_cont.size(0), 0), dtype=self.dtype, device=self.device)

        static_context_variable_selection = self.expand_static_context(
            self.static_context_variable_selection(static_embedding), timesteps
        )

        embeddings_varying_encoder = {
            name: input_vectors[name][:, :max_encoder_length] for name in self.encoder_variables
        }
        embeddings_varying_encoder, encoder_sparse_weights = self.encoder_variable_selection(
            embeddings_varying_encoder,
            static_context_variable_selection[:, :max_encoder_length],
        )

        embeddings_varying_decoder = {
            name: input_vectors[name][:, max_encoder_length:] for name in self.decoder_variables  # select decoder
        }
        embeddings_varying_decoder, decoder_sparse_weights = self.decoder_variable_selection(
            embeddings_varying_decoder,
            static_context_variable_selection[:, max_encoder_length:],
        )

        # LSTM
        # calculate initial state
        input_hidden = self.static_context_initial_hidden_lstm(static_embedding).expand(
            self.hparams.lstm_layers, -1, -1
        )
        input_cell = self.static_context_initial_cell_lstm(static_embedding).expand(self.hparams.lstm_layers, -1, -1)

        # run local encoder
        encoder_output, (hidden, cell) = self.lstm_encoder(
            embeddings_varying_encoder, (input_hidden, input_cell), lengths=encoder_lengths, enforce_sorted=False
        )

        # run local decoder
        decoder_output, _ = self.lstm_decoder(
            embeddings_varying_decoder,
            (hidden, cell),
            lengths=decoder_lengths,
            enforce_sorted=False,
        )

        # skip connection over lstm
        lstm_output_encoder = self.post_lstm_gate_encoder(encoder_output)
        lstm_output_encoder = self.post_lstm_add_norm_encoder(lstm_output_encoder, embeddings_varying_encoder)

        lstm_output_decoder = self.post_lstm_gate_decoder(decoder_output)
        lstm_output_decoder = self.post_lstm_add_norm_decoder(lstm_output_decoder, embeddings_varying_decoder)

        lstm_output = torch.cat([lstm_output_encoder, lstm_output_decoder], dim=1)

        # static enrichment
        static_context_enrichment = self.static_context_enrichment(static_embedding)
        attn_input = self.static_enrichment(
            lstm_output, self.expand_static_context(static_context_enrichment, timesteps)
        )

        # Attention
        attn_output, attn_output_weights = self.multihead_attn(
            q=attn_input[:, max_encoder_length:],  # query only for predictions
            k=attn_input,
            v=attn_input,
            mask=self.get_attention_mask(encoder_lengths=encoder_lengths, decoder_lengths=decoder_lengths),
        )

        # skip connection over attention
        attn_output = self.post_attn_gate_norm(attn_output, attn_input[:, max_encoder_length:])

        output = self.pos_wise_ff(attn_output)

        # skip connection over temporal fusion decoder (not LSTM decoder despite the LSTM output contains
        # a skip from the variable selection network)
        output = self.pre_output_gate_norm(output, lstm_output[:, max_encoder_length:])
        # if self.n_targets > 1 and not self.force_1D_output:  # if to use multi-target architecture
        #     output = [output_layer(output) for output_layer in self.output_layer]
        # else:
        output = self.output_layer(output)

        return self.to_network_output(
            prediction=self.transform_output(output, target_scale=x["target_scale"], x=x),
            encoder_attention=attn_output_weights[..., :max_encoder_length],
            decoder_attention=attn_output_weights[..., max_encoder_length:],
            static_variables=static_variable_selection,
            encoder_variables=encoder_sparse_weights,
            decoder_variables=decoder_sparse_weights,
            decoder_lengths=decoder_lengths,
            encoder_lengths=encoder_lengths,
        )

    def on_fit_end(self):
        if self.log_interval > 0:
            self.log_embeddings()

    def create_log(self, x, y, out, batch_idx, **kwargs):
        log = super().create_log(x, y, out, batch_idx, **kwargs)
        if self.log_interval > 0:
            log["interpretation"] = self._log_interpretation(out)
        return log

    def _log_interpretation(self, out):
        # calculate interpretations etc for latter logging
        interpretation = self.interpret_output(
            detach(out),
            reduction="sum",
            attention_prediction_horizon=0,  # attention only for first prediction horizon
        )
        return interpretation

    # def on_epoch_end(self, outputs):
    #     """
    #     run at epoch end for training or validation
    #     """
    #     if self.log_interval > 0 and not self.training:
    #         self.log_interpretation(outputs)

    def on_epoch_end(self):
        """
        run at epoch end for training or validation
        """
        pass

    def interpret_output(
        self,
        out: Dict[str, torch.Tensor],
        reduction: str = "none",
        attention_prediction_horizon: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        interpret output of model

        Args:
            out: output as produced by ``forward()``
            reduction: "none" for no averaging over batches, "sum" for summing attentions, "mean" for
                normalizing by encode lengths
            attention_prediction_horizon: which prediction horizon to use for attention

        Returns:
            interpretations that can be plotted with ``plot_interpretation()``
        """
        # take attention and concatenate if a list to proper attention object
        batch_size = len(out["decoder_attention"])
        if isinstance(out["decoder_attention"], (list, tuple)):
            # start with decoder attention
            # assume issue is in last dimension, we need to find max
            max_last_dimension = max(x.size(-1) for x in out["decoder_attention"])
            first_elm = out["decoder_attention"][0]
            # create new attention tensor into which we will scatter
            decoder_attention = torch.full(
                (batch_size, *first_elm.shape[:-1], max_last_dimension),
                float("nan"),
                dtype=first_elm.dtype,
                device=first_elm.device,
            )
            # scatter into tensor
            for idx, x in enumerate(out["decoder_attention"]):
                decoder_length = out["decoder_lengths"][idx]
                decoder_attention[idx, :, :, :decoder_length] = x[..., :decoder_length]
        else:
            decoder_attention = out["decoder_attention"].clone()
            decoder_mask = create_mask(out["decoder_attention"].size(1), out["decoder_lengths"])
            decoder_attention[decoder_mask[..., None, None].expand_as(decoder_attention)] = float("nan")

        if isinstance(out["encoder_attention"], (list, tuple)):
            # same game for encoder attention
            # create new attention tensor into which we will scatter
            first_elm = out["encoder_attention"][0]
            encoder_attention = torch.full(
                (batch_size, *first_elm.shape[:-1], self.hparams.max_encoder_length),
                float("nan"),
                dtype=first_elm.dtype,
                device=first_elm.device,
            )
            # scatter into tensor
            for idx, x in enumerate(out["encoder_attention"]):
                encoder_length = out["encoder_lengths"][idx]
                encoder_attention[idx, :, :, self.hparams.max_encoder_length - encoder_length :] = x[
                    ..., :encoder_length
                ]
        else:
            # roll encoder attention (so start last encoder value is on the right)
            encoder_attention = out["encoder_attention"].clone()
            shifts = encoder_attention.size(3) - out["encoder_lengths"]
            new_index = (
                torch.arange(encoder_attention.size(3), device=encoder_attention.device)[None, None, None].expand_as(
                    encoder_attention
                )
                - shifts[:, None, None, None]
            ) % encoder_attention.size(3)
            encoder_attention = torch.gather(encoder_attention, dim=3, index=new_index)
            # expand encoder_attentiont to full size
            if encoder_attention.size(-1) < self.hparams.max_encoder_length:
                encoder_attention = torch.concat(
                    [
                        torch.full(
                            (
                                *encoder_attention.shape[:-1],
                                self.hparams.max_encoder_length - out["encoder_lengths"].max(),
                            ),
                            float("nan"),
                            dtype=encoder_attention.dtype,
                            device=encoder_attention.device,
                        ),
                        encoder_attention,
                    ],
                    dim=-1,
                )

        # combine attention vector
        attention = torch.concat([encoder_attention, decoder_attention], dim=-1)
        attention[attention < 1e-5] = float("nan")

        # histogram of decode and encode lengths
        encoder_length_histogram = integer_histogram(out["encoder_lengths"], min=0, max=self.hparams.max_encoder_length)
        decoder_length_histogram = integer_histogram(
            out["decoder_lengths"], min=1, max=out["decoder_variables"].size(1)
        )

        # mask where decoder and encoder where not applied when averaging variable selection weights
        encoder_variables = out["encoder_variables"].squeeze(-2).clone()
        encode_mask = create_mask(encoder_variables.size(1), out["encoder_lengths"])
        encoder_variables = encoder_variables.masked_fill(encode_mask.unsqueeze(-1), 0.0).sum(dim=1)
        encoder_variables /= (
            out["encoder_lengths"]
            .where(out["encoder_lengths"] > 0, torch.ones_like(out["encoder_lengths"]))
            .unsqueeze(-1)
        )

        decoder_variables = out["decoder_variables"].squeeze(-2).clone()
        decode_mask = create_mask(decoder_variables.size(1), out["decoder_lengths"])
        decoder_variables = decoder_variables.masked_fill(decode_mask.unsqueeze(-1), 0.0).sum(dim=1)
        decoder_variables /= out["decoder_lengths"].unsqueeze(-1)

        # static variables need no masking
        static_variables = out["static_variables"].squeeze(1)
        # attention is batch x time x heads x time_to_attend
        # average over heads + only keep prediction attention and attention on observed timesteps
        attention = masked_op(
            attention[
                :, attention_prediction_horizon, :, : self.hparams.max_encoder_length + attention_prediction_horizon
            ],
            op="mean",
            dim=1,
        )

        if reduction != "none":  # if to average over batches
            static_variables = static_variables.sum(dim=0)
            encoder_variables = encoder_variables.sum(dim=0)
            decoder_variables = decoder_variables.sum(dim=0)

            attention = masked_op(attention, dim=0, op=reduction)
        else:
            attention = attention / masked_op(attention, dim=1, op="sum").unsqueeze(-1)  # renormalize

        interpretation = dict(
            attention=attention.masked_fill(torch.isnan(attention), 0.0),
            static_variables=static_variables,
            encoder_variables=encoder_variables,
            decoder_variables=decoder_variables,
            encoder_length_histogram=encoder_length_histogram,
            decoder_length_histogram=decoder_length_histogram,
        )
        return interpretation



    def plot_prediction(
        self,
        x: Dict[str, torch.Tensor],
        out: Dict[str, torch.Tensor],
        idx: int,
        plot_attention: bool = True,
        add_loss_to_title: bool = False,
        show_future_observed: bool = True,
        ax=None,
        **kwargs,
    ) -> plt.Figure:
        """
        Plot actuals vs prediction and attention

        Args:
            x (Dict[str, torch.Tensor]): network input
            out (Dict[str, torch.Tensor]): network output
            idx (int): sample index
            plot_attention: if to plot attention on secondary axis
            add_loss_to_title: if to add loss to title. Default to False.
            show_future_observed: if to show actuals for future. Defaults to True.
            ax: matplotlib axes to plot on

        Returns:
            plt.Figure: matplotlib figure
        """

        # plot prediction as normal
        fig = super().plot_prediction(
            x,
            out,
            idx=idx,
            add_loss_to_title=add_loss_to_title,
            show_future_observed=show_future_observed,
            ax=ax,
            **kwargs,
        )

        # add attention on secondary axis
        if plot_attention:
            interpretation = self.interpret_output(out.iget(slice(idx, idx + 1)))
            for f in to_list(fig):
                ax = f.axes[0]
                ax2 = ax.twinx()
                ax2.set_ylabel("Attention")
                encoder_length = x["encoder_lengths"][0]
                ax2.plot(
                    torch.arange(-encoder_length, 0),
                    interpretation["attention"][0, -encoder_length:].detach().cpu(),
                    alpha=0.2,
                    color="k",
                )
                f.tight_layout()
        return fig



    def plot_interpretation(self, interpretation: Dict[str, torch.Tensor]) -> Dict[str, plt.Figure]:
        """
        Make figures that interpret model.

        * Attention
        * Variable selection weights / importances

        Args:
            interpretation: as obtained from ``interpret_output()``

        Returns:
            dictionary of matplotlib figures
        """
        figs = {}

        # attention
        fig, ax = plt.subplots()
        attention = interpretation["attention"].detach().cpu()
        attention = attention / attention.sum(-1).unsqueeze(-1)
        ax.plot(
            np.arange(-self.hparams.max_encoder_length, attention.size(0) - self.hparams.max_encoder_length), attention
        )
        ax.set_xlabel("Time index")
        ax.set_ylabel("Attention")
        ax.set_title("Attention")
        figs["attention"] = fig

        # variable selection
        def make_selection_plot(title, values, labels):
            fig, ax = plt.subplots(figsize=(7, len(values) * 0.25 + 2))
            order = np.argsort(values)
            values = values / values.sum(-1).unsqueeze(-1)
            ax.barh(np.arange(len(values)), values[order] * 100, tick_label=np.asarray(labels)[order])
            ax.set_title(title)
            ax.set_xlabel("Importance in %")
            plt.tight_layout()
            return fig

        figs["static_variables"] = make_selection_plot(
            "Static variables importance", interpretation["static_variables"].detach().cpu(), self.static_variables
        )
        figs["encoder_variables"] = make_selection_plot(
            "Encoder variables importance", interpretation["encoder_variables"].detach().cpu(), self.encoder_variables
        )
        figs["decoder_variables"] = make_selection_plot(
            "Decoder variables importance", interpretation["decoder_variables"].detach().cpu(), self.decoder_variables
        )

        return figs



    def log_interpretation(self, outputs):
        """
        Log interpretation metrics to tensorboard.
        """
        # extract interpretations
        interpretation = {
            # use padded_stack because decoder length histogram can be of different length
            name: padded_stack([x["interpretation"][name].detach() for x in outputs], side="right", value=0).sum(0)
            for name in outputs[0]["interpretation"].keys()
        }
        # normalize attention with length histogram squared to account for: 1. zeros in attention and
        # 2. higher attention due to less values
        attention_occurances = interpretation["encoder_length_histogram"][1:].flip(0).float().cumsum(0)
        attention_occurances = attention_occurances / attention_occurances.max()
        attention_occurances = torch.cat(
            [
                attention_occurances,
                torch.ones(
                    interpretation["attention"].size(0) - attention_occurances.size(0),
                    dtype=attention_occurances.dtype,
                    device=attention_occurances.device,
                ),
            ],
            dim=0,
        )
        interpretation["attention"] = interpretation["attention"] / attention_occurances.pow(2).clamp(1.0)
        interpretation["attention"] = interpretation["attention"] / interpretation["attention"].sum()

        figs = self.plot_interpretation(interpretation)  # make interpretation figures
        label = self.current_stage
        # log to tensorboard
        for name, fig in figs.items():
            self.logger.experiment.add_figure(
                f"{label.capitalize()} {name} importance", fig, global_step=self.global_step
            )

        # log lengths of encoder/decoder
        for type in ["encoder", "decoder"]:
            fig, ax = plt.subplots()
            lengths = (
                padded_stack([out["interpretation"][f"{type}_length_histogram"] for out in outputs])
                .sum(0)
                .detach()
                .cpu()
            )
            if type == "decoder":
                start = 1
            else:
                start = 0
            ax.plot(torch.arange(start, start + len(lengths)), lengths)
            ax.set_xlabel(f"{type.capitalize()} length")
            ax.set_ylabel("Number of samples")
            ax.set_title(f"{type.capitalize()} length distribution in {label} epoch")

            self.logger.experiment.add_figure(
                f"{label.capitalize()} {type} length distribution", fig, global_step=self.global_step
            )



    def log_embeddings(self):
        """
        Log embeddings to tensorboard
        """
        for name, emb in self.input_embeddings.items():
            labels = self.hparams.embedding_labels[name]
            self.logger.experiment.add_embedding(
                emb.weight.data.detach().cpu(), metadata=labels, tag=name, global_step=self.global_step
                )

class TemporalFusionTransformerSpatial(BaseModelWithCovariates):
    def __init__(
        self,
        hidden_size: int = 16,
        lstm_layers: int = 1,
        dropout: float = 0.1,
        output_size: Union[int, List[int]] = 7,
        loss: MultiHorizonMetric = None,
        cumulative: bool = False,
        joint_prediction: bool = False,
        attention_head_size: int = 4,
        max_encoder_length: int = 10,
        static_categoricals: List[str] = [],
        static_reals: List[str] = [],
        time_varying_categoricals_encoder: List[str] = [],
        time_varying_categoricals_decoder: List[str] = [],
        categorical_groups: Dict[str, List[str]] = {},
        time_varying_reals_encoder: List[str] = [],
        time_varying_reals_decoder: List[str] = [],
        x_reals: List[str] = [],
        x_categoricals: List[str] = [],
        hidden_continuous_size: int = 8,
        hidden_continuous_sizes: Dict[str, int] = {},
        embedding_sizes: Dict[str, Tuple[int, int]] = {},
        embedding_paddings: List[str] = [],
        embedding_labels: Dict[str, np.ndarray] = {},
        learning_rate: float = 1e-3,
        log_interval: Union[int, float] = -1,
        log_val_interval: Union[int, float] = None,
        log_gradient_flow: bool = False,
        reduce_on_plateau_patience: int = 1000,
        monotone_constaints: Dict[str, int] = {},
        share_single_variable_networks: bool = False,
        causal_attention: bool = True,
        logging_metrics: nn.ModuleList = None,
        **kwargs,
    ):
        """
        Temporal Fusion Transformer for forecasting timeseries - use its :py:meth:`~from_dataset` method if possible.

        Implementation of the article
        `Temporal Fusion Transformers for Interpretable Multi-horizon Time Series
        Forecasting <https://arxiv.org/pdf/1912.09363.pdf>`_. The network outperforms DeepAR by Amazon by 36-69%
        in benchmarks.

        Enhancements compared to the original implementation (apart from capabilities added through base model
        such as monotone constraints):

        * static variables can be continuous
        * multiple categorical variables can be summarized with an EmbeddingBag
        * variable encoder and decoder length by sample
        * categorical embeddings are not transformed by variable selection network (because it is a redundant operation)
        * variable dimension in variable selection network are scaled up via linear interpolation to reduce
          number of parameters
        * non-linear variable processing in variable selection network can be shared among decoder and encoder
          (not shared by default)

        Tune its hyperparameters with
        :py:func:`~pytorch_forecasting.models.temporal_fusion_transformer.tuning.optimize_hyperparameters`.

        Args:

            hidden_size: hidden size of network which is its main hyperparameter and can range from 8 to 512
            lstm_layers: number of LSTM layers (2 is mostly optimal)
            dropout: dropout rate
            output_size: number of outputs (e.g. number of quantiles for QuantileLoss and one target or list
                of output sizes).
            loss: loss function taking prediction and targets
            attention_head_size: number of attention heads (4 is a good default)
            max_encoder_length: length to encode (can be far longer than the decoder length but does not have to be)
            static_categoricals: names of static categorical variables
            static_reals: names of static continuous variables
            time_varying_categoricals_encoder: names of categorical variables for encoder
            time_varying_categoricals_decoder: names of categorical variables for decoder
            time_varying_reals_encoder: names of continuous variables for encoder
            time_varying_reals_decoder: names of continuous variables for decoder
            categorical_groups: dictionary where values
                are list of categorical variables that are forming together a new categorical
                variable which is the key in the dictionary
            x_reals: order of continuous variables in tensor passed to forward function
            x_categoricals: order of categorical variables in tensor passed to forward function
            hidden_continuous_size: default for hidden size for processing continous variables (similar to categorical
                embedding size)
            hidden_continuous_sizes: dictionary mapping continuous input indices to sizes for variable selection
                (fallback to hidden_continuous_size if index is not in dictionary)
            embedding_sizes: dictionary mapping (string) indices to tuple of number of categorical classes and
                embedding size
            embedding_paddings: list of indices for embeddings which transform the zero's embedding to a zero vector
            embedding_labels: dictionary mapping (string) indices to list of categorical labels
            learning_rate: learning rate
            log_interval: log predictions every x batches, do not log if 0 or less, log interpretation if > 0. If < 1.0
                , will log multiple entries per batch. Defaults to -1.
            log_val_interval: frequency with which to log validation set metrics, defaults to log_interval
            log_gradient_flow: if to log gradient flow, this takes time and should be only done to diagnose training
                failures
            reduce_on_plateau_patience (int): patience after which learning rate is reduced by a factor of 10
            monotone_constaints (Dict[str, int]): dictionary of monotonicity constraints for continuous decoder
                variables mapping
                position (e.g. ``"0"`` for first position) to constraint (``-1`` for negative and ``+1`` for positive,
                larger numbers add more weight to the constraint vs. the loss but are usually not necessary).
                This constraint significantly slows down training. Defaults to {}.
            share_single_variable_networks (bool): if to share the single variable networks between the encoder and
                decoder. Defaults to False.
            causal_attention (bool): If to attend only at previous timesteps in the decoder or also include future
                predictions. Defaults to True.
            logging_metrics (nn.ModuleList[LightningMetric]): list of metrics that are logged during training.
                Defaults to nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE()]).
            **kwargs: additional arguments to :py:class:`~BaseModel`.
        """
        self.cumulative = cumulative
        self.joint_prediction = joint_prediction
        if logging_metrics is None:
            logging_metrics = nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE()])
        if loss is None:
            if self.cumulative:
                loss = CumQuantileLoss()
            else:
                loss = QuantileLoss()
        self.save_hyperparameters()
        # store loss function separately as it is a module
        assert isinstance(loss, LightningMetric), "Loss has to be a PyTorch Lightning `Metric`"
        super().__init__(loss=loss, logging_metrics=logging_metrics, **kwargs)

        # processing inputs
        # embeddings
        self.input_embeddings = MultiEmbedding(
            embedding_sizes=self.hparams.embedding_sizes,
            categorical_groups=self.hparams.categorical_groups,
            embedding_paddings=self.hparams.embedding_paddings,
            x_categoricals=self.hparams.x_categoricals,
            max_embedding_size=self.hparams.hidden_size,
        )

        # continuous variable processing
        self.prescalers = nn.ModuleDict(
            {
                name: nn.Linear(1, self.hparams.hidden_continuous_sizes.get(name, self.hparams.hidden_continuous_size))
                for name in self.reals
            }
        )

        # variable selection
        # variable selection for static variables
        static_input_sizes = {
            name: self.input_embeddings.output_size[name] for name in self.hparams.static_categoricals
        }
        static_input_sizes.update(
            {
                name: self.hparams.hidden_continuous_sizes.get(name, self.hparams.hidden_continuous_size)
                for name in self.hparams.static_reals
            }
        )
        self.static_variable_selection = VariableSelectionNetwork(
            input_sizes=static_input_sizes,
            hidden_size=self.hparams.hidden_size,
            input_embedding_flags={name: True for name in self.hparams.static_categoricals},
            dropout=self.hparams.dropout,
            prescalers=self.prescalers,
        )

        # variable selection for encoder and decoder
        encoder_input_sizes = {
            name: self.input_embeddings.output_size[name] for name in self.hparams.time_varying_categoricals_encoder
        }
        encoder_input_sizes.update(
            {
                name: self.hparams.hidden_continuous_sizes.get(name, self.hparams.hidden_continuous_size)
                for name in self.hparams.time_varying_reals_encoder
            }
        )

        decoder_input_sizes = {
            name: self.input_embeddings.output_size[name] for name in self.hparams.time_varying_categoricals_decoder
        }
        decoder_input_sizes.update(
            {
                name: self.hparams.hidden_continuous_sizes.get(name, self.hparams.hidden_continuous_size)
                for name in self.hparams.time_varying_reals_decoder
            }
        )

        # create single variable grns that are shared across decoder and encoder
        if self.hparams.share_single_variable_networks:
            self.shared_single_variable_grns = nn.ModuleDict()
            for name, input_size in encoder_input_sizes.items():
                self.shared_single_variable_grns[name] = GatedResidualNetwork(
                    input_size,
                    min(input_size, self.hparams.hidden_size),
                    self.hparams.hidden_size,
                    self.hparams.dropout,
                )
            for name, input_size in decoder_input_sizes.items():
                if name not in self.shared_single_variable_grns:
                    self.shared_single_variable_grns[name] = GatedResidualNetwork(
                        input_size,
                        min(input_size, self.hparams.hidden_size),
                        self.hparams.hidden_size,
                        self.hparams.dropout,
                    )

        self.encoder_variable_selection = VariableSelectionNetwork(
            input_sizes=encoder_input_sizes,
            hidden_size=self.hparams.hidden_size,
            input_embedding_flags={name: True for name in self.hparams.time_varying_categoricals_encoder},
            dropout=self.hparams.dropout,
            context_size=self.hparams.hidden_size,
            prescalers=self.prescalers,
            single_variable_grns={}
            if not self.hparams.share_single_variable_networks
            else self.shared_single_variable_grns,
        )

        self.decoder_variable_selection = VariableSelectionNetwork(
            input_sizes=decoder_input_sizes,
            hidden_size=self.hparams.hidden_size,
            input_embedding_flags={name: True for name in self.hparams.time_varying_categoricals_decoder},
            dropout=self.hparams.dropout,
            context_size=self.hparams.hidden_size,
            prescalers=self.prescalers,
            single_variable_grns={}
            if not self.hparams.share_single_variable_networks
            else self.shared_single_variable_grns,
        )

        # static encoders
        # for variable selection
        self.static_context_variable_selection = GatedResidualNetwork(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            output_size=self.hparams.hidden_size,
            dropout=self.hparams.dropout,
        )

        # for hidden state of the lstm
        self.static_context_initial_hidden_lstm = GatedResidualNetwork(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            output_size=self.hparams.hidden_size,
            dropout=self.hparams.dropout,
        )

        # for cell state of the lstm
        self.static_context_initial_cell_lstm = GatedResidualNetwork(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            output_size=self.hparams.hidden_size,
            dropout=self.hparams.dropout,
        )

        # for post lstm static enrichment
        self.static_context_enrichment = GatedResidualNetwork(
            self.hparams.hidden_size, self.hparams.hidden_size, self.hparams.hidden_size, self.hparams.dropout
        )

        # lstm encoder (history) and decoder (future) for local processing
        self.lstm_encoder = LSTM(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.lstm_layers,
            dropout=self.hparams.dropout if self.hparams.lstm_layers > 1 else 0,
            batch_first=True,
        )

        self.lstm_decoder = LSTM(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.lstm_layers,
            dropout=self.hparams.dropout if self.hparams.lstm_layers > 1 else 0,
            batch_first=True,
        )

        # skip connection for lstm
        self.post_lstm_gate_encoder = GatedLinearUnit(self.hparams.hidden_size, dropout=self.hparams.dropout)
        self.post_lstm_gate_decoder = self.post_lstm_gate_encoder
        # self.post_lstm_gate_decoder = GatedLinearUnit(self.hparams.hidden_size, dropout=self.hparams.dropout)
        self.post_lstm_add_norm_encoder = AddNorm(self.hparams.hidden_size, trainable_add=False)
        # self.post_lstm_add_norm_decoder = AddNorm(self.hparams.hidden_size, trainable_add=True)
        self.post_lstm_add_norm_decoder = self.post_lstm_add_norm_encoder

        # static enrichment and processing past LSTM
        self.static_enrichment = GatedResidualNetwork(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            output_size=self.hparams.hidden_size,
            dropout=self.hparams.dropout,
            context_size=self.hparams.hidden_size,
        )

        # attention for long-range processing
        self.multihead_attn = InterpretableMultiHeadAttention(
            d_model=self.hparams.hidden_size, n_head=self.hparams.attention_head_size, dropout=self.hparams.dropout
        )
        self.post_attn_gate_norm = GateAddNorm(
            self.hparams.hidden_size, dropout=self.hparams.dropout, trainable_add=False
        )
        self.pos_wise_ff = GatedResidualNetwork(
            self.hparams.hidden_size, self.hparams.hidden_size, self.hparams.hidden_size, dropout=self.hparams.dropout
        )

        # output processing -> no dropout at this late stage
        self.pre_output_gate_norm = GateAddNorm(self.hparams.hidden_size, dropout=None, trainable_add=False)

        if self.joint_prediction:
            print("Joint prediction")
            assert self.n_targets == 2, "Only two targets are supported (lat, lon)"
            assert self.hparams.output_size[0] == self.hparams.output_size[1], "Output sizes have to be the same"
            output_size = self.hparams.output_size[0]
            self.layer_xyz = nn.Linear(self.hparams.hidden_size, 3*output_size) # use tanh in forward
            self.forward_output = self._forward_joint_prediction
            # self.output_layer = nn.ModuleList(
            #     [nn.Linear(output_size, output_size) for target in range(self.n_targets)]
            # )
        else:
            print("Separate prediction")
            if self.n_targets > 1:  # if to run with multiple targets
                self.output_layer = nn.ModuleList(
                    [nn.Linear(self.hparams.hidden_size, output_size) for output_size in self.hparams.output_size]
                )
                self.forward_output = self._forward_separate_prediction_n_targets
            else:
                self.output_layer = nn.Linear(self.hparams.hidden_size, self.hparams.output_size)
                self.forward_output = self._forward_separate_prediction_1_target

    def _forward_joint_prediction(self, output):
        # to xyz
        output = self.layer_xyz(output)
        output = torch.tanh(output) # output in [-1, 1]
        # reshape expanding last dimension to have size 3: (x, y, z)
        output = output.reshape(output.size(0), output.size(1), -1, 3) # dim (N, T, Q, 3)
        # normalize to unit vector
        output = output / (torch.norm(output, dim=-1, keepdim=True) + 1e-8)
        # to lat, lon. Shapes (N, T, Q)
        output = [torch.asin(output[..., 2]),  # lat
                  torch.atan2(output[..., 1], output[..., 0])] # lon
        # output = [self.output_layer[0](output[0]),
        #           self.output_layer[1](output[1])]
        return output

    def _forward_separate_prediction_1_target(self, output):
        output = self.output_layer(output)
        return output

    def _forward_separate_prediction_n_targets(self, output):
        output = [output_layer(output) for output_layer in self.output_layer]
        return output

    @classmethod
    def from_dataset(
        cls,
        dataset: TimeSeriesDataSet,
        allowed_encoder_known_variable_names: List[str] = None,
        **kwargs,
    ):
        """
        Create model from dataset.

        Args:
            dataset: timeseries dataset
            allowed_encoder_known_variable_names: List of known variables that are allowed in encoder, defaults to all
            **kwargs: additional arguments such as hyperparameters for model (see ``__init__()``)

        Returns:
            TemporalFusionTransformer
        """
        # add maximum encoder length
        # update defaults
        new_kwargs = copy(kwargs)
        new_kwargs["max_encoder_length"] = dataset.max_encoder_length
        cumulative = kwargs.get("cumulative", False)
        if cumulative:
            loss = CumQuantileLoss()
        else:
            loss = QuantileLoss()
        new_kwargs.update(cls.deduce_default_output_parameters(dataset, kwargs, loss))

        # create class and return
        return super().from_dataset(
            dataset, allowed_encoder_known_variable_names=allowed_encoder_known_variable_names, **new_kwargs
        )

    def expand_static_context(self, context, timesteps):
        """
        add time dimension to static context
        """
        return context[:, None].expand(-1, timesteps, -1)

    def get_attention_mask(self, encoder_lengths: torch.LongTensor, decoder_lengths: torch.LongTensor):
        """
        Returns causal mask to apply for self-attention layer.
        """
        decoder_length = decoder_lengths.max()
        if self.hparams.causal_attention:
            # indices to which is attended
            attend_step = torch.arange(decoder_length, device=self.device)
            # indices for which is predicted
            predict_step = torch.arange(0, decoder_length, device=self.device)[:, None]
            # do not attend to steps to self or after prediction
            decoder_mask = (attend_step >= predict_step).unsqueeze(0).expand(encoder_lengths.size(0), -1, -1) # True -> 0 when computing attention
            # when predicting the token at position `j`, the model can only attend to tokens at positions `0` to `j-1`, inclusive.
        else:
            # there is value in attending to future forecasts if they are made with knowledge currently
            #   available
            #   one possibility is here to use a second attention layer for future attention (assuming different effects
            #   matter in the future than the past)
            #   or alternatively using the same layer but allowing forward attention - i.e. only
            #   masking out non-available data and self
            decoder_mask = create_mask(decoder_length, decoder_lengths).unsqueeze(1).expand(-1, decoder_length, -1)
        # NOTE: create_mask(size, lengths) Create boolean masks of shape len(lenghts) x size. An entry at (i, j) is True if lengths[i] > j.
        # in other words, it is true where there is data, and false where there is padding.

        # do not attend to steps where data is padded
        encoder_mask = create_mask(encoder_lengths.max(), encoder_lengths).unsqueeze(1).expand(-1, decoder_length, -1)
        # combine masks along attended time - first encoder and then decoder
        mask = torch.cat(
            (
                encoder_mask,
                decoder_mask,
            ),
            dim=2,
        )
        return mask

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        input dimensions: n_samples x time x variables
        """
        encoder_lengths = x["encoder_lengths"]
        decoder_lengths = x["decoder_lengths"]
        x_cat = torch.cat([x["encoder_cat"], x["decoder_cat"]], dim=1)  # concatenate in time dimension
        x_cont = torch.cat([x["encoder_cont"], x["decoder_cont"]], dim=1)  # concatenate in time dimension
        timesteps = x_cont.size(1)  # encode + decode length
        max_encoder_length = int(encoder_lengths.max())
        input_vectors = self.input_embeddings(x_cat)
        input_vectors.update(
            {
                name: x_cont[..., idx].unsqueeze(-1)
                for idx, name in enumerate(self.hparams.x_reals)
                if name in self.reals
            }
        )

        # Embedding and variable selection
        if len(self.static_variables) > 0:
            # static embeddings will be constant over entire batch
            static_embedding = {name: input_vectors[name][:, 0] for name in self.static_variables}
            static_embedding, static_variable_selection = self.static_variable_selection(static_embedding)
        else:
            static_embedding = torch.zeros(
                (x_cont.size(0), self.hparams.hidden_size), dtype=self.dtype, device=self.device
            )
            static_variable_selection = torch.zeros((x_cont.size(0), 0), dtype=self.dtype, device=self.device)

        static_context_variable_selection = self.expand_static_context(
            self.static_context_variable_selection(static_embedding), timesteps
        )

        embeddings_varying_encoder = {
            name: input_vectors[name][:, :max_encoder_length] for name in self.encoder_variables
        }
        embeddings_varying_encoder, encoder_sparse_weights = self.encoder_variable_selection(
            embeddings_varying_encoder,
            static_context_variable_selection[:, :max_encoder_length],
        )

        embeddings_varying_decoder = {
            name: input_vectors[name][:, max_encoder_length:] for name in self.decoder_variables  # select decoder
        }
        embeddings_varying_decoder, decoder_sparse_weights = self.decoder_variable_selection(
            embeddings_varying_decoder,
            static_context_variable_selection[:, max_encoder_length:],
        )

        # LSTM
        # calculate initial state
        input_hidden = self.static_context_initial_hidden_lstm(static_embedding).expand(
            self.hparams.lstm_layers, -1, -1
        )
        input_cell = self.static_context_initial_cell_lstm(static_embedding).expand(self.hparams.lstm_layers, -1, -1)

        # run local encoder
        encoder_output, (hidden, cell) = self.lstm_encoder(
            embeddings_varying_encoder, (input_hidden, input_cell), lengths=encoder_lengths, enforce_sorted=False
        )

        # run local decoder
        decoder_output, _ = self.lstm_decoder(
            embeddings_varying_decoder,
            (hidden, cell),
            lengths=decoder_lengths,
            enforce_sorted=False,
        )

        # skip connection over lstm
        lstm_output_encoder = self.post_lstm_gate_encoder(encoder_output)
        lstm_output_encoder = self.post_lstm_add_norm_encoder(lstm_output_encoder, embeddings_varying_encoder)

        lstm_output_decoder = self.post_lstm_gate_decoder(decoder_output)
        lstm_output_decoder = self.post_lstm_add_norm_decoder(lstm_output_decoder, embeddings_varying_decoder)

        lstm_output = torch.cat([lstm_output_encoder, lstm_output_decoder], dim=1)

        # static enrichment
        static_context_enrichment = self.static_context_enrichment(static_embedding)
        attn_input = self.static_enrichment(
            lstm_output, self.expand_static_context(static_context_enrichment, timesteps)
        )

        # Attention
        attn_output, attn_output_weights = self.multihead_attn(
            q=attn_input[:, max_encoder_length:],  # query only for predictions
            k=attn_input,
            v=attn_input,
            mask=self.get_attention_mask(encoder_lengths=encoder_lengths, decoder_lengths=decoder_lengths),
        )

        # skip connection over attention
        attn_output = self.post_attn_gate_norm(attn_output, attn_input[:, max_encoder_length:])

        output = self.pos_wise_ff(attn_output)

        # skip connection over temporal fusion decoder (not LSTM decoder despite the LSTM output contains
        # a skip from the variable selection network)
        output = self.pre_output_gate_norm(output, lstm_output[:, max_encoder_length:])

        output = self.forward_output(output)

        return self.to_network_output(
            prediction=self.transform_output(output, target_scale=x["target_scale"]),
            encoder_attention=attn_output_weights[..., :max_encoder_length],
            decoder_attention=attn_output_weights[..., max_encoder_length:],
            static_variables=static_variable_selection,
            encoder_variables=encoder_sparse_weights,
            decoder_variables=decoder_sparse_weights,
            decoder_lengths=decoder_lengths,
            encoder_lengths=encoder_lengths,
        )

    def on_fit_end(self):
        if self.log_interval > 0:
            self.log_embeddings()

    def create_log(self, x, y, out, batch_idx, **kwargs):
        log = super().create_log(x, y, out, batch_idx, **kwargs)
        if self.log_interval > 0:
            log["interpretation"] = self._log_interpretation(out)
        return log

    def _log_interpretation(self, out):
        # calculate interpretations etc for latter logging
        interpretation = self.interpret_output(
            detach(out),
            reduction="sum",
            attention_prediction_horizon=0,  # attention only for first prediction horizon
        )
        return interpretation

    def epoch_end(self, outputs):
        """
        run at epoch end for training or validation
        """
        if self.log_interval > 0 and not self.training:
            self.log_interpretation(outputs)

    def interpret_output(
        self,
        out: Dict[str, torch.Tensor],
        reduction: str = "none",
        attention_prediction_horizon: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        interpret output of model

        Args:
            out: output as produced by ``forward()``
            reduction: "none" for no averaging over batches, "sum" for summing attentions, "mean" for
                normalizing by encode lengths
            attention_prediction_horizon: which prediction horizon to use for attention

        Returns:
            interpretations that can be plotted with ``plot_interpretation()``
        """
        # take attention and concatenate if a list to proper attention object
        batch_size = len(out["decoder_attention"])
        if isinstance(out["decoder_attention"], (list, tuple)):
            # start with decoder attention
            # assume issue is in last dimension, we need to find max
            max_last_dimension = max(x.size(-1) for x in out["decoder_attention"])
            first_elm = out["decoder_attention"][0]
            # create new attention tensor into which we will scatter
            decoder_attention = torch.full(
                (batch_size, *first_elm.shape[:-1], max_last_dimension),
                float("nan"),
                dtype=first_elm.dtype,
                device=first_elm.device,
            )
            # scatter into tensor
            for idx, x in enumerate(out["decoder_attention"]):
                decoder_length = out["decoder_lengths"][idx]
                decoder_attention[idx, :, :, :decoder_length] = x[..., :decoder_length]
        else:
            decoder_attention = out["decoder_attention"]
            decoder_mask = create_mask(out["decoder_attention"].size(1), out["decoder_lengths"])
            decoder_attention[decoder_mask[..., None, None].expand_as(decoder_attention)] = float("nan")

        if isinstance(out["encoder_attention"], (list, tuple)):
            # same game for encoder attention
            # create new attention tensor into which we will scatter
            first_elm = out["encoder_attention"][0]
            encoder_attention = torch.full(
                (batch_size, *first_elm.shape[:-1], self.hparams.max_encoder_length),
                float("nan"),
                dtype=first_elm.dtype,
                device=first_elm.device,
            )
            # scatter into tensor
            for idx, x in enumerate(out["encoder_attention"]):
                encoder_length = out["encoder_lengths"][idx]
                encoder_attention[idx, :, :, self.hparams.max_encoder_length - encoder_length :] = x[
                    ..., :encoder_length
                ]
        else:
            # roll encoder attention (so start last encoder value is on the right)
            encoder_attention = out["encoder_attention"]
            shifts = encoder_attention.size(3) - out["encoder_lengths"]
            new_index = (
                torch.arange(encoder_attention.size(3), device=encoder_attention.device)[None, None, None].expand_as(
                    encoder_attention
                )
                - shifts[:, None, None, None]
            ) % encoder_attention.size(3)
            encoder_attention = torch.gather(encoder_attention, dim=3, index=new_index)
            # expand encoder_attentiont to full size
            if encoder_attention.size(-1) < self.hparams.max_encoder_length:
                encoder_attention = torch.concat(
                    [
                        torch.full(
                            (
                                *encoder_attention.shape[:-1],
                                self.hparams.max_encoder_length - out["encoder_lengths"].max(),
                            ),
                            float("nan"),
                            dtype=encoder_attention.dtype,
                            device=encoder_attention.device,
                        ),
                        encoder_attention,
                    ],
                    dim=-1,
                )

        # combine attention vector
        attention = torch.concat([encoder_attention, decoder_attention], dim=-1)
        attention[attention < 1e-5] = float("nan")

        # histogram of decode and encode lengths
        encoder_length_histogram = integer_histogram(out["encoder_lengths"], min=0, max=self.hparams.max_encoder_length)
        decoder_length_histogram = integer_histogram(
            out["decoder_lengths"], min=1, max=out["decoder_variables"].size(1)
        )

        # mask where decoder and encoder where not applied when averaging variable selection weights
        encoder_variables = out["encoder_variables"].squeeze(-2)
        encode_mask = create_mask(encoder_variables.size(1), out["encoder_lengths"])
        encoder_variables = encoder_variables.masked_fill(encode_mask.unsqueeze(-1), 0.0).sum(dim=1)
        encoder_variables /= (
            out["encoder_lengths"]
            .where(out["encoder_lengths"] > 0, torch.ones_like(out["encoder_lengths"]))
            .unsqueeze(-1)
        )

        decoder_variables = out["decoder_variables"].squeeze(-2)
        decode_mask = create_mask(decoder_variables.size(1), out["decoder_lengths"])
        decoder_variables = decoder_variables.masked_fill(decode_mask.unsqueeze(-1), 0.0).sum(dim=1)
        decoder_variables /= out["decoder_lengths"].unsqueeze(-1)

        # static variables need no masking
        static_variables = out["static_variables"].squeeze(1)
        # attention is batch x time x heads x time_to_attend
        # average over heads + only keep prediction attention and attention on observed timesteps
        attention = masked_op(
            attention[
                :, attention_prediction_horizon, :, : self.hparams.max_encoder_length + attention_prediction_horizon
            ],
            op="mean",
            dim=1,
        )

        if reduction != "none":  # if to average over batches
            static_variables = static_variables.sum(dim=0)
            encoder_variables = encoder_variables.sum(dim=0)
            decoder_variables = decoder_variables.sum(dim=0)

            attention = masked_op(attention, dim=0, op=reduction)
        else:
            attention = attention / masked_op(attention, dim=1, op="sum").unsqueeze(-1)  # renormalize

        interpretation = dict(
            attention=attention.masked_fill(torch.isnan(attention), 0.0),
            static_variables=static_variables,
            encoder_variables=encoder_variables,
            decoder_variables=decoder_variables,
            encoder_length_histogram=encoder_length_histogram,
            decoder_length_histogram=decoder_length_histogram,
        )
        return interpretation

    def plot_prediction(
        self,
        x: Dict[str, torch.Tensor],
        out: Dict[str, torch.Tensor],
        idx: int,
        plot_attention: bool = True,
        add_loss_to_title: bool = False,
        show_future_observed: bool = True,
        ax=None,
        **kwargs,
    ) -> plt.Figure:
        """
        Plot actuals vs prediction and attention

        Args:
            x (Dict[str, torch.Tensor]): network input
            out (Dict[str, torch.Tensor]): network output
            idx (int): sample index
            plot_attention: if to plot attention on secondary axis
            add_loss_to_title: if to add loss to title. Default to False.
            show_future_observed: if to show actuals for future. Defaults to True.
            ax: matplotlib axes to plot on

        Returns:
            plt.Figure: matplotlib figure
        """

        # plot prediction as normal
        fig = super().plot_prediction(
            x,
            out,
            idx=idx,
            add_loss_to_title=add_loss_to_title,
            show_future_observed=show_future_observed,
            ax=ax,
            **kwargs,
        )

        # add attention on secondary axis
        if plot_attention:
            interpretation = self.interpret_output(out.iget(slice(idx, idx + 1)))
            for f in to_list(fig):
                ax = f.axes[0]
                ax2 = ax.twinx()
                ax2.set_ylabel("Attention")
                encoder_length = x["encoder_lengths"][0]
                ax2.plot(
                    torch.arange(-encoder_length, 0),
                    interpretation["attention"][0, -encoder_length:].detach().cpu(),
                    alpha=0.2,
                    color="k",
                )
                f.tight_layout()
        return fig

    def plot_interpretation(self, interpretation: Dict[str, torch.Tensor]) -> Dict[str, plt.Figure]:
        """
        Make figures that interpret model.

        * Attention
        * Variable selection weights / importances

        Args:
            interpretation: as obtained from ``interpret_output()``

        Returns:
            dictionary of matplotlib figures
        """
        figs = {}

        # attention
        fig, ax = plt.subplots()
        attention = interpretation["attention"].detach().cpu()
        attention = attention / attention.sum(-1).unsqueeze(-1)
        ax.plot(
            np.arange(-self.hparams.max_encoder_length, attention.size(0) - self.hparams.max_encoder_length), attention
        )
        ax.set_xlabel("Time index")
        ax.set_ylabel("Attention")
        ax.set_title("Attention")
        figs["attention"] = fig

        # variable selection
        def make_selection_plot(title, values, labels):
            order = np.argsort(values)
            values = values / values.sum(-1).unsqueeze(-1)
            ax.barh(np.arange(len(values)), values[order] * 100, tick_label=np.asarray(labels)[order])
            ax.set_title(title)
            ax.set_xlabel("Importance in %")
            plt.tight_layout()
            return fig

        figs["static_variables"] = make_selection_plot(
            "Static variables importance", interpretation["static_variables"].detach().cpu(), self.static_variables
        )
        figs["encoder_variables"] = make_selection_plot(
            "Encoder variables importance", interpretation["encoder_variables"].detach().cpu(), self.encoder_variables
        )
        figs["decoder_variables"] = make_selection_plot(
            "Decoder variables importance", interpretation["decoder_variables"].detach().cpu(), self.decoder_variables
        )

        return figs

    def log_interpretation(self, outputs):
        """
        Log interpretation metrics to tensorboard.
        """
        # extract interpretations
        interpretation = {
            # use padded_stack because decoder length histogram can be of different length
            name: padded_stack([x["interpretation"][name].detach() for x in outputs], side="right", value=0).sum(0)
            for name in outputs[0]["interpretation"].keys()
        }
        # normalize attention with length histogram squared to account for: 1. zeros in attention and
        # 2. higher attention due to less values
        attention_occurances = interpretation["encoder_length_histogram"][1:].flip(0).cumsum(0).float()
        attention_occurances = attention_occurances / attention_occurances.max()
        attention_occurances = torch.cat(
            [
                attention_occurances,
                torch.ones(
                    interpretation["attention"].size(0) - attention_occurances.size(0),
                    dtype=attention_occurances.dtype,
                    device=attention_occurances.device,
                ),
            ],
            dim=0,
        )
        interpretation["attention"] = interpretation["attention"] / attention_occurances.pow(2).clamp(1.0)
        interpretation["attention"] = interpretation["attention"] / interpretation["attention"].sum()

        figs = self.plot_interpretation(interpretation)  # make interpretation figures
        label = self.current_stage
        # log to tensorboard
        for name, fig in figs.items():
            self.logger.experiment.add_figure(
                f"{label.capitalize()} {name} importance", fig, global_step=self.global_step
            )

        # log lengths of encoder/decoder
        for type in ["encoder", "decoder"]:
            fig, ax = plt.subplots()
            lengths = (
                padded_stack([out["interpretation"][f"{type}_length_histogram"] for out in outputs])
                .sum(0)
                .detach()
                .cpu()
            )
            if type == "decoder":
                start = 1
            else:
                start = 0
            ax.plot(torch.arange(start, start + len(lengths)), lengths)
            ax.set_xlabel(f"{type.capitalize()} length")
            ax.set_ylabel("Number of samples")
            ax.set_title(f"{type.capitalize()} length distribution in {label} epoch")

            self.logger.experiment.add_figure(
                f"{label.capitalize()} {type} length distribution", fig, global_step=self.global_step
            )

    def log_embeddings(self):
        """
        Log embeddings to tensorboard
        """
        for name, emb in self.input_embeddings.items():
            labels = self.hparams.embedding_labels[name]
            self.logger.experiment.add_embedding(
                emb.weight.data.detach().cpu(), metadata=labels, tag=name, global_step=self.global_step
            )

class TemporalFusionTransformerImputationOld(TemporalFusionTransformer):
    """
    Extend TemporalFusionTransformer to include future variables (imputation task)

    Does not take into account:
    - Difference in encoder and future lengths
    - Future attention
    """
    def __init__(self,
                 *args,
                 future_attention: bool = False,
                 idx_encoder_reals: List[int] = [],
                 idx_future_reals: List[int] = [],
                 idx_encoder_categoricals: List[int] = [],
                 idx_future_categoricals: List[int] = [],
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.future_attention = future_attention
        self.idx_encoder_reals = idx_encoder_reals
        self.idx_future_reals = idx_future_reals
        self.idx_encoder_categoricals = idx_encoder_categoricals
        self.idx_future_categoricals = idx_future_categoricals

    @staticmethod
    def dataset_specs(
        dataset: ImputationDataset,
        allowed_encoder_known_variable_names: List[str] = [],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create model from dataset and set parameters related to covariates.

        Returns: kwargs to pass to model.from_dataset
        """
        # assert fixed encoder and decoder length for the moment
        if not allowed_encoder_known_variable_names:
            allowed_encoder_known_variable_names = (
                dataset.time_varying_known_categoricals_with_future + dataset.time_varying_known_reals_with_future
            )

        # embeddings
        embedding_labels = {
            name: encoder.classes_
            for name, encoder in dataset.categorical_encoders_with_future.items()
            if name in dataset.categoricals_with_future
        }
        embedding_paddings = dataset.dropout_categoricals_with_future
        # determine embedding sizes based on heuristic
        embedding_sizes = {
            name: (len(encoder.classes_), get_embedding_size(len(encoder.classes_)))
            for name, encoder in dataset.categorical_encoders_with_future.items()
            if name in dataset.categoricals_with_future
        }
        embedding_sizes.update(kwargs.get("embedding_sizes", {}))
        kwargs.setdefault("embedding_sizes", embedding_sizes)

        new_kwargs = dict(
            static_categoricals=dataset.static_categoricals_with_future,
            time_varying_categoricals_encoder=[
                name for name in dataset.time_varying_known_categoricals_with_future if name in allowed_encoder_known_variable_names
            ]
            + dataset.time_varying_unknown_categoricals_with_future,
            time_varying_categoricals_decoder=dataset.time_varying_known_categoricals_with_future,
            static_reals=dataset.static_reals_with_future,
            time_varying_reals_encoder=[
                name for name in dataset.time_varying_known_reals_with_future if name in allowed_encoder_known_variable_names
            ]
            + dataset.time_varying_unknown_reals_with_future,
            time_varying_reals_decoder=dataset.time_varying_known_reals_with_future,
            x_reals=dataset.reals_with_future,
            x_categoricals=dataset.flat_categoricals_with_future,
            embedding_labels=embedding_labels,
            embedding_paddings=embedding_paddings,
            categorical_groups=dataset.variable_groups,
            idx_encoder_reals=dataset.idx_encoder_reals,
            idx_future_reals=dataset.idx_future_reals,
            idx_encoder_categoricals=dataset.idx_encoder_categoricals,
            idx_future_categoricals=dataset.idx_future_categoricals,
        )
        new_kwargs.update(kwargs)
        return new_kwargs

    @classmethod
    def from_dataset(
        cls,
        dataset: ImputationDataset,
        allowed_encoder_known_variable_names: List[str] = [],
        **kwargs,
    ):
        """
        Create model from dataset.

        Args:
            dataset: ImputationDataset
            allowed_encoder_known_variable_names: List of known variables that are allowed in encoder, defaults to all
            **kwargs: additional arguments such as hyperparameters for model (see ``__init__()``)

        Returns:
            TemporalFusionTransformer
        """
        assert isinstance(dataset, ImputationDataset), "dataset has to be ImputationDataset"
        # add maximum encoder length
        # update defaults
        new_kwargs = copy(kwargs)
        new_kwargs["max_encoder_length"] = dataset.max_encoder_length
        new_kwargs.update(cls.deduce_default_output_parameters(dataset, kwargs, QuantileLoss()))

        # create class and return using BaseModelWithCovariatesFuture's from_dataset
        specs = cls.dataset_specs(dataset, allowed_encoder_known_variable_names=allowed_encoder_known_variable_names)
        new_kwargs.update(specs)
        return super().from_dataset(
            dataset, allowed_encoder_known_variable_names=allowed_encoder_known_variable_names, **new_kwargs
        )

    def get_attention_mask(self,
                           encoder_lengths: torch.LongTensor,
                           decoder_lengths: torch.LongTensor,
                           future_lengths: torch.LongTensor):
        """
        Returns causal mask to apply for self-attention layer.
        """
        decoder_length = decoder_lengths.max()
        if self.hparams.causal_attention:
            # indices to which is attended
            attend_step = torch.arange(decoder_length, device=self.device)
            # indices for which is predicted
            predict_step = torch.arange(0, decoder_length, device=self.device)[:, None]
            # do not attend to steps to self or after prediction
            decoder_mask = (attend_step >= predict_step).unsqueeze(0).expand(encoder_lengths.size(0), -1, -1)
        else:
            # there is value in attending to future forecasts if they are made with knowledge currently
            #   available
            #   one possibility is here to use a second attention layer for future attention (assuming different effects
            #   matter in the future than the past)
            #   or alternatively using the same layer but allowing forward attention - i.e. only
            #   masking out non-available data and self
            decoder_mask = create_mask(decoder_length, decoder_lengths).unsqueeze(1).expand(-1, decoder_length, -1)
        # do not attend to steps where data is padded
        encoder_mask = create_mask(encoder_lengths.max(), encoder_lengths).unsqueeze(1).expand(-1, decoder_length, -1)
        future_mask = create_mask(future_lengths.max(), future_lengths).unsqueeze(1).expand(-1, decoder_length, -1)
        # combine masks along attended time - first encoder, then decoder and finally future
        mask = torch.cat(
            (
                encoder_mask,
                decoder_mask,
                future_mask,
            ),
            dim=2,
        )
        return mask

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        input dimensions: n_samples x time x variables

        Modified to account for future variables.
        """
        encoder_lengths = x["encoder_lengths"]
        decoder_lengths = x["decoder_lengths"]
        future_lengths = x["future_lengths"]

        x_cat = torch.cat([x["encoder_cat"], x["decoder_cat"]], dim=1)  # concatenate in time dimension
        x_cont = torch.cat([x["encoder_cont"], x["decoder_cont"]], dim=1)  # concatenate in time dimension
        timesteps = x_cont.size(1)  # encode + decode length
        max_encoder_length = int(encoder_lengths.max())
        input_vectors = self.input_embeddings(x_cat)
        input_vectors.update(
            {
                name: x_cont[..., idx].unsqueeze(-1)
                for idx, name in enumerate(self.hparams.x_reals)
                if name in self.reals
            }
        )

        # Embedding and variable selection
        if len(self.static_variables) > 0:
            # static embeddings will be constant over entire batch
            static_embedding = {name: input_vectors[name][:, 0] for name in self.static_variables}
            static_embedding, static_variable_selection = self.static_variable_selection(static_embedding)
        else:
            static_embedding = torch.zeros(
                (x_cont.size(0), self.hparams.hidden_size), dtype=self.dtype, device=self.device
            )
            static_variable_selection = torch.zeros((x_cont.size(0), 0), dtype=self.dtype, device=self.device)

        static_context_variable_selection = self.expand_static_context(
            self.static_context_variable_selection(static_embedding), timesteps
        )

        embeddings_varying_encoder = {
            name: input_vectors[name][:, :max_encoder_length] for name in self.encoder_variables
        }
        embeddings_varying_encoder, encoder_sparse_weights = self.encoder_variable_selection(
            embeddings_varying_encoder,
            static_context_variable_selection[:, :max_encoder_length],
        )

        embeddings_varying_decoder = {
            name: input_vectors[name][:, max_encoder_length:] for name in self.decoder_variables  # select decoder
        }
        embeddings_varying_decoder, decoder_sparse_weights = self.decoder_variable_selection(
            embeddings_varying_decoder,
            static_context_variable_selection[:, max_encoder_length:],
        )

        # LSTM
        # calculate initial state
        input_hidden = self.static_context_initial_hidden_lstm(static_embedding).expand(
            self.hparams.lstm_layers, -1, -1
        )
        input_cell = self.static_context_initial_cell_lstm(static_embedding).expand(self.hparams.lstm_layers, -1, -1)

        # run local encoder
        encoder_output, (hidden, cell) = self.lstm_encoder(
            embeddings_varying_encoder, (input_hidden, input_cell), lengths=encoder_lengths, enforce_sorted=False
        )

        # run local decoder
        decoder_output, _ = self.lstm_decoder(
            embeddings_varying_decoder,
            (hidden, cell),
            lengths=decoder_lengths,
            enforce_sorted=False,
        )

        # skip connection over lstm
        lstm_output_encoder = self.post_lstm_gate_encoder(encoder_output)
        lstm_output_encoder = self.post_lstm_add_norm_encoder(lstm_output_encoder, embeddings_varying_encoder)

        lstm_output_decoder = self.post_lstm_gate_decoder(decoder_output)
        lstm_output_decoder = self.post_lstm_add_norm_decoder(lstm_output_decoder, embeddings_varying_decoder)

        lstm_output = torch.cat([lstm_output_encoder, lstm_output_decoder], dim=1)

        # static enrichment
        static_context_enrichment = self.static_context_enrichment(static_embedding)
        attn_input = self.static_enrichment(
            lstm_output, self.expand_static_context(static_context_enrichment, timesteps)
        )

        # Attention
        if self.future_attention:
            attn_mask = self.get_attention_mask(encoder_lengths=encoder_lengths, decoder_lengths=decoder_lengths, future_lengths=future_lengths)
        else:
            attn_mask = super().get_attention_mask(encoder_lengths=encoder_lengths, decoder_lengths=decoder_lengths)
        attn_output, attn_output_weights = self.multihead_attn(
            q=attn_input[:, max_encoder_length:],  # query only for predictions
            k=attn_input,
            v=attn_input,
            mask=attn_mask,
        )

        # skip connection over attention
        attn_output = self.post_attn_gate_norm(attn_output, attn_input[:, max_encoder_length:])

        output = self.pos_wise_ff(attn_output)

        # skip connection over temporal fusion decoder (not LSTM decoder despite the LSTM output contains
        # a skip from the variable selection network)
        output = self.pre_output_gate_norm(output, lstm_output[:, max_encoder_length:])
        if self.n_targets > 1:  # if to use multi-target architecture
            output = [output_layer(output) for output_layer in self.output_layer]
        else:
            output = self.output_layer(output)

        return self.to_network_output(
            prediction=self.transform_output(output, target_scale=x["target_scale"]),
            encoder_attention=attn_output_weights[..., :max_encoder_length],
            decoder_attention=attn_output_weights[..., max_encoder_length:],
            static_variables=static_variable_selection,
            encoder_variables=encoder_sparse_weights,
            decoder_variables=decoder_sparse_weights,
            decoder_lengths=decoder_lengths,
            encoder_lengths=encoder_lengths,
            future_lengths=future_lengths,
        )

class TemporalFusionTransformerForecasting(TemporalFusionTransformer):
    """
    Extend TemporalFusionTransformer to include missing time steps
    """
    def __init__(self,
                 *args,
                 missing_values_zero_attention: bool = False,
                 causal_attention: bool = True,
                 monotonic_q: bool = False,
                 verbose: bool = True,
                 **kwargs):
        """
        causal_attention (bool): If to attend only at previous timesteps in the decoder or also include future predictions. Defaults to False (imputation can use information from the future).
        """
        super().__init__(*args, causal_attention=causal_attention, **kwargs)
        self.missing_values_zero_attention = missing_values_zero_attention
        self.monotonic_q = monotonic_q
        if monotonic_q:
            if verbose:
                print("Using monotonic quantile transformation")
            # extract quantiles
            if isinstance(self.loss, MultiLoss):
                self.quantiles = self.loss[0].quantiles
            else:
                self.quantiles = self.loss.quantiles
            self.quantile_transform = MonotonicQuantiles(quantiles=self.quantiles)

    @classmethod
    def from_dataset(
        cls,
        dataset: ForecastingDataset,
        allowed_encoder_known_variable_names: Union[None, List[str]] = None,
        **kwargs,
    ):
        """
        Create model from dataset.

        Args:
            dataset: ImputationDataset
            allowed_encoder_known_variable_names: List of known variables that are allowed in encoder, defaults to all
            **kwargs: additional arguments such as hyperparameters for model (see ``__init__()``)

        Returns:
            TemporalFusionTransformer
        """
        assert isinstance(dataset, ForecastingDataset), "dataset has to be ForecastingDataset"
        # add maximum encoder length
        # update defaults
        new_kwargs = copy(kwargs)
        new_kwargs["max_encoder_length"] = dataset.max_encoder_length
        new_kwargs.update(cls.deduce_default_output_parameters(dataset, kwargs, QuantileLoss()))

        # create class and return using BaseModelWithCovariatesFuture's from_dataset
        return super().from_dataset(
            dataset,
            allowed_encoder_known_variable_names=allowed_encoder_known_variable_names, **new_kwargs,
            missing_values_zero_attention=dataset.store_missing_idxs,
        )

    def get_attention_mask(self,
                           max_encoder_length: int,
                           max_decoder_length: int,
                           encoder_lengths: torch.LongTensor,
                           decoder_lengths: torch.LongTensor,
                           encoder_missing: Union[torch.Tensor, None] = None,
                           ) -> torch.Tensor:
        """
        Returns causal mask to apply for self-attention layer.
        """
        if self.hparams.causal_attention:
            # indices to which is attended
            attend_step = torch.arange(max_decoder_length, device=self.device)
            # indices for which is predicted
            predict_step = torch.arange(0, max_decoder_length, device=self.device)[:, None]
            # do not attend to steps to self or after prediction
            decoder_mask = (attend_step >= predict_step).unsqueeze(0).expand(encoder_lengths.size(0), -1, -1)
        else:
            # there is value in attending to future forecasts if they are made with knowledge currently
            #   available
            #   one possibility is here to use a second attention layer for future attention (assuming different effects
            #   matter in the future than the past)
            #   or alternatively using the same layer but allowing forward attention - i.e. only
            #   masking out non-available data and self
            decoder_mask = create_mask(max_decoder_length, decoder_lengths).unsqueeze(1).expand(-1, max_decoder_length, -1)
        # do not attend to steps where data is padded
        encoder_mask = create_mask(max_encoder_length, encoder_lengths).unsqueeze(1).expand(-1, max_decoder_length, -1)
        # combine masks along attended time - first encoder and then decoder
        if encoder_missing is not None:
            encoder_missing = encoder_missing.unsqueeze(1).expand(-1, max_decoder_length, -1) # do not attend to missing values
            encoder_mask = encoder_mask | encoder_missing

        mask = torch.cat(
            (
                encoder_mask,
                decoder_mask,
            ),
            dim=2,
        )
        return mask

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        input dimensions: n_samples x time x variables
        Modified to include missing values
        """
        encoder_lengths = x["encoder_lengths"]
        decoder_lengths = x["decoder_lengths"]

        encoder_missing = x["encoder_missing"]

        x_cat = torch.cat([x["encoder_cat"], x["decoder_cat"]], dim=1)  # concatenate in time dimension
        x_cont = torch.cat([x["encoder_cont"], x["decoder_cont"]], dim=1)  # concatenate in time dimension
        timesteps = x_cont.size(1)  # encode + decode length
        max_encoder_length = x['max_encoder_length']
        max_decoder_length = x['max_decoder_length']
        input_vectors = self.input_embeddings(x_cat)
        input_vectors.update(
            {
                name: x_cont[..., idx].unsqueeze(-1)
                for idx, name in enumerate(self.hparams.x_reals)
                if name in self.reals
            }
        )

        # Embedding and variable selection
        if len(self.static_variables) > 0:
            # static embeddings will be constant over entire batch
            static_embedding = {name: input_vectors[name][:, 0] for name in self.static_variables}
            static_embedding, static_variable_selection = self.static_variable_selection(static_embedding)
        else:
            static_embedding = torch.zeros(
                (x_cont.size(0), self.hparams.hidden_size), dtype=self.dtype, device=self.device
            )
            static_variable_selection = torch.zeros((x_cont.size(0), 0), dtype=self.dtype, device=self.device)

        static_context_variable_selection = self.expand_static_context(
            self.static_context_variable_selection(static_embedding), timesteps
        )

        embeddings_varying_encoder = {
            name: input_vectors[name][:, :max_encoder_length] for name in self.encoder_variables
        }
        embeddings_varying_encoder, encoder_sparse_weights = self.encoder_variable_selection(
            embeddings_varying_encoder,
            static_context_variable_selection[:, :max_encoder_length],
        )

        embeddings_varying_decoder = {
            name: input_vectors[name][:, max_encoder_length:] for name in self.decoder_variables  # select decoder
        }
        embeddings_varying_decoder, decoder_sparse_weights = self.decoder_variable_selection(
            embeddings_varying_decoder,
            static_context_variable_selection[:, max_encoder_length:],
        )

        # LSTM
        # calculate initial state
        input_hidden = self.static_context_initial_hidden_lstm(static_embedding).expand(
            self.hparams.lstm_layers, -1, -1
        )
        input_cell = self.static_context_initial_cell_lstm(static_embedding).expand(self.hparams.lstm_layers, -1, -1)

        # run local encoder
        encoder_output, (hidden, cell) = self.lstm_encoder(
            embeddings_varying_encoder, (input_hidden, input_cell), lengths=encoder_lengths, enforce_sorted=False
        )

        # run local decoder
        decoder_output, _ = self.lstm_decoder(
            embeddings_varying_decoder,
            (hidden, cell),
            lengths=decoder_lengths,
            enforce_sorted=False,
        )

        # skip connection over lstm
        lstm_output_encoder = self.post_lstm_gate_encoder(encoder_output)
        lstm_output_encoder = self.post_lstm_add_norm_encoder(lstm_output_encoder, embeddings_varying_encoder)

        lstm_output_decoder = self.post_lstm_gate_decoder(decoder_output)
        lstm_output_decoder = self.post_lstm_add_norm_decoder(lstm_output_decoder, embeddings_varying_decoder)

        lstm_output = torch.cat([lstm_output_encoder, lstm_output_decoder], dim=1)

        # static enrichment
        static_context_enrichment = self.static_context_enrichment(static_embedding)
        attn_input = self.static_enrichment(
            lstm_output, self.expand_static_context(static_context_enrichment, timesteps)
        )

        # Attention
        attn_output, attn_output_weights = self.multihead_attn(
            q=attn_input[:, max_encoder_length:],  # query only for predictions
            k=attn_input,
            v=attn_input,
            mask=self.get_attention_mask(encoder_lengths=encoder_lengths, decoder_lengths=decoder_lengths,
                                         max_encoder_length=max_encoder_length, max_decoder_length=max_decoder_length,
                                         encoder_missing=encoder_missing),
        )

        # skip connection over attention
        attn_output = self.post_attn_gate_norm(attn_output, attn_input[:, max_encoder_length:])

        output = self.pos_wise_ff(attn_output)

        # skip connection over temporal fusion decoder (not LSTM decoder despite the LSTM output contains
        # a skip from the variable selection network)
        output = self.pre_output_gate_norm(output, lstm_output[:, max_encoder_length:])
        if self.n_targets > 1:  # if to use multi-target architecture
            output = [output_layer(output) for output_layer in self.output_layer]
            if self.monotonic_q:
                output = [self.quantile_transform(out) for out in output]
        else:
            output = self.output_layer(output)
            if self.monotonic_q:
                output = self.quantile_transform(output)

        if encoder_missing is None:
            missing_values_kwargs = {}
        else:
            missing_values_kwargs = dict(encoder_missing=encoder_missing)

        return self.to_network_output(
            prediction=self.transform_output(output, target_scale=x["target_scale"]),
            encoder_attention=attn_output_weights[..., :max_encoder_length],
            decoder_attention=attn_output_weights[..., max_encoder_length:],
            static_variables=static_variable_selection,
            encoder_variables=encoder_sparse_weights,
            decoder_variables=decoder_sparse_weights,
            decoder_lengths=decoder_lengths,
            encoder_lengths=encoder_lengths,
            **missing_values_kwargs
        )

    def step(
        self, x: Dict[str, torch.Tensor], y: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, **kwargs
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Run for each train/val step.

        Args:
            x (Dict[str, torch.Tensor]): x as passed to the network by the dataloader
            y (Tuple[torch.Tensor, torch.Tensor]): y as passed to the loss function by the dataloader
            batch_idx (int): batch number
            **kwargs: additional arguments to pass to the network apart from ``x``

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]: tuple where the first
                entry is a dictionary to which additional logging results can be added for consumption in the
                ``epoch_end`` hook and the second entry is the model's output.
        """
        # pack y sequence if different encoder lengths exist
        if (x["decoder_lengths"] < x["decoder_lengths"].max()).any():
            if isinstance(y[0], (list, tuple)):
                y = (
                    [
                        rnn.pack_padded_sequence(
                            y_part, lengths=x["decoder_lengths"].cpu(), batch_first=True, enforce_sorted=False
                        )
                        for y_part in y[0]
                    ],
                    y[1],
                )
            else:
                y = (
                    rnn.pack_padded_sequence(
                        y[0], lengths=x["decoder_lengths"].cpu(), batch_first=True, enforce_sorted=False
                    ),
                    y[1],
                )

        if self.training and len(self.hparams.monotone_constaints) > 0:
            # calculate gradient with respect to continous decoder features
            x["decoder_cont"].requires_grad_(True)
            assert not torch._C._get_cudnn_enabled(), (
                "To use monotone constraints, wrap model and training in context "
                "`torch.backends.cudnn.flags(enable=False)`"
            )
            out = self(x, **kwargs)
            prediction = out["prediction"]

            # handle multiple targets
            prediction_list = to_list(prediction)
            gradient = 0
            # todo: should monotone constrains be applicable to certain targets?
            for pred in prediction_list:
                gradient = (
                    gradient
                    + torch.autograd.grad(
                        outputs=pred,
                        inputs=x["decoder_cont"],
                        grad_outputs=torch.ones_like(pred),  # t
                        create_graph=True,  # allows usage in graph
                        allow_unused=True,
                    )[0]
                )

            # select relevant features
            indices = torch.tensor(
                [self.hparams.x_reals.index(name) for name in self.hparams.monotone_constaints.keys()]
            )
            monotonicity = torch.tensor(
                [val for val in self.hparams.monotone_constaints.values()], dtype=gradient.dtype, device=gradient.device
            )
            # add additionl loss if gradient points in wrong direction
            gradient = gradient[..., indices] * monotonicity[None, None]
            monotinicity_loss = gradient.clamp_max(0).mean()
            # multiply monotinicity loss by large number to ensure relevance and take to the power of 2
            # for smoothness of loss function
            monotinicity_loss = 10 * torch.pow(monotinicity_loss, 2)
            if isinstance(self.loss, (MASE, MultiLoss)):
                loss = self.loss(
                    prediction, y, encoder_target=x["encoder_target"], encoder_lengths=x["encoder_lengths"]
                )
            else:
                loss = self.loss(prediction, y)

            loss = loss * (1 + monotinicity_loss)
        else:
            out = self(x, **kwargs)

            # calculate loss
            prediction = out["prediction"]
            if isinstance(self.loss, (MASE, MultiLoss)):
                if isinstance(self.loss, MASE):
                    loss_kwargs = dict(encoder_target=x["encoder_target"], encoder_lengths=x["encoder_lengths"])
                elif isinstance(self.loss[0], QuantileLossZeroAttention):
                    loss_kwargs = dict(decoder_missing=x["decoder_missing"])
                else:
                    loss_kwargs = {}
                loss = self.loss(prediction, y, **loss_kwargs)
            else:
                loss = self.loss(prediction, y)

        self.log(
            f"{self.current_stage}_loss",
            loss,
            on_step=self.training,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(x["decoder_target"]),
        )
        log = {"loss": loss, "n_samples": x["decoder_lengths"].size(0)}
        return log, out

# DISCARDED. Time alignment prior to masking
# class ScaledDotProductAttentionBidirectional(nn.Module):
#     def __init__(self, dropout: float = None, scale: bool = True, d_model: int = 31):
#         super(ScaledDotProductAttention, self).__init__()
#         if dropout is not None:
#             self.dropout = nn.Dropout(p=dropout)
#         else:
#             self.dropout = dropout
#         self.softmax = nn.Softmax(dim=2)
#         self.scale = scale
#         self.d_model = d_model
#         self.d_model_unidir = d_model // 2

#     def forward(self, q, k, v, mask=None):
#         attn = torch.bmm(q, k.permute(0, 2, 1))  # query-key overlap

#         if self.scale:
#             dimension = torch.as_tensor(k.size(-1), dtype=attn.dtype, device=attn.device).sqrt()
#             attn = attn / dimension

#         if mask is not None:
#             # bidirectional
#             mask_rev = torch.flip(mask, [2])
#             if isinstance(mask, tuple):
#                 attn[..., :self.d_model_unidir] = attn[..., :self.d_model_unidir].masked_fill(mask, -1e9)
#                 attn[..., self.d_model_unidir:] = attn[..., self.d_model_unidir:].masked_fill(mask_rev, -1e9)
#             else:
#                 attn = attn.masked_fill(mask, -1e9)
#         attn = self.softmax(attn)

#         if self.dropout is not None:
#             attn = self.dropout(attn)
#         output = torch.bmm(attn, v)
#         return output, attn
# class InterpretableMultiHeadAttentionBidirectional(InterpretableMultiHeadAttention):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.attention = ScaledDotProductAttentionBidirectional(d_model=self.d_model)

class TemporalFusionTransformerImputation(TemporalFusionTransformer):
    """
    Extend TemporalFusionTransformer to include future variables (imputation task)
    """
    def __init__(self,
                 *args,
                 future_attention: bool = False,
                 idx_encoder_reals: List[int] = [],
                 idx_future_reals: List[int] = [],
                 idx_encoder_categoricals: List[int] = [],
                 idx_future_categoricals: List[int] = [],
                 idx_decoder_reals: List[int] = [],
                 idx_decoder_categoricals: List[int] = [],
                 missing_values_zero_attention: bool = False,
                 causal_attention: bool = False,
                 reverse_future: bool = False,
                 hidden_size: int = 31,
                 hidden_size_encoder: Union[int, None] = None,
                 monotonic_q: bool = False,
                 verbose: bool = True,
                 **kwargs):
        """
        causal_attention (bool): If to attend only at previous timesteps in the decoder or also include future predictions. Defaults to False (imputation can use information from the future).
        """
        if reverse_future:
            # ensure hidden_size is even
            if hidden_size % 2 != 0:
                hidden_size += 1
                warnings.warn("Hidden size has to be even for bidirectional LSTM, increasing hidden size by 1")
        super().__init__(*args, causal_attention=causal_attention, hidden_size=hidden_size, **kwargs)

        self.future_attention = future_attention
        self.idx_encoder_reals = idx_encoder_reals
        self.idx_future_reals = idx_future_reals
        self.idx_encoder_categoricals = idx_encoder_categoricals
        self.idx_future_categoricals = idx_future_categoricals
        self.idx_decoder_reals = idx_decoder_reals
        self.idx_decoder_categoricals = idx_decoder_categoricals
        self.missing_values_zero_attention = missing_values_zero_attention
        self.reverse_future = reverse_future

        encoder_only_variables = []
        future_only_variables = []
        for f in self.encoder_variables:
            if f.startswith('future_'):
                future_only_variables.append(f)
            else:
                encoder_only_variables.append(f)
        self.encoder_only_variables = encoder_only_variables
        self.future_only_variables = future_only_variables

        encoder_input_sizes = {
            name: self.input_embeddings.output_size[name] for name in self.hparams.time_varying_categoricals_encoder if not name.startswith('future_')
        }
        encoder_input_sizes.update(
            {
                name: self.hparams.hidden_continuous_sizes.get(name, self.hparams.hidden_continuous_size)
                for name in self.hparams.time_varying_reals_encoder if not name.startswith('future_')
            }
        )

        future_input_sizes = {
            name: self.input_embeddings.output_size[name] for name in self.hparams.time_varying_categoricals_encoder if name.startswith('future_')
        }
        future_input_sizes.update(
            {
                name: self.hparams.hidden_continuous_sizes.get(name, self.hparams.hidden_continuous_size)
                for name in self.hparams.time_varying_reals_encoder if name.startswith('future_')
            }
        )

        if self.reverse_future:
            if verbose:
                print("Training LSTM layers bidirectionally")
            self.hparams.hidden_size_encoder = self.hparams.hidden_size // 2

            decoder_input_sizes = {
                name: self.input_embeddings.output_size[name] for name in self.hparams.time_varying_categoricals_decoder
            }
            decoder_input_sizes.update(
                {
                    name: self.hparams.hidden_continuous_sizes.get(name, self.hparams.hidden_continuous_size)
                    for name in self.hparams.time_varying_reals_decoder
                }
            )

            self.decoder_variable_selection = VariableSelectionNetwork(
                input_sizes=decoder_input_sizes,
                hidden_size=self.hparams.hidden_size_encoder,
                input_embedding_flags={name: True for name in self.hparams.time_varying_categoricals_decoder},
                dropout=self.hparams.dropout,
                context_size=self.hparams.hidden_size,
                prescalers=self.prescalers,
                single_variable_grns={}
                if not self.hparams.share_single_variable_networks
                else self.shared_single_variable_grns,
            )

            # for hidden state of the lstm
            self.static_context_initial_hidden_lstm = GatedResidualNetwork(
                input_size=self.hparams.hidden_size,
                hidden_size=self.hparams.hidden_size_encoder,
                output_size=self.hparams.hidden_size_encoder,
                dropout=self.hparams.dropout,
            )
            # for cell state of the lstm
            self.static_context_initial_cell_lstm = GatedResidualNetwork(
                input_size=self.hparams.hidden_size,
                hidden_size=self.hparams.hidden_size_encoder,
                output_size=self.hparams.hidden_size_encoder,
                dropout=self.hparams.dropout,
            )

            # lstm encoder (history) and decoder (future) for local processing
            self.lstm_encoder = LSTM(
                input_size=self.hparams.hidden_size_encoder,
                hidden_size=self.hparams.hidden_size_encoder,
                num_layers=self.hparams.lstm_layers,
                dropout=self.hparams.dropout if self.hparams.lstm_layers > 1 else 0,
                batch_first=True,
            )

            self.lstm_decoder = LSTM(
                input_size=self.hparams.hidden_size_encoder,
                hidden_size=self.hparams.hidden_size_encoder,
                num_layers=self.hparams.lstm_layers,
                dropout=self.hparams.dropout if self.hparams.lstm_layers > 1 else 0,
                batch_first=True,
            )
            # skip connection for lstm
            self.post_lstm_gate_encoder = GatedLinearUnit(self.hparams.hidden_size_encoder, dropout=self.hparams.dropout)
            self.post_lstm_gate_decoder = self.post_lstm_gate_encoder
            self.post_lstm_add_norm_encoder = AddNorm(self.hparams.hidden_size_encoder, trainable_add=False)
            self.post_lstm_add_norm_decoder = self.post_lstm_add_norm_encoder
        else:
            self.hparams.hidden_size_encoder = self.hparams.hidden_size


        self.encoder_variable_selection = VariableSelectionNetwork(
            input_sizes=encoder_input_sizes,
            hidden_size=self.hparams.hidden_size_encoder,
            input_embedding_flags={name: True for name in self.hparams.time_varying_categoricals_encoder if not name.startswith('future_')},
            dropout=self.hparams.dropout,
            context_size=self.hparams.hidden_size,
            prescalers=self.prescalers,
            single_variable_grns={}
            if not self.hparams.share_single_variable_networks
            else self.shared_single_variable_grns,
        )

        self.future_variable_selection = VariableSelectionNetwork(
            input_sizes=future_input_sizes,
            hidden_size=self.hparams.hidden_size_encoder,
            input_embedding_flags={name: True for name in self.hparams.time_varying_categoricals_encoder if name.startswith('future_')},
            dropout=self.hparams.dropout,
            context_size=self.hparams.hidden_size,
            prescalers=self.prescalers,
            single_variable_grns={}
            if not self.hparams.share_single_variable_networks
            else self.shared_single_variable_grns,
        )

        # lstm future for local processing
        self.lstm_future = LSTM(
            input_size=self.hparams.hidden_size_encoder,
            hidden_size=self.hparams.hidden_size_encoder,
            num_layers=self.hparams.lstm_layers,
            dropout=self.hparams.dropout if self.hparams.lstm_layers > 1 else 0,
            batch_first=True,
        )

        # skip connection for lstm
        self.post_lstm_gate_future = self.post_lstm_gate_encoder
        # self.post_lstm_gate_future = GatedLinearUnit(self.hparams.hidden_size, dropout=self.hparams.dropout)
        self.post_lstm_add_norm_future = self.post_lstm_add_norm_encoder
        # self.post_lstm_add_norm_future = AddNorm(self.hparams.hidden_size, trainable_add=True)

        self.monotonic_q = monotonic_q
        if monotonic_q:
            if verbose:
                print("Using monotonic quantile transformation")
            # extract quantiles
            if isinstance(self.loss, MultiLoss):
                self.quantiles = self.loss[0].quantiles
            else:
                self.quantiles = self.loss.quantiles
            self.quantile_transform = MonotonicQuantiles(quantiles=self.quantiles)


    @staticmethod
    def dataset_specs(
        dataset: ImputationDataset,
        allowed_encoder_known_variable_names: List[str] = [],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create model from dataset and set parameters related to covariates.

        Returns: kwargs to pass to model.from_dataset
        """
        # assert fixed encoder and decoder length for the moment
        if not allowed_encoder_known_variable_names:
            allowed_encoder_known_variable_names = (
                dataset.time_varying_known_categoricals_with_future + dataset.time_varying_known_reals_with_future
            )

        # embeddings
        embedding_labels = {
            name: encoder.classes_
            for name, encoder in dataset.categorical_encoders_with_future.items()
            if name in dataset.categoricals_with_future
        }
        embedding_paddings = dataset.dropout_categoricals_with_future
        # determine embedding sizes based on heuristic
        embedding_sizes = {
            name: (len(encoder.classes_), get_embedding_size(len(encoder.classes_)))
            for name, encoder in dataset.categorical_encoders_with_future.items()
            if name in dataset.categoricals_with_future
        }
        embedding_sizes.update(kwargs.get("embedding_sizes", {}))
        kwargs.setdefault("embedding_sizes", embedding_sizes)

        new_kwargs = dict(
            static_categoricals=dataset.static_categoricals_with_future,
            time_varying_categoricals_encoder=[
                name for name in dataset.time_varying_known_categoricals_with_future if name in allowed_encoder_known_variable_names
            ]
            + dataset.time_varying_unknown_categoricals_with_future,
            time_varying_categoricals_decoder=dataset.time_varying_known_categoricals,
            static_reals=dataset.static_reals_with_future,
            time_varying_reals_encoder=[
                name for name in dataset.time_varying_known_reals_with_future if name in allowed_encoder_known_variable_names
            ]
            + dataset.time_varying_unknown_reals_with_future,
            time_varying_reals_decoder=dataset.time_varying_known_reals,
            x_reals=dataset.reals_with_future,
            x_categoricals=dataset.flat_categoricals_with_future,
            embedding_labels=embedding_labels,
            embedding_paddings=embedding_paddings,
            categorical_groups=dataset.variable_groups,
            idx_encoder_reals=dataset.idx_encoder_reals,
            idx_future_reals=dataset.idx_future_reals,
            idx_encoder_categoricals=dataset.idx_encoder_categoricals,
            idx_future_categoricals=dataset.idx_future_categoricals,
            idx_decoder_reals=dataset.idx_decoder_reals,
            idx_decoder_categoricals=dataset.idx_decoder_categoricals,
            missing_values_zero_attention=dataset.store_missing_idxs,
            reverse_future=dataset.reverse_future,
        )
        new_kwargs.update(kwargs)
        return new_kwargs

    @classmethod
    def from_dataset(
        cls,
        dataset: ImputationDataset,
        allowed_encoder_known_variable_names: List[str] = [],
        **kwargs,
    ):
        """
        Create model from dataset.

        Args:
            dataset: ImputationDataset
            allowed_encoder_known_variable_names: List of known variables that are allowed in encoder, defaults to all
            **kwargs: additional arguments such as hyperparameters for model (see ``__init__()``)

        Returns:
            TemporalFusionTransformer
        """
        assert isinstance(dataset, ImputationDataset), "dataset has to be ImputationDataset"
        # add maximum encoder length
        # update defaults
        new_kwargs = copy(kwargs)
        new_kwargs["max_encoder_length"] = dataset.max_encoder_length
        new_kwargs.update(cls.deduce_default_output_parameters(dataset, kwargs, QuantileLoss()))

        # create class and return using BaseModelWithCovariatesFuture's from_dataset
        specs = cls.dataset_specs(dataset, allowed_encoder_known_variable_names=allowed_encoder_known_variable_names)
        new_kwargs.update(specs)
        return super().from_dataset(
            dataset, allowed_encoder_known_variable_names=allowed_encoder_known_variable_names, **new_kwargs
        )

    def get_attention_mask(self,
                           max_encoder_length: int,
                           max_decoder_length: int,
                           max_future_length: int,
                           encoder_lengths: torch.LongTensor,
                           decoder_lengths: torch.LongTensor,
                           future_lengths: torch.LongTensor,
                           encoder_missing: Union[torch.Tensor, None] = None,
                           future_missing: Union[torch.Tensor, None] = None,
                           ) -> torch.Tensor:
        """
        Returns causal mask to apply for self-attention layer.
        """
        if self.hparams.causal_attention:
            # indices to which is attended
            attend_step = torch.arange(max_decoder_length, device=self.device)
            # indices for which is predicted
            predict_step = torch.arange(0, max_decoder_length, device=self.device)[:, None]
            # do not attend to steps to self or after prediction
            decoder_mask = (attend_step >= predict_step).unsqueeze(0).expand(encoder_lengths.size(0), -1, -1)
        else:
            # there is value in attending to future forecasts if they are made with knowledge currently
            #   available
            #   one possibility is here to use a second attention layer for future attention (assuming different effects
            #   matter in the future than the past)
            #   or alternatively using the same layer but allowing forward attention - i.e. only
            #   masking out non-available data and self
            decoder_mask = create_mask(max_decoder_length, decoder_lengths).unsqueeze(1).expand(-1, max_decoder_length, -1)
        # do not attend to steps where data is padded
        encoder_mask = create_mask(max_encoder_length, encoder_lengths).unsqueeze(1).expand(-1, max_decoder_length, -1)
        future_mask = create_mask(max_future_length, future_lengths).unsqueeze(1).expand(-1, max_decoder_length, -1)
        # combine masks along attended time - first encoder and then decoder
        if encoder_missing is not None:
            encoder_missing = encoder_missing.unsqueeze(1).expand(-1, max_decoder_length, -1) # do not attend to missing values
            encoder_mask = encoder_mask | encoder_missing
        if future_missing is not None:
            future_missing = future_missing.unsqueeze(1).expand(-1, max_decoder_length, -1) # do not attend to missing values
            future_mask = future_mask | future_missing

        mask = torch.cat(
            (
                encoder_mask,
                decoder_mask,
                future_mask,
            ),
            dim=2,
        )
        return mask

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        input dimensions: n_samples x time x variables

        Modified to account for future variables.
        """
        encoder_lengths = x["encoder_lengths"]
        decoder_lengths = x["decoder_lengths"]
        future_lengths = x["future_lengths"]

        encoder_missing = x["encoder_missing"]
        future_missing = x["future_missing"]

        x_cat = torch.cat([x["encoder_cat"], x["decoder_cat"]], dim=1)  # concatenate in time dimension
        x_cont = torch.cat([x["encoder_cont"], x["decoder_cont"]], dim=1)  # concatenate in time dimension
        # max_encoder_length = int(encoder_future_max_lengths.max().item())
        # Possible length diagrams: if future > encoder
        # __encoder_fill_decoder__
        # _______future___________
        # If encoder > future
        # __encoder__decoder__
        #__future_fill________
        # In both cases decoder occupies the last max_decoder_length points
        max_encoder_length = x['max_encoder_length']
        max_decoder_length = x['max_decoder_length']
        max_future_length = x['max_future_length']

        end_decoder_step = max_encoder_length + max_decoder_length  # encode + decode length
        timesteps = end_decoder_step + max_future_length # encode + decode + future length
        input_vectors = self.input_embeddings(x_cat)
        input_vectors.update(
            {
                name: x_cont[..., idx].unsqueeze(-1)
                for idx, name in enumerate(self.hparams.x_reals)
                if name in self.reals
            }
        )

        # Embedding and variable selection
        if len(self.static_variables) > 0:
            # static embeddings will be constant over entire batch
            static_embedding = {name: input_vectors[name][:, 0] for name in self.static_variables}
            static_embedding, static_variable_selection = self.static_variable_selection(static_embedding)
        else:
            static_embedding = torch.zeros(
                (x_cont.size(0), self.hparams.hidden_size), dtype=self.dtype, device=self.device
            )
            static_variable_selection = torch.zeros((x_cont.size(0), 0), dtype=self.dtype, device=self.device)

        static_context_variable_selection = self.expand_static_context(
            self.static_context_variable_selection(static_embedding), timesteps
        )

        embeddings_varying_encoder = {
            name: input_vectors[name][:, :max_encoder_length] for name in self.encoder_only_variables
        }
        embeddings_varying_encoder, encoder_sparse_weights = self.encoder_variable_selection(
            embeddings_varying_encoder,
            static_context_variable_selection[:, :max_encoder_length],
        )

        embeddings_varying_decoder = {
            name: input_vectors[name][:, -max_decoder_length:] for name in self.decoder_variables  # select decoder
        }
        embeddings_varying_decoder, decoder_sparse_weights = self.decoder_variable_selection(
            embeddings_varying_decoder,
            static_context_variable_selection[:, max_encoder_length:end_decoder_step],
        )

        embeddings_varying_future = {
            name: input_vectors[name][:, :max_future_length] for name in self.future_only_variables  # select future
        }
        embeddings_varying_future, future_sparse_weights = self.future_variable_selection(
            embeddings_varying_future,
            static_context_variable_selection[:, end_decoder_step:],
        )

        # LSTM
        # calculate initial state
        input_hidden = self.static_context_initial_hidden_lstm(static_embedding).expand(
            self.hparams.lstm_layers, -1, -1
        )
        input_cell = self.static_context_initial_cell_lstm(static_embedding).expand(self.hparams.lstm_layers, -1, -1)

        # run local encoder
        encoder_output, encoder_context = self.lstm_encoder( # encoder_context = (hidden, cell)
            embeddings_varying_encoder, (input_hidden, input_cell), lengths=encoder_lengths, enforce_sorted=False
        )

        # run local decoder
        decoder_output, decoder_context = self.lstm_decoder(
            embeddings_varying_decoder,
            encoder_context,
            lengths=decoder_lengths,
            enforce_sorted=False,
        )

        # run local future
        future_output, future_context = self.lstm_future(
            embeddings_varying_future,
            decoder_context,
            lengths=future_lengths,
            enforce_sorted=False,
        )

        # skip connection over lstm
        lstm_output_encoder = self.post_lstm_gate_encoder(encoder_output)
        lstm_output_encoder = self.post_lstm_add_norm_encoder(lstm_output_encoder, embeddings_varying_encoder)

        lstm_output_decoder = self.post_lstm_gate_decoder(decoder_output)
        lstm_output_decoder = self.post_lstm_add_norm_decoder(lstm_output_decoder, embeddings_varying_decoder)

        lstm_output_future = self.post_lstm_gate_future(future_output)
        lstm_output_future = self.post_lstm_add_norm_future(lstm_output_future, embeddings_varying_future)

        lstm_output = torch.cat([lstm_output_encoder, lstm_output_decoder, lstm_output_future], dim=1)

        if self.reverse_future:
            embeddings_varying_encoder_rev = {
                name: torch.flip(input_vectors[name][:, :max_encoder_length], [1]) for name in self.encoder_only_variables
            }
            embeddings_varying_encoder_rev, encoder_sparse_weights_rev = self.encoder_variable_selection(
                    embeddings_varying_encoder_rev,
                    static_context_variable_selection[:, :max_encoder_length],
            )

            embeddings_varying_decoder_rev = {
                name: torch.flip(input_vectors[name][:, -max_decoder_length:], [1]) for name in self.decoder_variables
            }
            embeddings_varying_decoder_rev, decoder_sparse_weights_rev = self.decoder_variable_selection(
                embeddings_varying_decoder_rev,
                static_context_variable_selection[:, max_encoder_length:end_decoder_step],
            )

            embeddings_varying_future_rev = {
                name: torch.flip(input_vectors[name][:, :max_future_length], [1]) for name in self.future_only_variables
            }
            embeddings_varying_future_rev, future_sparse_weights_rev = self.future_variable_selection(
                embeddings_varying_future_rev,
                static_context_variable_selection[:, end_decoder_step:],
            )

            # Run LSTM from opposite direction
            # run local future (reversed time)
            future_rev, future_context = self.lstm_future(
                embeddings_varying_future_rev, (input_hidden, input_cell), lengths=future_lengths, enforce_sorted=False
            )

            # run decoder from future context
            decoder_rev, decoder_context = self.lstm_decoder(
                embeddings_varying_decoder_rev,
                future_context,
                lengths=decoder_lengths,
                enforce_sorted=False,
            )

            # run local encoder from future context
            encoder_rev, encoder_context = self.lstm_encoder(
                embeddings_varying_encoder_rev,
                decoder_context,
                lengths=encoder_lengths,
                enforce_sorted=False,
            )

            # skip connection over lstm
            lstm_output_encoder_rev = self.post_lstm_gate_encoder(encoder_rev)
            lstm_output_encoder_rev = self.post_lstm_add_norm_encoder(lstm_output_encoder_rev, embeddings_varying_encoder_rev)

            lstm_output_decoder_rev = self.post_lstm_gate_decoder(decoder_rev)
            lstm_output_decoder_rev = self.post_lstm_add_norm_decoder(lstm_output_decoder_rev, embeddings_varying_decoder_rev)

            lstm_output_future_rev = self.post_lstm_gate_future(future_rev)
            lstm_output_future_rev = self.post_lstm_add_norm_future(lstm_output_future_rev, embeddings_varying_future_rev)

            lstm_output_rev = torch.cat([lstm_output_future_rev, lstm_output_decoder_rev, lstm_output_encoder_rev], dim=1)
            lstm_output_rev = torch.flip(lstm_output_rev, [1]) # align time

            lstm_output = torch.cat([lstm_output, lstm_output_rev], dim=-1) # concatenate in feature dimension with time alignment


        # static enrichment
        static_context_enrichment = self.static_context_enrichment(static_embedding)
        attn_input = self.static_enrichment(
            lstm_output, self.expand_static_context(static_context_enrichment, timesteps)
        )

        # Attention
        attn_output, attn_output_weights = self.multihead_attn(
            # query only for predictions
            q=attn_input[:, max_encoder_length:end_decoder_step],
            k=attn_input,
            v=attn_input,
            mask=self.get_attention_mask(max_encoder_length=max_encoder_length,
                                         max_decoder_length=max_decoder_length,
                                         max_future_length=max_future_length,
                                         encoder_lengths=encoder_lengths,
                                         decoder_lengths=decoder_lengths,
                                         future_lengths=future_lengths,
                                         encoder_missing=encoder_missing,
                                         future_missing=future_missing),
        )

        # skip connection over attention
        attn_output = self.post_attn_gate_norm(attn_output, attn_input[:, max_encoder_length:end_decoder_step])

        output = self.pos_wise_ff(attn_output)

        # skip connection over temporal fusion decoder (not LSTM decoder despite the LSTM output contains
        # a skip from the variable selection network)
        output = self.pre_output_gate_norm(output, lstm_output[:, max_encoder_length:end_decoder_step])
        if self.n_targets > 1:  # if to use multi-target architecture
            output = [output_layer(output) for output_layer in self.output_layer]
            if self.monotonic_q:
                output = [self.quantile_transform(output_layer) for output_layer in output]
        else:
            output = self.output_layer(output)
            if self.monotonic_q:
                output = self.quantile_transform(output)

        if future_missing is None:
            missing_values_kwargs = {}
        else:
            missing_values_kwargs = dict(encoder_missing=encoder_missing, future_missing=future_missing)

        if self.reverse_future:
            sparse_weights_kwargs = dict(encoder_sparse_weights_rev=encoder_sparse_weights_rev,
                                         decoder_sparse_weights_rev=decoder_sparse_weights_rev,
                                         future_sparse_weights_rev=future_sparse_weights_rev,
                                         )
        else:
            sparse_weights_kwargs = {}

        return self.to_network_output(
            prediction=self.transform_output(output, target_scale=x["target_scale"]),
            encoder_attention=attn_output_weights[..., :max_encoder_length],
            decoder_attention=attn_output_weights[..., max_encoder_length:end_decoder_step],
            future_attention=attn_output_weights[..., end_decoder_step:],
            static_variables=static_variable_selection,
            encoder_variables=encoder_sparse_weights,
            decoder_variables=decoder_sparse_weights,
            future_variables=future_sparse_weights,
            decoder_lengths=decoder_lengths,
            encoder_lengths=encoder_lengths,
            future_lengths=future_lengths,
            **missing_values_kwargs,
            **sparse_weights_kwargs,
        )

    def step(
        self, x: Dict[str, torch.Tensor], y: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, **kwargs
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Run for each train/val step.

        Args:
            x (Dict[str, torch.Tensor]): x as passed to the network by the dataloader
            y (Tuple[torch.Tensor, torch.Tensor]): y as passed to the loss function by the dataloader
            batch_idx (int): batch number
            **kwargs: additional arguments to pass to the network apart from ``x``

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]: tuple where the first
                entry is a dictionary to which additional logging results can be added for consumption in the
                ``epoch_end`` hook and the second entry is the model's output.
        """
        # pack y sequence if different encoder lengths exist
        if (x["decoder_lengths"] < x["decoder_lengths"].max()).any():
            if isinstance(y[0], (list, tuple)):
                y = (
                    [
                        rnn.pack_padded_sequence(
                            y_part, lengths=x["decoder_lengths"].cpu(), batch_first=True, enforce_sorted=False
                        )
                        for y_part in y[0]
                    ],
                    y[1],
                )
            else:
                y = (
                    rnn.pack_padded_sequence(
                        y[0], lengths=x["decoder_lengths"].cpu(), batch_first=True, enforce_sorted=False
                    ),
                    y[1],
                )

        if self.training and len(self.hparams.monotone_constaints) > 0:
            # calculate gradient with respect to continous decoder features
            x["decoder_cont"].requires_grad_(True)
            assert not torch._C._get_cudnn_enabled(), (
                "To use monotone constraints, wrap model and training in context "
                "`torch.backends.cudnn.flags(enable=False)`"
            )
            out = self(x, **kwargs)
            prediction = out["prediction"]

            # handle multiple targets
            prediction_list = to_list(prediction)
            gradient = 0
            # todo: should monotone constrains be applicable to certain targets?
            for pred in prediction_list:
                gradient = (
                    gradient
                    + torch.autograd.grad(
                        outputs=pred,
                        inputs=x["decoder_cont"],
                        grad_outputs=torch.ones_like(pred),  # t
                        create_graph=True,  # allows usage in graph
                        allow_unused=True,
                    )[0]
                )

            # select relevant features
            indices = torch.tensor(
                [self.hparams.x_reals.index(name) for name in self.hparams.monotone_constaints.keys()]
            )
            monotonicity = torch.tensor(
                [val for val in self.hparams.monotone_constaints.values()], dtype=gradient.dtype, device=gradient.device
            )
            # add additionl loss if gradient points in wrong direction
            gradient = gradient[..., indices] * monotonicity[None, None]
            monotinicity_loss = gradient.clamp_max(0).mean()
            # multiply monotinicity loss by large number to ensure relevance and take to the power of 2
            # for smoothness of loss function
            monotinicity_loss = 10 * torch.pow(monotinicity_loss, 2)
            if isinstance(self.loss, (MASE, MultiLoss)):
                loss = self.loss(
                    prediction, y, encoder_target=x["encoder_target"], encoder_lengths=x["encoder_lengths"]
                )
            else:
                loss = self.loss(prediction, y)

            loss = loss * (1 + monotinicity_loss)
        else:
            out = self(x, **kwargs)

            # calculate loss
            prediction = out["prediction"]
            if isinstance(self.loss, (MASE, MultiLoss)):
                if isinstance(self.loss, MASE):
                    loss_kwargs = dict(encoder_target=x["encoder_target"], encoder_lengths=x["encoder_lengths"])
                elif isinstance(self.loss[0], QuantileLossZeroAttention):
                    loss_kwargs = dict(decoder_missing=x["decoder_missing"])
                else:
                    loss_kwargs = {}
                loss = self.loss(prediction, y, **loss_kwargs)
            else:
                loss = self.loss(prediction, y)

        self.log(
            f"{self.current_stage}_loss",
            loss,
            on_step=self.training,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(x["decoder_target"]),
        )
        log = {"loss": loss, "n_samples": x["decoder_lengths"].size(0)}
        return log, out

class ProbabilisticForecaster:
    """
    Forecaster class to train a model and predict future locations of a given trajectory.
    """
    def __init__(self, model_specs={}, batch_size=128, val_batch_mpl=10, seed=0, load_data=True, num_workers=None, **dataset_kwargs):
        self.batch_size = batch_size
        if load_data:
            print("Loading dataset...")
            training, validation, test, training_predict, center = load.load_dataset(**dataset_kwargs)
            self.dataset_train = training
            self.dataset_val = validation
            self.dataset_test = test
            self.dataset_train_predict = training_predict
            self._center = center
            self.get_dataloader(batch_size=batch_size, val_batch_mpl=val_batch_mpl, num_workers=num_workers)
            print("Loaded.")

        self.model_specs = {**params.default_model_specs, **model_specs}
        self._dataset_kwargs = dataset_kwargs
        self.seed = seed
        # self.model_cls = None
        # self.model = None
        # self.trainer = None
        # self.get_predictions = None
        # self.get_actuals = None
        # self.y_val = None
        # self.y_pred_val = None

    def get_dataloader(self, batch_size=None, val_batch_mpl=100, num_workers=None):
        """
        Get dataloaders for training and validation.

        NOTE: higher batch size does not speed up training.

        Args:
            batch_size: batch size for training
            val_batch_mpl: batch size multiplier for validation. This is used to increase the speed of validation.
        """
        if batch_size is None:
            batch_size = self.batch_size
        else:
            self.batch_size = batch_size
        if num_workers is None:
            available_workers = len(psutil.Process().cpu_affinity())
            num_workers = min(1, available_workers)
        print(f"Using {num_workers} workers for dataloading. NOTE: There is no benefit for num_workers > 1. There is benefit for num_workers > 0.")
        pin_memory = torch.cuda.is_available()
        self.train_dataloader = self.dataset_train.to_dataloader(train=True, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
        self.val_dataloader = self.dataset_val.to_dataloader(train=False, batch_size=batch_size * val_batch_mpl, num_workers=num_workers, pin_memory=pin_memory)
        self.test_dataloader = self.dataset_test.to_dataloader(train=False, batch_size=batch_size * val_batch_mpl, num_workers=num_workers, pin_memory=pin_memory)
        self.train_predict_dataloader = self.dataset_train_predict.to_dataloader(train=False, batch_size=batch_size * val_batch_mpl, num_workers=num_workers, pin_memory=pin_memory)
        return

    def get_trainer(self, epochs=100, limit_train_batches=None, limit_val_batches=None, patience=25, **training_specs):
        monitor = "val_loss"
        limit_kwargs = dict()
        if limit_train_batches is not None:
            limit_kwargs["limit_train_batches"] = limit_train_batches
        elif self._dataset_kwargs.get('dive_data', False):
            limit_kwargs["limit_train_batches"] = 512
        if limit_val_batches is not None:
            limit_kwargs["limit_val_batches"] = limit_val_batches
            if limit_val_batches == 0:
                monitor = "train_loss"

        early_stop_callback = EarlyStopping(monitor=monitor, min_delta=1e-4, patience=patience, verbose=False, mode="min")
        lr_logger = LearningRateMonitor()  # log the learning rate
        save_best_model = ModelCheckpoint(monitor=monitor, mode="min", save_top_k=1, verbose=False, save_weights_only=True)
        logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

        default_training_specs = params.default_training_specs
        if torch.cuda.is_available():
            default_training_specs["accelerator"] = "gpu"
            default_training_specs["deterministic"] = False
        else:
            default_training_specs["accelerator"] = "cpu"
        if pl.__version__ >= "2.0.0":
            default_training_specs['devices'] = "auto"
            # default_training_specs['devices'] = 0
        else:
            default_training_specs['auto_select_gpus'] = True
            default_training_specs["gpus"] = 0 if torch.cuda.is_available() else None
        training_specs = {**default_training_specs, **training_specs}

        trainer = pl.Trainer(
            max_epochs=epochs,
            callbacks=[lr_logger, early_stop_callback, save_best_model],
            logger=logger,
            **limit_kwargs,
            **training_specs,
            # fast_dev_run=true,  # comment in to check that network or dataset has no serious bugs
        )
        return trainer

    def train(self, epochs=100, limit_train_batches=None, **training_specs):
        self.set_training_step_method()
        pl.seed_everything(self.seed)
        self.trainer = self.get_trainer(epochs, limit_train_batches, **training_specs)
        self.trainer.fit(self.model,
                         train_dataloaders=self.train_dataloader,
                         val_dataloaders=self.val_dataloader)

        print("Training finished. Evaluating validation set...")
        self.best_model_path = self.trainer.checkpoint_callback.best_model_path
        self.model = self.model_cls.load_from_checkpoint(self.best_model_path)
        self.get_predictions(partition='val')
        self.get_predictions(partition='test')

    def set_training_step_method(self):
        if torch.cuda.is_available():
            self.model.training_step = types.MethodType(training_step_gpu, self.model)
        return

def get_quantiles(quantiles, s_q=1):
    if isinstance(quantiles, str):
        if quantiles == 'all':
            if s_q > 1:
                quantiles = np.linspace(0.01, 0.99, int(100/s_q)) # maintain 1% and 99% quantiles
            else:
                quantiles = np.linspace(0, 1, int(100/s_q))[1:-1] # exclude 0 and 1
            if not 0.5 in quantiles:
                warnings.warn("Quantile 0.5 not in quantiles. Inserting.")
                pos = np.searchsorted(quantiles, 0.5)
                quantiles = np.insert(quantiles, pos, 0.5)
            quantiles = quantiles.tolist()
        else:
            quantiles = params.default_quantiles[quantiles]
    elif not isinstance(quantiles, list):
        raise ValueError("Quantiles must be a list, tuple, or numpy array. Alternatively, use 'exact' or 'Bonferroni'.")
    return quantiles

class QuantileForecaster(ProbabilisticForecaster):
    """
    Predicts quantiles for each output variable.

    The prediction region is defined by the quantiles, and its confidence level is defined by the Bonferroni method:
    https://en.wikipedia.org/wiki/Bonferroni_correction

    Default quantile and confidence levels:
        [0.0125, 0.9875] -> 97.5% prediction interval (95% prediction region)
        [0.025, 0.975] -> 95% prediction interval (90% prediction region)
        [0.125, 0.875] -> 75% prediction interval (50% prediction region)
        0.5 -> median prediction (for point estimate).
    """
    def __init__(self, quantiles='exact', s_q=1, cumulative=False, joint_prediction=False, task='forecasting', deprecated=False, decoder_missing_zero_loss=False, mid_rmse=False, mid_weight=2, target='cds', monotonic_q=False, **kwargs):
        super().__init__(task=task, target=target, **kwargs)

        quantiles = get_quantiles(quantiles, s_q)

        print(f"Quantiles: {quantiles}")
        self.target = target
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        self.joint_prediction = joint_prediction
        self.cumulative = cumulative
        self.task = task
        self.deprecated = deprecated
        self.decoder_missing_zero_loss = decoder_missing_zero_loss
        self.mid_rmse = mid_rmse
        self.monotonic_q = monotonic_q

        cds = kwargs.get('cds', 'mercator')
        self.cds = cds
        loss_kwargs = {}
        if decoder_missing_zero_loss:
            print("Using QuantileLossZeroAttention for discarding decoder missing values.")
            self.loss_cls = QuantileLossZeroAttention
        elif mid_rmse:
            print("Using QuantileLossRMSE for using RMSE on the point prediction and optimize quantiles.")
            self.loss_cls = QuantileLossRMSE
            loss_kwargs = dict(mid_weight=mid_weight)
        else:
            self.loss_cls = QuantileLoss
        if task == 'imputation':
            if self.deprecated:
                self.model_cls = TemporalFusionTransformerImputationOld
            else:
                self.model_cls = TemporalFusionTransformerImputation
            extra_kwargs = dict()
            if cds == 'spatial-dx':
                raise NotImplementedError("Spatial-dx not implemented for imputation task.")
        elif cds == 'spherical-dx':
            self.model_cls = TemporalFusionTransformerSpatial
            extra_kwargs = dict(joint_prediction=joint_prediction, cumulative=cumulative)
            if self.cumulative:
                self.loss_cls = CumQuantileLoss
            else:
                self.loss_cls = QuantileLoss
        else:
            self.model_cls = TemporalFusionTransformerForecasting
            extra_kwargs = dict()

        print(f"Model class: {self.model_cls.__name__}")
        self.loss_kwargs = loss_kwargs
        if target == 'cds':
            loss = MultiLoss(metrics=[self.loss_cls(quantiles=self.quantiles, **loss_kwargs),
                                    self.loss_cls(quantiles=self.quantiles, **loss_kwargs)])
            output_size = (self.num_quantiles, self.num_quantiles)
        elif target == 'rho':
            loss = self.loss_cls(quantiles=self.quantiles, **loss_kwargs)
            output_size = self.num_quantiles
        else:
            raise NotImplementedError(f"Target {target} not implemented.")
        self.num_targets = np.asarray(output_size).size

        if hasattr(self, 'dataset_train'):
            self.model = self.model_cls.from_dataset(
                self.dataset_train,
                output_size=output_size,
                loss=loss,
                **extra_kwargs,
                **self.model_specs,
                monotonic_q=monotonic_q,
            )

    def get_actuals(self, partition='val'):
        """
        Get actual values from validation dataset.
        """
        if hasattr(self, f'y_{partition}'):
            return getattr(self, f'y_{partition}')
        else:
            dataloader = getattr(self, f'{partition}_dataloader')
            if self.num_targets > 1:
                setattr(self, f'y_{partition}', torch.cat([torch.stack(y, axis=0) for _, (y, weight) in dataloader], axis=1))
            else:
                setattr(self, f'y_{partition}', torch.cat([y for _, (y, weight) in dataloader], axis=1))
            return getattr(self, f'y_{partition}')

    def get_predictions(self, partition='val'):
        """
        Get predictions from trained model.
        """
        dataloader = getattr(self, f'{partition}_dataloader')
        if self.num_targets > 1:
            setattr(self, f'y_pred_{partition}', torch.stack(self.model.predict(dataloader), axis=0))
        else:
            setattr(self, f'y_pred_{partition}', self.model.predict(dataloader))
        return getattr(self, f'y_pred_{partition}')

    def baseline(self, partition='val'):
        """
        Forecaster that predicts the last observed value.
        """
        if hasattr(self, f'baseline_error_{partition}'):
            return getattr(self, f'baseline_error_{partition}')
        else:
            dataloader = getattr(self, f'{partition}_dataloader')
            if self.num_targets > 1:
                baseline_predictions = torch.stack(Baseline().predict(dataloader), axis=-1)
            else:
                baseline_predictions = Baseline().predict(dataloader)
            y_real = self.get_actuals(partition=partition).permute(1,2,0)
            baseline_error = (y_real - baseline_predictions).abs().mean(axis=1).mean(axis=0) # MAE by coordinate
            setattr(self, f'baseline_error_{partition}', baseline_error)
            setattr(self, f'baseline_predictions_{partition}', baseline_predictions)
            return getattr(self, f'baseline_error_{partition}')

    def optimize_hyperparameters(
        self,
        model_path: str = "forecasting_quantile_optuna",
        max_epochs: int = 30, # 14(28) min per trial for forecasting (imputation)
        n_trials: int = 1000, # 'no limit'
        timeout: float = 3600 * 17.0,  # 17 hours
        gradient_clip_val_range: Tuple[float, float] = (0.01, 100.0),
        hidden_size_range: Tuple[int, int] = (32, 512),  #(32, 128),
        hidden_continuous_size_range: Tuple[int, int] = (4, 512), #(4, 128),
        attention_head_size_range: Tuple[int, int] = (1, 24), #(1, 5), #(3, 24),
        dropout_range: Tuple[float, float] = (0., 0.3),
        lstm_layers_range: Tuple[int, int] = (1, 4),
        learning_rate_range: Tuple[float, float] = (3e-5, 0.1), # next: (1e-3, 0.01)
        use_learning_rate_finder: bool = True,
        max_train_days_range: Tuple[int, int] = (1, 84), # 1 day to 3 months
        trainer_kwargs: Dict[str, Any] = dict(limit_train_batches=60), # 0.5" per batch, 0.5 min per epoch
        log_dir: str = "lightning_logs",
        study: optuna.Study = None,
        verbose: Union[int, bool] = None,
        batch_size: int = 128,
        prune_memory_error: bool = False,
        **dataset_kwargs,
    ) -> optuna.Study:
        """
        Optimize Temporal Fusion Transformer hyperparameters.
        Calls `pytorch_forecasting.models.temporal_fusion_transformer.tuning.optimize_hyperparameters`. This method only modifies the default arguments.
        Returns:
            optuna.Study: optuna study results
        """
        print(f"Optimizing hyperparameters for model {self.model_cls.__name__} with loss {self.loss_cls.__name__}")
        if self.cds.startswith('spherical'):
            gradient_clip_val_range = (1e-4, 5) # much smaller output for spherical
        if self.cumulative:
            extra_kwargs = dict(joint_prediction=self.joint_prediction, cumulative=True)
        else:
            extra_kwargs = dict()
        if self.monotonic_q:
            extra_kwargs['monotonic_q'] = True

        import optuna.logging
        optuna_logger = logging.getLogger("optuna")


        pl.seed_everything(self.seed)

        dive_data = dataset_kwargs.pop('dive_data', False)
        print(f"Loading {'dive' if dive_data else 'default'} data...")
        if dive_data:
            del dataset_kwargs['cds']
            gradient_clip_val_range = (0.01, 10) # short-term predictions with lower gradients
            print("Setting HP search for dive data (smaller models)")
            hidden_size_range = (16, 256)
            hidden_continuous_size_range = (4, 128)
            attention_head_size_range = (1, 16)
            lstm_layers_range = (1, 2)
            add_z = dataset_kwargs.pop('add_z', True)
            data, static_categoricals, known_reals, features, features_target = load.load_dive_data(add_z=add_z)
            self.data_specs = dict(data=data, static_categoricals=static_categoricals, known_reals=known_reals, features=features, features_target=features_target, cds='mercator', dive_data=True)
        else:
            sampling_freq = dataset_kwargs.pop('sampling_freq', 6)
            cds = dataset_kwargs.pop('cds', 'mercator')
            data, features, features_target, center  = load.load_data(sampling_freq=sampling_freq, cds=cds)
            self.data_specs = dict(data=data, features=features, features_target=features_target, center=center, sampling_freq=sampling_freq, cds=cds)
        print("Data loaded.")

        if dataset_kwargs.get('reverse_future', False):
            # increase hidden_size max range (uses half for the forward LSTM and half for the reverse LSTM)
            hidden_size_range = (hidden_size_range[0], hidden_size_range[1] * 2)
            print(f"Reversed future: increasing hidden_size_range to {hidden_size_range}")


        def _get_dataloaders(max_train_days):
            training, validation, test, training_predict, _ = load.load_dataset(max_train_days=max_train_days, **self.data_specs, **dataset_kwargs)
            self.dataset_train = training
            self.dataset_val = validation
            self.dataset_test = test # not to be used in hyperparameter optimization
            self.dataset_train_predict = training_predict
            self.get_dataloader(batch_size=batch_size, val_batch_mpl=1000) # single batch for val
            if dive_data:
                print("Dataset loaded for dive data.")
            else:
                print(f"Dataset loaded for max_train_days={max_train_days}")
            train_dataloader = deepcopy(self.train_dataloader)
            val_dataloader = deepcopy(self.val_dataloader)
            return train_dataloader, val_dataloader

        if dive_data or 'max_train_days' in dataset_kwargs: # optimize for a fixed max_train_days
            max_train_days = dataset_kwargs.get('max_train_days', 4)
            train_dataloader, val_dataloader = _get_dataloaders(max_train_days)
            optimize_max_train_days = False
            if not dive_data:
                print(f"Optimizing for fixed max_train_days={max_train_days}")
        else:
            optimize_max_train_days = True
            train_dataloader = None
            val_dataloader = None

        def optimizer(
            train_dataloaders: Union[DataLoader, None] = None,
            val_dataloaders: Union[DataLoader, None] = None,
            model_path: Union[str, None] = None,
            max_epochs: int = 20,
            n_trials: int = 100,
            timeout: float = 3600 * 8.0,  # 8 hours
            gradient_clip_val_range: Tuple[float, float] = (0.01, 100.0),
            hidden_size_range: Tuple[int, int] = (16, 265),
            hidden_continuous_size_range: Tuple[int, int] = (8, 64),
            attention_head_size_range: Tuple[int, int] = (1, 4),
            dropout_range: Tuple[float, float] = (0.1, 0.3),
            lstm_layers_range: Tuple[int, int] = (1, 4),
            learning_rate_range: Tuple[float, float] = (1e-5, 1.0),
            use_learning_rate_finder: bool = True,
            max_train_days_range: Tuple[int, int] = (1, 84),
            trainer_kwargs: Dict[str, Any] = {},
            log_dir: str = "lightning_logs",
            study: optuna.Study = None,
            verbose: Union[int, bool] = None,
            prune_memory_error: bool = False,
            pruner: optuna.pruners.BasePruner = optuna.pruners.SuccessiveHalvingPruner(),
            **kwargs,
        ) -> optuna.Study:
            """
            Optimize Temporal Fusion Transformer hyperparameters.

            Run hyperparameter optimization. Learning rate for is determined with
            the PyTorch Lightning learning rate finder.

            Args:
                model_path (str): folder to which model checkpoints are saved
                max_epochs (int, optional): Maximum number of epochs to run training. Defaults to 20.
                n_trials (int, optional): Number of hyperparameter trials to run. Defaults to 100.
                timeout (float, optional): Time in seconds after which training is stopped regardless of number of epochs
                    or validation metric. Defaults to 3600*8.0.
                hidden_size_range (Tuple[int, int], optional): Minimum and maximum of ``hidden_size`` hyperparameter. Defaults
                    to (16, 265).
                hidden_continuous_size_range (Tuple[int, int], optional):  Minimum and maximum of ``hidden_continuous_size``
                    hyperparameter. Defaults to (8, 64).
                attention_head_size_range (Tuple[int, int], optional):  Minimum and maximum of ``attention_head_size``
                    hyperparameter. Defaults to (1, 4).
                dropout_range (Tuple[float, float], optional):  Minimum and maximum of ``dropout`` hyperparameter. Defaults to
                    (0.1, 0.3).
                learning_rate_range (Tuple[float, float], optional): Learning rate range. Defaults to (1e-5, 1.0).
                use_learning_rate_finder (bool): If to use learning rate finder or optimize as part of hyperparameters.
                    Defaults to True.
                trainer_kwargs (Dict[str, Any], optional): Additional arguments to the
                    `PyTorch Lightning trainer <https://pytorch-lightning.readthedocs.io/en/latest/trainer.html>`_ such
                    as ``limit_train_batches``. Defaults to {}.
                log_dir (str, optional): Folder into which to log results for tensorboard. Defaults to "lightning_logs".
                study (optuna.Study, optional): study to resume. Will create new study by default.
                verbose (Union[int, bool]): level of verbosity.
                    * None: no change in verbosity level (equivalent to verbose=1 by optuna-set default).
                    * 0 or False: log only warnings.
                    * 1 or True: log pruning events.
                    * 2: optuna logging level at debug level.
                    Defaults to None.
                pruner (optuna.pruners.BasePruner, optional): The optuna pruner to use.
                    Defaults to optuna.pruners.SuccessiveHalvingPruner().

                **kwargs: Additional arguments for the :py:class:`~TemporalFusionTransformer`.

            Returns:
                optuna.Study: optuna study results
            """
            logging_level = {
                None: optuna.logging.get_verbosity(),
                0: optuna.logging.WARNING,
                1: optuna.logging.INFO,
                2: optuna.logging.DEBUG,
            }
            optuna_verbose = logging_level[verbose]
            optuna.logging.set_verbosity(optuna_verbose)

            # create objective function
            def objective(trial: optuna.Trial) -> float:
                nonlocal train_dataloaders, val_dataloaders
                # Filenames for each trial must be made unique in order to access each checkpoint.
                checkpoint_callback = pl.callbacks.ModelCheckpoint(
                    dirpath=os.path.join(model_path, "trial_{}".format(trial.number)), filename="{epoch}", monitor="val_loss"
                )

                # The default logger in PyTorch Lightning writes to event files to be consumed by
                # TensorBoard. We don't use any logger here as it requires us to implement several abstract
                # methods. Instead we setup a simple callback, that saves metrics from each validation step.
                metrics_callback = MetricsCallback()
                learning_rate_callback = LearningRateMonitor()
                logger = TensorBoardLogger(log_dir, name="optuna", version=trial.number)
                gradient_clip_val = trial.suggest_loguniform("gradient_clip_val", *gradient_clip_val_range)
                default_trainer_kwargs = dict(
                    gpus=[0] if torch.cuda.is_available() else None,
                    max_epochs=max_epochs,
                    gradient_clip_val=gradient_clip_val,
                    callbacks=[
                        metrics_callback,
                        learning_rate_callback,
                        checkpoint_callback,
                        PyTorchLightningPruningCallback(trial, monitor="val_loss"),
                    ],
                    logger=logger,
                    enable_progress_bar=optuna_verbose < optuna.logging.INFO,
                    enable_model_summary=[False, True][optuna_verbose < optuna.logging.INFO],
                )
                default_trainer_kwargs.update(trainer_kwargs)
                trainer = pl.Trainer(
                    **default_trainer_kwargs,
                )
                if optimize_max_train_days:
                    # create dataset
                    max_train_days = trial.suggest_int("max_train_days", *max_train_days_range)
                    train_dataloaders, val_dataloaders = _get_dataloaders(max_train_days)

                # create model
                hidden_size = trial.suggest_int("hidden_size", *hidden_size_range, log=True)
                lstm_layers = trial.suggest_int("lstm_layers", *lstm_layers_range)
                if lstm_layers > 1:
                    dropout = trial.suggest_uniform("dropout", *dropout_range)
                else: # dropout only has an effect when using multiple lstm layers
                    dropout = 0.0

                if self.num_targets > 1:
                    loss = MultiLoss(metrics=[self.loss_cls(quantiles=self.quantiles, **self.loss_kwargs),
                                              self.loss_cls(quantiles=self.quantiles, **self.loss_kwargs)])
                    output_size = (self.num_quantiles, self.num_quantiles)
                else:
                    loss = self.loss_cls(quantiles=self.quantiles, **self.loss_kwargs)
                    output_size = self.num_quantiles

                model = self.model_cls.from_dataset(
                    train_dataloaders.dataset,
                    dropout=dropout,
                    lstm_layers=lstm_layers,
                    hidden_size=hidden_size,
                    hidden_continuous_size=trial.suggest_int(
                        "hidden_continuous_size",
                        hidden_continuous_size_range[0],
                        min(hidden_continuous_size_range[1], hidden_size),
                        log=True,
                    ),
                    attention_head_size=trial.suggest_int("attention_head_size", *attention_head_size_range),
                    log_interval=-1,
                    loss=loss,
                    output_size=output_size,
                    verbose=trial.number == 0,
                    **extra_kwargs,
                    **kwargs,
                )
                if torch.cuda.is_available():
                    model.training_step = types.MethodType(training_step_gpu, model)

                # find good learning rate
                if use_learning_rate_finder:
                    lr_trainer = pl.Trainer(
                        gradient_clip_val=gradient_clip_val,
                        gpus=[0] if torch.cuda.is_available() else None,
                        logger=False,
                        enable_progress_bar=False,
                        enable_model_summary=False,
                    )
                    try:
                        res = lr_trainer.tuner.lr_find(
                            model,
                            train_dataloaders=train_dataloaders,
                            val_dataloaders=val_dataloaders,
                            early_stop_threshold=10000,
                            min_lr=learning_rate_range[0],
                            num_training=100,
                            max_lr=learning_rate_range[1],
                        )

                        loss_finite = np.isfinite(res.results["loss"])
                        if loss_finite.sum() > 3:  # at least 3 valid values required for learning rate finder
                            lr_smoothed, loss_smoothed = sm.nonparametric.lowess(
                                np.asarray(res.results["loss"])[loss_finite],
                                np.asarray(res.results["lr"])[loss_finite],
                                frac=1.0 / 10.0,
                            )[min(loss_finite.sum() - 3, 10) : -1].T
                            optimal_idx = np.gradient(loss_smoothed).argmin()
                            optimal_lr = lr_smoothed[optimal_idx]
                        else:
                            optimal_idx = np.asarray(res.results["loss"]).argmin()
                            optimal_lr = res.results["lr"][optimal_idx]
                        optuna_logger.info(f"Using learning rate of {optimal_lr:.3g}")
                        # add learning rate artificially
                        model.hparams.learning_rate = trial.suggest_loguniform("learning_rate", optimal_lr, optimal_lr) # loguniform for consistency with except block
                    except Exception as e:
                        warnings.warn(f"Learning rate finder failed, using default learning rate range. Exception: {e}")
                        model.hparams.learning_rate = trial.suggest_loguniform("learning_rate", *learning_rate_range)
                else:
                    model.hparams.learning_rate = trial.suggest_loguniform("learning_rate", *learning_rate_range)

                if prune_memory_error:
                    def empty_cache():
                        nonlocal train_dataloaders, val_dataloaders, trainer, model
                        # delete previous dataloaders for memory management
                        del self.train_dataloader
                        del self.val_dataloader
                        del train_dataloaders
                        del val_dataloaders
                        del trainer
                        del model
                        gc.collect()
                        torch.cuda.empty_cache()
                        return
                    try:
                        # fit
                        trainer.fit(model, train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders)
                        if optimize_max_train_days:
                            empty_cache()

                    except Exception as e:
                        print(e)
                        # CUDA out of memory
                        print("Assuming CUDA out of memory error, skipping trial.")
                        if optimize_max_train_days:
                            empty_cache()
                        raise optuna.TrialPruned()
                else:
                    trainer.fit(model, train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders)

                # report result
                return metrics_callback.metrics[-1]["val_loss"].item()

            # setup optuna and run
            if study is None:
                study = optuna.create_study(direction="minimize", pruner=pruner)
            study.optimize(objective, n_trials=n_trials, timeout=timeout)
            return study

        study = optimizer(train_dataloaders=train_dataloader,
                          val_dataloaders=val_dataloader,
                          model_path=model_path,
                          max_epochs=max_epochs,
                          n_trials=n_trials,
                          timeout=timeout,
                          gradient_clip_val_range=gradient_clip_val_range,
                          hidden_size_range=hidden_size_range,
                          hidden_continuous_size_range=hidden_continuous_size_range,
                          attention_head_size_range=attention_head_size_range,
                          dropout_range=dropout_range,
                          lstm_layers_range=lstm_layers_range,
                          learning_rate_range=learning_rate_range,
                          use_learning_rate_finder=use_learning_rate_finder,
                          max_train_days_range=max_train_days_range,
                          trainer_kwargs=trainer_kwargs,
                          log_dir=log_dir,
                          study=study,
                          verbose=verbose,
                          prune_memory_error=prune_memory_error,
                          )
        return study

    def get_results(self, partition='val'): # the real y can be found in x['decoder_target']
        if self.num_targets > 1:
            stack_targets = lambda t: torch.stack(t, axis=0).numpy()
        else:
            stack_targets = lambda t: t.numpy()

        dataloader = getattr(self, f"{partition}_dataloader")
        out_raw, x, *_ = self.model.predict(dataloader, mode="raw", return_x=True)
        y_pred_raw = stack_targets(out_raw['prediction'])
        y_pred = stack_targets(self.model.to_prediction(out_raw))
        y_pred_quantiles = stack_targets(self.model.to_quantiles(out_raw))

        # TupleOutputMixIn.to_network_output is not pickable. We need to store the output as a dictionary
        out_dict = {k: v for k, v in out_raw.items()}

        results = dict(x = x,
                       out = out_dict,
                       y_pred_raw=y_pred_raw,
                       y_pred=y_pred,
                       y_pred_quantiles=y_pred_quantiles,
                      )
        if self.task == 'forecasting':
            interpretation = self.model.interpret_output(out_raw, reduction="sum")
            interpretation_raw = self.model.interpret_output(out_raw, reduction="none")
            results['interpretation'] = interpretation
            results['interpretation_raw'] = interpretation_raw

        partition_str = partition.replace("_predict", "") # 'train_predict' -> 'train'
        results = {f'{k}_{partition_str}': v for k, v in results.items()}
        results['static_variables'] = self.model.static_variables
        results['encoder_variables'] = self.model.encoder_variables
        results['decoder_variables'] = self.model.decoder_variables
        if self.task == 'imputation':
            results['encoder_only_variables'] = self.model.encoder_only_variables
            results['future_only_variables'] = self.model.future_only_variables

        dive_data = self._dataset_kwargs.get('dive_data', False)
        if dive_data:
            add_z = self._dataset_kwargs.get('add_z', False)
            if add_z:
                scaler_z = self.dataset_train.scalers['z']
                results['z_loc'] = scaler_z.mean_
                results['z_scale'] = scaler_z.scale_

            # sampling rate
            dataset = self.dataset_train
            dt_idx = dataset.reals.index('dt')
            dt = x['encoder_cont'][..., dt_idx]
            scaler_dt = dataset.scalers['dt']
            dt = dt*scaler_dt.scale_ + scaler_dt.mean_
            dt_mode = dt.ravel().mode()[0].item() * 24
            results['dt_mode'] = dt_mode
        return results

class DistributionForecaster(ProbabilisticForecaster):
    """
    Predicts the parameters of a bivariate gaussian mixture distribution.

    The prediction region is defined by the alpha shape or the convex hull of the PDF.
    The confidence level is computed numerically by sampling from the PDF and computing the fraction of samples that lie within the prediction region.

    Output shape: (batch_size, max_prediction_length, num_mixtures * 6)
    Prediction output shape: (batch_size, max_prediction_length, 2)
    Quantile output shape: (batch_size, max_prediction_length, num_quantiles, 2)
    """
    def __init__(self, num_mixtures=3, epsilon=0, lambda_area=5., affine_transform=True, is_energy_score=False, beta=1, R=100, **kwargs):
        super().__init__(**kwargs)
        self.num_mixtures = num_mixtures
        self.epsilon = epsilon
        self.lambda_area = lambda_area
        self.affine_transform = affine_transform
        self.is_energy_score = is_energy_score
        self.model_cls = TemporalFusionTransformerDistribution
        if hasattr(self, 'dataset_train'):
            self.model = self.model_cls.from_dataset(
                self.dataset_train,
                loss_type='mixture',
                num_mixtures=num_mixtures,
                output_size=num_mixtures * 6,
                affine_transform=affine_transform,
                is_energy_score=is_energy_score,
                beta=beta,
                R=R,
                loss=BivariateGaussianMixtureLoss(num_mixtures=num_mixtures, epsilon=epsilon, lambda_area=lambda_area, affine_transform=affine_transform,
                                                  is_energy_score=is_energy_score, beta=beta, R=R),
                **self.model_specs,
            )


    def get_actuals(self, partition='val'):
        """
        Get actual values from validation dataset.
        """
        if hasattr(self, f'y_{partition}'):
            return getattr(self, f'y_{partition}')
        else:
            dataloader = getattr(self, f'{partition}_dataloader')
            setattr(self, f'y_{partition}', torch.cat([torch.stack(y, dim=-1) for _, (y, weight) in dataloader], axis=0))
            return getattr(self, f'y_{partition}')

    def get_predictions(self, partition='val'):
        """
        Get predictions from trained model.
        """
        if hasattr(self, f'y_pred_{partition}'):
            return getattr(self, f'y_pred_{partition}')
        else:
            dataloader = getattr(self, f'{partition}_dataloader')
            setattr(self, f'y_pred_{partition}', self.model.predict(dataloader))
            return getattr(self, f'y_pred_{partition}')

    def baseline(self, partition='val'):
        """
        Forecaster that predicts the last observed value.
        """
        if hasattr(self, f'baseline_error_{partition}'):
            return getattr(self, f'baseline_error_{partition}')
        else:
            dataloader = getattr(self, f'{partition}_dataloader')
            baseline_predictions = torch.stack(Baseline().predict(dataloader), axis=-1)
            y_real = self.get_actuals(partition=partition)
            baseline_error = (y_real - baseline_predictions).abs().mean(axis=1).mean(axis=0) # MAE by coordinate
            setattr(self, f'baseline_error_{partition}', baseline_error)
            setattr(self, f'baseline_predictions_{partition}', baseline_predictions)
            return getattr(self, f'baseline_error_{partition}')

    def optimize_hyperparameters(
        self,
        model_path: str = "forecasting_dist_optuna",
        max_epochs: int = 30, # 14 min per trial
        n_trials: int = 1000, # 23 hours
        timeout: float = 3600 * 17.0,  # 17 hours
        epsilon: float = 0,
        is_energy_score: bool = False,
        beta: float = 1,
        R: int = 100,
        gradient_clip_val_range: Tuple[float, float] = (0.01, 1.0),
        num_mixtures_range: Tuple[int, int] = (1, 10),
        hidden_size_range: Tuple[int, int] = (32, 128),
        hidden_continuous_size_range: Tuple[int, int] = (8, 64),
        attention_head_size_range: Tuple[int, int] = (1, 4),
        dropout_range: Tuple[float, float] = (0., 0.3),
        learning_rate_range: Tuple[float, float] = (1e-5, 0.1),
        use_learning_rate_finder: bool = True,
        trainer_kwargs: Dict[str, Any] = dict(limit_train_batches=60), # 0.5" per batch, 0.5 min per epoch
        log_dir: str = "lightning_logs",
        study: optuna.Study = None,
        verbose: Union[int, bool] = None,
        batch_size: int = 128,
        **dataset_kwargs,
    ) -> optuna.Study:
        """
        Optimize Temporal Fusion Transformer hyperparameters.

        Run hyperparameter optimization. Learning rate for is determined with
        the PyTorch Lightning learning rate finder.

        Args:
            train_dataloader (DataLoader): dataloader for training model
            val_dataloader (DataLoader): dataloader for validating model
            model_path (str): folder to which model checkpoints are saved
            max_epochs (int, optional): Maximum number of epochs to run training. Defaults to 20.
            n_trials (int, optional): Number of hyperparameter trials to run. Defaults to 100.
            timeout (float, optional): Time in seconds after which training is stopped regardless of number of epochs
                or validation metric. Defaults to 3600*8.0.
            hidden_size_range (Tuple[int, int], optional): Minimum and maximum of ``hidden_size`` hyperparameter. Defaults
                to (16, 265).
            hidden_continuous_size_range (Tuple[int, int], optional):  Minimum and maximum of ``hidden_continuous_size``
                hyperparameter. Defaults to (8, 64).
            attention_head_size_range (Tuple[int, int], optional):  Minimum and maximum of ``attention_head_size``
                hyperparameter. Defaults to (1, 4).
            dropout_range (Tuple[float, float], optional):  Minimum and maximum of ``dropout`` hyperparameter. Defaults to
                (0.1, 0.3).
            learning_rate_range (Tuple[float, float], optional): Learning rate range. Defaults to (1e-5, 1.0).
            use_learning_rate_finder (bool): If to use learning rate finder or optimize as part of hyperparameters.
                Defaults to True.
            trainer_kwargs (Dict[str, Any], optional): Additional arguments to the
                `PyTorch Lightning trainer <https://pytorch-lightning.readthedocs.io/en/latest/trainer.html>`_ such
                as ``limit_train_batches``. Defaults to {}.
            log_dir (str, optional): Folder into which to log results for tensorboard. Defaults to "lightning_logs".
            study (optuna.Study, optional): study to resume. Will create new study by default.
            verbose (Union[int, bool]): level of verbosity.
                * None: no change in verbosity level (equivalent to verbose=1 by optuna-set default).
                * 0 or False: log only warnings.
                * 1 or True: log pruning events.
                * 2: optuna logging level at debug level.
                Defaults to None.

            **dataset_kwargs: Additional arguments for load.load_dataset()

        Returns:
            optuna.Study: optuna study results
        """
        pl.seed_everything(self.seed)

        print("Loading dataset...")
        training, validation, test, training_predict,  _ = load.load_dataset(**dataset_kwargs)
        self.dataset_train = training
        self.dataset_val = validation
        self.dataset_test = test # not to be used in hyperparameter optimization
        self.dataset_train_predict = training_predict
        self.get_dataloader(batch_size=batch_size, val_batch_mpl=4)
        print("Loaded.")

        assert isinstance(self.train_dataloader.dataset, TimeSeriesDataSet) and isinstance(
            self.val_dataloader.dataset, TimeSeriesDataSet
        ), "dataloaders must be built from timeseriesdataset"

        model_path = f"{model_path}/{self.model_cls.__name__}"

        optuna_logger = logging.getLogger("optuna")
        logging_level = {
            None: optuna.logging.get_verbosity(),
            0: optuna.logging.WARNING,
            1: optuna.logging.INFO,
            2: optuna.logging.DEBUG,
        }
        verbose = logging_level[verbose]
        optuna.logging.set_verbosity(verbose)

        # create objective function
        def objective(trial: optuna.Trial) -> float:
            # Filenames for each trial must be made unique in order to access each checkpoint.
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                os.path.join(model_path, "trial_{}".format(trial.number), "{epoch}"), monitor="val_loss"
            )

            # The default logger in PyTorch Lightning writes to event files to be consumed by
            # TensorBoard. We don't use any logger here as it requires us to implement several abstract
            # methods. Instead we setup a simple callback, that saves metrics from each validation step.
            metrics_callback = MetricsCallback()
            learning_rate_callback = LearningRateMonitor()
            logger = TensorBoardLogger(log_dir, name="optuna", version=trial.number)
            gradient_clip_val = trial.suggest_loguniform("gradient_clip_val", *gradient_clip_val_range)
            trainer_kwargs.setdefault("gpus", [0] if torch.cuda.is_available() else None)
            trainer = pl.Trainer(
                max_epochs=max_epochs,
                gradient_clip_val=gradient_clip_val,
                callbacks=[
                    metrics_callback,
                    learning_rate_callback,
                    PyTorchLightningPruningCallback(trial, monitor="val_loss"),
                    checkpoint_callback,
                ],
                logger=logger,
                **trainer_kwargs,
            )

            # create model
            hidden_size = trial.suggest_int("hidden_size", *hidden_size_range, log=True)
            num_mixtures = trial.suggest_int("num_mixtures", *num_mixtures_range)
            loss = BivariateGaussianMixtureLoss(num_mixtures=num_mixtures, epsilon=epsilon, is_energy_score=is_energy_score, beta=beta, R=R)
            model = self.model_cls.from_dataset(
                self.train_dataloader.dataset,
                num_mixtures=num_mixtures,
                dropout=trial.suggest_uniform("dropout", *dropout_range),
                hidden_size=hidden_size,
                hidden_continuous_size=trial.suggest_int(
                    "hidden_continuous_size",
                    hidden_continuous_size_range[0],
                    min(hidden_continuous_size_range[1], hidden_size),
                    log=True,
                ),
                attention_head_size=trial.suggest_int("attention_head_size", *attention_head_size_range),
                log_interval=-1,
                loss=loss,
            )
            # find good learning rate
            if use_learning_rate_finder:
                lr_trainer = pl.Trainer(
                    gradient_clip_val=gradient_clip_val,
                    gpus=[0] if torch.cuda.is_available() else None,
                    logger=False,
                )
                res = lr_trainer.tuner.lr_find(
                    model,
                    train_dataloaders=self.train_dataloader,
                    val_dataloaders=self.val_dataloader,
                    early_stop_threshold=10000,
                    min_lr=learning_rate_range[0],
                    num_training=100,
                    max_lr=learning_rate_range[1],
                )

                loss_finite = np.isfinite(res.results["loss"])
                lr_smoothed, loss_smoothed = sm.nonparametric.lowess(
                    np.asarray(res.results["loss"])[loss_finite],
                    np.asarray(res.results["lr"])[loss_finite],
                    frac=1.0 / 10.0,
                )[10:-1].T
                # ensure loss_smoothed has minimum length required
                if len(loss_smoothed) < 3: # Fallback
                    warnings.warn("Smoothed loss has less than 3 non-finite elements, gradient can not be computed. Therefore, optimal_lr can not be found. Sampling lr from the suggested range.")
                    model.hparams.learning_rate = trial.suggest_loguniform("learning_rate", *learning_rate_range)
                else:
                    optimal_idx = np.gradient(loss_smoothed).argmin()
                    optimal_lr = lr_smoothed[optimal_idx]
                    optuna_logger.info(f"Using learning rate of {optimal_lr:.3g}")
                    # add learning rate artificially
                    model.hparams.learning_rate = trial.suggest_loguniform("learning_rate", optimal_lr, optimal_lr)
            else:
                model.hparams.learning_rate = trial.suggest_loguniform("learning_rate", *learning_rate_range)

            # fit
            trainer.fit(model, train_dataloaders=self.train_dataloader, val_dataloaders=self.val_dataloader)

            # report result
            return metrics_callback.metrics[-1]["val_loss"].item()

        # setup optuna and run
        pruner = optuna.pruners.SuccessiveHalvingPruner()
        if study is None:
            study = optuna.create_study(direction="minimize", pruner=pruner)
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        return study

class QuantileForecaster2D(ProbabilisticForecaster):
    """
    Predicts the parameters of a bivariate gaussian mixture distribution.

    The prediction region is defined by the alpha shape or the convex hull of the PDF.
    The confidence level is computed numerically by sampling from the PDF and computing the fraction of samples that lie within the prediction region.

    Output shape: (batch_size, max_prediction_length, num_mixtures * 6)
    Prediction output shape: (batch_size, max_prediction_length, 2)
    Quantile output shape: (batch_size, max_prediction_length, num_quantiles, 2)
    """
    def __init__(self, quantiles_2D=[0.5, 0.9, 0.95], epsilon=0, affine_transform=True, **kwargs):
        super().__init__(**kwargs)
        self.model_cls = TemporalFusionTransformerDistribution
        self.afine_transform = affine_transform
        self.epsilon = epsilon
        self.quantiles_2D = quantiles_2D
        if hasattr(self, 'dataset_train'):
            self.model = self.model_cls.from_dataset(
                self.dataset_train,
                loss_type='quantile',
                num_mixtures=1,
                output_size=5,
                quantiles_2D=quantiles_2D,
                epsilon=epsilon,
                affine_transform=affine_transform,
                loss=BivariateGaussianQuantileLoss(quantiles_2D=quantiles_2D, epsilon=epsilon, affine_transform=affine_transform),
                **self.model_specs,
            )


    def get_actuals(self, partition='val'):
        """
        Get actual values from validation dataset.
        """
        dataloader = getattr(self, f'{partition}_dataloader')
        setattr(self, f'y_{partition}', torch.cat([torch.stack(y, dim=-1) for _, (y, weight) in dataloader], axis=0))
        return getattr(self, f'y_{partition}')

    def get_predictions(self, partition='val'):
        """
        Get predictions from trained model.
        """
        dataloader = getattr(self, f'{partition}_dataloader')
        setattr(self, f'y_pred_{partition}', self.model.predict(dataloader))
        return getattr(self, f'y_pred_{partition}')

    def baseline(self):
        """
        Forecaster that predicts the last observed value.
        """
        self.baseline_predictions = torch.stack(Baseline().predict(self.val_dataloader), axis=-1)
        self.baseline_error = (self.y_val - self.baseline_predictions).abs().mean(axis=1).mean(axis=0) # MAE by coordinate
        return self.baseline_error

    def optimize_hyperparameters(
        self,
        model_path: str = "forecasting_dist_optuna",
        max_epochs: int = 30, # 14 min per trial
        n_trials: int = 100, # 23 hours
        timeout: float = 3600 * 17.0,  # 17 hours
        quantiles_2D: List[float] = [0.5, 0.9, 0.95],
        epsilon: float = 0,
        gradient_clip_val_range: Tuple[float, float] = (0.01, 1.0),
        hidden_size_range: Tuple[int, int] = (32, 128),
        hidden_continuous_size_range: Tuple[int, int] = (8, 64),
        attention_head_size_range: Tuple[int, int] = (1, 4),
        dropout_range: Tuple[float, float] = (0., 0.3),
        learning_rate_range: Tuple[float, float] = (1e-5, 0.1),
        use_learning_rate_finder: bool = True,
        trainer_kwargs: Dict[str, Any] = dict(limit_train_batches=60), # 0.5" per batch, 0.5 min per epoch
        log_dir: str = "lightning_logs",
        study: optuna.Study = None,
        verbose: Union[int, bool] = None,
        batch_size: int = 128,
        **dataset_kwargs,
    ) -> optuna.Study:
        """
        Optimize Temporal Fusion Transformer hyperparameters.

        Run hyperparameter optimization. Learning rate for is determined with
        the PyTorch Lightning learning rate finder.

        Args:
            train_dataloader (DataLoader): dataloader for training model
            val_dataloader (DataLoader): dataloader for validating model
            model_path (str): folder to which model checkpoints are saved
            max_epochs (int, optional): Maximum number of epochs to run training. Defaults to 20.
            n_trials (int, optional): Number of hyperparameter trials to run. Defaults to 100.
            timeout (float, optional): Time in seconds after which training is stopped regardless of number of epochs
                or validation metric. Defaults to 3600*8.0.
            hidden_size_range (Tuple[int, int], optional): Minimum and maximum of ``hidden_size`` hyperparameter. Defaults
                to (16, 265).
            hidden_continuous_size_range (Tuple[int, int], optional):  Minimum and maximum of ``hidden_continuous_size``
                hyperparameter. Defaults to (8, 64).
            attention_head_size_range (Tuple[int, int], optional):  Minimum and maximum of ``attention_head_size``
                hyperparameter. Defaults to (1, 4).
            dropout_range (Tuple[float, float], optional):  Minimum and maximum of ``dropout`` hyperparameter. Defaults to
                (0.1, 0.3).
            learning_rate_range (Tuple[float, float], optional): Learning rate range. Defaults to (1e-5, 1.0).
            use_learning_rate_finder (bool): If to use learning rate finder or optimize as part of hyperparameters.
                Defaults to True.
            trainer_kwargs (Dict[str, Any], optional): Additional arguments to the
                `PyTorch Lightning trainer <https://pytorch-lightning.readthedocs.io/en/latest/trainer.html>`_ such
                as ``limit_train_batches``. Defaults to {}.
            log_dir (str, optional): Folder into which to log results for tensorboard. Defaults to "lightning_logs".
            study (optuna.Study, optional): study to resume. Will create new study by default.
            verbose (Union[int, bool]): level of verbosity.
                * None: no change in verbosity level (equivalent to verbose=1 by optuna-set default).
                * 0 or False: log only warnings.
                * 1 or True: log pruning events.
                * 2: optuna logging level at debug level.
                Defaults to None.

            **dataset_kwargs: Additional arguments for load.load_dataset()

        Returns:
            optuna.Study: optuna study results
        """
        pl.seed_everything(self.seed)

        print("Loading dataset...")
        training, validation, test, training_predict, _ = load.load_dataset(**dataset_kwargs)
        self.dataset_train = training
        self.dataset_val = validation
        self.dataset_test = test # not to be used in hyperparameter optimization
        self.dataset_train_predict = training_predict
        self.get_dataloader(batch_size=batch_size, val_batch_mpl=4)
        print("Loaded.")

        assert isinstance(self.train_dataloader.dataset, TimeSeriesDataSet) and isinstance(
            self.val_dataloader.dataset, TimeSeriesDataSet
        ), "dataloaders must be built from timeseriesdataset"

        model_path = f"{model_path}/{self.model_cls.__name__}"

        optuna_logger = logging.getLogger("optuna")
        logging_level = {
            None: optuna.logging.get_verbosity(),
            0: optuna.logging.WARNING,
            1: optuna.logging.INFO,
            2: optuna.logging.DEBUG,
        }
        verbose = logging_level[verbose]
        optuna.logging.set_verbosity(verbose)

        # create objective function
        def objective(trial: optuna.Trial) -> float:
            # Filenames for each trial must be made unique in order to access each checkpoint.
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                os.path.join(model_path, "trial_{}".format(trial.number), "{epoch}"), monitor="val_loss"
            )

            # The default logger in PyTorch Lightning writes to event files to be consumed by
            # TensorBoard. We don't use any logger here as it requires us to implement several abstract
            # methods. Instead we setup a simple callback, that saves metrics from each validation step.
            metrics_callback = MetricsCallback()
            learning_rate_callback = LearningRateMonitor()
            logger = TensorBoardLogger(log_dir, name="optuna", version=trial.number)
            gradient_clip_val = trial.suggest_loguniform("gradient_clip_val", *gradient_clip_val_range)
            trainer_kwargs.setdefault("gpus", [0] if torch.cuda.is_available() else None)
            trainer = pl.Trainer(
                max_epochs=max_epochs,
                gradient_clip_val=gradient_clip_val,
                callbacks=[
                    metrics_callback,
                    learning_rate_callback,
                    PyTorchLightningPruningCallback(trial, monitor="val_loss"),
                    checkpoint_callback,
                ],
                logger=logger,
                **trainer_kwargs,
            )

            # create model
            hidden_size = trial.suggest_int("hidden_size", *hidden_size_range, log=True)
            loss = BivariateGaussianQuantileLoss(quantiles=quantiles_2D, epsilon=epsilon)
            model = self.model_cls.from_dataset(
                self.train_dataloader.dataset,
                mode='quantiles',
                num_mixtures=1,
                dropout=trial.suggest_uniform("dropout", *dropout_range),
                hidden_size=hidden_size,
                hidden_continuous_size=trial.suggest_int(
                    "hidden_continuous_size",
                    hidden_continuous_size_range[0],
                    min(hidden_continuous_size_range[1], hidden_size),
                    log=True,
                ),
                attention_head_size=trial.suggest_int("attention_head_size", *attention_head_size_range),
                log_interval=-1,
                loss=loss,
            )
            # find good learning rate
            if use_learning_rate_finder:
                lr_trainer = pl.Trainer(
                    gradient_clip_val=gradient_clip_val,
                    gpus=[0] if torch.cuda.is_available() else None,
                    logger=False,
                )
                res = lr_trainer.tuner.lr_find(
                    model,
                    train_dataloaders=self.train_dataloader,
                    val_dataloaders=self.val_dataloader,
                    early_stop_threshold=10000,
                    min_lr=learning_rate_range[0],
                    num_training=100,
                    max_lr=learning_rate_range[1],
                )

                loss_finite = np.isfinite(res.results["loss"])
                lr_smoothed, loss_smoothed = sm.nonparametric.lowess(
                    np.asarray(res.results["loss"])[loss_finite],
                    np.asarray(res.results["lr"])[loss_finite],
                    frac=1.0 / 10.0,
                )[10:-1].T
                # ensure loss_smoothed has minimum length required
                if len(loss_smoothed) < 3: # Fallback
                    warnings.warn("Smoothed loss has less than 3 non-finite elements, gradient can not be computed. Therefore, optimal_lr can not be found. Sampling lr from the suggested range.")
                    model.hparams.learning_rate = trial.suggest_loguniform("learning_rate", *learning_rate_range)
                else:
                    optimal_idx = np.gradient(loss_smoothed).argmin()
                    optimal_lr = lr_smoothed[optimal_idx]
                    optuna_logger.info(f"Using learning rate of {optimal_lr:.3g}")
                    # add learning rate artificially
                    model.hparams.learning_rate = trial.suggest_loguniform("learning_rate", optimal_lr, optimal_lr)
            else:
                model.hparams.learning_rate = trial.suggest_loguniform("learning_rate", *learning_rate_range)

            # fit
            trainer.fit(model, train_dataloaders=self.train_dataloader, val_dataloaders=self.val_dataloader)

            # report result
            return metrics_callback.metrics[-1]["val_loss"].item()

        # setup optuna and run
        pruner = optuna.pruners.SuccessiveHalvingPruner()
        if study is None:
            study = optuna.create_study(direction="minimize", pruner=pruner)
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        return study
