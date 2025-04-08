import numpy as np
import pandas as pd
import torch
from numba import njit, boolean
from copy import deepcopy
from pytorch_forecasting.data.timeseries import check_for_nonfinite
from pytorch_forecasting import TimeSeriesDataSet
from torch.distributions import Beta
from torch.nn.utils import rnn
from typing import Any, Callable, Dict, List, Tuple, Union
from pytorch_forecasting.data.encoders import EncoderNormalizer, GroupNormalizer, MultiNormalizer, NaNLabelEncoder
import inspect
import warnings

@njit
def find_trajectories(t, min_shift):
    """
    Parameters:
    -----------
    t: np.ndarray
        Time index
    min_shift: int
        Minimum time shift between consecutive trajectories

    Returns:
    --------
    mask: np.ndarray
        Boolean array with True values for the time steps where a new trajectory starts.
    """
    t_size = t.size
    mask = np.zeros(t_size, dtype=boolean)

    start = 0
    mask[start] = True
    dt = t[start:] - t[start] # take whole array because times can be repeated
    shifts = np.where(dt > min_shift)[0]
    if shifts.size > 0:
        shift = shifts[0] + 1
        start = start + shift
        while start < t_size:
            mask[start] = True
            dt = t[start:] - t[start]
            shifts = np.where(dt > min_shift)[0]
            if shifts.size > 0:
                shift = shifts[0] + 1
                start = start + shift
            else:
                break
    return mask

def prune_trajectories(df_ID,*, min_shift):
    """
    Prunes trajectories in the given DataFrame based on a minimum shift between consecutive trajectories.

    Parameters:
    -----------
    df_ID: DataFrame
        Input DataFrame which contains the trajectories. It must have 'time' and 'time_first' columns.
    min_shift: int
        Minimum time shift between consecutive trajectories. This ensures that every trajectory starts at least 'min_shift' time steps later than the preceding one.

    Returns:
    --------
    DataFrame
        A pruned DataFrame where each trajectory starts at least 'min_shift' time steps later than the preceding one.
    """
    df_ID = df_ID.sort_values('time')
    t = (df_ID['time'] - df_ID['time_first']).values
    mask = find_trajectories(t, min_shift)
    return df_ID[mask]


def compute_future_gap(row, *, time, max_index_end):
    if row.index_end == time.size - 1 or row.index_end == max_index_end:
        return -1
    return (time[row.index_end + 1] - time[row.index_end]) - 1

def compute_decoder_gap(row, *, time, compute_decoder_length):
    t_r = time[row.index_start: row.index_end+1].copy()
    t_r -= t_r[0]
    decoder_length = compute_decoder_length(t_r[-1], row.sequence_length)
    encoder_length = row.sequence_length - decoder_length

    encoder_last_time_idx = np.where(t_r <= encoder_length-1)[0][-1]
    decoder_first_time_idx = encoder_last_time_idx + 1
    decoder_gap = (t_r[decoder_first_time_idx] - t_r[encoder_last_time_idx]) - 1
    return decoder_gap

def prune_by_gap_length(df_ID,*, time, compute_decoder_length,  max_future_gap=1, max_decoder_gap=0):
    """
    Ensure the size of the prediction time window.
    """
    max_index_end = df_ID.index_end.iloc[-1]
    future_gap = df_ID.apply(compute_future_gap, time=time, max_index_end=max_index_end, axis=1).values
    decoder_gap = df_ID.apply(compute_decoder_gap, time=time, compute_decoder_length=compute_decoder_length, axis=1).values
    valid = ((future_gap <= max_future_gap) & (future_gap >= 0)
             & (decoder_gap <= max_decoder_gap) & (decoder_gap >= 0))
    return df_ID[valid]

@njit
def nb_test_val_start(t, max_encoder_length, max_prediction_length, max_decoder_gap=0, max_future_gap=1):
    """
    Find train_end, val_start, val_end, test_start indices for a single sequence ID.

    Ensures a fixed decoder size with max_decoder_gap and max_future_gap.
    """
    imputation_max_length = 2*max_encoder_length + max_prediction_length
    null_out = np.nan, np.nan, np.nan, np.nan
    if t[-1] < imputation_max_length: # not one full imputation sequence
        return null_out

    test_start = t[(t - (t[-1] - imputation_max_length)) >= 0][0] - 1 #

    if t[-1] - test_start < max_prediction_length+max_encoder_length: # not room for test
        return null_out

    def compute_gaps(start):
        decoder_start = start + max_encoder_length
        encoder_end_onwards = t[t >= decoder_start-1]
        if encoder_end_onwards.size < max_encoder_length:
            decoder_gap, future_gap = np.nan, np.nan
        else:
            encoder_end, decoder_start = t[t >= decoder_start-1][:2]
            decoder_gap = decoder_start - encoder_end - 1 # 0 = no gap
            future_start = decoder_start + max_prediction_length
            decoder_end, future_start = t[t >= future_start-1][:2]
            future_gap = future_start - decoder_end - 1 # 0 = no gap
        return decoder_gap, future_gap

    def find_start(start):
        decoder_gap, future_gap = compute_gaps(start)
        while decoder_gap > max_decoder_gap or future_gap > max_future_gap:
            start -= 1
            decoder_gap, future_gap = compute_gaps(start)
            if np.isnan(decoder_gap):
                return np.nan
        return start

    test_start = find_start(test_start)
    if np.isnan(test_start):
        return null_out
    t_pruned = t[t <= test_start - max_encoder_length - max_prediction_length]
    if t_pruned.size == 0:
        return null_out
    else:
        val_start = t_pruned[(t_pruned - (t_pruned[-1] - imputation_max_length - 1)) >= 0][0] + max_encoder_length - 1
        val_start = find_start(val_start)
        val_end = val_start + imputation_max_length
        train_end = val_start + max_encoder_length
    if val_start < 0 or test_start == val_start or train_end < imputation_max_length:
        return null_out
    return train_end, val_start, val_end, test_start

def test_val_start(S, **kwargs):
    """
    Find train_end, val_start, val_end, test_start indices for a single sequence ID.
    kwargs: max_encoder_length, max_prediction_length, max_decoder_gap, max_future_gap
    """
    t = S.time_idx.values
    t -= t[0]
    return nb_test_val_start(t, **kwargs)

def impute_all_index(df_ID, *, min_encoder_length, max_encoder_length, max_prediction_length, time_data):
    """
    This function sets the TimeSeriesDataset index for imputing all missing values on time series data.

    Parameters:
    df_ID (DataFrame): The input DataFrame which needs to be imputed, corresponding to a single sequence ID.
    min_encoder_length (int): The minimum length of the encoder.
    max_encoder_length (int): The maximum length of the encoder.
    max_prediction_length (int): The maximum length of the prediction.

    Returns:
    df_ID_pruned (DataFrame): The DataFrame with the index for imputing all missing values.

    The function first sorts the DataFrame based on time. It then calculates the gaps between time steps and determines the start and end indices for the encoder. It also adjusts the start and end times to account for missing positions in the time_ID. Finally, it prunes the DataFrame to ensure the encoder length is preserved and is greater than or equal to the minimum encoder length.
    """
    df_ID = df_ID.sort_values('time')
    t = (df_ID['time'] - df_ID['time_first']).values
    unique_t = np.unique(t)
    valid = unique_t >= min_encoder_length
    unique_t = unique_t[valid]

    # calculate gaps between time steps
    encoder_end = unique_t[:-1]
    time_gap = np.diff(unique_t) - 1
    gaps = (time_gap > 0) & (time_gap <= max_prediction_length)
    encoder_end = encoder_end[gaps]
    decoder_length = time_gap[gaps]

    encoder_length = np.clip(encoder_end, None, max_encoder_length)
    sequence_length = encoder_length + decoder_length
    t_index_start = df_ID.iloc[0].index_start
    t_index_end = df_ID.iloc[-1].index_end
    time_ID = time_data[t_index_start:t_index_end+1]

    # determine index start and end
    time_start = encoder_end - encoder_length + df_ID['time_first'].iloc[0]
    time_end = time_start + sequence_length
    index_end = np.searchsorted(time_ID, time_end) # accurate
    # account for the fact that starting possition may be missing from time_ID
    index_start = np.searchsorted(time_ID, time_start)

    # # assert both times are in time_ID
    # time_start = time_ID[index_start]
    # index_start = np.searchsorted(time_ID, time_start)
    # index_end = np.searchsorted(time_ID, time_end)
    # (time_data[index_start + t_index_start] == time_start).all(), (time_data[index_end + t_index_start] == time_end).all()

    dt = time_ID[index_start] - time_start
    time_start += dt
    encoder_length -= dt
    sequence_length = time_end - time_start + 1
    index_start += t_index_start
    index_end += t_index_start
    time_idx_last_observed = encoder_length + time_start

    # assert encoder length is preserved
    # index_last_observed = np.searchsorted(time_ID, time_idx_last_observed)
    # (time_data[index_last_observed + t_index_start] == time_idx_last_observed).all()

    row = df_ID.iloc[0]
    df_ID_pruned = pd.DataFrame(dict(index_start = index_start,
                                     index_end = index_end,
                                     time_first = row.time_first,
                                     time_last = row.time_last,
                                     time_idx_last_observed = time_idx_last_observed,
                                     time = time_start,
                                     count = row['count'],
                                     sequence_id = row.sequence_id,
                                     sequence_length = sequence_length,
                                     ))
    df_ID_pruned = df_ID_pruned[encoder_length >= min_encoder_length]
    return df_ID_pruned

class CustomDataset(TimeSeriesDataSet):
    """
    Add _getitem_old method to include information of missing steps (encoder & decoder).
    """
    def __init__(self, *args, store_missing_idxs=False, predict_shift=None, dive_data=False, **kwargs):
        self.store_missing_idxs = store_missing_idxs
        self.predict_shift = predict_shift
        self.dive_data = dive_data
        super().__init__(*args, **kwargs)

    def _getitem_old(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get sample for model. Includes information of missing steps (encoder & decoder).

        Args:
            idx (int): index of prediction (between ``0`` and ``len(dataset) - 1``)

        Returns:
            Tuple[Dict[str, torch.Tensor], torch.Tensor]: x and y for model
        """
        index = self.index.iloc[idx]
        # get index data
        data_cont = self.data["reals"][index.index_start : index.index_end + 1].clone()
        data_cat = self.data["categoricals"][index.index_start : index.index_end + 1].clone()
        time = self.data["time"][index.index_start : index.index_end + 1].clone()
        target = [d[index.index_start : index.index_end + 1].clone() for d in self.data["target"]]
        groups = self.data["groups"][index.index_start].clone()
        if self.data["weight"] is None:
            weight = None
        else:
            weight = self.data["weight"][index.index_start : index.index_end + 1].clone()
        # get target scale in the form of a list
        target_scale = self.target_normalizer.get_parameters(groups, self.group_ids)
        if not isinstance(self.target_normalizer, MultiNormalizer):
            target_scale = [target_scale]

        # fill in missing values (if not all time indices are specified
        sequence_length = len(time)
        if sequence_length < index.sequence_length:
            assert self.allow_missing_timesteps, "allow_missing_timesteps should be True if sequences have gaps"
            repetitions = torch.cat([time[1:] - time[:-1], torch.ones(1, dtype=time.dtype)])
            indices = torch.repeat_interleave(torch.arange(len(time)), repetitions)
            repetition_indices = torch.cat([torch.tensor([False], dtype=torch.bool), indices[1:] == indices[:-1]])

            # select data
            data_cat = data_cat[indices]
            data_cont = data_cont[indices]
            target = [d[indices] for d in target]
            if weight is not None:
                weight = weight[indices]

            # reset index
            if self.time_idx in self.reals:
                time_idx = self.reals.index(self.time_idx)
                data_cont[:, time_idx] = torch.linspace(
                    data_cont[0, time_idx], data_cont[-1, time_idx], len(target[0]), dtype=data_cont.dtype
                )

            # make replacements to fill in categories
            for name, value in self.encoded_constant_fill_strategy.items():
                if name in self.reals:
                    data_cont[repetition_indices, self.reals.index(name)] = value
                elif name in [f"__target__{target_name}" for target_name in self.target_names]:
                    target_pos = self.target_names.index(name[len("__target__") :])
                    target[target_pos][repetition_indices] = value
                elif name in self.flat_categoricals:
                    data_cat[repetition_indices, self.flat_categoricals.index(name)] = value
                elif name in self.target_names:  # target is just not an input value
                    pass
                else:
                    raise KeyError(f"Variable {name} is not known and thus cannot be filled in")

            sequence_length = len(target[0])
        else:
            repetition_indices = torch.zeros(sequence_length, dtype=torch.bool)

        # determine data window
        assert (
            sequence_length >= self.min_prediction_length
        ), "Sequence length should be at least minimum prediction length"
        # determine prediction/decode length and encode length
        if self.predict_mode == 'all':
            encoder_length = index.time_idx_last_observed - index.time + 1
            decoder_length = index.sequence_length - encoder_length
            # ensure max_encoder_length
            start_idx = max(encoder_length - self.max_encoder_length , 0)
            encoder_length -= start_idx
            sequence_length = encoder_length + decoder_length
            end_idx = start_idx + sequence_length
            data_cat = data_cat[start_idx:end_idx]
            data_cont = data_cont[start_idx:end_idx]
            target = [t[start_idx:end_idx] for t in target]
        else:
            decoder_length = self.calculate_decoder_length(time[-1], sequence_length)
            encoder_length = sequence_length - decoder_length
        assert (
            decoder_length >= self.min_prediction_length
        ), "Decoder length should be at least minimum prediction length"
        assert encoder_length >= self.min_encoder_length, "Encoder length should be at least minimum encoder length"

        if self.randomize_length is not None:  # randomization improves generalization
            # modify encode and decode lengths
            modifiable_encoder_length = encoder_length - self.min_encoder_length
            encoder_length_probability = Beta(self.randomize_length[0], self.randomize_length[1]).sample()

            # subsample a new/smaller encode length
            new_encoder_length = self.min_encoder_length + int(
                (modifiable_encoder_length * encoder_length_probability).round()
            )

            # extend decode length if possible
            new_decoder_length = min(decoder_length + (encoder_length - new_encoder_length), self.max_prediction_length)

            # select subset of sequence of new sequence
            if new_encoder_length + new_decoder_length < len(target[0]):
                data_cat = data_cat[encoder_length - new_encoder_length : encoder_length + new_decoder_length]
                data_cont = data_cont[encoder_length - new_encoder_length : encoder_length + new_decoder_length]
                target = [t[encoder_length - new_encoder_length : encoder_length + new_decoder_length] for t in target]
                repetition_indices = repetition_indices[encoder_length - new_encoder_length : encoder_length + new_decoder_length]
                encoder_length = new_encoder_length
                decoder_length = new_decoder_length

            # switch some variables to nan if encode length is 0
            if encoder_length == 0 and len(self.dropout_categoricals) > 0:
                data_cat[
                    :, [self.flat_categoricals.index(c) for c in self.dropout_categoricals]
                ] = 0  # zero is encoded nan

        assert decoder_length > 0, "Decoder length should be greater than 0"
        assert encoder_length >= 0, "Encoder length should be at least 0"

        if self.add_relative_time_idx:
            data_cont[:, self.reals.index("relative_time_idx")] = (
                torch.arange(-encoder_length, decoder_length, dtype=data_cont.dtype) / self.max_encoder_length
            )

        if self.add_encoder_length:
            data_cont[:, self.reals.index("encoder_length")] = (
                (encoder_length - 0.5 * self.max_encoder_length) / self.max_encoder_length * 2.0
            )

        # rescale target
        for idx, target_normalizer in enumerate(self.target_normalizers):
            if isinstance(target_normalizer, EncoderNormalizer):
                target_name = self.target_names[idx]
                # fit and transform
                target_normalizer.fit(target[idx][:encoder_length])
                # get new scale
                single_target_scale = target_normalizer.get_parameters()
                # modify input data
                if target_name in self.reals:
                    data_cont[:, self.reals.index(target_name)] = target_normalizer.transform(target[idx])
                if self.add_target_scales:
                    data_cont[:, self.reals.index(f"{target_name}_center")] = self.transform_values(
                        f"{target_name}_center", single_target_scale[0]
                    )[0]
                    data_cont[:, self.reals.index(f"{target_name}_scale")] = self.transform_values(
                        f"{target_name}_scale", single_target_scale[1]
                    )[0]
                # scale needs to be numpy to be consistent with GroupNormalizer
                target_scale[idx] = single_target_scale.numpy()

        # rescale covariates
        for name in self.reals:
            if name not in self.target_names and name not in self.lagged_variables:
                normalizer = self.get_transformer(name)
                if isinstance(normalizer, EncoderNormalizer):
                    # fit and transform
                    pos = self.reals.index(name)
                    normalizer.fit(data_cont[:encoder_length, pos])
                    # transform
                    data_cont[:, pos] = normalizer.transform(data_cont[:, pos])

        # also normalize lagged variables
        for name in self.reals:
            if name in self.lagged_variables:
                normalizer = self.get_transformer(name)
                if isinstance(normalizer, EncoderNormalizer):
                    pos = self.reals.index(name)
                    data_cont[:, pos] = normalizer.transform(data_cont[:, pos])

        # overwrite values
        if self._overwrite_values is not None:
            if isinstance(self._overwrite_values["target"], slice):
                positions = self._overwrite_values["target"]
            elif self._overwrite_values["target"] == "all":
                positions = slice(None)
            elif self._overwrite_values["target"] == "encoder":
                positions = slice(None, encoder_length)
            else:  # decoder
                positions = slice(encoder_length, None)

            if self._overwrite_values["variable"] in self.reals:
                idx = self.reals.index(self._overwrite_values["variable"])
                data_cont[positions, idx] = self._overwrite_values["values"]
            else:
                assert (
                    self._overwrite_values["variable"] in self.flat_categoricals
                ), "overwrite values variable has to be either in real or categorical variables"
                idx = self.flat_categoricals.index(self._overwrite_values["variable"])
                data_cat[positions, idx] = self._overwrite_values["values"]

        # weight is only required for decoder
        if weight is not None:
            weight = weight[encoder_length:]

        # if user defined target as list, output should be list, otherwise tensor
        if self.multi_target:
            encoder_target = [t[:encoder_length] for t in target]
            target = [t[encoder_length:] for t in target]
        else:
            encoder_target = target[0][:encoder_length]
            target = target[0][encoder_length:]
            target_scale = target_scale[0]

        return (
            dict(
                x_cat=data_cat,
                x_cont=data_cont,
                encoder_length=encoder_length,
                decoder_length=decoder_length,
                encoder_target=encoder_target,
                encoder_time_idx_start=time[0],
                groups=groups,
                target_scale=target_scale,
                encoder_missing=repetition_indices[:encoder_length] if self.store_missing_idxs else None,
                decoder_missing=repetition_indices[encoder_length:] if self.store_missing_idxs else None,
            ),
            (target, weight),
        )
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        return self._getitem_old(idx) # Override __getitem__ method if needed

class ImputationDataset(CustomDataset):
    """
    Extension of TimeSeriesDataSet to include future variables and imputation of missing values.
    """
    def __init__(self, *args, min_encoder_length=56, min_future_length=None, min_future_obs=None,
                 fix_training_prediction_length=False, max_future_gap=1, max_decoder_gap=0, expand_encoder_until_future_length=True,
                 reverse_future=False,
                 **kwargs):
        if min_future_length is None:
            self.min_future_length = min_encoder_length
        else:
            self.min_future_length = min_future_length
        if min_future_obs is None:
            self.min_future_obs = self.min_future_length // 2
        else:
            self.min_future_obs = min_future_obs
        self.fix_training_prediction_length = fix_training_prediction_length
        self.max_future_gap = max_future_gap
        self.max_decoder_gap = max_decoder_gap
        self.expand_encoder_until_future_length = expand_encoder_until_future_length
        self.reverse_future = reverse_future
        super().__init__(*args, min_encoder_length=min_encoder_length, **kwargs)

        self.static_var_types = ['static_categoricals', 'static_reals']
        self.varying_var_types = ['time_varying_known_categoricals', 'time_varying_known_reals',
                             'time_varying_unknown_categoricals', 'time_varying_unknown_reals']
        self.var_types = self.static_var_types + self.varying_var_types
        self.joint_var_types = ['reals', 'categoricals']

        parse_future = lambda l: [f'future_{v}' for v in l]
        self.future_static_reals = ['future_encoder_length'] if self.add_encoder_length else []
        self.future_static_categoricals = []
        for v in self.varying_var_types + self.joint_var_types: # future vars: v -> future_v
            if v == 'categoricals':
                self.future_categoricals = self.future_time_varying_known_categoricals + self.future_time_varying_unknown_categoricals
            else:
                setattr(self, f'future_{v}', parse_future(getattr(self, v)))
        for v in self.var_types: # aggregate: v + future_v
            setattr(self, f'{v}_with_future', getattr(self, v) + getattr(self, f'future_{v}'))

        self.static_variables = (self.static_categoricals_with_future
                                 + self.static_reals_with_future)
        self.decoder_variables = (self.time_varying_known_categoricals # exclude future variables (decoder = imputation window)
                                  + self.time_varying_known_reals)
        self.encoder_variables = (  self.time_varying_known_categoricals_with_future
                                  + self.time_varying_known_reals_with_future
                                  + self.time_varying_unknown_categoricals_with_future
                                  + self.time_varying_unknown_reals_with_future)

        self.reals_with_future = (self.static_reals_with_future
                                  + self.time_varying_known_reals_with_future
                                  + self.time_varying_unknown_reals_with_future)
        self.categoricals_with_future = (self.static_categoricals_with_future
                                         + self.time_varying_known_categoricals_with_future
                                         + self.time_varying_unknown_categoricals_with_future)

        # Get indices to use when joining past + future features in __getitem__
        reals_joint = self.reals + self.future_reals
        categoricals_joint = self.categoricals + self.static_categoricals + self.future_categoricals
        self.categorical_order = [categoricals_joint.index(c) for c in self.categoricals_with_future]
        self.reals_order = [reals_joint.index(r) for r in self.reals_with_future]

        idx_encoder_reals = []
        idx_future_reals = []
        for i, r in enumerate(self.reals_with_future):
            if r.startswith('future_'):
                idx_future_reals.append(i)
            else:
                idx_encoder_reals.append(i)

        idx_encoder_categoricals = []
        idx_future_categoricals = []
        for i, c in enumerate(self.categoricals_with_future):
            if c.startswith('future_'):
                idx_future_categoricals.append(i)
            else:
                idx_encoder_categoricals.append(i)

        self.idx_encoder_reals = idx_encoder_reals
        self.idx_future_reals = idx_future_reals
        self.idx_encoder_categoricals = idx_encoder_categoricals
        self.idx_future_categoricals = idx_future_categoricals
        self.idx_decoder_reals = [self.reals_with_future.index(r) for r in self.decoder_variables if r in self.reals_with_future]
        self.idx_decoder_categoricals = [self.categoricals_with_future.index(c) for c in self.decoder_variables if c in self.categoricals_with_future]

    @classmethod
    def from_dataset(cls, dataset, data: pd.DataFrame, stop_randomization: bool = False, predict: Union[bool, str] = False, **update_kwargs):
        """
        Generate dataset with different underlying data but same variable encoders and scalers, etc.

        Calls :py:meth:`~from_parameters` under the hood.

        Args:
            dataset (TimeSeriesDataSet): dataset from which to copy parameters
            data (pd.DataFrame): data from which new dataset will be generated
            stop_randomization (bool, optional): If to stop randomizing encoder and decoder lengths,
                e.g. useful for validation set. Defaults to False.
            predict (bool|str): True or 'last': predict the decoder length on the last entries in the
                time index (i.e. one prediction per group only).
                'all': impute all missing values.
                Defaults to False.
            **kwargs: keyword arguments overriding parameters in the original dataset

        Returns:
            TimeSeriesDataSet: new dataset
        """
        return cls.from_parameters(
            dataset.get_parameters(with_future=False), data, stop_randomization=stop_randomization, predict=predict, **update_kwargs
        )

    @classmethod
    def from_parameters(
        cls,
        parameters: Dict[str, Any],
        data: pd.DataFrame,
        stop_randomization: bool = None,
        predict: Union[bool, str] = False,
        **update_kwargs,
    ):
        """
        Generate dataset with different underlying data but same variable encoders and scalers, etc.

        Args:
            parameters (Dict[str, Any]): dataset parameters which to use for the new dataset
            data (pd.DataFrame): data from which new dataset will be generated
            stop_randomization (bool, optional): If to stop randomizing encoder and decoder lengths,
                e.g. useful for validation set. Defaults to False.
            predict (bool, optional): If to predict the decoder length on the last entries in the
                time index (i.e. one prediction per group only). Defaults to False.
            **kwargs: keyword arguments overriding parameters

        Returns:
            TimeSeriesDataSet: new dataset
        """
        parameters = deepcopy(parameters)
        if predict:
            parameters["predict_mode"] = predict
            if stop_randomization is None:
                stop_randomization = True
            elif not stop_randomization:
                warnings.warn(
                    "If predicting, no randomization should be possible - setting stop_randomization=True", UserWarning
                )
                stop_randomization = True
            if predict == 'all':
                parameters['min_prediction_length'] = 1
                parameters['min_encoder_length'] = 1
                parameters['min_future_length'] = 1
            else:
                parameters["min_prediction_length"] = parameters["max_prediction_length"]
        elif stop_randomization is None:
            stop_randomization = False

        if stop_randomization:
            parameters["randomize_length"] = None
        parameters.update(update_kwargs)

        new = cls(data, **parameters)
        return new

    @property
    def flat_categoricals_with_future(self) -> List[str]:
        """
        Categorical variables as defined in input data.

        Returns:
            List[str]: list of variables
        """
        categories = self.flat_categoricals.copy()
        for name in self.future_categoricals:
            if name in self.variable_groups:
                categories.extend(self.variable_groups[name])
            else:
                categories.append(name)
        return categories

    @property
    def categorical_encoders_with_future(self):
        encoders = self.categorical_encoders.copy()
        for f in self.future_categoricals:
            encoders[f] = deepcopy(self.categorical_encoders[f.replace("future_", "")])
        return encoders

    @property
    def dropout_categoricals_with_future(self) -> List[str]:
        """
        list of categorical variables that are unknown when making a
        forecast without observed history
        """
        return [name for name, encoder in self.categorical_encoders_with_future.items() if encoder.add_nan]

    def get_parameters(self, with_future=True) -> Dict[str, Any]:
        """
        Get parameters that can be used with :py:meth:`~from_parameters` to create a new dataset with the same scalers.

        Returns:
            Dict[str, Any]: dictionary of parameters
        """
        parent_cls = TimeSeriesDataSet
        exclude = ["data", "self"]
        variables = self.var_types + self.joint_var_types

        kwargs = {}
        for name in inspect.signature(parent_cls.__init__).parameters.keys():
            if name not in exclude:
                if name in variables and with_future:
                    kwargs[name] = getattr(self, f'{name}_with_future')
                else:
                    kwargs[name] = getattr(self, name)

        kwargs["categorical_encoders"] = self.categorical_encoders
        kwargs["scalers"] = self.scalers
        kwargs['min_future_length'] = self.min_future_length
        kwargs['min_future_obs'] = self.min_future_obs
        kwargs['fix_training_prediction_length'] = self.fix_training_prediction_length
        kwargs['max_future_gap'] = self.max_future_gap
        kwargs['max_decoder_gap'] = self.max_decoder_gap
        kwargs['expand_encoder_until_future_length'] = self.expand_encoder_until_future_length
        kwargs['store_missing_idxs'] = self.store_missing_idxs
        kwargs['predict_shift'] = self.predict_shift
        kwargs['reverse_future'] = self.reverse_future
        return kwargs

    def _construct_index(self, data: pd.DataFrame, predict_mode: Union[bool, str]) -> pd.DataFrame:
        index = super()._construct_index(data, predict_mode=False)
        self.index_raw = index.copy()
        apply_to_IDs = lambda index, func, **kwargs: index.groupby('sequence_id', group_keys=False).apply(func, **kwargs).reset_index(drop=True)
        time = check_for_nonfinite(
            torch.tensor(data["__time_idx__"].to_numpy(np.int64), dtype=torch.int64), self.time_idx
        ).numpy()

        if predict_mode == 'all':
            return apply_to_IDs(index, impute_all_index, time_data=time,
                                min_encoder_length=self.min_encoder_length, max_encoder_length=self.max_encoder_length, max_prediction_length=self.max_prediction_length)
        else:
            max_future_length = self.max_encoder_length

            @njit
            def num_future_observations(index_end):
                future_start = index_end + 1
                t0 = time[index_end] + 1
                tf_candidates = time[future_start: future_start+max_future_length]
                gap = np.where(np.diff(tf_candidates - t0) < 0)[0]
                if gap.size > 0:
                    tf_candidates = tf_candidates[:gap[0]+1]
                valid = (tf_candidates - t0) <= (max_future_length - 1) # best = tf_candidates[valid][-1]
                return valid.sum()

            def valid_index(df_ID):
                """
                Leave space for encoder of future variables.

                Drop items where time + max_encoder_length + max_prediction_length + max_encoder_length(future) - 1 > time_last (-1 since time is 0-indexed)
                 """
                enough_time_length = df_ID['time'] + 2*self.max_encoder_length + self.max_prediction_length - 1 <= df_ID['time_last']
                n_future_obs = df_ID['index_end'].apply(num_future_observations)
                enough_future_observations = n_future_obs >= self.min_future_obs
                valid = enough_time_length & enough_future_observations
                return df_ID[valid]

            index = apply_to_IDs(index, valid_index)

            if self.fix_training_prediction_length:
                print("Ensuring fixed prediction length")
                index = apply_to_IDs(index, prune_by_gap_length, time=time,
                                     compute_decoder_length=self.calculate_decoder_length,  max_future_gap=self.max_future_gap, max_decoder_gap=self.max_decoder_gap)

            if predict_mode:
                if predict_mode == 'non-overlapping' or isinstance(predict_mode, bool):
                    if self.predict_shift is None:
                        min_shift = self.max_encoder_length + self.max_prediction_length # max_sequence_length
                    else:
                        min_shift = self.predict_shift
                    index = apply_to_IDs(index, prune_trajectories, min_shift=min_shift)
                elif predict_mode == 'encoder-overlap':
                    min_shift = self.max_prediction_length
                    index = apply_to_IDs(index, prune_trajectories, min_shift=min_shift)
                else:
                    raise ValueError("predict_mode should be a boolean, 'non-overlapping', 'encoder-overlap' or 'all'")
            return index


    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        index = self.index.iloc[idx]
        x, (decoder_target, weight_orig) = self._getitem_old(idx)
        encoder_length = x['encoder_length']
        decoder_length = x['decoder_length']
        orig_sequence_length = encoder_length + decoder_length

        decoder_end = index.index_end # start at last decoder point (later will be removed)
        max_future_length = self.max_encoder_length
        t_decoder_end = self.data['time'][decoder_end]
        tf_candidates = self.data['time'][decoder_end: decoder_end+max_future_length]
        gap = np.where(np.diff(tf_candidates - t_decoder_end) < 0)[0]
        if gap.size > 0:
            tf_candidates = tf_candidates[:gap[0]+1]
        valid = (tf_candidates - t_decoder_end) < max_future_length
        future_end = decoder_end + valid.sum().item()
        # ensure future idx from same ID
        future_groups = self.data['groups'][decoder_end: future_end+1]
        future_end -= (future_groups != future_groups[0].item()).sum().item()

        # Future data
        data_cont = self.data["reals"][decoder_end : future_end+1].clone()
        data_cat = self.data["categoricals"][decoder_end : future_end+1].clone()
        time = self.data["time"][decoder_end : future_end+1].clone()
        target = [d[decoder_end : future_end+1].clone() for d in self.data["target"]]
        groups = self.data["groups"][decoder_end].clone()
        assert groups.item() == x['groups'].item()

        # get target scale in the form of a list
        target_scale = self.target_normalizer.get_parameters(groups, self.group_ids)
        if not isinstance(self.target_normalizer, MultiNormalizer):
            target_scale = [target_scale]

        def expand_sequences(time, data_cat, data_cont, target):
            assert self.allow_missing_timesteps, "allow_missing_timesteps should be True if sequences have gaps"
            max_time = time[0] + self.max_encoder_length
            time_clipped = torch.clip(time, 0, max_time)
            repetitions = torch.cat([time_clipped[1:] - time_clipped[:-1], torch.ones(1, dtype=time_clipped.dtype)])
            indices = torch.repeat_interleave(torch.arange(len(time_clipped)), repetitions)
            repetition_indices = torch.cat([torch.tensor([False], dtype=torch.bool), indices[1:] == indices[:-1]])
            if time[-1] > max_time:
                indices = indices[:-1]
                repetition_indices = repetition_indices[:-1]
            indices = indices[1:] # skip decoder last point
            indices[indices == 0] = 1 # backfill first observation from future
            repetition_indices = repetition_indices[1:]

            # select data
            time = time[indices]
            data_cat = data_cat[indices]
            data_cont = data_cont[indices]
            target = [d[indices] for d in target]

            # reset index
            if self.time_idx in self.reals:
                time_idx = self.reals.index(self.time_idx)
                data_cont[:, time_idx] = torch.linspace(
                    data_cont[0, time_idx], data_cont[-1, time_idx], len(target[0]), dtype=data_cont.dtype
                )

            # make replacements to fill in categories
            for name, value in self.encoded_constant_fill_strategy.items():
                if name in self.reals:
                    data_cont[repetition_indices, self.reals.index(name)] = value
                elif name in [f"__target__{target_name}" for target_name in self.target_names]:
                    target_pos = self.target_names.index(name[len("__target__") :])
                    target[target_pos][repetition_indices] = value
                elif name in self.flat_categoricals:
                    data_cat[repetition_indices, self.flat_categoricals.index(name)] = value
                elif name in self.target_names:  # target is just not an input value
                    pass
                else:
                    raise KeyError(f"Variable {name} is not known and thus cannot be filled in")


            future_length = len(target[0])
            assert future_length >= self.min_future_length, "Encoder future length should be at least min_future_length"
            assert future_length <= self.max_encoder_length, "Encoder future length should be at most maximum encoder length"
            return future_length, data_cat, data_cont, target, time, repetition_indices

        # fill in missing values (if not all time indices are specified
        future_length = len(time)
        if future_length < max_future_length + 1:
            future_length, data_cat, data_cont, target, time, repetition_indices = expand_sequences(time, data_cat, data_cont, target)
        else:
            future_length = max_future_length
            data_cat = data_cat[1:] # skip decoder last point
            data_cont = data_cont[1:]
            target = [t[1:] for t in target]
            time = time[1:]
            repetition_indices = torch.zeros(future_length, dtype=torch.bool)

        if self.randomize_length is not None:  # randomization improves generalization
            # modify future length
            modifiable_future_length = future_length - self.min_future_length
            future_length_probability = Beta(self.randomize_length[0], self.randomize_length[1]).sample()

            # subsample a new/smaller encode length
            new_future_length = self.min_future_length + int(
                (modifiable_future_length * future_length_probability).round()
            )

            # select subset of sequence of new sequence
            if new_future_length < len(target[0]):
                time = time[:new_future_length]
                data_cat = data_cat[:new_future_length]
                data_cont = data_cont[:new_future_length]
                target = [t[:new_future_length] for t in target]
                future_length = new_future_length

        if self.add_relative_time_idx:
            pos = self.reals.index("relative_time_idx")
            # rel_time_init = time[0] - t_decoder_end # possible gap between decoder and future
            # rel_time = torch.arange(rel_time_init, rel_time_init+future_length)
            # data_cont[:, self.reals.index("relative_time_idx")] = (decoder_length + rel_time - 1) / self.max_encoder_length
            data_cont[:, self.reals.index("relative_time_idx")] = torch.arange(decoder_length, decoder_length+future_length) / self.max_encoder_length

        if self.add_encoder_length:
            data_cont[:, self.reals.index("encoder_length")] = (
                (future_length - 0.5 * self.max_encoder_length) / self.max_encoder_length * 2.0
            )

        # rescale target
        for idx, target_normalizer in enumerate(self.target_normalizers):
            if isinstance(target_normalizer, EncoderNormalizer):
                target_name = self.target_names[idx]
                if target_name in self.reals:
                    pos = self.reals.index(target_name)
                    x_orig = target_normalizer.inverse_transform(x['x_cont'][:, pos].clone())
                else:
                    if isinstance(decoder_target, list):
                        x_orig = decoder_target[idx].clone()
                    else:
                        x_orig = decoder_target.clone()
                if isinstance(target, list):
                    x_future = target[idx].clone()
                else:
                    x_future = target.clone()
                # fit and transform
                target_normalizer.fit(torch.cat((x_orig[:encoder_length], x_future)))
                # get new scale
                single_target_scale = target_normalizer.get_parameters()
                if target_name in self.reals:
                    data_cont[:, pos] = target_normalizer.transform(x_future)
                    x['x_cont'][:, pos] = target_normalizer.transform(x_orig)
                if self.add_target_scales:
                    x['x_cont'][:, self.reals.index(f"{target_name}_center")] = self.transform_values(
                        f"{target_name}_center", single_target_scale[0]
                    )[0]
                    x['x_cont'][:, self.reals.index(f"{target_name}_scale")] = self.transform_values(
                        f"{target_name}_scale", single_target_scale[1]
                    )[0]
                # scale needs to be numpy to be consistent with GroupNormalizer
                target_scale[idx] = single_target_scale.numpy()

        # rescale covariates
        for name in self.reals:
            if name not in self.target_names and name not in self.lagged_variables:
                normalizer = self.get_transformer(name)
                if isinstance(normalizer, EncoderNormalizer):
                    # fit and transform
                    pos = self.reals.index(name)
                    x_orig = normalizer.inverse_transform(x['x_cont'][:, pos].clone())
                    x_future = data_cont[:, pos].clone()
                    normalizer.fit(torch.cat((x_orig[:encoder_length], x_future)))
                    # transform
                    data_cont[:, pos] = normalizer.transform(x_future)
                    x['x_cont'][:, pos] = normalizer.transform(x_orig)

        # also normalize lagged variables
        for name in self.reals:
            if name in self.lagged_variables:
                normalizer = self.get_transformer(name)
                if isinstance(normalizer, EncoderNormalizer):
                    pos = self.reals.index(name)
                    data_cont[:, pos] = normalizer.transform(data_cont[:, pos])

        # overwrite values
        if self._overwrite_values is not None:
            if isinstance(self._overwrite_values["target"], slice):
                positions = self._overwrite_values["target"]
            elif self._overwrite_values["target"] == "all":
                positions = slice(None)
            elif self._overwrite_values["target"] == "encoder":
                positions = slice(None, future_length)
            else:  # decoder
                positions = slice(future_length, None)

            if self._overwrite_values["variable"] in self.reals:
                idx = self.reals.index(self._overwrite_values["variable"])
                data_cont[positions, idx] = self._overwrite_values["values"]
            else:
                assert (
                    self._overwrite_values["variable"] in self.flat_categoricals
                ), "overwrite values variable has to be either in real or categorical variables"
                idx = self.flat_categoricals.index(self._overwrite_values["variable"])
                data_cat[positions, idx] = self._overwrite_values["values"]

        # if user defined target as list, output should be list, otherwise tensor
        if not self.multi_target:
            target_scale = target_scale[0]

        if future_length > orig_sequence_length and (self.expand_encoder_until_future_length or self.predict_mode == 'all'):
            # repeat values of encoder until orig_sequence_length = future_length
            def fill_orig_data(data_type):
                encoder_data = x[data_type][:encoder_length]
                decoder_data = x[data_type][encoder_length:]
                encoder_data = torch.cat((encoder_data,
                                         encoder_data[-1:].repeat(future_length - orig_sequence_length, 1)),
                                         dim=0)
                data_orig = torch.cat((encoder_data, decoder_data), dim=0)
                return data_orig
            data_orig_cat = fill_orig_data('x_cat')
            data_orig_cont = fill_orig_data('x_cont')
            orig_sequence_length = future_length
            # Possible length diagrams: if future > encoder
            # __encoder_fill_decoder__
            # _______future___________
            # If encoder > future
            # __encoder__decoder__
            #__future_fill________
            # In both cases decoder occupies the last max_decoder_length points
        else:
            data_orig_cat = x['x_cat']
            data_orig_cont = x['x_cont']

        def fill_data(z):
            """fill data with last observation until same size as orig_sequence_length"""
            length_diff = orig_sequence_length - future_length
            if length_diff > 0:
                return torch.cat((z,
                                  z[-1].unsqueeze(0).repeat(orig_sequence_length - future_length, 1)), axis=0)
            elif length_diff < 0:
                return z[:orig_sequence_length]
            else:
                return z

        # if self.reverse_future: # Changed for back and forward passes in TFTImputation
        #     data_cat = torch.flip(data_cat, [0])
        #     data_cont = torch.flip(data_cont, [0])
        #     repetition_indices = torch.flip(repetition_indices, [0])

        data_cat_full = torch.cat((data_orig_cat, fill_data(data_cat)), axis=1)[:, self.categorical_order]
        data_cont_full = torch.cat((data_orig_cont, fill_data(data_cont)), axis=1)[:, self.reals_order]

        return (
            dict(
                x_cat=data_cat_full,
                x_cont=data_cont_full,
                encoder_length=encoder_length,
                decoder_length=decoder_length,
                future_length=future_length,
                encoder_target=x['encoder_target'],
                encoder_time_idx_start=x['encoder_time_idx_start'],
                groups=groups,
                target_scale=target_scale,
                encoder_missing=x['encoder_missing'],
                decoder_missing=x['decoder_missing'],
                future_missing=repetition_indices if self.store_missing_idxs else None,
            ),
            (decoder_target, weight_orig),
        )

    @staticmethod
    def _collate_fn(
        batches: List[Tuple[Dict[str, torch.Tensor], torch.Tensor]]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Collate function to combine items into mini-batch for dataloader.

        Args:
            batches (List[Tuple[Dict[str, torch.Tensor], torch.Tensor]]): List of samples generated with
                :py:meth:`~__getitem__`.

        Returns:
            Tuple[Dict[str, torch.Tensor], Tuple[Union[torch.Tensor, List[torch.Tensor]], torch.Tensor]: minibatch
        """
        # collate function for dataloader
        # lengths
        encoder_lengths = torch.tensor([batch[0]["encoder_length"] for batch in batches], dtype=torch.long)
        decoder_lengths = torch.tensor([batch[0]["decoder_length"] for batch in batches], dtype=torch.long)
        future_lengths = torch.tensor([batch[0]["future_length"] for batch in batches], dtype=torch.long)

        # ids
        decoder_time_idx_start = (
            torch.tensor([batch[0]["encoder_time_idx_start"] for batch in batches], dtype=torch.long) + encoder_lengths
        )
        decoder_time_idx = decoder_time_idx_start.unsqueeze(1) + torch.arange(decoder_lengths.max()).unsqueeze(0)
        groups = torch.stack([batch[0]["groups"] for batch in batches])

        encoder_future_max_lengths = torch.fmax(encoder_lengths, future_lengths) # we will take care of it while masking

        # features
        encoder_cont = rnn.pad_sequence(
            [batch[0]["x_cont"][:length] for length, batch in zip(encoder_future_max_lengths, batches)], batch_first=True
        )
        encoder_cat = rnn.pad_sequence(
            [batch[0]["x_cat"][:length] for length, batch in zip(encoder_future_max_lengths, batches)], batch_first=True
        )

        decoder_cont = rnn.pad_sequence(
            [batch[0]["x_cont"][length:] for length, batch in zip(encoder_future_max_lengths, batches)], batch_first=True
        )
        decoder_cat = rnn.pad_sequence(
            [batch[0]["x_cat"][length:] for length, batch in zip(encoder_future_max_lengths, batches)], batch_first=True
        )

        # target scale
        if isinstance(batches[0][0]["target_scale"], torch.Tensor):  # stack tensor
            target_scale = torch.stack([batch[0]["target_scale"] for batch in batches])
        elif isinstance(batches[0][0]["target_scale"], (list, tuple)):
            target_scale = []
            for idx in range(len(batches[0][0]["target_scale"])):
                if isinstance(batches[0][0]["target_scale"][idx], torch.Tensor):  # stack tensor
                    scale = torch.stack([batch[0]["target_scale"][idx] for batch in batches])
                else:
                    scale = torch.from_numpy(
                        np.array([batch[0]["target_scale"][idx] for batch in batches], dtype=np.float32),
                    )
                target_scale.append(scale)
        else:  # convert to tensor
            target_scale = torch.from_numpy(
                np.array([batch[0]["target_scale"] for batch in batches], dtype=np.float32),
            )

        # target and weight
        if isinstance(batches[0][1][0], (tuple, list)):
            target = [
                rnn.pad_sequence([batch[1][0][idx] for batch in batches], batch_first=True)
                for idx in range(len(batches[0][1][0]))
            ]
            encoder_target = [
                rnn.pad_sequence([batch[0]["encoder_target"][idx] for batch in batches], batch_first=True)
                for idx in range(len(batches[0][1][0]))
            ]
        else:
            target = rnn.pad_sequence([batch[1][0] for batch in batches], batch_first=True)
            encoder_target = rnn.pad_sequence([batch[0]["encoder_target"] for batch in batches], batch_first=True)

        if batches[0][1][1] is not None:
            weight = rnn.pad_sequence([batch[1][1] for batch in batches], batch_first=True)
        else:
            weight = None

        if batches[0][0]["encoder_missing"] is not None: # add missing idxs
            encoder_missing = rnn.pad_sequence([batch[0]["encoder_missing"] for batch in batches], batch_first=True)
            decoder_missing = rnn.pad_sequence([batch[0]["decoder_missing"] for batch in batches], batch_first=True)
            future_missing = rnn.pad_sequence([batch[0]["future_missing"] for batch in batches], batch_first=True)
        else:
            encoder_missing = None
            decoder_missing = None
            future_missing = None

        # Due as many computations as possible on the CPU for the data loading.
        max_encoder_length = int(encoder_lengths.max().item())
        max_decoder_length = int(decoder_lengths.max().item())
        max_future_length = int(future_lengths.max().item())

        return (
            dict(
                encoder_cat=encoder_cat,
                encoder_cont=encoder_cont,
                encoder_target=encoder_target,
                encoder_lengths=encoder_lengths,
                future_lengths=future_lengths,
                encoder_future_max_lengths=encoder_future_max_lengths,
                decoder_cat=decoder_cat,
                decoder_cont=decoder_cont,
                decoder_target=target,
                decoder_lengths=decoder_lengths,
                decoder_time_idx=decoder_time_idx,
                groups=groups,
                target_scale=target_scale,
                encoder_missing=encoder_missing,
                decoder_missing=decoder_missing,
                future_missing=future_missing,
                max_encoder_length=max_encoder_length,
                max_decoder_length=max_decoder_length,
                max_future_length=max_future_length,
            ),
            (target, weight),
        )


class ForecastingDataset(CustomDataset):
    """
    Modification of _construct index to produce non-overlapping sequences for evaluation. (Multiple sequences per ID, compared to the original that selects only one)

    predict_mode: str
        'non-overlapping': Selects non-overlapping sequences for evaluation. Useful for the validation set.
        'encoder-overlap': Selects overlapping sequences for evaluation. Useful for error analysis (evaluation is performed in most of the trajectory).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _construct_index(self, data: pd.DataFrame, predict_mode: Union[bool, str]) -> pd.DataFrame:
        if predict_mode == 'original':
            index = super()._construct_index(data, predict_mode=True)
        else:
            index = super()._construct_index(data, predict_mode=False)
            apply_to_IDs = lambda index, func, **kwargs: index.groupby('sequence_id', group_keys=False).apply(func, **kwargs).reset_index(drop=True)

            if predict_mode:
                if predict_mode == 'non-overlapping' or isinstance(predict_mode, bool):
                    if self.predict_shift is None:
                        min_shift = self.max_encoder_length + self.max_prediction_length # max_sequence_length
                    else:
                        min_shift = self.predict_shift
                elif predict_mode == 'encoder-overlap':
                    min_shift = self.max_prediction_length
                elif predict_mode == 'default':
                    min_shift = 112
                else:
                    raise ValueError("predict_mode should be a boolean or 'non-overlapping' or 'encoder-overlap'")
                index = apply_to_IDs(index, prune_trajectories, min_shift=min_shift)
        return index

    @classmethod
    def from_parameters(
        cls,
        parameters: Dict[str, Any],
        data: pd.DataFrame,
        stop_randomization: bool = None,
        predict: Union[bool, str] = False,
        **update_kwargs,
    ):
        """
        Generate dataset with different underlying data but same variable encoders and scalers, etc.

        Args:
            parameters (Dict[str, Any]): dataset parameters which to use for the new dataset
            data (pd.DataFrame): data from which new dataset will be generated
            stop_randomization (bool, optional): If to stop randomizing encoder and decoder lengths,
                e.g. useful for validation set. Defaults to False.
            predict (bool, optional): If to predict the decoder length on the last entries in the
                time index (i.e. one prediction per group only). Defaults to False.
            **kwargs: keyword arguments overriding parameters

        Returns:
            TimeSeriesDataSet: new dataset
        """
        parameters = deepcopy(parameters)
        if predict:
            if stop_randomization is None:
                stop_randomization = True
            elif not stop_randomization:
                warnings.warn(
                    "If predicting, no randomization should be possible - setting stop_randomization=True", UserWarning
                )
                stop_randomization = True
            parameters["min_prediction_length"] = parameters["max_prediction_length"]
            parameters["predict_mode"] = predict
        elif stop_randomization is None:
            stop_randomization = False

        if stop_randomization:
            parameters["randomize_length"] = None
        parameters.update(update_kwargs)

        new = cls(data, **parameters)
        return new

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get parameters that can be used with :py:meth:`~from_parameters` to create a new dataset with the same scalers.

        Returns:
            Dict[str, Any]: dictionary of parameters
        """
        parent_cls = TimeSeriesDataSet
        exclude = ["data", "self"]

        kwargs = {}
        for name in inspect.signature(parent_cls.__init__).parameters.keys():
            if name not in exclude:
                kwargs[name] = getattr(self, name)

        kwargs["categorical_encoders"] = self.categorical_encoders
        kwargs["scalers"] = self.scalers
        kwargs['store_missing_idxs'] = self.store_missing_idxs
        return kwargs

    @staticmethod
    def _collate_fn(
        batches: List[Tuple[Dict[str, torch.Tensor], torch.Tensor]]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Collate function to combine items into mini-batch for dataloader.

        Adds encoder missing time steps and decoder missing time steps to the output.

        Args:
            batches (List[Tuple[Dict[str, torch.Tensor], torch.Tensor]]): List of samples generated with
                :py:meth:`~__getitem__`.

        Returns:
            Tuple[Dict[str, torch.Tensor], Tuple[Union[torch.Tensor, List[torch.Tensor]], torch.Tensor]: minibatch
        """
        # collate function for dataloader
        # lengths
        encoder_lengths = torch.tensor([batch[0]["encoder_length"] for batch in batches], dtype=torch.long)
        decoder_lengths = torch.tensor([batch[0]["decoder_length"] for batch in batches], dtype=torch.long)

        # ids
        decoder_time_idx_start = (
            torch.tensor([batch[0]["encoder_time_idx_start"] for batch in batches], dtype=torch.long) + encoder_lengths
        )
        decoder_time_idx = decoder_time_idx_start.unsqueeze(1) + torch.arange(decoder_lengths.max()).unsqueeze(0)
        groups = torch.stack([batch[0]["groups"] for batch in batches])

        # features
        encoder_cont = rnn.pad_sequence(
            [batch[0]["x_cont"][:length] for length, batch in zip(encoder_lengths, batches)], batch_first=True
        )
        encoder_cat = rnn.pad_sequence(
            [batch[0]["x_cat"][:length] for length, batch in zip(encoder_lengths, batches)], batch_first=True
        )

        decoder_cont = rnn.pad_sequence(
            [batch[0]["x_cont"][length:] for length, batch in zip(encoder_lengths, batches)], batch_first=True
        )
        decoder_cat = rnn.pad_sequence(
            [batch[0]["x_cat"][length:] for length, batch in zip(encoder_lengths, batches)], batch_first=True
        )

        # target scale
        if isinstance(batches[0][0]["target_scale"], torch.Tensor):  # stack tensor
            target_scale = torch.stack([batch[0]["target_scale"] for batch in batches])
        elif isinstance(batches[0][0]["target_scale"], (list, tuple)):
            target_scale = []
            for idx in range(len(batches[0][0]["target_scale"])):
                if isinstance(batches[0][0]["target_scale"][idx], torch.Tensor):  # stack tensor
                    scale = torch.stack([batch[0]["target_scale"][idx] for batch in batches])
                else:
                    scale = torch.from_numpy(
                        np.array([batch[0]["target_scale"][idx] for batch in batches], dtype=np.float32),
                    )
                target_scale.append(scale)
        else:  # convert to tensor
            target_scale = torch.from_numpy(
                np.array([batch[0]["target_scale"] for batch in batches], dtype=np.float32),
            )

        # target and weight
        if isinstance(batches[0][1][0], (tuple, list)):
            target = [
                rnn.pad_sequence([batch[1][0][idx] for batch in batches], batch_first=True)
                for idx in range(len(batches[0][1][0]))
            ]
            encoder_target = [
                rnn.pad_sequence([batch[0]["encoder_target"][idx] for batch in batches], batch_first=True)
                for idx in range(len(batches[0][1][0]))
            ]
        else:
            target = rnn.pad_sequence([batch[1][0] for batch in batches], batch_first=True)
            encoder_target = rnn.pad_sequence([batch[0]["encoder_target"] for batch in batches], batch_first=True)

        if batches[0][1][1] is not None:
            weight = rnn.pad_sequence([batch[1][1] for batch in batches], batch_first=True)
        else:
            weight = None

        if batches[0][0]["encoder_missing"] is not None: # add missing idxs
            encoder_missing = rnn.pad_sequence([batch[0]["encoder_missing"] for batch in batches], batch_first=True)
            decoder_missing = rnn.pad_sequence([batch[0]["decoder_missing"] for batch in batches], batch_first=True)
        else:
            encoder_missing = None
            decoder_missing = None

        # Due as many computations as possible on the CPU for the data loading.
        max_encoder_length = int(encoder_lengths.max().item())
        max_decoder_length = int(decoder_lengths.max().item())

        return (
            dict(
                encoder_cat=encoder_cat,
                encoder_cont=encoder_cont,
                encoder_target=encoder_target,
                encoder_lengths=encoder_lengths,
                decoder_cat=decoder_cat,
                decoder_cont=decoder_cont,
                decoder_target=target,
                decoder_lengths=decoder_lengths,
                decoder_time_idx=decoder_time_idx,
                groups=groups,
                target_scale=target_scale,
                encoder_missing=encoder_missing,
                decoder_missing=decoder_missing,
                max_encoder_length=max_encoder_length,
                max_decoder_length=max_decoder_length,
            ),
            (target, weight),
        )
