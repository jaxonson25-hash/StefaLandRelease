import torch
import numpy as np
import pandas as pd

class Validator:

    def __int__(self):
        pass

    def check_indices_for_loss_calculation(self, indices_for_loss_calculation: list or str, seq_len: int) -> list:

        if isinstance(indices_for_loss_calculation, str):
            if indices_for_loss_calculation == "all":
                indices_for_loss_calculation = list(range(seq_len))
            else:
                raise ValueError("indices_for_loss_calculation is only 'all' or list")

        elif isinstance(indices_for_loss_calculation, list):
            indices_for_loss_calculation = [index if index >= 0 else seq_len + index for index in indices_for_loss_calculation]
            self.is_consecutive(indices_for_loss_calculation)
            assert indices_for_loss_calculation[-1] == seq_len - 1, "Should be end with seq_len - 1"

        else:
            raise ValueError("indices_for_loss_calculation is only 'all' or list")

        return indices_for_loss_calculation

    def is_consecutive(self, indices):
        indices = np.array(indices)
        diff = np.diff(indices)
        assert np.all(diff == 1), "indices should be consecutive"

    @staticmethod
    def combine_timeseries_and_statics(time_series_data, static_data,):
        """
           Combine time series data and static data.

           Parameters:
           - time_series_data: Tensor or ndarray of shape [bs, seq_len, features]
           - static_data: Tensor or ndarray of shape [bs, features]

           Returns:
           Combined data of shape [bs, seq_len, features + static_features]
       """
        if isinstance(time_series_data, torch.Tensor):
            time_series_data = time_series_data.transpose(0, 1)  # transpose to [seq_length, batch_size, n_features]
            static_data = static_data.unsqueeze(0).repeat(time_series_data.shape[0], 1, 1)
            time_series_data = torch.cat([time_series_data, static_data], dim=-1)
            time_series_data = time_series_data.transpose(0, 1)  # transpose to [batch_size, seq_length, n_features]

        elif isinstance(time_series_data, np.ndarray):
            time_series_data = np.swapaxes(time_series_data, 0, 1)
            static_data = np.expand_dims(static_data, axis=0).repeat(time_series_data.shape[0], axis=0)
            time_series_data = np.concatenate([time_series_data, static_data], axis=-1)
            time_series_data = np.swapaxes(time_series_data, 0, 1)

        else:
            raise ValueError("time_series_data should be torch.Tensor or np.ndarray")

        return time_series_data
