"""
Modified from:
Kratzert, F., Klotz, D., Hochreiter, S., and Nearing, G. S.: A note on leveraging synergy in multiple meteorological
datasets with deep learning for rainfall-runoff modeling, Hydrol. Earth Syst. Sci. Discuss.,
https://doi.org/10.5194/hess-2020-221, in review, 2020.
"""
import torch.nn as nn
from MFFormer.utils.validation_tools import Validator

class LSTM(nn.Module):

    def __init__(self, c_in, c_out, hidden_size, dropout, initial_forget_bias=None):
        super().__init__()

        self.initial_forget_bias = initial_forget_bias
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=c_in, hidden_size=hidden_size, batch_first=True)

        self.dropout = nn.Dropout(p=dropout)

        self.project = nn.Linear(hidden_size, c_out)
        self.reset_parameters()

    def reset_parameters(self):
        if self.initial_forget_bias is not None:
            hidden_size = self.hidden_size
            self.lstm.bias_hh_l0.data[hidden_size:2 * hidden_size] = self.initial_forget_bias

    def forward(self, data_dict, *args, **kwargs):
        """
        x=batch_x, c=batch_c, time_mark=batch_time_stamp, stage_name="train"
        """
        x = data_dict['batch_x']
        c = data_dict['batch_c']

        if c is not None:
            x = Validator.combine_timeseries_and_statics(x, c) # (batch_size, seq_len, num_x + num_c)

        lstm_output, (h_n, c_n) = self.lstm(input=x)

        y_pred = self.project(self.dropout(lstm_output))

        return y_pred



class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        c_in = len(configs.time_series_variables) + len(configs.static_variables)
        c_out = len(configs.target_variables)

        self.model = LSTM(c_in=c_in, c_out=c_out, hidden_size=configs.d_model,
                          dropout=configs.dropout, initial_forget_bias=configs.initial_forget_bias)

    def forecast(self, data_dict):

        dec_out = self.model(data_dict)

        return dec_out


    def forward(self, data_dict):
        if self.task_name in ['forecast', 'regression']:
            dec_out = self.forecast(data_dict)
            output_dict = {
                "outputs_time_series": dec_out,
            }
            return output_dict
