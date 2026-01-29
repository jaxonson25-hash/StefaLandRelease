import torch
import numpy as np

def align_dim(data, dim=None, sample=None):

    if not sample is None:
        dim = len(sample.shape)
    assert not dim is None
    data_dim = len(data.shape)
    while not data_dim == dim:
        if data_dim > dim:
            data = data[...,-1]
        elif data_dim < dim:
            data = data[...,None]
        else:
            return data
        data_dim = len(data.shape)
    return data

def convert_to_numpy(data):
    data_type = type(data)
    if data_type == torch.Tensor:
        return data.cpu().detach().numpy()
    else:
        return data

def covert_to_tensor(data):
    pass

import os
import json
import pandas as pd
import numpy as np
from collections import OrderedDict

def norm_streamflow():
    pass

def de_norm_streamflow():
    pass

def normalization(data, stat_list, from_raw=True,**kwargs):
    """

    data size: [..., features]. last dimension is features
    stat_list: [[90 percentile,10 percentile, mean,std], [90 percentile,10 percentile, mean,std]...]

    The feature order must correspond to the variable list order.

    from_raw = True
        normalization
    from_raw = False
        de_normalization

    """
    if data is None:
        return None

    if len(stat_list) == 1:
        data = cal_norm(data, stat=stat_list[0], from_raw=from_raw)
    else:
        for ndx, stat in enumerate(stat_list):
            data[...,ndx] = cal_norm(data[...,ndx], stat=stat, from_raw=from_raw)

    return data


def cal_norm(data, stat, from_raw=False):
    """
    e.g.
    data = np.arange(12).reshape(1, 3, 4),  # any order
    stat = [90 percentile, 10 percentile, mean, std]
    """
    if from_raw:
        # normalization
        data_out = (data - stat[2]) / stat[3]
    else:
        # de_normalization
        data_out = data * stat[3] + stat[2]
    return data_out


def load_stat(csv_folder=None, var_x_list=[], var_c_list=[], var_y_list=[],y_state_list=None):
    """
    e.g.
    var_timeseries = ["precipitation", "temperature"]
    vat_constant = ["soil", "DEM"]
    var_target = ["soil_moisture"]
    csv_path = "/data/format_data/")

    csv_path must include "statistics" subpath.

    stat_x_list, stat_y_list = load(csv_path, var_timeseries+var_constant, var_target)
    output:
        stat_x_list = {"precipitation":[left_p10, left_p90, mean, std],
                        "temperature":[left_p10, left_p90, mean, std],...]}
        stat_y_list: same as stat_x_list
        stat_list = {"x":stat_x_dict, "c":stat_c_dict, "y":stat_y_dict}

    """
    # Python 3.7 dicts in all Python implementations must preserve insertion order.
    # So you can use dict or OrderDict()

    stat_x_dict = OrderedDict()
    stat_c_dict = OrderedDict()
    stat_y_dict = OrderedDict()
    for var_s in var_x_list:
        csv_file = os.path.join(csv_folder, 'statistics', var_s + "_stat.csv")
        stat = pd.read_csv(csv_file, dtype=np.float, header=None).values.flatten()
        stat_x_dict[var_s] = stat.tolist()

    for var_s in var_c_list:
        csv_file = os.path.join(csv_folder, 'statistics', var_s + "_stat.csv")
        stat = pd.read_csv(csv_file, dtype=np.float, header=None).values.flatten()
        stat_c_dict[var_s] = stat.tolist()

    if not var_y_list is None:
        for var_s in var_y_list:
            csv_file = os.path.join(csv_folder, 'statistics', var_s + "_stat.csv")
            stat = pd.read_csv(csv_file, dtype=np.float, header=None).values.flatten()
            stat_y_dict[var_s] = stat.tolist()
    else:
        if y_state_list is None:
            stat_y_dict["y"] = [-9999., -9999., -9999., -9999]
        else:
            stat_y_dict["y"] = y_state_list

    stat_dict = {"x":stat_x_dict, "c":stat_c_dict, "y":stat_y_dict}

    return stat_dict


def batch_norma(data, ):
    """
    data size: [..., features]

    :param data:
    :return:
    """
    num_features = data.shape[-1]
    data_flatten = data.reshape([-1,num_features])
    mean = np.nanmean(data_flatten, axis=0)
    std = np.nanstd(data_flatten, axis=0)

    data_norm = (data - mean) / std
    return data_norm



def align_dim(data, dim=None, sample=None):

    if not sample is None:
        dim = len(sample.shape)
    assert not dim is None
    data_dim = len(data.shape)
    while data_dim == dim:
        if data_dim > dim:
            data = data[...,-1]
        elif data_dim < dim:
            data = data[...,None]
        else:
            return data
        data_dim = len(data.shape)
    return data


def de_normalization(data, stat_list, **kwargs):
    """

    data size: [..., features]. last dimension is features
    stat_list: [[90 percentile,10 percentile, mean,std], [90 percentile,10 percentile, mean,std]...]

    The feature order must correspond to the variable list order.


    """

    if len(stat_list) == 1:
        data = cal_de_norm(data, stat=stat_list[0])
    else:
        for ndx, stat in enumerate(stat_list):
            data[..., ndx] = cal_de_norm(data[..., ndx], stat=stat)

    return data


def cal_de_norm(data, stat):
    """
    e.g.
    data = np.arange(12).reshape(1, 3, 4),  # any order
    stat = [90 percentile, 10 percentile, mean, std]
    """
    return data * stat[3] + stat[2]

# def cal_statistics(raw_x, raw_y, raw_c, seq_length):
#     """
#     raw_x (basins, time, features)
#     raw_y (basins, time, 1)
#     raw_c (basins, features)
#     """
#     scalers = {
#         'x_mean': np.zeros(raw_x.shape[-1]),
#         'x_std': np.zeros(raw_x.shape[-1]),
#         'y_mean': np.zeros(raw_y.shape[-1]),
#         'y_std': np.zeros(raw_y.shape[-1])
#     }
#     num_sites = raw_x.shape[0]
#     total_samples = 0
#     target_std_list = []
#     for idx_site in range(num_sites):
#         sub_x = raw_x[idx_site, :, :]
#         sub_y = raw_y[idx_site, :, :]
#
#         # calculate statistics
#         x_mean = np.nanmean(sub_x, axis=0)
#         x_std = np.nanstd(sub_x, axis=0)  #, ddof=1
#         y_mean = np.nanmean(sub_y, axis=0)
#         y_std = np.nanstd(sub_y, axis=0)  # , ddof=1
#         target_std = y_std[0]
#
#         num_samples = (len(sub_x) - seq_length + 1)
#
#         total_samples = total_samples + num_samples
#
#         scalers["x_mean"] += num_samples * x_mean
#         scalers["x_std"] += num_samples * x_std
#         scalers["y_mean"] += num_samples * y_mean
#         scalers["y_std"] += num_samples * y_std
#
#         target_std_array = np.array([target_std] * num_samples, dtype=np.float32).reshape(-1, 1)
#         target_std_list.append(target_std_array)
#
#     target_std_list = np.concatenate(target_std_list, axis=0) # shape = (num_samples, 1)
#
#     for key in scalers:
#         scalers[key] /= total_samples
#
#     if raw_c is not None:
#         scalers["c_mean"] = np.nanmean(raw_c, axis=0)
#         scalers["c_std"] = np.nanstd(raw_c, axis=0, ddof=1)
#     else:
#         scalers["c_mean"] = None
#         scalers["c_std"] = None
#
#     scalers["y_std_samples"] = target_std_list
#

def filter_sites(raw_x, raw_y):

    valid_sites_x = ~np.isnan(raw_x).all(axis=(1, 2))
    valid_sites_y = ~np.isnan(raw_y).all(axis=(1, 2))

    filtered_x = raw_x[valid_sites_x]
    filtered_y = raw_y[valid_sites_y]

    return filtered_x, filtered_y

def cal_statistics(raw_x, raw_y, raw_c, seq_length):

    filtered_x, filtered_y = filter_sites(raw_x, raw_y)
    filtered_c = raw_c

    if filtered_x.size == 0 or filtered_y.size == 0:
        raise ValueError("No valid sites found!")

    num_samples = filtered_x.shape[1] - seq_length + 1
    total_samples = num_samples * filtered_x.shape[0]

    # x_mean = np.nanmean(filtered_x, axis=1)
    # x_std = np.nanstd(filtered_x, axis=1)
    # y_mean = np.nanmean(filtered_y, axis=1)
    # y_std = np.nanstd(filtered_y, axis=1)
    # scalers = {
    #     'x_mean': np.average(x_mean, axis=0, weights=np.full(x_mean.shape[0], num_samples)),
    #     'x_std': np.average(x_std, axis=0, weights=np.full(x_std.shape[0], num_samples)),
    #     'y_mean': np.average(y_mean, axis=0, weights=np.full(y_mean.shape[0], num_samples)),
    #     'y_std': np.average(y_std, axis=0, weights=np.full(y_std.shape[0], num_samples)),
    #     'y_std_samples': y_std[:, 0].repeat(num_samples).reshape(-1, 1) if filtered_y.size > 0 else None
    # }

    scalers = {
        "x_mean": np.nanmean(filtered_x, axis=(0, 1)),
        "x_std": np.nanstd(filtered_x, axis=(0, 1)),
        "y_mean": np.nanmean(filtered_y, axis=(0, 1)),
        "y_std": np.nanstd(filtered_y, axis=(0, 1)),
        # "y_std_samples": np.tile(np.nanstd(filtered_y, axis=1), (1, num_samples)).reshape(-1, 1) if filtered_y.size > 0 else None,
        # "x_std_samples": np.tile(np.nanstd(filtered_x, axis=1)[:,None,:], (1, num_samples, 1)).reshape(-1, filtered_x.shape[-1]) if filtered_x.size > 0 else None
        "y_std_samples": np.nanstd(filtered_y, axis=1) if filtered_y.size > 0 else None, # [num_sites, 1]
        "x_std_samples": np.nanstd(filtered_x, axis=1) if filtered_x.size > 0 else None, # [num_sites, features]
    }

    if filtered_c is not None and filtered_c.size > 0:
        scalers["c_mean"] = np.nanmean(filtered_c, axis=0)
        scalers["c_std"] = np.nanstd(filtered_c, axis=0, ddof=1)
    else:
        scalers["c_mean"] = None
        scalers["c_std"] = None

    return scalers

def cal_statistics_hydroDL(data, seriesLst, statDict):
    """
    data: (basin, time, variable) or (basin, variable)
    seriesLst: list of variable names
    statDict: {}
    """
    for k in range(len(seriesLst)):
        var = seriesLst[k]
        sub_data = data[..., k]
        sub_data_flat = sub_data.flatten()
        sub_data_remove_nan = sub_data_flat[~np.isnan(sub_data_flat)]
        mean = np.mean(sub_data_remove_nan).astype(float)
        std = np.std(sub_data_remove_nan).astype(float)
        std = np.maximum(std, 0.001)
        statDict[var] = [None, None, mean, std]
    return statDict