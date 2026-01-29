import numpy as np
from MFFormer.utils.data.load_nc import NetCDFDataset
from MFFormer.utils.stats.transform import cal_statistics


def read_nc_data(config_dataset, warmup_days, mode="train", scaler=None, sample_len=None, do_norm=True,
                 data_type="site"):

    # Check if this is grid fire data
    if (hasattr(config_dataset, 'data_type') and config_dataset.data_type == "grid" and 
        hasattr(config_dataset, 'input_nc_file') and 'fire_prediction' in str(config_dataset.input_nc_file)):
        
        # Use the fire grid specific loader
        from MFFormer.data_provider.grid_data_loader import read_nc_data_grid_fire
        return read_nc_data_grid_fire(config_dataset, warmup_days, mode, scaler, sample_len, do_norm)
    
    # Use existing loaders for other data types
    nc_tool = NetCDFDataset()

    if data_type in ["site", "station", "basin"]:
        nc_loader = nc_tool.nc2array
        config_dataset.sampling_interval = None
    elif data_type == "grid":
        nc_loader = nc_tool.nc2array_grid
    else:
        raise NotImplementedError

    time_series_variables = config_dataset.time_series_variables + config_dataset.target_variables

    # load dataset
    time_series_data, static_data, date_range = nc_loader(
        config_dataset.input_nc_file,
        station_ids=config_dataset.station_ids,
        time_range=getattr(config_dataset, "{}_date_list".format(mode)),
        time_series_variables=time_series_variables,
        static_variables=config_dataset.static_variables,
        add_coords=True,
        warmup_days=warmup_days,
        sampling_interval=config_dataset.sampling_interval,
    )

    lon = static_data[:, -1]
    lat = static_data[:, -2]

    if not config_dataset.add_coords:
        # remove the coordinates
        static_data = static_data[:, :-2]

    # remove negative values
    negative_value_variables = config_dataset.negative_value_variables
    positive_value_variables_in_time_series = list(set(time_series_variables) - set(negative_value_variables))
    positive_value_variables_in_static = list(set(config_dataset.static_variables) - set(negative_value_variables))
    if len(positive_value_variables_in_time_series) > 0:
        # time series variables
        idx_positive_value_variables = [time_series_variables.index(var) for var in positive_value_variables_in_time_series]
        for idx_positive_value in idx_positive_value_variables:
            temp_data = time_series_data[..., idx_positive_value]
            temp_data[temp_data < 0] = 0
            time_series_data[..., idx_positive_value] = temp_data
    if len(positive_value_variables_in_static) > 0:
        # static variables
        idx_positive_value_variables = [config_dataset.static_variables.index(var) for var in positive_value_variables_in_static]
        for idx_positive_value in idx_positive_value_variables:
            temp_data = static_data[..., idx_positive_value]
            temp_data[temp_data < 0] = 0
            static_data[..., idx_positive_value] = temp_data

    # model uncertainty
    sigma = config_dataset.Gaussian_noise_sigma
    if sigma > 0:
        if config_dataset.time_series_variables_add_noise is not None:
            idx_time_series_variables_add_noise = [time_series_variables.index(var) for var in
                                                   config_dataset.time_series_variables_add_noise]
            for idx_add_noise in idx_time_series_variables_add_noise:
                noise = np.abs(np.random.normal(loc=0, scale=sigma, size=time_series_data[..., idx_add_noise].shape))
                time_series_data[..., idx_add_noise] += noise

        if config_dataset.static_variables_add_noise is not None:
            idx_static_variables_add_noise = [config_dataset.static_variables.index(var) for var in
                                              config_dataset.static_variables_add_noise]
            for idx_add_noise in idx_static_variables_add_noise:
                noise = np.abs(np.random.normal(loc=0, scale=sigma, size=static_data[..., idx_add_noise].shape))
                static_data[..., idx_add_noise] += noise

    num_time_variables_except_target = len(config_dataset.time_series_variables)
    num_static_variables = len(config_dataset.static_variables)
    num_target_variables = len(config_dataset.target_variables)

    assert time_series_data.shape[-1] == num_time_variables_except_target + num_target_variables

    data_x = time_series_data[:, :, :num_time_variables_except_target]
    data_y = time_series_data[:, :, num_time_variables_except_target:]

    # detect (bs, seq_len, 0)
    if 0 in data_y.shape:
        data_y = np.full_like(data_x[:, :, -1:], -9999)

    assert not np.isnan(data_y).all(), "data_y can not be all nan"

    if not do_norm:
        return data_x, data_y, static_data, date_range, None

    if scaler is None:
        # calculate statistics
        scaler = cal_statistics(raw_x=data_x, raw_y=data_y, raw_c=static_data, seq_length=sample_len)

    # normalize
    epsilon = 1e-5
    static_norm = (static_data - scaler["c_mean"]) / (scaler["c_std"] + epsilon)
    data_x_norm = (data_x - scaler["x_mean"]) / (scaler["x_std"] + epsilon)
    data_y_norm = (data_y - scaler["y_mean"]) / (scaler["y_std"] + epsilon)

    # restore the categorical variables
    if len(config_dataset.static_variables_category) > 0:
        static_variables_category_dict = {}
        for categorical_name in config_dataset.static_variables_category:
            categorical_index = config_dataset.static_variables.index(categorical_name)

            static_data[:, categorical_index] = static_data[:, categorical_index].astype(np.int64)

            unique_classes = np.unique(static_data[:, categorical_index])

            if mode in ['train']:
                class_to_index = dict(zip(unique_classes, range(len(unique_classes))))
                index_to_class = dict(zip(range(len(unique_classes)), unique_classes))
            else:
                class_to_index = config_dataset.static_variables_category_dict[categorical_name]["class_to_index"]
                index_to_class = config_dataset.static_variables_category_dict[categorical_name]["index_to_class"]

            static_norm[:, categorical_index] = np.vectorize(class_to_index.get)(static_data[:, categorical_index])

            static_variables_category_dict[categorical_name] = {
                "class_to_index": class_to_index,
                "index_to_class": index_to_class,
            }
    else:
        static_variables_category_dict = None

    config_dataset.static_variables_category_dict = static_variables_category_dict
    config_dataset.lat = lat
    config_dataset.lon = lon

    return data_x_norm, data_y_norm, static_norm, date_range, scaler, config_dataset