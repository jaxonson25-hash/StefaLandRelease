"""
MFFormer/utils/data/grid_data_loader.py
Fire Grid NetCDF Data Loader - Specialized loader for gridded fire prediction data
"""
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime
from MFFormer.utils.stats.transform import cal_statistics

def nc2array_grid_fire(nc_file, station_ids, time_range, time_series_variables, 
                      static_variables, add_coords=True, warmup_days=0, 
                      sampling_interval=None, land_mask_var='land_mask'):
    """
    Load ALL land grid cells from fire prediction NetCDF file.
    """
    
    with xr.open_dataset(nc_file) as ds:
        # Get land mask and find all land coordinates
        land_mask = ds[land_mask_var].values
        land_coords = [(y, x) for y in range(land_mask.shape[0]) 
                       for x in range(land_mask.shape[1]) if land_mask[y, x]]
        
        print(f"Loading {len(land_coords)} land grid cells")
        
        # Convert time coordinate
        if 'time' in ds.dims:
            time_values = ds['time'].values
            if hasattr(ds['time'], 'units') and 'days since' in str(ds['time'].units):
                ref_date = pd.to_datetime('1992-01-01')
                dates = [ref_date + pd.Timedelta(days=int(t)) for t in time_values]
            else:
                dates = pd.to_datetime(time_values)
        else:
            raise ValueError("No time dimension found in dataset")
        
        # Filter dates to requested range
        start_date = pd.to_datetime(time_range[0])
        end_date = pd.to_datetime(time_range[1])
        actual_start = start_date - pd.Timedelta(days=warmup_days)
        
        date_mask = [(d >= actual_start) and (d <= end_date) for d in dates]
        filtered_dates = [d for d, m in zip(dates, date_mask) if m]
        time_indices = [i for i, m in enumerate(date_mask) if m]
        
        if sampling_interval:
            time_indices = time_indices[::sampling_interval]
            filtered_dates = filtered_dates[::sampling_interval]
        
        print(f"Time range: {filtered_dates[0]} to {filtered_dates[-1]} ({len(filtered_dates)} steps)")
        
        n_stations = len(land_coords)
        n_times = len(time_indices)
        n_time_vars = len(time_series_variables)
        n_static_vars = len(static_variables)
        
        # Initialize arrays
        time_series_data = np.full((n_stations, n_times, n_time_vars), np.nan, dtype=np.float32)
        static_data = np.full((n_stations, n_static_vars), np.nan, dtype=np.float32)
        
        # Load time series data
        for var_idx, var_name in enumerate(time_series_variables):
            if var_name in ds.variables:
                var_data = ds[var_name].values
                for station_idx, (y, x) in enumerate(land_coords):
                    if var_data.ndim == 3:
                        time_series_data[station_idx, :, var_idx] = var_data[time_indices, y, x]
        
        # Load static data
        for var_idx, var_name in enumerate(static_variables):
            if var_name in ds.variables:
                var_data = ds[var_name].values
                for station_idx, (y, x) in enumerate(land_coords):
                    if var_data.ndim == 2:
                        static_data[station_idx, var_idx] = var_data[y, x]
                    elif var_data.ndim == 0:
                        static_data[station_idx, var_idx] = var_data
        
        # Add coordinates if requested
        if add_coords:
            coords_data = np.full((n_stations, 2), np.nan, dtype=np.float32)
            if 'latitude' in ds.variables and 'longitude' in ds.variables:
                lat_data = ds['latitude'].values
                lon_data = ds['longitude'].values
                for station_idx, (y, x) in enumerate(land_coords):
                    coords_data[station_idx, 0] = lat_data[y, x]
                    coords_data[station_idx, 1] = lon_data[y, x]
                static_data = np.concatenate([static_data, coords_data], axis=1)
        
        # Check fire data
        fire_idx = next((i for i, var in enumerate(time_series_variables) if var == 'fire_occurrence'), -1)
        if fire_idx >= 0:
            fire_data = time_series_data[:, :, fire_idx]
            fire_count = np.count_nonzero(~np.isnan(fire_data) & (fire_data > 0))
            print(f"Fire events loaded: {fire_count}")
            if fire_count == 0:
                print("WARNING: No fire events found in data")

    return time_series_data, static_data, filtered_dates
            

def read_nc_data_grid_fire(config_dataset, warmup_days, mode="train", scaler=None, 
                          sample_len=None, do_norm=True):
    """
    Modified version of read_nc_data specifically for gridded fire prediction data.
    Integrates with existing MFFormer pipeline with minimal changes.
    """
    
    print(f"Reading fire grid data for {mode} mode...")
    
    # Use the new grid loader
    time_series_variables = config_dataset.time_series_variables + config_dataset.target_variables

    # Load dataset using grid-specific loader
    time_series_data, static_data, date_range = nc2array_grid_fire(
        config_dataset.input_nc_file,
        station_ids=config_dataset.station_ids,
        time_range=getattr(config_dataset, f"{mode}_date_list"),
        time_series_variables=time_series_variables,
        static_variables=config_dataset.static_variables,
        add_coords=config_dataset.add_coords,
        warmup_days=warmup_days,
        sampling_interval=getattr(config_dataset, 'sampling_interval', None),
        land_mask_var=getattr(config_dataset, 'land_mask_var', 'land_mask')
    )

    # Extract coordinates if added
    if config_dataset.add_coords:
        lon = static_data[:, -1]
        lat = static_data[:, -2]
        static_data = static_data[:, :-2]  # Remove coords from static data
    else:
        lon = None
        lat = None

    # Handle negative values (same as original)
    negative_value_variables = getattr(config_dataset, 'negative_value_variables', [])
    positive_value_variables_in_time_series = list(set(time_series_variables) - set(negative_value_variables))
    positive_value_variables_in_static = list(set(config_dataset.static_variables) - set(negative_value_variables))
    
    if len(positive_value_variables_in_time_series) > 0:
        idx_positive_value_variables = [time_series_variables.index(var) for var in positive_value_variables_in_time_series]
        for idx_positive_value in idx_positive_value_variables:
            temp_data = time_series_data[..., idx_positive_value]
            temp_data[temp_data < 0] = 0
            time_series_data[..., idx_positive_value] = temp_data
    
    if len(positive_value_variables_in_static) > 0:
        idx_positive_value_variables = [config_dataset.static_variables.index(var) for var in positive_value_variables_in_static]
        for idx_positive_value in idx_positive_value_variables:
            temp_data = static_data[..., idx_positive_value]
            temp_data[temp_data < 0] = 0
            static_data[..., idx_positive_value] = temp_data

    # Add noise if specified (same as original)
    sigma = getattr(config_dataset, 'Gaussian_noise_sigma', 0)
    if sigma > 0:
        if hasattr(config_dataset, 'time_series_variables_add_noise') and config_dataset.time_series_variables_add_noise is not None:
            idx_time_series_variables_add_noise = [time_series_variables.index(var) for var in
                                                   config_dataset.time_series_variables_add_noise]
            for idx_add_noise in idx_time_series_variables_add_noise:
                noise = np.abs(np.random.normal(loc=0, scale=sigma, size=time_series_data[..., idx_add_noise].shape))
                time_series_data[..., idx_add_noise] += noise

        if hasattr(config_dataset, 'static_variables_add_noise') and config_dataset.static_variables_add_noise is not None:
            idx_static_variables_add_noise = [config_dataset.static_variables.index(var) for var in
                                              config_dataset.static_variables_add_noise]
            for idx_add_noise in idx_static_variables_add_noise:
                noise = np.abs(np.random.normal(loc=0, scale=sigma, size=static_data[..., idx_add_noise].shape))
                static_data[..., idx_add_noise] += noise

    # Split time series and target data
    num_time_variables_except_target = len(config_dataset.time_series_variables)
    num_static_variables = len(config_dataset.static_variables)
    num_target_variables = len(config_dataset.target_variables)

    assert time_series_data.shape[-1] == num_time_variables_except_target + num_target_variables

    data_x = time_series_data[:, :, :num_time_variables_except_target]
    data_y = time_series_data[:, :, num_time_variables_except_target:]

    assert not np.isnan(data_y).all(), "data_y cannot be all nan"

    print(f"Data shapes: X={data_x.shape}, Y={data_y.shape}, Static={static_data.shape}")

    # Normalization (same as original)
    if not do_norm:
        return data_x, data_y, static_data, date_range, None

    if scaler is None:
        print("Calculating statistics for normalization...")
        scaler = cal_statistics(raw_x=data_x, raw_y=data_y, raw_c=static_data, seq_length=sample_len)

    # Apply normalization
    epsilon = 1e-5
    static_norm = (static_data - scaler["c_mean"]) / (scaler["c_std"] + epsilon)
    data_x_norm = (data_x - scaler["x_mean"]) / (scaler["x_std"] + epsilon)
    # DON'T normalize fire targets - keep as raw 0/1
    if 'fire_occurrence' in config_dataset.target_variables:
        # Keep fire data as raw 0/1 values
        data_y_norm = data_y  # No normalization for binary fire data
        # Set fake scaler values so denormalization doesn't break
        scaler["y_mean"] = np.zeros_like(scaler["y_mean"])
        scaler["y_std"] = np.ones_like(scaler["y_std"])
    else:
        # Normal normalization for non-fire targets
        data_y_norm = (data_y - scaler["y_mean"]) / (scaler["y_std"] + epsilon)

    # Handle categorical variables if any
    static_variables_category_dict = None
    if hasattr(config_dataset, 'static_variables_category') and len(config_dataset.static_variables_category) > 0:
        # Fire grid data doesn't have categorical variables, but keep for compatibility
        static_variables_category_dict = {}

    # Set coordinates and categorical dict
    config_dataset.lat = lat
    config_dataset.lon = lon
    config_dataset.static_variables_category_dict = static_variables_category_dict

    print(f"Fire grid data processing complete!")
    return data_x_norm, data_y_norm, static_norm, date_range, scaler, config_dataset