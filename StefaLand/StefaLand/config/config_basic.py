import os
import sys
import json
import MFFormer
import argparse
import pandas as pd
import numpy as np
from MFFormer.utils.sys_tools import re_folder, fix_seed, print_args
from MFFormer.datasets.load_path import LoadPath

server_dir = LoadPath.server()

def parse_list(option_str):
    return option_str.split(',')


def get_config():
    parser = argparse.ArgumentParser(description='Time series modeling')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='regression',
                        help='task name, options:[pretrain, pretrain_time_series, regression, forecast]')
    parser.add_argument('--model', type=str, required=True, default='MFFormer',
                        help='model name, options: [LSTM, Transformer, MFFormer]')
    parser.add_argument('--output_dir', type=str, default=os.path.join(server_dir, 'output', '30.MFFormer'),
                        help='output directory')
    parser.add_argument('--des', type=str, default='', help='exp description')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='true or specified path to load weights from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained model path')
    parser.add_argument('--do_eval', action='store_true', default=False, help='whether to do evaluation')
    parser.add_argument('--do_test', action='store_true', default=False, help='only do test')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer')
    parser.add_argument('--criterion', type=str, default='MaskedNSE', help='loss function')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    # data loader
    parser.add_argument('--data', type=parse_list, required=True, default='CAMELS_Frederik',
                        help='dataset name in data folder')
    parser.add_argument('--input_nc_file', type=str, default=None, help='input nc file')
    parser.add_argument('--time_series_variables', type=parse_list, default=None, help='time series data names')
    parser.add_argument('--target_variables', type=parse_list, default=None, help='time series data names')
    parser.add_argument('--static_variables', type=parse_list, default=None, help='numerical static data names')
    parser.add_argument('--static_variables_category', type=parse_list, default=None,
                        help='categorical static data names')
    parser.add_argument('--train_date_list', type=parse_list, default=None, help='train date list')
    parser.add_argument('--val_date_list', type=parse_list, default=None, help='val date list')
    parser.add_argument('--test_date_list', type=parse_list, default=None, help='test date list')
    parser.add_argument('--station_ids', type=parse_list, default=None, help='station ids')
    parser.add_argument('--station_ids_file', type=parse_list, default=None, help='station ids file')
    parser.add_argument('--add_coords', action="store_true", default=False, help='whether to add coords')
    parser.add_argument('--num_kfold', type=int, default=None, help='num of kfold')
    parser.add_argument('--nth_kfold', type=int, default=None, help='nth kfold')
    parser.add_argument('--data_type', type=str, default=None, help='data type, options: [site, basin, grid]')
    parser.add_argument('--sampling_interval', type=int, default=None, help='sampling interval only for grid data')
    parser.add_argument('--regions', type=parse_list, default=None, help='regions that station ids belong to')
    parser.add_argument('--train_regions', type=parse_list, default=None, help='train regions')  # PUR
    parser.add_argument('--test_regions', type=parse_list, default=None, help='test regions')  # PUR
    parser.add_argument('--sampling_regions', type=parse_list, default=None, help='sampling in regional order')
    parser.add_argument('--scaler_file', type=str, default=None, help='scaler file path')
    parser.add_argument('--negative_value_variables', type=parse_list, default=None, help='variables include negative value')

    parser.add_argument('--test_mode', type=str, default='CV', choices=['CV', 'PUR', 'PUB'],
                        help='Test mode: CV (Cross-Validation), PUR (Partially Ungauged Regions), or PUB (Prediction in Ungauged Basins)')
    parser.add_argument('--huc_regions', type=str, nargs='+', action='append', 
                    help='Groups of HUC regions for PUR mode. Each group is a space-separated list of HUC codes.')
    parser.add_argument('--test_region_index', type=int, default=0,
                    help='Index of the HUC region group to use for testing in PUR mode')
    parser.add_argument('--test_pub', type=float, default=1, help='index of pub for testing')
    parser.add_argument('--gage_info_file', type=str, default='/storage/home/nrk5343/scratch/Data/gages_list_with_pub.csv',
                    help='Path to the CAMELS gage info CSV file')

    # dataset
    parser.add_argument('--num_data_chunks', type=int, default=0, help='split the dataset into chunks')
    parser.add_argument('--use_complete_reindex', action='store_true', default=False, help='Uses older file reindexing method')

    # forecasting and regression task
    parser.add_argument('--seq_len', type=int, default=365, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=365, help='start token length')
    parser.add_argument('--pred_len', type=int, default=365, help='prediction sequence length')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=True)
    parser.add_argument('--sampling_stride', type=int, default=1, help='each sampling data interval, for example, 1 means each data, 2 means every two data')

    # pretrain task
    parser.add_argument('--mask_ratio_time_series', type=float, default=0.5, help='mask ratio for time series data')
    parser.add_argument('--mask_ratio_static', type=float, default=0.5, help='mask ratio for static data')
    parser.add_argument('--mask_skip_variables', type=parse_list, default=None, help='the variables to skip masking')
    parser.add_argument('--mask_all_variables', type=parse_list, default=None, help='the variables to mask all')
    parser.add_argument('--group_mask_dict', type=parse_list, default=None, help='the group mask dict')
    parser.add_argument('--min_window_size', type=int, default=30, help='min window size')
    parser.add_argument('--max_window_size', type=int, default=90, help='max window size')
    parser.add_argument('--static_pred_start_point', type=int, default=0, help='static data prediction start point')
    parser.add_argument('--inference_variables', type=parse_list, default=None, help='inference variables')

    # model define
    parser.add_argument('--enc_in', type=int, default=None, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=None, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=None, help='output size')
    parser.add_argument('--d_model', type=int, default=256, help='dimension of model, or hidden size')
    parser.add_argument('--num_heads', type=int, default=4, help='num of heads')
    parser.add_argument('--num_enc_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--num_dec_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ffd', type=int, default=512, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--initial_forget_bias', type=float, default=3.0,
                        help='forget bias for the initial LSTM of decoder in LSTM-based models')
    parser.add_argument('--init_weight', type=float, default=0.02, help='init weight')
    parser.add_argument('--init_bias', type=float, default=0.02, help='init bias')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--epochs', type=int, default=30, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=30, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='optimizer weight decay')
    parser.add_argument('--lradj', type=str, default=None, help='adjust learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=0, help='warm up epochs')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--clip_grad', type=float or None, help='gradient clipping, None means no clipping',
                        default=None)
    parser.add_argument('--use_target_stds', action='store_true', help='use stds of targets to normalize loss',
                        default=False)
    parser.add_argument('--warmup_train', action='store_true', default=False,
                        help='if warm up train, use half mini-batch train data')
    parser.add_argument('--ratio_time_series_loss', type=float, default=1.0, help='ratio of time series for loss')
    parser.add_argument('--ratio_static_loss', type=float, default=0.5, help='ratio of static for loss')
    parser.add_argument('--ratio_time_series_variables', type=parse_list, default=None,
                        help='ratio of time series variables')
    parser.add_argument('--ratio_static_variables', type=parse_list, default=None, help='ratio of static variables')
    parser.add_argument('--calculate_time_series_each_variable_loss', action='store_true', default=False,
                        help='calculate each variable loss')
    parser.add_argument('--calculate_static_each_variable_loss', action='store_true', default=False,
                        help='calculate each variable loss')

    # model uncertainty related
    parser.add_argument('--num_MC_dropout_samples', type=int, default=0, help='num of MC dropout samples')
    parser.add_argument('--add_input_noise', action='store_true', default=False, help='add input data noise')
    parser.add_argument('--Gaussian_noise_sigma', type=float, default=0, help='Gaussian noise sigma')
    parser.add_argument('--time_series_variables_add_noise', type=parse_list, default=None,
                        help='time series variables add noise')
    parser.add_argument('--static_variables_add_noise', type=parse_list, default=None, help='static variables add noise')

    #StefaLand Patcher
    parser.add_argument('--use_patches',action='store_true',default=True,  help='use patch embeddings')
    parser.add_argument('--patch_len', type=int,  default=16, help='patch length')
    parser.add_argument('--patch_stride',type=int, default=8, help='patch stride')

    # GPU
    parser.add_argument('--use_gpu', action='store_true', default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='use one gpu, which device is used')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    parser.add_argument('--num_nothing', type=float, help='num of nothing')

    args = parser.parse_args()

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        # args.gpu = args.device_ids[0]

    if len(args.data) > 1:
        args.use_multi_data_sources = True
    else:
        args.use_multi_data_sources = False

    args.print_params = True

    # update the station_ids from station_ids_file
    if args.station_ids_file is not None:
        args.station_ids = get_station_ids_from_file(args.station_ids_file)

    # update the model uncertainty related parameters
    if (not args.time_series_variables_add_noise is None) and (len(args.time_series_variables_add_noise) > 0):
        args.add_input_noise = True

    # format the save path
    args = format_save_path(args)

    return args


def get_station_ids_from_file(station_ids_file_list):

    station_ids_list = []

    for station_ids_file in station_ids_file_list:
        if station_ids_file.endswith('.json'):
            with open(station_ids_file, 'r') as f:
                station_ids = json.load(f)
        elif station_ids_file.endswith('.txt'):
            station_ids = pd.read_csv(station_ids_file, header=None, dtype=str).values[:, 0].tolist()
        else:
            station_ids_file = LoadPath.station_ids(station_ids_file)
            station_ids = pd.read_csv(station_ids_file, header=None, dtype=str).values[:, 0].tolist()

        station_ids_list.extend(station_ids)

    station_ids = np.array(station_ids_list)

    return station_ids

def format_save_path(config):
    data = "_".join(config.data)
    if (not config.regions is None) and (len(config.regions) > 0):
        regions = "_" + "_".join(config.regions)
    else:
        regions = ""

    if (not config.test_regions is None) and (len(config.test_regions) > 0):
        PUR = "_PUR_test_" + "_".join(config.test_regions)
    else:
        PUR = ""
    saved_folder_name = f"{config.task_name}_{config.model}_{data}{regions}{PUR}_{config.des}"
    saved_dir = os.path.join(config.output_dir, saved_folder_name)
    checkpoints_dir = os.path.join(saved_dir, "checkpoints")
    results_dir = os.path.join(saved_dir, "results")
    index_dir = os.path.join(saved_dir, "index")

    re_folder(checkpoints_dir)
    re_folder(results_dir)
    re_folder(index_dir)
    config.saved_folder_name = saved_folder_name
    config.saved_dir = saved_dir
    config.checkpoints_dir = checkpoints_dir
    config.results_dir = results_dir
    config.index_dir = index_dir
    return config


def get_dataset_config(dataset_name):
    from MFFormer.config.config_dataset_camels import config_dataset_camels
    # from MFFormer.config.config_dataset_camels_chem import config_dataset_camels_chem
    # from MFFormer.config.config_dataset_global_streamflow_4299_basins import config_dataset_global_4299
    # from MFFormer.config.config_dataset_global_6738 import config_dataset_global_6738
    from MFFormer.config.config_dataset_soil_cmd import config_soil_moisture
    # from MFFormer.config.config_dataset_fire_occurence import config_dataset_fire_grid
    # from MFFormer.config.config_dataset_global_2D_contour import config_dataset_2D_contour
    # from MFFormer.config.config_dataset_global_3D_contour import config_dataset_3D_contour
    # from MFFormer.config.config_dataset_global_streamflow_4299_selection import config_dataset_global_4299_selection
    # from MFFormer.config.config_dataset_global_streamflow_4299_basins_ERA5 import config_dataset_global_4299_ERA5
    # from MFFormer.config.config_dataset_global_MSWX import config_dataset_global_grid_MSWEX
    # from MFFormer.config.config_dataset_camels_dapeng import config_dataset_camels_dapeng
    # # from MFFormer.config.config_dataset_WQP import config_dataset_WQP
    # # from MFFormer.config.config_dataset_MERIT import config_dataset_MERIT
    # from MFFormer.config.config_dataset_global_streamflow_except_4299_basins import config_dataset_global_except_4299
    # from MFFormer.config.config_dataset_global_streamflow_level_8 import config_dataset_global_level_8
    # from MFFormer.config.config_dataset_global_streamflow_4299_and_level_8 import config_dataset_global_4299_and_level_8
    # from MFFormer.config.config_dataset_global_streamflow_4299_and_except import config_dataset_global_4299_and_except
    # from MFFormer.config.config_dataset_global_streamflow_3434_3662 import config_dataset_global_3434_3262
    # from MFFormer.config.config_dataset_global_streamflow_3434_3662_additional import config_dataset_global_3434_3262_additional
    # from MFFormer.config.config_dataset_global_streamflow_3434_3248_additional import config_dataset_global_3434_3248_additional
    from MFFormer.config.config_dataset_global_streamflow_3434_3248_additional_1952 import config_dataset_global_3434_3248_additional_1952
    from MFFormer.config.config_dataset_global_streamflow_175373_1992_2005 import config_dataset_global_streamflow_175373_1992_2005
    from MFFormer.config.config_dataset_global_streamflow_175373_2006_2018 import config_dataset_global_streamflow_175373_2006_2018
    from MFFormer.config.config_dataset_global_streamflow_StefalandSubsetA import config_dataset_global_streamflow_StefalandSubsetA
    # from MFFormer.config.config_dataset_global_HydroBasin_level08 import config_dataset_global_hydrobasin_level08
    # from MFFormer.config.config_dataset_global_20000 import config_dataset_global_20000
    # from MFFormer.config.config_dataset_global_4D_contour_Australia_204046 import config_dataset_4D_contour_Australia_204046
    from MFFormer.config.config_dataset_WoSIS_sample_5000 import config_dataset_WoSIS_sample_5000 

    config_dataset_dict = {
        # 'CAMELS': config_dataset_camels,
        # 'CAMELS_Chem': config_dataset_camels_chem,
        # 'GLOBAL_4299': config_dataset_global_4299,
        # 'GLOBAL_6738': config_dataset_global_6738,
        'WoSIS' : config_dataset_WoSIS_sample_5000, 
        'GMD' : config_soil_moisture,
        # 'FIRE_GRID' : config_dataset_fire_grid, 
        # 'GLOBAL_2D_contour': config_dataset_2D_contour,
        # 'GLOBAL_3D_contour': config_dataset_3D_contour,
        # 'GLOBAL_4299_selection': config_dataset_global_4299_selection,
        # 'GLOBAL_4299_ERA5': config_dataset_global_4299_ERA5,
        # 'GLOBAL_4299_except': config_dataset_global_except_4299,
        # 'GLOBAL_4299_and_except': config_dataset_global_4299_and_except,
        # 'GLOBAL_level_8': config_dataset_global_level_8,
        # 'GLOBAL_4299_and_level_8': config_dataset_global_4299_and_level_8,
        # 'GLOBAL_GRID_MSWX': config_dataset_global_grid_MSWEX,
        # 'CAMELS_Dapeng': config_dataset_camels_dapeng,
        # 'GLOBAL_3434_3262': config_dataset_global_3434_3262,
        # 'GLOBAL_3434_3262_additional': config_dataset_global_3434_3262_additional,
        # 'GLOBAL_3434_3248_additional': config_dataset_global_3434_3248_additional,
        'GLOBAL_3434_3248_additional_1952': config_dataset_global_3434_3248_additional_1952,
        'GLOBAL_175373_1992_2005' : config_dataset_global_streamflow_175373_1992_2005,
        'GLOBAL_175373_2006_2018' : config_dataset_global_streamflow_175373_2006_2018,
        'StefaLandSubsetA' : config_dataset_global_streamflow_StefalandSubsetA
        # 'WQP': config_dataset_WQP,
        # 'MERIT': config_dataset_MERIT,
        # 'Australia_204046': config_dataset_4D_contour_Australia_204046,
        # 'GLOBAL_HydroBasin_level08': config_dataset_global_hydrobasin_level08,
        # 'GLOBAL_20000': config_dataset_global_20000,
    }
    config_dataset = config_dataset_dict[dataset_name]
    return config_dataset


def update_configs(configs, config_dataset):
    variables_to_update = ['input_nc_file', 'train_date_list', 'val_date_list', 'test_date_list',
                           'time_series_variables', 'target_variables', 'static_variables', 'static_variables_category',
                           'station_ids', 'regions', 'add_coords', 'group_mask_dict', 'data_type', 'mask_all_variables',
                           'mask_skip_variables', 'station_ids', 'negative_value_variables']

    for var in variables_to_update:
        config_value = getattr(configs, var)
        dataset_value = getattr(config_dataset, var)

        if config_value is None:
            setattr(configs, var, dataset_value)
        else:
            setattr(config_dataset, var, config_value)

    # optional variables may not exist in config_dataset
    variables_optional = ['sampling_interval']
    for var in variables_optional:
        try:
            config_value = getattr(configs, var)
            dataset_value = getattr(config_dataset, var)
            if config_value is None:
                setattr(configs, var, dataset_value)
            else:
                setattr(config_dataset, var, config_value)
        except:
            pass

    # parameters add to config_dataset
    variables_to_add = ['add_input_noise', 'Gaussian_noise_sigma', 'time_series_variables_add_noise', 'static_variables_add_noise', 'num_nothing']
    for var in variables_to_add:
        config_value = getattr(configs, var)
        setattr(config_dataset, var, config_value)

    # update enc_in, dec_in, c_out
    configs.enc_in = len(configs.time_series_variables) + len(configs.static_variables)
    configs.dec_in = len(configs.target_variables)
    configs.c_out = len(configs.target_variables)

    if configs.print_params:
        print_args(configs)
        configs.print_params = False

    # update the station_ids
    if len(configs.regions) > 0:
        station_ids, lat, lon = get_station_ids_by_regions(configs, config_dataset, configs.regions)
        setattr(configs, 'station_ids', station_ids)
        setattr(config_dataset, 'station_ids', station_ids)

    return configs, config_dataset


def update_configs_multi_data_sources(configs, config_dataset):
    variables_to_update = ['input_nc_file', 'static_variables_category', 'station_ids', 'group_mask_dict', 'data_type',
                           'mask_all_variables', 'mask_skip_variables']

    for var in variables_to_update:
        config_value = getattr(configs, var)
        dataset_value = getattr(config_dataset, var)
        setattr(configs, var, dataset_value)

    variables_not_update = ['train_date_list', 'val_date_list', 'test_date_list', 'add_coords', 'regions',
                            'time_series_variables', 'target_variables', 'static_variables', ]
    for var in variables_not_update:
        config_value = getattr(configs, var)
        dataset_value = getattr(config_dataset, var)
        setattr(config_dataset, var, config_value)

    # update the station_ids
    if len(configs.regions) > 0:
        station_ids, lat, lon = get_station_ids_by_regions(configs, config_dataset, configs.regions)
        setattr(configs, 'station_ids', station_ids)
        setattr(config_dataset, 'station_ids', station_ids)

    return configs, config_dataset


def get_station_ids_by_regions(configs, config_dataset, regions):
    import json
    import time
    import xarray as xr
    from MFFormer.utils.gis.vector_tools import VectorTools

    vts = VectorTools()
    index_dir = configs.index_dir

    ds = xr.open_dataset(config_dataset.input_nc_file)
    station_ids = ds.station_ids.values
    lat = ds.lat.values
    lon = ds.lon.values

    index_within_regions = []
    for region in regions:
        shp_path = LoadPath.shapefile(region)
        index_within_region = vts.pts_within_shp(inp_shp=shp_path, lon_list=lon, lat_list=lat, opt_path=index_dir, )
        index_within_regions.extend(index_within_region)

    index_within_regions = list(set(index_within_regions))
    station_ids = station_ids[index_within_regions]
    lat = lat[index_within_regions]
    lon = lon[index_within_regions]

    # save the station ids, lat, lon as json file
    regions_info = {
        'station_ids': station_ids.tolist(),
        'lat': lat.tolist(),
        'lon': lon.tolist(),
    }

    # regions info file name with timestamp
    regions_info_file = os.path.join(index_dir, f"index_regions_info_{int(time.time())}.json")
    with open(regions_info_file, 'w') as f:
        json.dump(regions_info, f)

    return station_ids, lat, lon

def get_station_ids_by_regionsHUC(configs, config_dataset, regions):
    import pandas as pd
    import time
    import os
    import json
    from MFFormer.datasets.load_path import LoadPath

    # Load CAMELS gage info
    gageinfo = pd.read_csv(configs.gage_info_file, dtype={"huc": int, "gage": str, "LAT": float, "LONG": float})

    # Process HUC regions
    all_station_ids = []
    all_lats = []
    all_lons = []
 
    for region_group in regions:
        region_hucs = [int(huc) for huc in region_group]
        region_gages = gageinfo[gageinfo['huc'].isin(region_hucs)]
        all_station_ids.extend(region_gages['gage'].tolist())
        all_lats.extend(region_gages['LAT'].tolist())
        all_lons.extend(region_gages['LONG'].tolist())
        # print(f'individual region hucs{region_hucs}')

    # Remove duplicates while preserving order
    station_ids = list(dict.fromkeys(all_station_ids))
    lat = [all_lats[all_station_ids.index(sid)] for sid in station_ids]
    lon = [all_lons[all_station_ids.index(sid)] for sid in station_ids]

    # Save the station ids, lat, lon as json file
    regions_info = {
        'station_ids': station_ids,
        # 'lat': lat,
        # 'lon': lon,
    }
    print(f'config huc regions {regions}: {len(station_ids)} stations')

    # Regions info file name with timestamp
    # index_dir = configs.index_dir
    # regions_info_file = os.path.join(index_dir, f"index_regions_info_{int(time.time())}.json")
    # with open(regions_info_file, 'w') as f:
    #     json.dump(regions_info, f)

    return station_ids, lat, lon

def get_station_ids_by_pub(configs, config_dataset, pub_ids):
    import pandas as pd
    import time
    import os
    import json
    from MFFormer.datasets.load_path import LoadPath
    # Load CAMELS gage info
    gageinfo = pd.read_csv(configs.gage_info_file, dtype={"PUB_ID": int, "gage": str, "LAT": float, "LONG": float})

    # Ensure pub_ids is a list
    if not isinstance(pub_ids, list):
        pub_ids = [pub_ids]

    # Filter gages by the specified PUB IDs
    pub_gages = gageinfo[gageinfo['PUB_ID'].isin(pub_ids)]

    # Extract station IDs, latitudes, and longitudes
    all_station_ids = pub_gages['gage'].tolist()
    all_lats = pub_gages['LAT'].tolist()
    all_lons = pub_gages['LONG'].tolist()

    # Remove duplicates while preserving order
    station_ids = list(dict.fromkeys(all_station_ids))
    lat = [all_lats[all_station_ids.index(sid)] for sid in station_ids]
    lon = [all_lons[all_station_ids.index(sid)] for sid in station_ids]

    # Save the station ids, lat, lon as json file
    regions_info = {
        'station_ids': station_ids,
        # 'lat': lat,
        # 'lon': lon,
    }

    # # Regions info file name with timestamp
    # index_dir = configs.index_dir
    # regions_info_file = os.path.join(index_dir, f"index_pub{pub_ids}_info_{int(time.time())}.json")
    # with open(regions_info_file, 'w') as f:
    #     json.dump(regions_info, f)

    print(f"PUB ID {pub_ids}: {len(station_ids)} stations")

    return station_ids, lat, lon



def save_as_bash_script(configs, filename):
    """
    Save the parameters from an argparse.Namespace object into a bash script file,
    using the installation path of the MFFormer package.

    :param configs: argparse.Namespace object containing the configuration parameters to be saved.
    :param filename: Name of the bash script file to be created.
    """
    # Get the directory of the MFFormer package
    mfformer_dir = os.path.dirname(MFFormer.__file__)

    # input parameters
    save_name = 'configs_input.sh'
    input_args = ' '.join(sys.argv[1:])
    with open(os.path.join(os.path.dirname(filename), save_name), 'w') as f:
        f.write("#!/bin/bash\n\n")
        f.write(f"cd {mfformer_dir}\n\n")  # Change to the MFFormer directory
        f.write("python -u run.py \\\n")
        f.write(input_args)
        f.write(" --resume_from_checkpoint True ")

    with open(filename, 'w') as file:
        file.write("#!/bin/bash\n\n")
        file.write(f"cd {mfformer_dir}\n\n")  # Change to the MFFormer directory
        file.write("python -u run.py \\\n")
        for key, value in vars(configs).items():

            if key in ['station_ids']:
                continue

            if value is not None:
                if isinstance(value, list):
                    # Convert list values into comma-separated strings
                    value_str = ','.join(map(str, value))
                    file.write(f"  --{key} {value_str} \\\n")
                else:
                    file.write(f"  --{key} {value} \\\n")
            else:
                # Write parameters with None value as comments
                file.write(f"  #--{key} None \\\n")
