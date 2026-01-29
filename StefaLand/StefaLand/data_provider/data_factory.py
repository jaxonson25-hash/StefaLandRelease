import os
import copy
import json
import pickle
import numpy as np
import torch
from MFFormer.data_provider.dataset import Dataset
from MFFormer.data_provider.dataloader import DataLoader
from MFFormer.config.config_basic import get_config, get_dataset_config, update_configs_multi_data_sources
from MFFormer.config.config_basic import get_station_ids_by_regionsHUC, get_station_ids_by_pub
from MFFormer.utils.tools import get_kfold_index


def data_provider(config, config_dataset, flag, scaler=None):
    dataset = Dataset

    data_set = dataset(
        config=config,
        config_dataset=config_dataset,
        flag=flag,
        scaler=scaler,
    )

    dataloader = DataLoader(data_set)
    return data_set, dataloader


def get_train_val_test_dataset(config, return_val=True, return_test=True):
    saved_dir = os.path.join(config.output_dir, config.saved_folder_name)
    index_dir = os.path.join(saved_dir, "index")

    do_CV = True if config.nth_kfold is not None else False
    do_PUR = config.test_mode == 'PUR'
    do_PUB = config.test_mode == 'PUB'

    train_data, val_data, test_data = None, None, None
    train_loader, val_loader, test_loader = None, None, None

    dataset_dict = {}
    for ndx, data_name in enumerate(config.data):

        config_dataset = get_dataset_config(data_name)

        if ndx > 0:
            config, config_dataset = update_configs_multi_data_sources(config, config_dataset)

        config_train = copy.deepcopy(config)
        config_dataset_train = copy.deepcopy(config_dataset)
        config_train.flag = 'train'
        config_val = copy.deepcopy(config)
        config_dataset_val = copy.deepcopy(config_dataset)
        config_val.flag = 'val'
        config_test = copy.deepcopy(config)
        config_dataset_test = copy.deepcopy(config_dataset)
        config_test.flag = 'test'

        if config.huc_regions is not None:
            # print(config.huc_regions)
            config.huc_test = [config.huc_regions[config.test_region_index]]
            # print(config.huc_test)
            trainhuc =  [item for index, item in enumerate(config.huc_regions) if index is not config.test_region_index]
            config.huc_train = trainhuc
            # print(config.huc_train)
        
        if do_PUB:
            testpubs = [config.test_pub]
            trainpubs = [item for item in range(1, 11) if item not in testpubs]


        # save index file
        index_train_file = f"index_cv_num_kfold_{config.num_kfold}_nth_kfold_{config.nth_kfold}_train_{data_name}.json"
        index_test_file = f"index_cv_num_kfold_{config.num_kfold}_nth_kfold_{config.nth_kfold}_test_{data_name}.json"
        index_train_station_ids_file = f"index_cv_num_kfold_{config.num_kfold}_nth_kfold_{config.nth_kfold}_train_station_ids_{data_name}.json"
        index_test_station_ids_file = f"index_cv_num_kfold_{config.num_kfold}_nth_kfold_{config.nth_kfold}_test_station_ids_{data_name}.json"
        scaler_file = f"scaler_{data_name}.pkl"

        if config.scaler_file is not None:
            # load the scaler file using pickle
            with open(os.path.join(index_dir, config.scaler_file), 'rb') as file:
                scaler = pickle.load(file)
        else:
            scaler = None

        # Get original station_ids as numpy array for CV indexing
        original_station_ids = np.array(config_dataset.station_ids)
        
        if do_CV:
            station_ids = original_station_ids  # Keep as numpy array for CV
            num_pixel = len(station_ids)
            index_train_list, index_test_list = get_kfold_index(num_pixel=num_pixel, num_kfold=config.num_kfold,
                                                                seed=config.seed)
            cv_index_train = index_train_list[config.nth_kfold]
            cv_index_test = index_test_list[config.nth_kfold]

            with open(os.path.join(index_dir, index_train_file), 'w') as file:
                json.dump(cv_index_train, file)

            with open(os.path.join(index_dir, index_test_file), 'w') as file:
                json.dump(cv_index_test, file)

            with open(os.path.join(index_dir, index_train_station_ids_file), 'w') as file:
                json.dump(station_ids[cv_index_train].tolist(), file)

            with open(os.path.join(index_dir, index_test_station_ids_file), 'w') as file:
                json.dump(station_ids[cv_index_test].tolist(), file)

        if do_PUR:
            print("Train ", end='')
            station_ids_train, lat, lon = get_station_ids_by_regionsHUC(config, config_dataset, config.huc_train)
            setattr(config_train, 'station_ids', station_ids_train)
            setattr(config_dataset_train, 'station_ids', station_ids_train)
        if do_PUB:
            print("Train ", end='')
            station_ids_train, lat, lon = get_station_ids_by_pub(config, config_dataset, trainpubs)
            setattr(config_train, 'station_ids', station_ids_train)
            setattr(config_dataset_train, 'station_ids', station_ids_train)
        elif do_CV:
            # Use numpy array indexing for CV
            setattr(config_train, 'station_ids', original_station_ids[cv_index_train].tolist())
            setattr(config_dataset_train, 'station_ids', original_station_ids[cv_index_train].tolist())

        train_data, train_loader = data_provider(config=config_train, config_dataset=config_dataset_train, flag='train', scaler=scaler)
        scaler = train_data.scaler

        config_dataset.static_variables_category_dict = train_data.static_variables_category_dict
        config.static_variables_category_dict = train_data.static_variables_category_dict
        config.static_variables_category_dict = train_data.static_variables_category_dict

        if return_val:

            if do_PUR:
                print("Validation ", end='')
                station_ids_val, lat, lon = get_station_ids_by_regionsHUC(config, config_dataset, config.huc_test)
                setattr(config_val, 'station_ids', station_ids_val)
                setattr(config_dataset_val, 'station_ids', station_ids_val)
            elif do_PUB:
                print("Validation ", end='')
                station_ids_val, lat, lon = get_station_ids_by_pub(config, config_dataset, testpubs)
                setattr(config_val, 'station_ids', station_ids_val)
                setattr(config_dataset_val, 'station_ids', station_ids_val)
            
            
            elif do_CV:
                # Use numpy array indexing for CV
                setattr(config_val, 'station_ids', original_station_ids[cv_index_test].tolist())
                setattr(config_dataset_val, 'station_ids', original_station_ids[cv_index_test].tolist())

            val_data, val_loader = data_provider(config=config_val, config_dataset=config_dataset_val, flag='val',
                                                 scaler=scaler)


        if return_test:

            if do_PUR:
                print("Test ", end='')
                station_ids_test, lat, lon = get_station_ids_by_regionsHUC(config, config_dataset, config.huc_test)
                setattr(config_test, 'station_ids', station_ids_test)
                setattr(config_dataset_test, 'station_ids', station_ids_test)
            elif do_PUB:
                print("Test ", end='')
                station_ids_test, lat, lon = get_station_ids_by_pub(config, config_dataset, testpubs)
                setattr(config_test, 'station_ids', station_ids_test)
                setattr(config_dataset_test, 'station_ids', station_ids_test)
            
            elif do_CV:
                # Use numpy array indexing for CV
                setattr(config_test, 'station_ids', original_station_ids[cv_index_test].tolist())
                setattr(config_dataset_test, 'station_ids', original_station_ids[cv_index_test].tolist())

            test_data, test_loader = data_provider(config=config_test, config_dataset=config_dataset_test, flag='test',
                                                   scaler=scaler)

        dataset_dict[data_name] = {
            "train_data": train_data,
            "train_loader": train_loader,
            "val_data": val_data,
            "val_loader": val_loader,
            "test_data": test_data,
            "test_loader": test_loader,
            "scaler": scaler,
            "config_dataset": config_dataset,
            "config": config
        }

        # save the scaler file using pickle
        with open(os.path.join(index_dir, scaler_file), 'wb') as file:
            pickle.dump(scaler, file)

    return dataset_dict



# def get_train_val_test_datasetCamels(config, return_val=True, return_test=True):
#     saved_dir = os.path.join(config.output_dir, config.saved_folder_name)
#     index_dir = os.path.join(saved_dir, "index")

#     do_CV = config.nth_kfold is not None
#     do_PUR = config.test_mode == 'PUR'
#     do_PUB = config.test_mode == 'PUB'
    
#     # Validate configuration for PUR mode
#     if do_PUR:
#         if config.huc_regions is None or len(config.huc_regions) == 0:
#             raise ValueError("For PUR mode, huc_regions must be specified and non-empty.")
#         if config.test_region_index is None:
#             raise ValueError("For PUR mode, test_region_index must be specified.")

#     train_data, val_data, test_data = None, None, None
#     train_loader, val_loader, test_loader = None, None, None

#     dataset_dict = {}
#     for ndx, data_name in enumerate(config.data):
#         config_dataset = get_dataset_config(data_name)

#         if ndx > 0:
#             config, config_dataset = update_configs_multi_data_sources(config, config_dataset)

#         config_train = copy.deepcopy(config)
#         config_dataset_train = copy.deepcopy(config_dataset)
#         config_train.flag = 'train'
#         config_val = copy.deepcopy(config)
#         config_dataset_val = copy.deepcopy(config_dataset)
#         config_val.flag = 'val'
#         config_test = copy.deepcopy(config)
#         config_dataset_test = copy.deepcopy(config_dataset)
#         config_test.flag = 'test'

#         # Load or create scaler
#         scaler_file = f"scaler_{data_name}.pkl"
#         scaler = load_scaler(config, index_dir, scaler_file)

#         # save index file
#         index_train_file = f"index_cv_num_kfold_{config.num_kfold}_nth_kfold_{config.nth_kfold}_train_{data_name}.json"
#         index_test_file = f"index_cv_num_kfold_{config.num_kfold}_nth_kfold_{config.nth_kfold}_test_{data_name}.json"
#         index_train_station_ids_file = f"index_cv_num_kfold_{config.num_kfold}_nth_kfold_{config.nth_kfold}_train_station_ids_{data_name}.json"
#         index_test_station_ids_file = f"index_cv_num_kfold_{config.num_kfold}_nth_kfold_{config.nth_kfold}_test_station_ids_{data_name}.json"

        

#         # Initialize CAMELSDataset
#         full_dataset = CAMELSDataset(config=config, config_dataset=config_dataset, flag='train', scaler=scaler)
        
#         # Perform data splitting based on the chosen mode
#         if do_CV:
#             train_indices, test_indices = get_cv_indices(config, full_dataset, index_dir, data_name)
#         elif do_PUR:
#             train_indices, test_indices = full_dataset.split_pur(config.test_region_index)
#         elif do_PUB:
#             train_indices, test_indices = full_dataset.split_pub(test_fraction=config.test_fraction)
#         else:
#             raise ValueError(f"Invalid test_mode: {config.test_mode}. Choose 'CV', 'PUR', or 'PUB'.")

#         # Further split train into train and validation
#         if return_val:
#             train_size = int(0.8 * len(train_indices))
#             train_indices, val_indices = train_indices[:train_size], train_indices[train_size:]
#         else:
#             val_indices = []

#         # # Create subsets of the dataset
#         # train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
#         # val_dataset = torch.utils.data.Subset(full_dataset, val_indices) if return_val else None
#         # test_dataset = torch.utils.data.Subset(full_dataset, test_indices) if return_test else None

#         # # Create data loaders
#         # train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
#         # val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False) if return_val else None
#         # test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False) if return_test else None

   
#         setattr(config_train, 'station_ids', train_indices)
#         setattr(config_dataset_train, 'station_ids', train_indices)
   

#         train_data, train_loader = data_provider(config=config_train, config_dataset=config_dataset_train, flag='train', scaler=scaler)
#         scaler = train_data.scaler

#         config_dataset.static_variables_category_dict = train_data.static_variables_category_dict
#         config.static_variables_category_dict = train_data.static_variables_category_dict
#         config.static_variables_category_dict = train_data.static_variables_category_dict

#         if return_val:
#             if do_PUR:
#                 setattr(config_val, 'station_ids', val_indices)
#                 setattr(config_dataset_val, 'station_ids', val_indices)
#             elif do_CV:
#                 setattr(config_val, 'station_ids', val_indices)
#                 setattr(config_dataset_val, 'station_ids', val_indices)

#             val_data, val_loader = data_provider(config=config_val, config_dataset=config_dataset_val, flag='val',
#                                                  scaler=scaler)


#         if return_test:
#             if do_PUR:
#                 setattr(config_test, 'station_ids', test_indices)
#                 setattr(config_dataset_test, 'station_ids', test_indices)
#             elif do_CV:
#                 setattr(config_test, 'station_ids', test_indices)
#                 setattr(config_dataset_test, 'station_ids', test_indices)

#             test_data, test_loader = data_provider(config=config_test, config_dataset=config_dataset_test, flag='test',
#                                                    scaler=scaler)


#         # Update scaler and configurations
#         scaler = full_dataset.scaler

#         dataset_dict[data_name] = {
#             "train_data": train_dataset,
#             "train_loader": train_loader,
#             "val_data": val_dataset,
#             "val_loader": val_loader,
#             "test_data": test_dataset,
#             "test_loader": test_loader,
#             "scaler": scaler,
#             "config_dataset": config_dataset,
#             "config": config
#         }

#         # Save the scaler
#         save_scaler(scaler, index_dir, scaler_file)

#     return dataset_dict

# def get_cv_indices(config, dataset, index_dir, data_name):
#     num_samples = len(dataset)
#     index_train_list, index_test_list = get_kfold_index(num_pixel=num_samples, num_kfold=config.num_kfold, seed=config.seed)
#     cv_index_train = index_train_list[config.nth_kfold]
#     cv_index_test = index_test_list[config.nth_kfold]

#     save_indices(index_dir, data_name, config.num_kfold, config.nth_kfold, cv_index_train, cv_index_test, dataset.gages_df.index)

#     return cv_index_train, cv_index_test

# def load_scaler(config, index_dir, scaler_file):
#     if config.scaler_file is not None:
#         with open(os.path.join(index_dir, config.scaler_file), 'rb') as file:
#             return pickle.load(file)
#     return None


# def save_indices(index_dir, data_name, num_kfold, nth_kfold, cv_index_train, cv_index_test, all_indices):
#     for name, indices in [('train', cv_index_train), ('test', cv_index_test)]:
#         index_file = f"index_cv_num_kfold_{num_kfold}_nth_kfold_{nth_kfold}_{name}_{data_name}.json"
#         station_ids_file = f"index_cv_num_kfold_{num_kfold}_nth_kfold_{nth_kfold}_{name}_station_ids_{data_name}.json"
        
#         with open(os.path.join(index_dir, index_file), 'w') as file:
#             json.dump(indices.tolist(), file)
        
#         with open(os.path.join(index_dir, station_ids_file), 'w') as file:
#             json.dump(all_indices[indices].tolist(), file)

# def save_scaler(scaler, index_dir, scaler_file):
#     with open(os.path.join(index_dir, scaler_file), 'wb') as file:
#         pickle.dump(scaler, file)
