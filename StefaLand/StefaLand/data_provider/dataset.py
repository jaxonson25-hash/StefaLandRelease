import os
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as DS
from MFFormer.utils.data.timefeatures import time_features
import multiprocessing
from MFFormer.layers.mask import get_batch_mask
from MFFormer.data_provider.read_data import read_nc_data

warnings.filterwarnings('ignore')


class Dataset(DS):
    def __init__(self, config, config_dataset, flag='train', scaler=None, max_samples=None):

        self.config = config
        self.config_dataset = config_dataset
        self.mode = flag
        self.config.mode = flag

        self.seq_len = config.seq_len
        self.label_len = config.label_len
        self.pred_len = config.pred_len
        self.use_target_stds = config.use_target_stds
        self.batch_size = config.batch_size
        self.static_variables = config.static_variables

        self.index_dir = config.index_dir

        self.sample_len = self.get_sample_len()

        warmup_days = self.sample_len - self.pred_len
        self.raw_x, self.raw_y, self.raw_c, self.date_range, self.scaler, self.config_dataset = read_nc_data(
            config_dataset,
            warmup_days=warmup_days,
            mode=flag,
            scaler=scaler,
            sample_len=self.sample_len,
            data_type=config.data_type, )

        self.static_variables_category_dict = self.config_dataset.static_variables_category_dict

        assert self.mode in ["train", "valid", "test", "val"]
        assert len(self.date_range) == self.raw_x.shape[1]

        self.num_col = self.raw_x.shape[1] - self.sample_len + 1
        self.num_row = self.raw_x.shape[0]

        # gpu_num = min([30, multiprocessing.cpu_count()])
        # self.pool = multiprocessing.Pool(gpu_num)
        #reducing the total number as its unnessary and may restrict counts in training 
        ncpu = multiprocessing.cpu_count()
        nworkers = min(8, max(1, ncpu - 1))
        self.pool = multiprocessing.Pool(nworkers)
        return_mask = True if "pretrain" in self.config.task_name else False
        if (not config.sampling_regions is None) and (self.mode in ["train"]):
            self.get_batch_sample_index_by_regions(self.config_dataset.lon, self.config_dataset.lat,
                                                   config.sampling_regions, return_mask)
        else:
            self.get_batch_sample_index(return_mask)

        # load the first index
        self.load_index(0)
        self._len = len(self.slice_grid_list)

        self.time_stamp = time_features(pd.to_datetime(self.date_range), freq="1D").transpose(1, 0)  # shape = (8640, 4)

    def load_index(self, nth_file):
        index_file = os.path.join(self.index_dir, f"sampling_index_{self.mode}_{nth_file}.npz")
        index_dict = np.load(index_file, allow_pickle=True, mmap_mode="r")
        self.slice_grid_list, self.slice_time_list, self.slice_idx_list, self.slice_time_series_mask_index_list, \
        self.slice_static_mask_index_list = index_dict.values()
        assert len(self.slice_grid_list) == len(self.slice_time_list) == len(self.slice_idx_list) == len(
            self.slice_time_series_mask_index_list) == len(self.slice_static_mask_index_list)

    def set_length(self, new_length):
        self._len = new_length

    def get_sample_len(self):

        if self.config.task_name in ["forecast"]:
            sample_len = self.seq_len + self.pred_len
        else:
            sample_len = self.seq_len

        return sample_len

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        slice_grid = self.slice_grid_list[idx]
        slice_time = self.slice_time_list[idx]
        slice_idx = self.slice_idx_list[idx]
        slice_time_series_mask_index = self.slice_time_series_mask_index_list[idx]
        slice_static_mask_index = self.slice_static_mask_index_list[idx]
        slice_row = slice_idx // self.num_col
        slice_col = slice_idx % self.num_col

        # raw_x: basins, time, features
        sub_x = self.raw_x[slice_grid[:, None], slice_time[:, :self.seq_len], :]      # [B, T, F]
        sub_y = self.raw_y[slice_grid[:, None], slice_time[:, -self.pred_len:], :]    # [B, P, F]
        if self.raw_c is not None:
            sub_c = self.raw_c[slice_grid]                                            # [B, C]
        else:
            sub_c = np.empty(0)
        sub_x_original = sub_x.copy()

        # Timestamps (build BEFORE any pinning)
        x_flat_time = slice_time[:, :self.seq_len].reshape(-1)
        x_extracted_times = self.time_stamp[x_flat_time, :]
        x_time_stamp = x_extracted_times.reshape(len(slice_grid), -1, self.time_stamp.shape[-1])
        x_time_stamp = torch.from_numpy(x_time_stamp.astype(np.float32))

        y_flat_time = slice_time[:, -self.pred_len:].reshape(-1)
        y_extracted_times = self.time_stamp[y_flat_time, :]
        y_time_stamp = y_extracted_times.reshape(len(slice_grid), -1, self.time_stamp.shape[-1])
        y_time_stamp = torch.from_numpy(y_time_stamp.astype(np.float32))

        # Core tensors
        x_torch = torch.from_numpy(sub_x.astype(np.float32))
        x_original_torch = torch.from_numpy(sub_x_original.astype(np.float32))
        y_torch = torch.from_numpy(sub_y.astype(np.float32))
        c_torch = torch.from_numpy(sub_c.astype(np.float32))
        slice_time_series_mask_index_torch = torch.from_numpy(slice_time_series_mask_index)
        slice_static_mask_index_torch = torch.from_numpy(slice_static_mask_index)

        # Optional stds
        if self.use_target_stds:
            target_std_list = self.scaler["y_std_samples"][slice_row].astype(np.float32).copy()
            target_std = torch.from_numpy(target_std_list)  # [B, 1]
            x_std_list = self.scaler["x_std_samples"][slice_row].astype(np.float32).copy()
            x_std = torch.from_numpy(x_std_list)            # [B, F]
        else:
            target_std = torch.empty(0)
            x_std = torch.empty(0)

        # Pin only after everything exists
        if getattr(self.config, "use_gpu", False):
            x_torch = x_torch.pin_memory()
            x_original_torch = x_original_torch.pin_memory()
            y_torch = y_torch.pin_memory()
            c_torch = c_torch.pin_memory()
            x_time_stamp = x_time_stamp.pin_memory()
            y_time_stamp = y_time_stamp.pin_memory()
            slice_time_series_mask_index_torch = slice_time_series_mask_index_torch.pin_memory()
            slice_static_mask_index_torch = slice_static_mask_index_torch.pin_memory()
            target_std = target_std.pin_memory() if target_std.numel() > 0 else target_std
            x_std = x_std.pin_memory() if x_std.numel() > 0 else x_std

        data_dict = {
            "batch_x": x_torch,
            "batch_x_original": x_original_torch,
            "batch_y": y_torch,
            "batch_c": c_torch,
            "batch_target_std": target_std,
            "batch_x_std": x_std,
            "batch_x_time_stamp": x_time_stamp,
            "batch_y_time_stamp": y_time_stamp,
            "batch_sample_idx": slice_idx,
            "batch_sample_row": slice_row,
            "batch_sample_col": slice_col,
            "batch_time_series_mask_index": slice_time_series_mask_index_torch,
            "batch_static_mask_index": slice_static_mask_index_torch,
            "sample_len": self.sample_len,
        }
        return data_dict


    def get_batch_sample_index_by_regions(self, lon, lat, regions, return_mask=False):
        import copy
        from MFFormer.datasets.load_path import LoadPath
        from MFFormer.utils.gis.vector_tools import VectorTools

        assert len(lon) == len(lat) == self.num_row
        assert len(regions) > 1
        self.num_index_files = len(regions)

        all_num_index_matrix = self.num_row * self.num_col
        all_index_matrix = np.arange(0, all_num_index_matrix)  # shape: num_grid * num_time

        if self.config.sampling_stride > 1:
            all_index_matrix = all_index_matrix[::self.config.sampling_stride]
            all_num_index_matrix = len(all_index_matrix)

        vts = VectorTools()
        index_dir = self.index_dir

        for idx_region, region in enumerate(regions):
            shp_path = LoadPath.shapefile(region)
            index_within_region = vts.pts_within_shp(inp_shp=shp_path, lon_list=lon, lat_list=lat, opt_path=index_dir)
            print(region, len(index_within_region))

            all_index_matrix_within_region = copy.deepcopy(all_index_matrix)[index_within_region]

            if self.mode in ["train"]:
                np.random.shuffle(all_index_matrix_within_region)

            self.get_batch_sample_index_per_file(all_index_matrix_within_region, idx_region, return_mask)

    def get_batch_sample_index(self, return_mask=False):

        max_num_samples_per_time = 7000000
        self.num_index_files = 1

        all_num_index_matrix = self.num_row * self.num_col
        all_index_matrix = np.arange(0, all_num_index_matrix)  # shape: num_grid * num_time

        if self.config.sampling_stride > 1:
            all_index_matrix = all_index_matrix[::self.config.sampling_stride]
            all_num_index_matrix = len(all_index_matrix)

        if self.mode in ["train"]:
            np.random.shuffle(all_index_matrix)  # shuffle index matrix

        # split the index matrix into multiple files
        if all_num_index_matrix > max_num_samples_per_time:
            num_files = all_num_index_matrix // max_num_samples_per_time + bool(all_num_index_matrix % max_num_samples_per_time)
            index_matrix_list = np.array_split(all_index_matrix, num_files)
            for idx_file, index_matrix in enumerate(index_matrix_list):
                self.get_batch_sample_index_per_file(index_matrix, idx_file, return_mask)
            self.num_index_files = num_files
        else:
            self.get_batch_sample_index_per_file(all_index_matrix, 0, return_mask)


    def get_batch_sample_index_per_file(self, index_matrix, nth_file, return_mask):

        num_index_matrix = len(index_matrix)
        num_batch_samples = num_index_matrix // self.batch_size
        # split index matrix into batches
        slice_grid_list = []
        slice_time_list = []
        slice_index_list = []
        batch_index_list = np.array_split(index_matrix,
                                          num_batch_samples + bool(num_index_matrix % self.batch_size))
        each_batch_num_list = [len(batch) for batch in batch_index_list]

        for idx, index_matrix in enumerate(batch_index_list):
            # get index_grid and index_time
            grid_list = index_matrix // self.num_col
            time_start_list = index_matrix % self.num_col
            time_list = time_start_list[:, None] + np.arange(self.sample_len)

            slice_grid_list.append(grid_list)
            slice_time_list.append(time_list)
            slice_index_list.append(index_matrix)

        if return_mask:
            args_list = [(num, self.config) for num in each_batch_num_list]
            results = self.pool.starmap(get_batch_mask, args_list)
            slice_time_series_mask_index_list, slice_static_mask_index_list = zip(*results)
        else:
            slice_time_series_mask_index_list = [np.empty(0)] * len(batch_index_list)
            slice_static_mask_index_list = [np.empty(0)] * len(batch_index_list)

        index_dict = {
            "slice_grid_list": slice_grid_list,
            "slice_time_list": slice_time_list,
            "slice_index_list": slice_index_list,
            "slice_time_series_mask_index_list": slice_time_series_mask_index_list,
            "slice_static_mask_index_list": slice_static_mask_index_list,
        }

        # saving the index
        index_save_name = f"sampling_index_{self.mode}_{nth_file}.npz"
        np.savez(os.path.join(self.index_dir, index_save_name), **{k: (np.array(v, dtype=object) if isinstance(v, (list, tuple)) else v) for k, v in index_dict.items()})

        self.slice_grid_list = slice_grid_list

    def __del__(self):
        self.pool.close()
        self.pool.join()

    def update_samples_index(self, return_mask):
        self.get_batch_sample_index(return_mask)

    #new method to replace update_samples_index(return_mask) in every epoch
    def reshuffle_batches(self):
        n = len(self.slice_idx_list)
        perm = np.random.permutation(n)

        def reord(lst): 
            return [lst[i] for i in perm]

        self.slice_grid_list = reord(self.slice_grid_list)
        self.slice_time_list = reord(self.slice_time_list)
        self.slice_idx_list  = reord(self.slice_idx_list)
        self.slice_time_series_mask_index_list = reord(self.slice_time_series_mask_index_list)
        self.slice_static_mask_index_list      = reord(self.slice_static_mask_index_list)
        self._len = len(self.slice_idx_list)


    @staticmethod
    def restore_data(data: np.ndarray, num_stations: int) -> np.ndarray:
        """
        data: num_total_sample, seq_len, feature_dim  --> num_stations, num_valid_time, seq_len, feature_dim --> num_stations, num_valid_time, feature_dim
        data_obs: num_sites, num_time, feature_dim
        """
        assert len(data.shape) == 3
        num_total_sample, seq_len, feature_dim = data.shape
        reshaped_data = data.reshape(num_stations, -1, seq_len, feature_dim)
        num_stations, num_time_len, seq_len, feature_dim = reshaped_data.shape

        padding_num = (num_time_len - 1) % seq_len

        if padding_num != 0:
            padding_data = reshaped_data[:, -1, :, :][:, -padding_num:, :]
        else:
            padding_data = np.empty([num_stations, 0, feature_dim], dtype=reshaped_data.dtype)

        sliced_data = reshaped_data[:, ::seq_len, :, :].reshape(num_stations, -1, feature_dim)

        restored_data = np.concatenate([sliced_data, padding_data], axis=1)

        return restored_data

    def inverse_transform(self, data, mean=None, std=None, inverse_categorical=False, single_variable_name=None):

        if mean is None:
            mean = self.scaler["y_mean"]
        if std is None:
            std = self.scaler["y_std"]

        data_rescale = data * std + mean

        if inverse_categorical:
            if single_variable_name is None:
                if not self.static_variables_category_dict is None:
                    for categorical_name in self.static_variables_category_dict.keys():
                        index_to_class = self.static_variables_category_dict[categorical_name]["index_to_class"]
                        categorical_index = self.static_variables.index(categorical_name)
                        data_rescale[..., categorical_index] = np.vectorize(index_to_class.get)(
                            data[..., categorical_index])
            elif (not single_variable_name is None) & (single_variable_name in self.config.static_variables_category):
                assert data.shape[-1] == 1
                index_to_class = self.static_variables_category_dict[single_variable_name]["index_to_class"]
                data_rescale = np.vectorize(index_to_class.get)(data)

        return data_rescale
