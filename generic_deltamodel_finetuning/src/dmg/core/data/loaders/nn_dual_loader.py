import copy
import logging
import os
import numpy as np
import torch
from typing import Dict, Any, Optional
import pandas as pd

from dmg.core.data.loaders.base import BaseLoader
from dmg.core.data.loaders.load_nc import NetCDFDataset
from dmg.core.data.data import extract_temporal_features
from dmg.core.data.loader_utils import (
    load_nn_data,
    load_norm_stats,
    normalize_data,
    to_tensor,
    split_by_basin,
)

log = logging.getLogger(__name__)


class NnDualLoader(BaseLoader):
    """
    NnDirectLoader + an additional pretrained NN input stream loaded from
    config['pretrained_path'].

    Keeps ALL original keys (x_phy, c_phy, x_nn, c_nn, xc_nn_norm, target, temporal_features).
    Adds:
      - xc_pretrained_norm
    """

    def __init__(
        self,
        config: Dict[str, Any],
        test_split: Optional[bool] = False,
        overwrite: Optional[bool] = False,
        holdout_index: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.nc_tool = NetCDFDataset()
        self.config = config
        self.test_split = test_split
        self.overwrite = overwrite

        # task vars (unchanged)
        self.nn_attributes = config["delta_model"]["nn_model"].get("attributes", [])
        self.nn_forcings = config["delta_model"]["nn_model"].get("forcings", [])
        self.target = config["train"]["target"]

        # pretrained vars
        nn_cfg = config["delta_model"]["nn_model"]
        self.pretrained_ts_vars = nn_cfg.get("pretrained_time_series_vars", [])
        self.pretrained_static_vars = nn_cfg.get("pretrained_static_vars", [])

        # paths
        self.task_nc_path = config["data_path"]
        self.pretrained_nc_path = config["pretrained_path"]

        # norms
        self.log_norm_vars = config["delta_model"]["phy_model"].get("use_log_norm", [])
        out_base = config.get("out_path", "results")
        self.task_norm_path = os.path.join(out_base, "normalization_statistics.json")
        self.pre_norm_path = os.path.join(out_base, "normalization_statistics_pretrained.json")

        self.device = config["device"]
        self.dtype = torch.float32

        # spatial testing config (unchanged)
        self.test = config.get("test", {})
        self.is_spatial_test = (self.test and self.test.get("type") == "spatial")
        if holdout_index is not None:
            self.holdout_index = holdout_index
        elif self.is_spatial_test and "current_holdout_index" in self.test:
            self.holdout_index = self.test["current_holdout_index"]
        elif self.is_spatial_test and self.test.get("holdout_indexs"):
            self.holdout_index = self.test["holdout_indexs"][0]
        else:
            self.holdout_index = None

        self.train_dataset = None
        self.eval_dataset = None
        self.dataset = None
        self.norm_stats = None
        self.pre_norm_stats = None

        self.load_dataset()

    def load_dataset(self) -> None:
        train_range = {"start": self.config["train"]["start_time"], "end": self.config["train"]["end_time"]}
        test_range = {"start": self.config["test"]["start_time"], "end": self.config["test"]["end_time"]}

        if self.is_spatial_test:
            train_data = self._preprocess_data("train", train_range)
            test_data = self._preprocess_data("test", test_range)

            self.train_dataset, _ = split_by_basin(train_data, self.config, self.test, self.holdout_index)
            _, self.eval_dataset = split_by_basin(test_data, self.config, self.test, self.holdout_index)
        else:
            if self.test_split:
                self.train_dataset = self._preprocess_data("train", train_range)
                self.eval_dataset = self._preprocess_data("test", test_range)
            else:
                full_range = {"start": self.config["train"]["start_time"], "end": self.config["test"]["end_time"]}
                self.dataset = self._preprocess_data("all", full_range)

    def _preprocess_data(self, scope: str, t_range: Dict[str, str]) -> Dict[str, torch.Tensor]:
        # --- task data: EXACTLY like NnDirectLoader
        nn_data = load_nn_data(
            self._cfg_with_data_path(self.task_nc_path),
            scope,
            t_range,
            self.nn_forcings,
            self.nn_attributes,
            self.target,
            self.device,
            self.nc_tool,
        )

        target = nn_data["target"]
        x_nn = nn_data["x_nn"].cpu().numpy() if torch.is_tensor(nn_data["x_nn"]) else nn_data["x_nn"]
        c_nn = nn_data["c_nn"].cpu().numpy() if torch.is_tensor(nn_data["c_nn"]) else nn_data["c_nn"]

        # placeholders required by HydroSampler
        x_phy = np.zeros((target.shape[0], target.shape[1], 0), dtype=np.float32)
        c_phy = np.zeros((c_nn.shape[0], 0), dtype=np.float32)

        # temporal features (same as your existing loader)
        start_date = pd.to_datetime(t_range["start"].replace("/", "-"))
        end_date = pd.to_datetime(t_range["end"].replace("/", "-"))
        warmup_days = self.config["delta_model"]["phy_model"]["warm_up"]
        date_range = pd.date_range(start_date - pd.Timedelta(days=warmup_days), end_date, freq="D")
        temporal_features = extract_temporal_features(date_range)

        # task normalization (unchanged)
        self.norm_stats = load_norm_stats(
            self.task_norm_path,
            self.overwrite,
            x_nn,
            c_nn,
            target,
            self.nn_forcings,
            self.nn_attributes,
            self.target,
            self.log_norm_vars,
            self.config,
        )
        xc_nn_norm = normalize_data(x_nn, c_nn, self.nn_forcings, self.nn_attributes, self.norm_stats, self.log_norm_vars)

        # --- pretrained data: additional read from pretrained_path
        pre_nn = load_nn_data(
            self._cfg_with_data_path(self.pretrained_nc_path),
            scope,
            t_range,
            self.pretrained_ts_vars,
            self.pretrained_static_vars,
            [],  # no target from pretrained file
            self.device,
            self.nc_tool,
        )
        x_pre = pre_nn["x_nn"].cpu().numpy() if torch.is_tensor(pre_nn["x_nn"]) else pre_nn["x_nn"]
        c_pre = pre_nn["c_nn"].cpu().numpy() if torch.is_tensor(pre_nn["c_nn"]) else pre_nn["c_nn"]

        # pretrained normalization (separate stats file, no log norms)
        dummy_target = np.zeros((x_pre.shape[0], x_pre.shape[1], 1), dtype=np.float32)
        self.pre_norm_stats = load_norm_stats(
            self.pre_norm_path,
            self.overwrite,
            x_pre,
            c_pre,
            dummy_target,
            self.pretrained_ts_vars,
            self.pretrained_static_vars,
            ["_dummy_"],
            [],
            self.config,
        )
        xc_pretrained_norm = normalize_data(
            x_pre, c_pre, self.pretrained_ts_vars, self.pretrained_static_vars, self.pre_norm_stats, []
        )

        # dataset dict: keep everything old, add one new key
        return {
            "x_phy": to_tensor(x_phy, self.device, self.dtype),
            "c_phy": to_tensor(c_phy, self.device, self.dtype),
            "x_nn": to_tensor(x_nn, self.device, self.dtype),
            "c_nn": to_tensor(c_nn, self.device, self.dtype),
            "xc_nn_norm": to_tensor(xc_nn_norm, self.device, self.dtype),
            "temporal_features": to_tensor(temporal_features, self.device, self.dtype),
            "target": to_tensor(target, self.device, self.dtype),

            # new
            "xc_pretrained_norm": to_tensor(xc_pretrained_norm, self.device, self.dtype),
        }

    def _cfg_with_data_path(self, data_path: str) -> Dict[str, Any]:
        # load_nn_data reads config['data_path'], so override that only for this call
        cfg = dict(self.config)
        cfg["data_path"] = data_path
        return cfg
