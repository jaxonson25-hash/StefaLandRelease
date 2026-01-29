import csv
import json
import logging
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt
import scipy.stats as stats
from pydantic import BaseModel, ConfigDict, model_validator

log = logging.getLogger()


class Metrics(BaseModel):
    """Metrics for model evaluation.

    Using Pydantic BaseModel for validation.
    Metrics are calculated at each grid point and are listed below.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    pred: npt.NDArray[np.float32]
    target: npt.NDArray[np.float32]
    
    # Metrics from paste-2.txt
    bias: npt.NDArray[np.float32] = np.ndarray([])
    rmse: npt.NDArray[np.float32] = np.ndarray([])
    rmse_ub: npt.NDArray[np.float32] = np.ndarray([])
    corr: npt.NDArray[np.float32] = np.ndarray([])
    corr_spearman: npt.NDArray[np.float32] = np.ndarray([])
    r2: npt.NDArray[np.float32] = np.ndarray([])
    nse: npt.NDArray[np.float32] = np.ndarray([])
    
    flv: npt.NDArray[np.float32] = np.ndarray([])
    fhv: npt.NDArray[np.float32] = np.ndarray([])
    pbias: npt.NDArray[np.float32] = np.ndarray([])
    pbias_mid: npt.NDArray[np.float32] = np.ndarray([])
    
    kge: npt.NDArray[np.float32] = np.ndarray([])
    kge_12: npt.NDArray[np.float32] = np.ndarray([])
    
    rmse_low: npt.NDArray[np.float32] = np.ndarray([])
    rmse_mid: npt.NDArray[np.float32] = np.ndarray([])
    rmse_high: npt.NDArray[np.float32] = np.ndarray([])

    def __init__(
        self,
        pred: npt.NDArray[np.float32],
        target: npt.NDArray[np.float32],
    ) -> None:
        if pred.ndim == 1:
            pred = np.expand_dims(pred, axis=0)
        if target.ndim == 1:
            target = np.expand_dims(target, axis=0)

        super().__init__(pred=pred, target=target)

    def model_post_init(self, __context: Any) -> Any:
        """Calculate metrics.

        This method is called after the model is initialized.

        Parameters
        ----------
        __context : Any
            Context object.

        Returns
        -------
        Any
            Context object.
        """
        self.bias = np.nanmean(self.pred - self.target, axis=1)  # From statError function
        self.rmse = self._rmse(self.pred, self.target)
        self.rmse_ub = self._rmse_ub(self.pred, self.target)

        self.corr = np.full(self.ngrid, np.nan)
        self.corr_spearman = np.full(self.ngrid, np.nan)
        self.r2 = np.full(self.ngrid, np.nan)
        self.nse = np.full(self.ngrid, np.nan)

        self.flv = np.full(self.ngrid, np.nan)
        self.fhv = np.full(self.ngrid, np.nan)
        self.pbias = np.full(self.ngrid, np.nan)
        self.pbias_mid = np.full(self.ngrid, np.nan)

        self.kge = np.full(self.ngrid, np.nan)
        self.kge_12 = np.full(self.ngrid, np.nan)

        self.rmse_low = np.full(self.ngrid, np.nan)
        self.rmse_mid = np.full(self.ngrid, np.nan)
        self.rmse_high = np.full(self.ngrid, np.nan)

        for i in range(0, self.ngrid):
            _pred = self.pred[i]
            _target = self.target[i]
            idx = np.where(
                np.logical_and(~np.isnan(_pred), ~np.isnan(_target))
            )[0]

            if idx.shape[0] > 0:
                pred = _pred[idx]
                target = _target[idx]

                # Calculating FHV and FLV metrics as in statError function
                pred_sort = np.sort(pred)
                target_sort = np.sort(target)
                index_low = round(0.3 * pred_sort.shape[0])
                index_high = round(0.98 * pred_sort.shape[0])

                low_pred = pred_sort[:index_low]
                mid_pred = pred_sort[index_low:index_high]
                high_pred = pred_sort[index_high:]

                low_target = target_sort[:index_low]
                mid_target = target_sort[index_low:index_high]
                high_target = target_sort[index_high:]
                
                # Using the same calculation method as statError
                self.pbias[i] = np.sum(pred - target) / np.sum(target) * 100
                self.flv[i] = np.sum(low_pred - low_target) / np.sum(low_target) * 100
                self.fhv[i] = np.sum(high_pred - high_target) / np.sum(high_target) * 100
                self.pbias_mid[i] = np.sum(mid_pred - mid_target) / np.sum(mid_target) * 100
                
                self.rmse_low[i] = np.sqrt(np.nanmean((low_pred - low_target) ** 2))
                self.rmse_mid[i] = np.sqrt(np.nanmean((mid_pred - mid_target) ** 2))
                self.rmse_high[i] = np.sqrt(np.nanmean((high_pred - high_target) ** 2))

                if idx.shape[0] > 1:
                    # At least two points needed for correlation.
                    self.corr[i] = stats.pearsonr(pred, target)[0]
                    self.corr_spearman[i] = stats.spearmanr(pred, target)[0]

                    _pred_mean = pred.mean()
                    _target_mean = target.mean()
                    _pred_std = np.std(pred)
                    _target_std = np.std(target)
                    
                    # KGE calculation from statError
                    self.kge[i] = 1 - np.sqrt(
                        (self.corr[i] - 1) ** 2 + 
                        (_pred_std / _target_std - 1) ** 2 + 
                        (_pred_mean / _target_mean - 1) ** 2
                    )
                    
                    # KGE12 calculation from statError
                    self.kge_12[i] = 1 - np.sqrt(
                        (self.corr[i] - 1) ** 2 + 
                        ((_pred_std * _target_mean) / (_target_std * _pred_mean) - 1) ** 2 + 
                        (_pred_mean / _target_mean - 1) ** 2
                    )
                    
                    # NSE/R2 calculation from statError
                    sst = np.sum((target - _target_mean) ** 2)
                    ssres = np.sum((target - pred) ** 2)
                    self.nse[i] = self.r2[i] = 1 - ssres / sst

        return super().model_post_init(__context)

    @model_validator(mode='after')
    @classmethod
    def validate_pred(cls, metrics: Any) -> Any:
        """Checks that there are no NaN predictions."""
        pred = metrics.pred
        if np.isnan(pred).sum() > 0:
            msg = "Pred contains NaN, check your gradient chain"
            log.exception(msg)
            raise ValueError(msg)
        return metrics
    
    def calc_stats(self, *args, **kwargs) -> Dict[str, Dict[str, float]]:
        """Calculate aggregate statistics of metrics."""
        stats = {}
        model_dict = self.model_dump()
        model_dict.pop('pred', None)
        model_dict.pop('target', None)

        # Calculate statistics
        for key, value in model_dict.items():
            if isinstance(value, np.ndarray) and value.size > 0:
                stats[key] = {
                    'median': float(np.nanmedian(value)),
                    'mean': float(np.nanmean(value)),
                    'std': float(np.nanstd(value)),
                }
        return stats

    def model_dump_agg_stats(self, path: str) -> None:
        """Dump aggregate statistics (median, mean, std) to json or csv."""
        stats = self.calc_stats()
        
        if path.endswith('.json'):
            with open(path, 'w') as f:
                json.dump(stats, f, indent=4)
        elif path.endswith('.csv'):
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Metric', 'Median', 'Mean', 'Std'])
                for metric, values in stats.items():
                    writer.writerow([metric, values['median'], values['mean'], values['std']])
        else:
            raise ValueError("Provide either a .json or .csv file path.")
    
    def model_dump_json(self, *args, **kwargs) -> str:
        """Dump raw metrics to json."""
        model_dict = self.model_dump()
        for key, value in model_dict.items():
            if isinstance(value, np.ndarray):
                setattr(self, key, value.tolist())

        if hasattr(self, 'pred'):
            del self.pred
        if hasattr(self, 'target'):
            del self.target

        return super().model_dump_json(*args, **kwargs)
    
    def dump_metrics(self, path: str) -> None:
        """Dump all metrics and aggregate statistics (median, mean, std) to json."""
        # Save aggregate statistics
        save_path = os.path.join(path, 'metrics_agg.json')
        self.model_dump_agg_stats(save_path)

        # Save raw metrics
        save_path = os.path.join(path, f'metrics.json')
        json_dat = self.model_dump_json(indent=4)
        
        with open(save_path, "w") as f:
            json.dump(json_dat, f)        

    def tile_mean(self, data: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Calculate mean of data and tile it."""
        return np.tile(np.nanmean(data, axis=1), (self.nt, 1)).transpose()
    
    @property
    def ngrid(self) -> int:
        """Calculate number of items in grid."""
        return self.pred.shape[0]
    
    @property
    def nt(self) -> int:
        """Calculate number of time steps."""
        return self.pred.shape[1]
    
    @staticmethod
    def _rmse(
        pred: npt.NDArray[np.float32],
        target: npt.NDArray[np.float32],
        axis: Optional[int] = 1,
    ) -> npt.NDArray[np.float32]:
        """Calculate root mean square error."""
        return np.sqrt(np.nanmean((pred - target) ** 2, axis=axis))
    
    def _rmse_ub(
        self,
        pred: npt.NDArray[np.float32],
        target: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """Calculate unbiased root mean square error."""
        pred_mean = self.tile_mean(self.pred)
        target_mean = self.tile_mean(self.target)
        pred_anom = self.pred - pred_mean
        target_anom = self.target - target_mean
        return self._rmse(pred_anom, target_anom)