import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from contextlib import nullcontext
import os

import numpy as np
import torch
import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from numpy.typing import NDArray

from dmg.trainers.base import BaseTrainer
from dmg.core.calc.metrics import Metrics
from dmg.core.calc.fire_metrics import FireMetrics
from dmg.core.utils.factory import import_data_sampler, load_criterion
from dmg.core.data import create_training_grid

log = logging.getLogger(__name__)


class FinetuneTrainer(BaseTrainer):
    def __init__(
        self,
        config: Dict[str, Any],
        model: torch.nn.Module = None,
        train_dataset: Optional[dict] = None,
        eval_dataset: Optional[dict] = None,
        dataset: Optional[dict] = None,
        loss_func: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        verbose: Optional[bool] = False,
    ) -> None:
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = eval_dataset
        self.dataset = dataset
        self.device = config['device']
        self.verbose = verbose

        self.is_in_train = False
        self.epoch_train_loss_list: List[float] = []
        self.epoch_val_loss_list: List[float] = []

        self.sampler = import_data_sampler(config['data_sampler'])(config)
        self.metrics_type = self.config.get('metrics_type', 'standard')

        # forecast_days only matters for fire; keep 1 for standard
        self.forecast_days = getattr(self.sampler, 'forecast_days', 1) if self.metrics_type == 'fire' else 1
        self.model_type = self._determine_model_type()

        if 'train' in config['mode']:
            self.loss_func = loss_func or load_criterion(
                self.train_dataset['target'],
                config['loss_function'],
                device=config['device'],
            )
            self.model.loss_func = self.loss_func
            self.optimizer = optimizer or self.init_optimizer()
            self.start_epoch = self.config['train'].get('start_epoch', 0) + 1
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=self.config.get('lr_patience', 5),
                factor=self.config.get('lr_factor', 0.1),
            )

    def _determine_model_type(self) -> str:
        return 'hbv'

    def init_optimizer(self) -> torch.optim.Optimizer:
        optimizer_name = self.config['train']['optimizer']
        learning_rate = self.config['delta_model']['nn_model']['learning_rate']
        optimizer_dict = {
            'Adadelta': torch.optim.Adadelta,
            'Adam': torch.optim.Adam,
            'AdamW': torch.optim.AdamW,
            'SGD': torch.optim.SGD,
        }
        optimizer_cls = optimizer_dict.get(optimizer_name)
        if optimizer_cls is None:
            raise ValueError(f"Optimizer '{optimizer_name}' not recognized. "
                             f"Available options are: {list(optimizer_dict.keys())}")
        trainable_params = self.model.get_parameters()
        if len(trainable_params) == 0:
            raise ValueError("No trainable parameters found.")
        return optimizer_cls(trainable_params, lr=learning_rate)

    def train(self) -> None:
        log.info(f"Training model: Beginning {self.start_epoch} of {self.config['train']['epochs']} epochs")
        if self.metrics_type == 'fire':
            log.info(f"Using fixed forecast horizon: {self.forecast_days} days")

        results_dir = self.config.get('save_path', 'results')
        os.makedirs(results_dir, exist_ok=True)
        results_file = open(os.path.join(results_dir, "results.txt"), 'a')
        results_file.write(f"\nTrain start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        if self.metrics_type == 'fire':
            results_file.write(f"Forecast days: {self.forecast_days}\n")

        use_amp = self.config.get('use_amp', False)
        scaler = torch.cuda.amp.GradScaler() if use_amp and torch.cuda.is_available() else None

        n_samples, n_minibatch, n_timesteps = create_training_grid(
            self.train_dataset['xc_nn_norm'],
            self.config
        )
        log.info(f"training grid - basins: {n_samples}, timesteps: {n_timesteps}, mini_batches: {n_minibatch}")

        for epoch in range(self.start_epoch, self.config['train']['epochs'] + 1):
            train_loss: List[float] = []
            epoch_time = time.time()
            self.model.train()

            for _ in range(1, n_minibatch + 1):
                self.optimizer.zero_grad()
                batch_data = self.sampler.get_training_sample(
                    self.train_dataset,
                    n_samples,
                    n_timesteps
                )
                if self.metrics_type == 'fire':
                    batch_data['current_forecast_days'] = self.forecast_days

                cm = torch.cuda.amp.autocast() if scaler is not None else nullcontext()
                with cm:
                    outputs = self.model(batch_data)
                    target = batch_data['target']

                    if self.model_type == 'hbv':
                        hbv_output = outputs['Hbv_1_1p']
                        model_output = hbv_output['streamflow'] if isinstance(hbv_output, dict) and 'streamflow' in hbv_output else hbv_output
                    else:
                        model_output = outputs['Hbv_1_1p']
                        
                    sample_ids = batch_data.get('batch_sample', None)
                    if sample_ids is None:
                        raise KeyError("batch_data is missing 'batch_sample' required for NseBatchLoss")

                    loss = self.loss_func(model_output, target, sample_ids=sample_ids)

                    if not torch.isnan(loss):
                        train_loss.append(loss.item())

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

            self._log_epoch_stats(epoch, train_loss, [], epoch_time, results_file)

            if epoch % self.config['train']['save_epoch'] == 0:
                self.model.save_model(epoch)

        results_file.close()
        log.info("Training complete")

    # -------- Evaluation dispatcher -------- #
    def evaluate(self) -> Tuple[np.ndarray, np.ndarray]:
        self.is_in_train = False
        if self.metrics_type == 'fire':
            return self._evaluate_fire()
        else:
            return self._evaluate_standard()


    # -------- Standard evaluation -------- #
    def _evaluate_standard(self) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()

        obs = self.test_dataset["target"]
        obs_t = obs if torch.is_tensor(obs) else torch.as_tensor(obs)

        if obs_t.ndim != 3:
            raise ValueError(f"Expected target ndim=3, got {obs_t.ndim} with shape {tuple(obs_t.shape)}")

        # detect whether target is [N,T,1] or [T,N,1] by picking the larger dim as time
        # (works for CAMELS where T=6940 and N=531)
        if obs_t.shape[1] >= obs_t.shape[0]:
            # [N,T,1]
            N_total = int(obs_t.shape[0])
            T_total = int(obs_t.shape[1])
            target_time_major = False
        else:
            # [T,N,1]
            T_total = int(obs_t.shape[0])
            N_total = int(obs_t.shape[1])
            target_time_major = True

        # targets_np as [T,N]
        targets_np = obs_t.detach().cpu().numpy()
        if target_time_major:
            targets_np = targets_np[:, :, 0]  # [T,N]
        else:
            targets_np = targets_np[:, :, 0].transpose(1, 0)  # [T,N]

        # basin batching
        batch_size = int(self.config.get("test", {}).get("batch_size", N_total))
        starts = np.arange(0, N_total, batch_size, dtype=int)
        ends = np.append(starts[1:], N_total).astype(int)

        rho = int(self.config.get("train", {}).get("rho", 365))
        use_amp = bool(self.config.get("use_amp", False) and torch.cuda.is_available())
        model_name = self.config["delta_model"]["phy_model"]["model"][0]

        # time windows (stride=rho + final overlapping window)
        last_start = max(0, T_total - rho)
        t_starts = list(range(0, last_start + 1, rho))
        if len(t_starts) == 0:
            t_starts = [0]
        if t_starts[-1] != last_start:
            t_starts.append(last_start)

        def _time_slice_batched(sample_full: dict, t0: int, t1: int) -> dict:
            """
            sample_full is already basin-sliced by get_validation_sample(dataset, s, e).
            Only slice time dimension (the dimension matching T_total) for any 3D tensor.
            Move tensors to device.
            """
            out = {}
            for k, v in sample_full.items():
                if torch.is_tensor(v):
                    vt = v
                else:
                    # keep non-tensors as-is (sampler sometimes returns metadata)
                    out[k] = v
                    continue

                if vt.ndim == 3:
                    # find which axis is time by matching T_total
                    if vt.shape[0] == T_total:
                        # [T,B,C]
                        out[k] = vt[t0:t1, :, :].to(self.device)
                    elif vt.shape[1] == T_total:
                        # [B,T,C]
                        out[k] = vt[:, t0:t1, :].to(self.device)
                    else:
                        # not a time series tensor, just move it
                        out[k] = vt.to(self.device)
                else:
                    out[k] = vt.to(self.device)

            return out

        parts = []

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
            for s, e in zip(starts, ends):
                sample_full = self.sampler.get_validation_sample(self.test_dataset, int(s), int(e))

                # infer basin batch size B from any 3D tensor that has a T_total axis
                B = None
                for v in sample_full.values():
                    if torch.is_tensor(v) and v.ndim == 3 and (v.shape[0] == T_total or v.shape[1] == T_total):
                        B = int(v.shape[1] if v.shape[0] == T_total else v.shape[0])
                        break
                if B is None:
                    B = int(e - s)

                pred_be = torch.empty((T_total, B), device="cpu", dtype=torch.float32)

                for t0 in t_starts:
                    t1 = t0 + rho
                    sample_w = _time_slice_batched(sample_full, t0, t1)
                    out_dict = self.model(sample_w, eval=True)

                    out = out_dict[model_name]
                    if isinstance(out, dict) and "streamflow" in out:
                        out = out["streamflow"]

                    # normalize to [rho,B] in float
                    if torch.is_tensor(out):
                        o = out
                    else:
                        raise ValueError(f"Unexpected model output type: {type(out)}")

                    if o.ndim == 3 and o.shape[2] == 1:
                        o = o[:, :, 0]

                    # o might be [B,rho] or [rho,B]
                    if o.ndim == 2 and o.shape == (B, rho):
                        o = o.transpose(0, 1)
                    elif o.ndim == 2 and o.shape == (rho, B):
                        pass
                    else:
                        raise ValueError(
                            f"eval window got {tuple(o.shape)} expected {(rho,B)} or {(B,rho)} "
                            f"at basins [{s}:{e}] time [{t0}:{t1}]"
                        )

                    pred_be[t0:t1, :] = o.detach().cpu()

                parts.append(pred_be)

        pred = torch.cat(parts, dim=1).numpy()  # [T_total, N_total]

        os.makedirs(self.config["out_path"], exist_ok=True)
        Metrics(pred.swapaxes(0, 1), targets_np.swapaxes(0, 1)).dump_metrics(self.config["out_path"])
        np.save(os.path.join(self.config["out_path"], "pred.npy"), pred)
        np.save(os.path.join(self.config["out_path"], "target.npy"), targets_np)

        log.info(f"Saved eval metrics + arrays to: {self.config['out_path']}")
        return pred, obs




    # -------- Fire evaluation -------- #
    def _evaluate_fire(self) -> Tuple[np.ndarray, np.ndarray]:
        observations = self.test_dataset['target']
        n_time_steps = observations.shape[0]
        n_grid_cells = observations.shape[1]

        log.info(f"Starting temporal evaluation: {n_time_steps} time steps, {n_grid_cells} grid cells")
        max_prediction_time = n_time_steps - self.forecast_days + 1
        log.info(f"Will make predictions for {max_prediction_time} time steps (keeping {self.forecast_days} days for targets)")

        all_predictions = np.zeros((max_prediction_time, n_grid_cells, self.forecast_days))

        for t in tqdm.tqdm(range(max_prediction_time), desc='Temporal Evaluation'):
            time_step_data = self._get_time_step_data(self.test_dataset, t)
            prediction = self.model(time_step_data, eval=True)
            model_name = self.config['delta_model']['phy_model']['model'][0]
            model_output = prediction[model_name]

            if isinstance(model_output, torch.Tensor):
                pred_array = model_output.cpu().detach().numpy()
            else:
                raise ValueError(f"Unexpected model output type: {type(model_output)}")

            if pred_array.shape != (n_grid_cells, self.forecast_days):
                raise ValueError(f"Expected prediction shape {(n_grid_cells, self.forecast_days)}, got {pred_array.shape}")

            all_predictions[t] = pred_array

        log.info(f"Completed temporal evaluation. Final predictions shape: {all_predictions.shape}")

        targets_np = observations.cpu().numpy() if isinstance(observations, torch.Tensor) else observations
        self.calc_metrics_temporal(all_predictions, targets_np)
        return all_predictions, observations

    # -------- Helpers -------- #
    def _get_time_step_data(self, dataset: dict, time_step: int) -> dict:
        batch_data = {}
        for key, value in dataset.items():
            if isinstance(value, torch.Tensor):
                if value.ndim == 3:
                    start_time = max(0, time_step - self.config['delta_model']['rho'] + 1)
                    end_time = time_step + 1
                    batch_data[key] = value[start_time:end_time].to(device=self.device)
                elif value.ndim == 2:
                    batch_data[key] = value.to(device=self.device)
                else:
                    batch_data[key] = value.to(device=self.device)
            else:
                batch_data[key] = value
        if self.metrics_type == 'fire':
            batch_data['current_forecast_days'] = self.forecast_days
        return batch_data

    # ===== Fire metrics path ===== #
    def calc_metrics_temporal(self, predictions: np.ndarray, targets: np.ndarray) -> None:
        if targets.ndim == 3 and targets.shape[2] == 1:
            targets = targets.squeeze(-1)

        n_pred_time, n_grid_cells, forecast_days = predictions.shape
        n_target_time, n_target_grid_cells = targets.shape

        if n_grid_cells != n_target_grid_cells:
            raise ValueError(f"Grid cell mismatch: predictions {n_grid_cells} vs targets {n_target_grid_cells}")

        max_forecast_time = n_pred_time + forecast_days - 1
        if max_forecast_time > n_target_time:
            raise ValueError(f"Not enough target data: need up to time {max_forecast_time}, have {n_target_time}")

        pred_probs = 1 / (1 + np.exp(-np.clip(predictions, -500, 500))) if (predictions.min() < 0 or predictions.max() > 1) else predictions

        day_results = {}
        for day in range(forecast_days):
            day_predictions = []
            day_targets = []
            for t in range(n_pred_time):
                target_time = t + day
                if target_time < n_target_time:
                    day_predictions.append(pred_probs[t, :, day])
                    day_targets.append(targets[target_time, :])

            if day_predictions:
                day_pred_array = np.stack(day_predictions, axis=0)
                day_target_array = np.stack(day_targets, axis=0)
                day_metrics = self._compute_day_metrics(day_pred_array, day_target_array, day + 1)
                day_results[day] = day_metrics

        self._save_temporal_fire_metrics(day_results, forecast_days)

    def _compute_day_metrics(self, predictions: np.ndarray, targets: np.ndarray, day_num: int) -> dict:
        n_times, n_cells = predictions.shape
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()

        pred_for_metrics = pred_flat.reshape(-1, 1)
        target_for_metrics = target_flat.reshape(-1, 1)

        effective_height = min(81, n_times)
        effective_width = min(132, n_cells)

        fm = FireMetrics(
            pred_for_metrics,
            target_for_metrics,
            forecast_days=1,
            grid_resolution_km=40.0,
            grid_height=effective_height,
            grid_width=effective_width
        )

        result_40km = fm.spatial_results_40km.get(0, {'csi': 0.0, 'pod': 0.0, 'far': 1.0, 'hits': 0, 'misses': 0, 'false_alarms': 0})
        result_80km = fm.spatial_results_80km.get(0, {'csi': 0.0, 'pod': 0.0, 'far': 1.0, 'hits': 0, 'misses': 0, 'false_alarms': 0})
        result_120km = fm.spatial_results_120km.get(0, {'csi': 0.0, 'pod': 0.0, 'far': 1.0, 'hits': 0, 'misses': 0, 'false_alarms': 0})

        return {
            '40km': result_40km,
            '80km': result_80km,
            '120km': result_120km,
            'total_predictions': float(pred_flat.sum()),
            'total_targets': float(target_flat.sum()),
            'n_samples': int(len(pred_flat))
        }

    def _save_temporal_fire_metrics(self, day_results: dict, forecast_days: int) -> None:
        summary = {
            'forecast_days': forecast_days,
            'day_by_day_results': {},
            'degradation_analysis': {}
        }

        for neighborhood in ['40km', '80km', '120km']:
            csi_values = []
            pod_values = []
            far_values = []
            for day in range(forecast_days):
                if day in day_results and neighborhood in day_results[day]:
                    result = day_results[day][neighborhood]
                    if isinstance(result.get('csi'), (int, float)) and result['csi'] is not None:
                        csi_values.append(result['csi'])
                        pod_values.append(result.get('pod', 0.0))
                        far_values.append(result.get('far', 1.0))
                        summary['day_by_day_results'][f'day_{day+1}_{neighborhood}'] = result

            if len(csi_values) > 1:
                total_degradation = csi_values[0] - csi_values[-1]
                daily_degradation = total_degradation / (len(csi_values) - 1)
                summary['degradation_analysis'][neighborhood] = {
                    'total_degradation': total_degradation,
                    'daily_degradation': daily_degradation,
                    'day_1_csi': csi_values[0],
                    'final_day_csi': csi_values[-1],
                    'mean_csi': float(np.mean(csi_values)),
                    'all_csi_values': csi_values
                }

        os.makedirs(self.config['out_path'], exist_ok=True)
        results_path = os.path.join(self.config['out_path'], 'temporal_fire_metrics.json')

        import json
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            return obj

        with open(results_path, 'w') as f:
            json.dump(summary, f, indent=2, default=convert_numpy)

        log.info(f"Temporal fire metrics saved to {results_path}")

    # ===== Implement abstract methods required by BaseTrainer ===== #
    def calc_metrics(
        self,
        batch_predictions: List[Dict[str, torch.Tensor]],
        observations: torch.Tensor,
    ) -> None:
        tensors = [d['prediction'] for d in batch_predictions]
        pred = torch.cat(tensors, dim=1 if tensors[0].ndim == 3 else 0).detach().cpu().numpy()

        if pred.ndim == 3 and pred.shape[2] == 1:
            pred = pred[:, :, 0]  # [time, cells]

        target = observations.detach().cpu().numpy() if isinstance(observations, torch.Tensor) else observations
        if target.ndim == 3 and target.shape[2] == 1:
            target = target[:, :, 0]

        if pred.shape != target.shape and pred.T.shape == target.shape:
            pred = pred.T

        if pred.shape != target.shape:
            raise ValueError(f"calc_metrics: shape mismatch pred {pred.shape} vs target {target.shape}")

        os.makedirs(self.config['out_path'], exist_ok=True)
        Metrics(pred.swapaxes(0, 1), target.swapaxes(0, 1)).dump_metrics(self.config['out_path'])

    def inference(self) -> None:
        raise NotImplementedError("Inference is not implemented for FinetuneTrainer.")

    # ===== Private logging helpers (called by train) ===== #
    def _log_epoch_stats(
        self,
        epoch: int,
        train_loss: List[float],
        val_loss: Optional[List[float]],
        epoch_time: float,
        results_file,
    ) -> None:
        train_loss_avg = float(np.mean(train_loss)) if len(train_loss) > 0 else float('nan')
        val_loss_avg = float(np.mean(val_loss)) if val_loss else None

        self.epoch_train_loss_list.append(train_loss_avg)
        if val_loss_avg is not None:
            self.epoch_val_loss_list.append(val_loss_avg)

        msg = (
            f"Epoch {epoch}: train_loss={train_loss_avg:.3f}"
            f"{f', val_loss={val_loss_avg:.3f}' if val_loss_avg is not None else ''}"
            f" ({time.time() - epoch_time:.2f}s)"
        )
        log.info(msg)
        results_file.write(msg + "\n")
        results_file.flush()

        self._save_loss_data(epoch, train_loss_avg, val_loss_avg)

    def _save_loss_data(self, epoch: int, train_loss: float, val_loss: Optional[float]) -> None:
        results_dir = self.config.get('save_path', 'results')
        os.makedirs(results_dir, exist_ok=True)
        loss_path = os.path.join(results_dir, "loss_data.csv")
        with open(loss_path, 'a') as f:
            f.write(f"{epoch},{train_loss},{'' if val_loss is None else val_loss}\n")


def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
