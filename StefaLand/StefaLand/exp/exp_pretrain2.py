import os
import time
import datetime
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch

from MFFormer.data_provider.data_factory import data_provider
from MFFormer.exp.exp_basic import Exp_Basic
from MFFormer.utils.tools import adjust_learning_rate, visual  # EarlyStopping,

from MFFormer.utils.stats.metrics import cal_stations_metrics
from MFFormer.data_provider.data_factory import get_train_val_test_dataset
from MFFormer.layers.dropout import apply_dropout

warnings.filterwarnings('ignore')


class ExpPretrain(Exp_Basic):
    def __init__(self, config):
        super().__init__(config)
        self.dataset_dict = get_train_val_test_dataset(self.config)
        config = self.dataset_dict[config.data[0]]["config"]

        # set weight for loss
        if config.ratio_time_series_variables is None:
            weights_time_series_variables = torch.ones(len(config.time_series_variables)) * 1.0
            weights_time_series_variables[-1] = 1.0
        else:
            assert len(config.ratio_time_series_variables) == len(config.time_series_variables)
            weights_time_series_variables = torch.from_numpy(np.array(config.ratio_time_series_variables).astype(float))

        self.weights_time_series_variables = torch.nn.Parameter(weights_time_series_variables)
        if getattr(config, "ratio_static_variables", None) is None:
            w_static = torch.ones(len(getattr(config, "static_variables", []))) * 1.0
        else:
            assert len(config.ratio_static_variables) == len(config.static_variables)
            w_static = torch.from_numpy(np.array(config.ratio_static_variables).astype(float))

        self.weights_static_variables = torch.nn.Parameter(w_static)

        # global mixing (defaults)
        self.lambda_time  = float(getattr(config, "lambda_time", 1.0))
        self.lambda_static = float(getattr(config, "lambda_static", 1.0))
        # self.experiment_verifier()

    def experiment_verifier(self):

        if self.add_input_noise:
            assert self.config.calculate_time_series_each_variable_loss, "add_input_noise must be used with calculate_time_series_each_variable_loss"
            assert self.config.calculate_static_each_variable_loss, "add_input_noise must be used with calculate_static_each_variable_loss"

    def compute_loss(self, batch_data_dict, output_dict):
        # -------------------------
        # time series loss 
        # -------------------------
        batch_y = batch_data_dict["batch_x"]
        outputs_time_series = output_dict["outputs_time_series"]
        x_stds = batch_data_dict.get("batch_x_std", None)

        m_full = output_dict["masked_time_series_index"]
        mm_full = output_dict["masked_missing_time_series_index"]

        Lp = self.config.pred_len
        outputs_time_series = outputs_time_series[:, -Lp:, :]
        batch_y             = batch_y[:,             -Lp:, :].to(self.device)
        m_tail  = m_full[:,  -Lp:, :]
        mm_tail = mm_full[:, -Lp:, :]

        loss_mask_ts = (m_tail & (~mm_tail))

        M_final = int(loss_mask_ts.sum().item())
        if M_final == 0:
            raise RuntimeError(
                f"No supervised elements in loss window. "
                f"M_full={int(m_full.sum().item())}, M_tail={int(m_tail.sum().item())}, "
                f"M_final={M_final}, pred_len={Lp}, seq_len={self.config.seq_len}"
            )

        # per-variable TS loss (same structure you already have)
        if self.config.calculate_time_series_each_variable_loss:
            w_ts = self.weights_time_series_variables.to(outputs_time_series.device).to(outputs_time_series.dtype)
            loss_time_series = outputs_time_series.sum() * 0.0
            have_any = False

            for i in range(batch_y.shape[-1]):
                m = loss_mask_ts[..., i]
                if not m.any():
                    continue

                if (x_stds is not None) and (0 not in x_stds.size()):
                    masked_x_stds = x_stds.unsqueeze(1).expand_as(batch_y)[..., i][m]
                else:
                    masked_x_stds = None

                li = self.criterion(outputs_time_series[..., i][m],
                                    batch_y[..., i][m],
                                    target_stds=masked_x_stds)
                if not torch.isfinite(li):
                    raise RuntimeError(f"Sub-loss is non-finite for TS feature {i}.")
                loss_time_series = loss_time_series + li * w_ts[i]
                have_any = True

            if not have_any:
                raise RuntimeError("TS per-variable path reached with no valid features after masking.")
        else:
            if (x_stds is not None) and (0 not in x_stds.size()):
                masked_x_stds = x_stds.unsqueeze(1).expand_as(batch_y)[loss_mask_ts]
            else:
                masked_x_stds = None

            w_ts = self.weights_time_series_variables.to(outputs_time_series.device).to(outputs_time_series.dtype)
            weighted_outputs = outputs_time_series * w_ts
            loss_time_series = self.criterion(weighted_outputs[loss_mask_ts],
                                            batch_y[loss_mask_ts],
                                            target_stds=masked_x_stds)
            if not torch.isfinite(loss_time_series):
                raise RuntimeError("TS loss is non-finite after masking.")

        # -------------------------
        # static loss (numeric only)
        # -------------------------
        loss_static = outputs_time_series.sum() * 0.0  # seed on device

        do_static = bool(getattr(self.config, "calculate_static_loss", True))
        if do_static and ("outputs_static" in output_dict) and ("masked_static_index" in output_dict) and ("batch_c" in batch_data_dict):

            # numeric-only indices (skip categorical names listed in static_variables_category)
            cat = set(getattr(self.config, "static_variables_category", []) or [])
            num_idx_list = [i for i, v in enumerate(self.config.static_variables) if v not in cat]

            if len(num_idx_list) > 0:
                pred_s = output_dict["outputs_static"]          # [B, F_static] for your current models
                true_s = batch_data_dict["batch_c"].to(self.device)  # [B, F_static]
                m_s    = output_dict["masked_static_index"]     # [B, F_static] bool
                mm_s   = output_dict.get("masked_missing_static_index", None)  # [B, F_static] bool or None

                if mm_s is not None:
                    loss_mask_s = (m_s & (~mm_s))
                else:
                    loss_mask_s = m_s

                w_s = self.weights_static_variables.to(pred_s.device).to(pred_s.dtype)

                # per-variable static loss, numeric-only
                have_s = False
                for gi in num_idx_list:
                    m = loss_mask_s[:, gi]
                    if not m.any():
                        continue

                    li = self.criterion(pred_s[:, gi][m], true_s[:, gi][m], target_stds=None)
                    if not torch.isfinite(li):
                        raise RuntimeError(f"Sub-loss is non-finite for static feature {gi}.")
                    loss_static = loss_static + li * w_s[gi]
                    have_s = True

                # if you want a hard fail when static is enabled but no static supervision exists:
                if getattr(self.config, "require_static_supervision", False) and (not have_s):
                    raise RuntimeError("Static loss enabled but no supervised static elements in this batch.")
        return self.lambda_time * loss_time_series + self.lambda_static * loss_static


    def train(self):
        import gc
        from time import perf_counter #just to track things 

        f = open(os.path.join(self.saved_dir, "results.txt"), 'a')
        f.write("\n")
        f.write("train start time: " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + '\n')

        # resume from checkpoint
        self.load_model_resume()

        use_cuda = torch.cuda.is_available() and str(self.device).startswith("cuda")

        for epoch in np.arange(self.start_epoch, self.config.epochs):
            epoch += 1

            # running averages (no big lists)
            train_loss_sum, train_loss_cnt = 0.0, 0
            val_loss_sum,   val_loss_cnt   = 0.0, 0

            t_epoch0 = perf_counter()
            epoch_time = time.time()

            for data_name in self.config.data:
                train_loader = self.dataset_dict[data_name]["train_loader"]
                val_loader   = self.dataset_dict[data_name]["val_loader"]
                self.config  = self.dataset_dict[data_name]["config"]
                # create AMP scaler
                scaler = torch.cuda.amp.GradScaler(enabled=getattr(self.config, "use_amp", False))

                self.model.train()

                for idx_index_file in range(train_loader.dataset.num_index_files):
                    if idx_index_file > 0:
                        train_loader.dataset.load_index(idx_index_file)
                        train_loader.dataset.set_length(len(train_loader.dataset.slice_grid_list))

                    t_data = t_fwd = t_bwd = 0.0
                    num_batches = 0

                    for i, batch_data_dict in enumerate(train_loader):
                        num_batches += 1
                        # zero grads as None (saves RAM)
                        self.model_optim.zero_grad(set_to_none=True)

                        # to device
                        t0 = perf_counter()
                        for k, v in list(batch_data_dict.items()):
                            if torch.is_tensor(v) and v.dtype != torch.bool:
                                batch_data_dict[k] = v.float().to(self.device, non_blocking=use_cuda) if use_cuda else v.float()
                        batch_data_dict['mode'] = 'train'
                        t_data += perf_counter() - t0

                        # forward + loss (timed with sync)
                        if self.config.use_amp:
                            if use_cuda: torch.cuda.synchronize()
                            t1 = perf_counter()
                            with torch.cuda.amp.autocast():
                                output_dict = self.model(batch_data_dict)
                                loss = self.compute_loss(batch_data_dict, output_dict)
                            if use_cuda: torch.cuda.synchronize()
                            t_fwd += perf_counter() - t1

                            if torch.isnan(loss):
                                del output_dict, loss, batch_data_dict
                                continue

                            # running avg
                            train_loss_sum += float(loss.detach())
                            train_loss_cnt += 1

                            # backward + step 
                            if use_cuda: torch.cuda.synchronize()
                            t2 = perf_counter()
                            scaler.scale(loss).backward()
                            if self.config.clip_grad is not None:
                                scaler.unscale_(self.model_optim)
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad)
                            scaler.step(self.model_optim)
                            scaler.update()
                            if use_cuda: torch.cuda.synchronize()
                            t_bwd += perf_counter() - t2
                        else:
                            if use_cuda: torch.cuda.synchronize()
                            t1 = perf_counter()
                            output_dict = self.model(batch_data_dict)
                            loss = self.compute_loss(batch_data_dict, output_dict)
                            if use_cuda: torch.cuda.synchronize()
                            t_fwd += perf_counter() - t1

                            if torch.isnan(loss):
                                del output_dict, loss, batch_data_dict
                                continue

                            train_loss_sum += float(loss.detach())
                            train_loss_cnt += 1

                            if use_cuda: torch.cuda.synchronize()
                            t2 = perf_counter()
                            loss.backward()
                            if self.config.clip_grad is not None:
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad)
                            self.model_optim.step()
                            if use_cuda: torch.cuda.synchronize()
                            t_bwd += perf_counter() - t2

                        # drop graph-holding refs ASAP
                        del output_dict, loss, batch_data_dict

                        # light cleanup every 200 steps
                        if (i + 1) % 500 == 0:
                            gc.collect()
                            if use_cuda:
                                torch.cuda.empty_cache()

                    per_batch = (t_data + t_fwd + t_bwd) / max(1, num_batches)
                    print(f"[Profile][{data_name}] file={idx_index_file}  "
                        f"data {t_data:.2f}s | fwd {t_fwd:.2f}s | bwd {t_bwd:.2f}s "
                        f"| per-batch {per_batch:.3f}s")

                # validation running average
                if self.config.do_eval:
                    v = self.val(val_loader)
                    if v is not None:
                        val_loss_sum += v
                        val_loss_cnt += 1

                t_update = time.time()
                if self.config.use_complete_reindex:
                    train_loader.dataset.update_samples_index(True)
                    msg = "update_samples_index(True)"
                else:
                    train_loader.dataset.reshuffle_batches()
                    msg = "reshuffle_batches()"
                dt = time.time() - t_update
                print(f"{msg} cost time: {dt:.2f} s")
                f.write(f"{msg} cost time: {dt:.2f} s\n")

            # save, LR, logs
            self.save_model(self.checkpoints_dir, epoch, self.model, self.model_optim,
                            self.epoch_train_loss_list, self.epoch_val_loss_list)

            adjust_learning_rate(self.model_optim, epoch + 1, self.config)

            train_loss = (train_loss_sum / max(1, train_loss_cnt)) if train_loss_cnt else float('nan')
            val_loss   = (val_loss_sum   / max(1, val_loss_cnt  )) if val_loss_cnt   else None

            print("Epoch: {} train loss: {:.3f} cost time: {:.2f} s".format(
                epoch, train_loss, time.time() - epoch_time))
            print("Epoch: {} val loss: {:.3f} ".format(epoch, val_loss if val_loss is not None else float('nan')))
            f.write("Epoch: {} train loss: {:.3f} cost time: {:.2f} s\n".format(
                epoch, train_loss, time.time() - epoch_time))
            f.write("Epoch: {} val loss: {:.3f}\n".format(epoch, val_loss if val_loss is not None else float('nan')))
            self.epoch_train_loss_list.append(train_loss)
            self.epoch_val_loss_list.append(val_loss)

            # write/plot loss
            loss_csv_file = os.path.join(self.results_dir, "loss_data.csv")
            loss_pic_file = os.path.join(self.saved_dir, "loss_curve.png")
            self.write_loss_to_csv(epoch, train_loss, val_loss, loss_csv_file)
            self.plot_loss_curve_from_csv(loss_csv_file, loss_pic_file)

            print(f"[Profile] EPOCH {epoch} wall {perf_counter() - t_epoch0:.2f}s")

            # end-of-epoch cleanup
            gc.collect()
            if use_cuda:
                torch.cuda.empty_cache()
        f.close()
        print("results saved in: ", os.path.abspath(self.results_dir))
        return self.model


    def val(self, val_loader):
        import gc
        total_sum, total_cnt = 0.0, 0
        self.model.eval()
        with torch.no_grad():
            for idx_index_file in range(val_loader.dataset.num_index_files):
                if idx_index_file > 0:
                    val_loader.dataset.load_index(idx_index_file)
                    val_loader.dataset.set_length(len(val_loader.dataset.slice_grid_list))

                for i, batch_data_dict in enumerate(val_loader):
                    for key, value in list(batch_data_dict.items()):
                        if torch.is_tensor(value) and not value.dtype == torch.bool:
                            batch_data_dict[key] = value.float().to(self.device)
                    batch_data_dict['mode'] = 'val'

                    if self.config.use_amp:
                        with torch.cuda.amp.autocast():
                            output_dict = self.model(batch_data_dict)
                    else:
                        output_dict = self.model(batch_data_dict)

                    loss = self.compute_loss(batch_data_dict, output_dict)
                    if not torch.isnan(loss):
                        total_sum += float(loss)
                        total_cnt += 1

                    del output_dict, loss, batch_data_dict
                    if (i + 1) % 200 == 0:
                        gc.collect()
                        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                            torch.cuda.empty_cache()

        avg = (total_sum / max(1, total_cnt)) if total_cnt else None
        self.model.train()
        return avg
    
    def _static_numeric_idx(self):
        cat = set(getattr(self.config, "static_variables_category", []) or [])
        return [i for i, v in enumerate(self.config.static_variables) if v not in cat]

    def test(self, test=0):

        f = open(os.path.join(self.saved_dir, "results.txt"), 'a')
        f.write("test start time: " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + '\n')
        test_start_time = time.time()

        if self.config.do_test:
            print('loading model')
            self.load_model_resume()

        for data_name in self.config.data:
            print("testing on {}".format(data_name))
            f.write(f'{data_name}: \n')

            test_data = self.dataset_dict[data_name]["test_data"]
            test_loader = self.dataset_dict[data_name]["test_loader"]
            self.config = self.dataset_dict[data_name]["config"]

            n_feat = len(self.config.time_series_variables)
            preds_F  = {fi: [] for fi in range(n_feat)}
            trues_F  = {fi: [] for fi in range(n_feat)}
            maskF_F  = {fi: [] for fi in range(n_feat)}

            num_idx_list = self._static_numeric_idx()
            has_static = (len(getattr(self.config, "static_variables", [])) > 0) and (len(num_idx_list) > 0)
            preds_static_F = {j: [] for j in range(len(num_idx_list))}
            trues_static_F = {j: [] for j in range(len(num_idx_list))}
            maskS_F        = {j: [] for j in range(len(num_idx_list))}

            self.model.eval()
            with torch.no_grad():
                for idx_index_file in range(test_loader.dataset.num_index_files):
                    if idx_index_file > 0:
                        test_loader.dataset.load_index(idx_index_file)
                        test_loader.dataset.set_length(len(test_loader.dataset.slice_grid_list))

                    for i, batch_data_dict in enumerate(test_loader):
                        batch_data_dict = {
                            key: value.float().to(self.device) if torch.is_tensor(value) and not value.dtype == torch.bool
                            else value for key, value in batch_data_dict.items()
                        }
                        batch_data_dict['mode'] = 'test'
                        batch_y = batch_data_dict["batch_x"]

                        if self.config.use_amp:
                            with torch.cuda.amp.autocast():
                                output_dict = self.model(batch_data_dict)
                        else:
                            output_dict = self.model(batch_data_dict)

                        outputs_time_series = output_dict["outputs_time_series"]
                        masked_time_series_index = output_dict["masked_time_series_index"]

                        pred = outputs_time_series[:, -self.config.pred_len:, :].detach().cpu()
                        true = batch_y[:,        -self.config.pred_len:, :].detach().cpu()
                        mask = masked_time_series_index[:, -self.config.pred_len:, :].detach().cpu()

                        for fi in range(pred.shape[-1]):
                            preds_F[fi].append(pred[..., fi].numpy())
                            trues_F[fi].append(true[..., fi].numpy())
                            maskF_F[fi].append(mask[..., fi].numpy())

                        if has_static and ("outputs_static" in output_dict) and ("masked_static_index" in output_dict):
                            p_st = output_dict["outputs_static"].detach().cpu().numpy()   # [B, F_static]
                            y_st = batch_data_dict["batch_c"].detach().cpu().numpy()      # [B, F_static]
                            m_st = output_dict["masked_static_index"].detach().cpu().numpy().astype(bool)  # [B, F_static]
                            for j, gi in enumerate(num_idx_list):
                                # collect as [B, 1, 1] so restore_data([N,1,1]) -> [S,L,1]
                                preds_static_F[j].append(p_st[:, gi:gi+1][:, None, :])
                                trues_static_F[j].append(y_st[:, gi:gi+1][:, None, :])
                                maskS_F[j].append(m_st[:, gi:gi+1][:, None, :])

                        del pred, true, mask, outputs_time_series, output_dict

            time_series_median_metrics_dict = {key: [] for key in ["NSE", "KGE", "Corr", "L1", "L2"]}

            f.write("Test date: {}, time: {}\n".format(datetime.datetime.now().strftime("%Y-%m-%d"),
                                                    datetime.datetime.now().strftime("%H:%M:%S")))

            for fi, var in enumerate(self.config.time_series_variables):
                pred_f = np.concatenate(preds_F[fi], axis=0)  # [N,T]
                true_f = np.concatenate(trues_F[fi], axis=0)  # [N,T]
                mask_f = np.concatenate(maskF_F[fi], axis=0)  # [N,T]

                pred_f = pred_f[..., None]  # [N,T,1]
                true_f = true_f[..., None]
                mask_f = mask_f[..., None]

                preds_restored_f = test_data.restore_data(data=pred_f, num_stations=test_data.num_row)  # [S,L,1]
                trues_restored_f = test_data.restore_data(data=true_f, num_stations=test_data.num_row)
                mask_restored_f  = test_data.restore_data(data=mask_f, num_stations=test_data.num_row)

                nominal_len = len(pd.date_range(
                    test_data.config_dataset.test_date_list[0],
                    test_data.config_dataset.test_date_list[-1],
                    inclusive="both"
                ))
                actual_len = preds_restored_f.shape[1]
                if nominal_len != actual_len:
                    print(f"[warn] calendar span ({nominal_len}) != restored length ({actual_len}); continuing with actual_len for {var}.")

                preds_rescale_f = test_data.inverse_transform(
                    preds_restored_f,
                    mean=test_data.scaler["x_mean"][fi],
                    std =test_data.scaler["x_std"][fi]
                )
                trues_rescale_f = test_data.inverse_transform(
                    trues_restored_f,
                    mean=test_data.scaler["x_mean"][fi],
                    std =test_data.scaler["x_std"][fi]
                )

                unmasked = np.logical_not(mask_restored_f)
                preds_rescale_f[unmasked] = np.nan
                trues_rescale_f[unmasked] = np.nan

                y = trues_rescale_f[..., 0]
                x = preds_rescale_f[..., 0]
                if var not in self.config.negative_value_variables:
                    y[y < 0] = 0
                    x[x < 0] = 0

                metrics_list = ["NSE", "KGE", "Corr"]
                metrics_dict = cal_stations_metrics(y, x, metrics_list)

                mv = np.isfinite(y) & np.isfinite(x)
                d  = (x - y)[mv]
                L1 = np.nan if d.size == 0 else float(np.mean(np.abs(d)))
                L2 = np.nan if d.size == 0 else float(np.sqrt(np.mean(d * d)))

                print(f"{var}. NSE: {np.nanmedian(metrics_dict['NSE']):.3f}, "
                    f"KGE: {np.nanmedian(metrics_dict['KGE']):.3f}, "
                    f"Corr: {np.nanmedian(metrics_dict['Corr']):.3f}, "
                    f"L1: {L1:.3f}, L2: {L2:.3f}")
                f.write(f"{var}. NSE: {np.nanmedian(metrics_dict['NSE']):.3f}, "
                        f"KGE: {np.nanmedian(metrics_dict['KGE']):.3f}, "
                        f"Corr: {np.nanmedian(metrics_dict['Corr']):.3f}, "
                        f"L1: {L1:.3f}, L2: {L2:.3f}\n")

                time_series_median_metrics_dict["NSE"].append(np.nanmedian(metrics_dict["NSE"]))
                time_series_median_metrics_dict["KGE"].append(np.nanmedian(metrics_dict["KGE"]))
                time_series_median_metrics_dict["Corr"].append(np.nanmedian(metrics_dict["Corr"]))
                time_series_median_metrics_dict["L1"].append(L1)
                time_series_median_metrics_dict["L2"].append(L2)

            # stable means for logging
            def _safe_mean(a): 
                aa = np.asarray(a, dtype=float)
                aa = aa[np.isfinite(aa)]
                return float(np.mean(aa)) if aa.size else float('nan')
            def _safe_median(a): 
                aa = np.asarray(a, dtype=float)
                aa = aa[np.isfinite(aa)]
                return float(np.median(aa)) if aa.size else float('nan')

            print("obs_mean: {:.3f}, obs_median: {:.7f}".format(
                _safe_mean([np.mean(np.concatenate(trues_F[fi], axis=0)) for fi in range(n_feat)]),
                _safe_median([np.median(np.concatenate(trues_F[fi], axis=0)) for fi in range(n_feat)]))
            )
            print("pred_mean: {:.3f}, pred_median: {:.7f}".format(
                _safe_mean([np.mean(np.concatenate(preds_F[fi], axis=0)) for fi in range(n_feat)]),
                _safe_median([np.median(np.concatenate(preds_F[fi], axis=0)) for fi in range(n_feat)]))
            )

            preds_rescale = np.concatenate([test_data.inverse_transform(
                test_data.restore_data(data=np.concatenate(preds_F[fi], axis=0)[..., None], num_stations=test_data.num_row),
                mean=test_data.scaler["x_mean"][fi],
                std=test_data.scaler["x_std"][fi]
            ) for fi in range(n_feat)], axis=-1)

            trues_rescale = np.concatenate([test_data.inverse_transform(
                test_data.restore_data(data=np.concatenate(trues_F[fi], axis=0)[..., None], num_stations=test_data.num_row),
                mean=test_data.scaler["x_mean"][fi],
                std=test_data.scaler["x_std"][fi]
            ) for fi in range(n_feat)], axis=-1)

            np.save(os.path.join(self.results_dir, 'pred_time_series.npy'), preds_rescale)
            np.save(os.path.join(self.results_dir, 'true_time_series.npy'), trues_rescale)

            # statics after dynamics
            if has_static and len(num_idx_list) > 0:
                c_mean = np.asarray(getattr(test_data, "scaler", {}).get("c_mean")) if hasattr(test_data, "scaler") else None
                c_std  = np.asarray(getattr(test_data, "scaler", {}).get("c_std"))  if hasattr(test_data, "scaler") else None
                static_names = [self.config.static_variables[gi] for gi in num_idx_list]
                all_pred_static_rescaled, all_true_static_rescaled = [], []

                for j, name in enumerate(static_names):
                    pred_s = np.concatenate(preds_static_F[j], axis=0)   # [N,1,1]
                    true_s = np.concatenate(trues_static_F[j], axis=0)   # [N,1,1]
                    mask_s = np.concatenate(maskS_F[j],        axis=0)   # [N,1,1] bool

                    pred_s_r = test_data.restore_data(data=pred_s, num_stations=test_data.num_row)  # [S,L,1]
                    true_s_r = test_data.restore_data(data=true_s, num_stations=test_data.num_row)  # [S,L,1]
                    mask_s_r = test_data.restore_data(data=mask_s, num_stations=test_data.num_row)  # [S,L,1]

                    if c_mean is not None and c_std is not None:
                        gi = num_idx_list[j]
                        pred_s_r = pred_s_r * c_std[gi] + c_mean[gi]
                        true_s_r = true_s_r * c_std[gi] + c_mean[gi]

                    unmasked = np.logical_not(mask_s_r)
                    pred_s_r[unmasked] = np.nan
                    true_s_r[unmasked] = np.nan

                    y = true_s_r[..., 0]
                    x = pred_s_r[..., 0]

                    metrics_list = ["NSE", "KGE", "Corr"]
                    md = cal_stations_metrics(y, x, metrics_list)

                    mv = np.isfinite(y) & np.isfinite(x)
                    d  = (x - y)[mv]
                    L1 = np.nan if d.size == 0 else float(np.mean(np.abs(d)))
                    L2 = np.nan if d.size == 0 else float(np.sqrt(np.mean(d * d)))

                    print(f"Static {name}. NSE: {np.nanmedian(md['NSE']):.3f}, "
                        f"KGE: {np.nanmedian(md['KGE']):.3f}, "
                        f"Corr: {np.nanmedian(md['Corr']):.3f}, "
                        f"L1: {L1:.3f}, L2: {L2:.3f}")
                    f.write(f"Static {name}. NSE: {np.nanmedian(md['NSE']):.3f}, "
                            f"KGE: {np.nanmedian(md['KGE']):.3f}, "
                            f"Corr: {np.nanmedian(md['Corr']):.3f}, "
                            f"L1: {L1:.3f}, L2: {L2:.3f}\n")

                    all_pred_static_rescaled.append(pred_s_r)
                    all_true_static_rescaled.append(true_s_r)

                if len(all_pred_static_rescaled):
                    pred_static_cat = np.concatenate(all_pred_static_rescaled, axis=-1)
                    true_static_cat = np.concatenate(all_true_static_rescaled, axis=-1)
                    np.save(os.path.join(self.results_dir, 'pred_static.npy'), pred_static_cat)
                    np.save(os.path.join(self.results_dir, 'true_static.npy'), true_static_cat)

            f.write("test cost time: {:.2f} s\n".format(time.time() - test_start_time))
        f.close()
        return


    def inference(self):

        f = open(os.path.join(self.saved_dir, "results.txt"), 'a')
        f.write("Inference date: {}, time: {}".format(datetime.datetime.now().strftime("%Y-%m-%d"),
                                                    datetime.datetime.now().strftime("%H:%M:%S")))
        f.write('\n')

        if self.config.resume_from_checkpoint is None:
            self.config.resume_from_checkpoint = True
        self.load_model_resume()

        for data_name in self.config.data:

            f.write(f'{data_name}: \n')

            test_data = self.dataset_dict[data_name]["test_data"]
            test_loader = self.dataset_dict[data_name]["test_loader"]
            self.config = self.dataset_dict[data_name]["config"]

            time_series_order = self.config.time_series_variables

            if self.config.inference_variables is None:
                inference_time_series_variables = self.config.time_series_variables
            else:
                inference_time_series_variables = [var for var in self.config.inference_variables if var in
                                                self.config.time_series_variables]

            metrics_list = ["NSE", "KGE", "Corr"]

            if len(inference_time_series_variables) > 0:

                for time_series_variable in inference_time_series_variables:

                    idx_time_series_variable = time_series_order.index(time_series_variable)

                    preds, trues = [], []

                    self.model.eval()
                    with torch.no_grad():

                        for idx_index_file in range(test_loader.dataset.num_index_files):
                            if idx_index_file > 0:
                                test_loader.dataset.load_index(idx_index_file)
                                test_loader.dataset.set_length(len(test_loader.dataset.slice_grid_list))

                            for i, batch_data_dict in enumerate(test_loader):

                                batch_data_dict = {
                                    key: value.float().to(self.device) if torch.is_tensor(
                                        value) and not value.dtype == torch.bool
                                    else value for key, value in batch_data_dict.items()
                                }
                                batch_time_series_mask_index = torch.zeros_like(batch_data_dict["batch_x"],
                                                                                dtype=torch.bool)
                                batch_time_series_mask_index[:, :, idx_time_series_variable] = True

                                batch_data_dict["batch_time_series_mask_index"] = batch_time_series_mask_index

                                batch_y = batch_data_dict["batch_x"]
                                batch_data_dict['mode'] = 'test'
                                output_dict = self.model(batch_data_dict)

                                outputs_time_series = output_dict["outputs_time_series"]

                                pred = outputs_time_series[:, -self.config.pred_len:, idx_time_series_variable][..., None]
                                true = batch_y[:, -self.config.pred_len:, idx_time_series_variable][..., None]

                                preds.append(pred.detach().cpu().numpy())
                                trues.append(true.detach().cpu().numpy())

                    preds = np.concatenate(preds, axis=0)
                    trues = np.concatenate(trues, axis=0)

                    preds_restored = test_data.restore_data(data=preds, num_stations=test_data.num_row)
                    trues_restored = test_data.restore_data(data=trues, num_stations=test_data.num_row)

                    nominal_len = len(pd.date_range(
                        test_data.config_dataset.test_date_list[0],
                        test_data.config_dataset.test_date_list[-1],
                        inclusive="both"
                    ))
                    actual_len = preds_restored.shape[1]
                    if nominal_len != actual_len:
                        print(f"[warn] calendar span ({nominal_len}) != restored length ({actual_len}); continuing with actual_len.")

                    preds_rescale = test_data.inverse_transform(preds_restored,
                                                                mean=test_data.scaler["x_mean"][
                                                                    idx_time_series_variable],
                                                                std=test_data.scaler["x_std"][idx_time_series_variable])
                    trues_rescale = test_data.inverse_transform(trues_restored,
                                                                mean=test_data.scaler["x_mean"][
                                                                    idx_time_series_variable],
                                                                std=test_data.scaler["x_std"][idx_time_series_variable])

                    y = trues_rescale[..., 0]
                    x = preds_rescale[..., 0]
                    if not self.config.time_series_variables[idx_time_series_variable] in self.config.negative_value_variables:
                        y[y < 0] = 0
                        x[x < 0] = 0

                    metrics_dict = cal_stations_metrics(y, x, metrics_list)
                    mv = np.isfinite(y) & np.isfinite(x)
                    d  = (x - y)[mv]
                    L1 = np.nan if d.size == 0 else float(np.mean(np.abs(d)))
                    L2 = np.nan if d.size == 0 else float(np.sqrt(np.mean(d * d)))

                    varname = self.config.time_series_variables[idx_time_series_variable]
                    print(f"{varname}. NSE: {np.nanmedian(metrics_dict['NSE']):.3f}, "
                        f"KGE: {np.nanmedian(metrics_dict['KGE']):.3f}, "
                        f"Corr: {np.nanmedian(metrics_dict['Corr']):.3f}, "
                        f"L1: {L1:.3f}, L2: {L2:.3f}")
                    f.write(f"{varname}. NSE: {np.nanmedian(metrics_dict['NSE']):.3f}, "
                            f"KGE: {np.nanmedian(metrics_dict['KGE']):.3f}, "
                            f"Corr: {np.nanmedian(metrics_dict['Corr']):.3f}, "
                            f"L1: {L1:.3f}, L2: {L2:.3f}\n")

            # statics after dynamics (per variable)
            num_idx_list = self._static_numeric_idx()
            if len(getattr(self.config, "static_variables", [])) > 0 and len(num_idx_list) > 0:

                c_mean = np.asarray(getattr(test_data, "scaler", {}).get("c_mean")) if hasattr(test_data, "scaler") else None
                c_std  = np.asarray(getattr(test_data, "scaler", {}).get("c_std"))  if hasattr(test_data, "scaler") else None

                for j, gi in enumerate(num_idx_list):
                    name = self.config.static_variables[gi]
                    preds_s, trues_s, masks_s = [], [], []

                    self.model.eval()
                    with torch.no_grad():
                        for idx_index_file in range(test_loader.dataset.num_index_files):
                            if idx_index_file > 0:
                                test_loader.dataset.load_index(idx_index_file)
                                test_loader.dataset.set_length(len(test_loader.dataset.slice_grid_list))

                            for i, batch_data_dict in enumerate(test_loader):
                                bd = {
                                    key: value.float().to(self.device) if torch.is_tensor(value) and not value.dtype == torch.bool
                                    else value for key, value in batch_data_dict.items()
                                }
                                bd['mode'] = 'test'

                                bsm = torch.zeros_like(bd["batch_c"], dtype=torch.bool, device=self.device)
                                bsm[:, gi] = True
                                bd["batch_static_mask_index"] = bsm

                                if self.config.use_amp:
                                    with torch.cuda.amp.autocast():
                                        od = self.model(bd)
                                else:
                                    od = self.model(bd)

                                p = od["outputs_static"].detach().cpu().numpy()[:, gi:gi+1]  # [B,1]
                                y = bd["batch_c"].detach().cpu().numpy()[:, gi:gi+1]       # [B,1]
                                m = od["masked_static_index"].detach().cpu().numpy()[:, gi:gi+1].astype(bool)

                                # make [B,1,1] so restore_data accepts it
                                preds_s.append(p[:, None, :])
                                trues_s.append(y[:, None, :])
                                masks_s.append(m[:, None, :])

                    pred_s = np.concatenate(preds_s, axis=0)  # [N,1,1]
                    true_s = np.concatenate(trues_s, axis=0)  # [N,1,1]
                    mask_s = np.concatenate(masks_s, axis=0)  # [N,1,1]

                    pred_s_r = test_data.restore_data(data=pred_s, num_stations=test_data.num_row)  # [S,L,1]
                    true_s_r = test_data.restore_data(data=true_s, num_stations=test_data.num_row)
                    mask_s_r = test_data.restore_data(data=mask_s, num_stations=test_data.num_row)

                    if c_mean is not None and c_std is not None:
                        pred_s_r = pred_s_r * c_std[gi] + c_mean[gi]
                        true_s_r = true_s_r * c_std[gi] + c_mean[gi]

                    unmasked = np.logical_not(mask_s_r)
                    pred_s_r[unmasked] = np.nan
                    true_s_r[unmasked] = np.nan

                    y = true_s_r[..., 0]
                    x = pred_s_r[..., 0]

                    md = cal_stations_metrics(y, x, metrics_list)
                    mv = np.isfinite(y) & np.isfinite(x)
                    d  = (x - y)[mv]
                    L1 = np.nan if d.size == 0 else float(np.mean(np.abs(d)))
                    L2 = np.nan if d.size == 0 else float(np.sqrt(np.mean(d * d)))

                    print(f"Static {name}. NSE: {np.nanmedian(md['NSE']):.3f}, "
                        f"KGE: {np.nanmedian(md['KGE']):.3f}, "
                        f"Corr: {np.nanmedian(md['Corr']):.3f}, "
                        f"L1: {L1:.3f}, L2: {L2:.3f}")
                    f.write(f"Static {name}. NSE: {np.nanmedian(md['NSE']):.3f}, "
                            f"KGE: {np.nanmedian(md['KGE']):.3f}, "
                            f"Corr: {np.nanmedian(md['Corr']):.3f}, "
                            f"L1: {L1:.3f}, L2: {L2:.3f}\n")

        f.close()
        return