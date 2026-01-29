import os
import time
import warnings
import numpy as np
import pandas as pd

import torch
from MFFormer.exp.exp_basic import Exp_Basic
from MFFormer.data_provider.data_factory import data_provider
from MFFormer.utils.tools import adjust_learning_rate, visual  # EarlyStopping,
from MFFormer.utils.stats.metrics import cal_stations_metrics
from MFFormer.data_provider.data_factory import get_train_val_test_dataset

warnings.filterwarnings('ignore')

class ExpForecast(Exp_Basic):
    def __init__(self, config):
        super().__init__(config)
        self.dataset_dict = get_train_val_test_dataset(self.config)
        config = self.dataset_dict[config.data[0]]["config"]

    def model_forward(self, batch_data_dict):
        batch_x = batch_data_dict["batch_x"]
        batch_c = batch_data_dict["batch_c"]
        batch_y = batch_data_dict["batch_y"]
        batch_x_mark = batch_data_dict["batch_x_time_stamp"]
        batch_y_mark = batch_data_dict["batch_y_time_stamp"]
        batch_target_std = batch_data_dict["batch_target_std"]

        num_target_variables = len(self.config.target_variables)
        batch_y_history = batch_x[:, -self.config.label_len:, -num_target_variables:]
        batch_y_mark_history = batch_x_mark[:, -self.config.label_len:, :]

        batch_y_mark = torch.cat([batch_y_mark_history, batch_y_mark], dim=1)

        # decoder input
        dec_inp = torch.zeros_like(batch_y).float()
        dec_inp = torch.cat([batch_y_history, dec_inp], dim=1).float().to(self.device)

        batch_data_dict["dec_inp"] = dec_inp

        outputs_dict = self.model(batch_data_dict)
        return outputs_dict

    def compute_loss(self, batch_data_dict, output_dict):
        batch_y = batch_data_dict["batch_y"]
        outputs_time_series = output_dict["outputs_time_series"]
        target_stds = batch_data_dict["batch_target_std"]

        outputs_time_series = outputs_time_series[:, -self.config.pred_len:, :]
        batch_y = batch_y[:, -self.config.pred_len:, :]

        loss = self.criterion(outputs_time_series, batch_y, target_stds=target_stds)

        return loss

    def train(self):
        # resume from checkpoint
        self.load_model_resume()

        # Initialize AMP scaler if enabled
        if self.config.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in np.arange(self.start_epoch, self.config.epochs):
            epoch += 1
            for data_name in self.config.data:
                train_loader = self.dataset_dict[data_name]["train_loader"]
                val_loader = self.dataset_dict[data_name]["val_loader"]
                self.config = self.dataset_dict[data_name]["config"]

                iter_count = 0
                train_loss = []
                adjust_learning_rate(self.model_optim, epoch, self.config)

                self.model.train()
                epoch_time = time.time()
                for i, batch_data_dict in enumerate(train_loader):
                    iter_count += 1
                    self.model_optim.zero_grad()

                    batch_data_dict = {
                        key: value.float().to(self.device) if torch.is_tensor(value) and not value.dtype == torch.bool
                        else value for key, value in batch_data_dict.items()
                    }
                    batch_data_dict['mode'] = 'train'

                    # Use AMP if enabled
                    if self.config.use_amp:
                        with torch.cuda.amp.autocast():
                            output_dict = self.model_forward(batch_data_dict)
                            loss = self.compute_loss(batch_data_dict, output_dict)
                        
                        # skip nan loss
                        if torch.isnan(loss):
                            continue
                            
                        train_loss.append(loss.item())
                        
                        # Scale loss and backward
                        scaler.scale(loss).backward()
                        
                        # Unscale gradients and clip if needed
                        scaler.unscale_(self.model_optim)
                        if self.config.clip_grad is not None:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad)
                        
                        # Step optimizer and update scaler
                        scaler.step(self.model_optim)
                        scaler.update()
                    else:
                        output_dict = self.model_forward(batch_data_dict)
                        loss = self.compute_loss(batch_data_dict, output_dict)

                        # skip nan loss
                        if torch.isnan(loss):
                            continue

                        train_loss.append(loss.item())
                        loss.backward()

                        # clip gradients
                        if self.config.clip_grad is not None:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad)

                        self.model_optim.step()

            self.save_model(self.checkpoints_dir, epoch, self.model, self.model_optim, self.epoch_train_loss_list,
                            self.epoch_val_loss_list)

            adjust_learning_rate(self.model_optim, epoch + 1, self.config)

            train_loss = np.average(train_loss)

            # print the time of each epoch
            print("Epoch: {} loss: {:.3f} cost time: {:.3f}".format(epoch, train_loss, time.time() - epoch_time))

            # update the dataset index
            update_time = time.time()
            train_loader.dataset.update_samples_index(False)
            print("update dataset index cost time: {:.3f}".format(time.time() - update_time))

            self.epoch_train_loss_list.append(train_loss)

            if self.config.do_eval:
                # validation
                val_loss = self.val(val_loader)
            else:
                val_loss = None

            self.epoch_val_loss_list.append(val_loss)

            # save the loss
            loss_csv_file = os.path.join(self.results_dir, "loss_data.csv")
            loss_pic_file = os.path.join(self.saved_dir, "loss_curve.png")
            self.write_loss_to_csv(epoch, train_loss, val_loss, loss_csv_file)
            # plot the loss curve
            self.plot_loss_curve_from_csv(loss_csv_file, loss_pic_file)

        print("results saved in: ", os.path.abspath(self.results_dir))

        return self.model

    def val(self, val_loader):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, batch_data_dict in enumerate(val_loader):

                batch_data_dict = {
                    key: value.float().to(self.device) if torch.is_tensor(value) and not value.dtype == torch.bool
                    else value for key, value in batch_data_dict.items()
                }
                batch_data_dict['mode'] = 'val'
                
                # Use AMP if enabled
                if self.config.use_amp:
                    with torch.cuda.amp.autocast():
                        output_dict = self.model_forward(batch_data_dict)
                        loss = self.compute_loss(batch_data_dict, output_dict)
                else:
                    output_dict = self.model_forward(batch_data_dict)
                    loss = self.compute_loss(batch_data_dict, output_dict)

                # skip nan loss
                if torch.isnan(loss):
                    continue

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, test=0):
        f = open(os.path.join(self.saved_dir, "results.txt"), 'a')
        
        for data_name in self.config.data:
            print("testing on {}".format(data_name))
            f.write(f'{data_name}: \n')

            test_data = self.dataset_dict[data_name]["test_data"]
            test_loader = self.dataset_dict[data_name]["test_loader"]
            self.config = self.dataset_dict[data_name]["config"]


            if self.config.do_test:
                print('loading model')
                self.load_model_resume()

            preds, trues = [], []
            self.model.eval()
            with torch.no_grad():
                for i, batch_data_dict in enumerate(test_loader):
                    batch_data_dict = {
                        key: value.float().to(self.device) if torch.is_tensor(value) and not value.dtype == torch.bool
                        else value for key, value in batch_data_dict.items()
                    }
                    batch_data_dict['mode'] = 'test'
                    batch_y = batch_data_dict["batch_y"]

                    # encoder - decoder
                    # Use AMP if enabled
                    if self.config.use_amp:
                        with torch.cuda.amp.autocast():
                            output_dict = self.model_forward(batch_data_dict)
                    else:
                        output_dict = self.model_forward(batch_data_dict)

                    outputs_time_series = output_dict["outputs_time_series"]  # [batch_size, pred_len, num_features]

                    pred = outputs_time_series[:, -self.config.pred_len:, :]
                    true = batch_y[:, -self.config.pred_len:, :]

                    preds.append(pred.detach().cpu().numpy())
                    trues.append(true.detach().cpu().numpy())

            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)

        # result save
        preds_restored = test_data.restore_data(data=preds, num_stations=test_data.num_row)
        trues_restored = test_data.restore_data(data=trues, num_stations=test_data.num_row)

        raw_date_len = len(pd.date_range(test_data.config_dataset.test_date_list[0],
                                         test_data.config_dataset.test_date_list[-1]))
        assert raw_date_len == preds_restored.shape[1]

        # rescale the data
        preds_rescale = test_data.inverse_transform(preds_restored, mean=test_data.scaler["y_mean"],
                                                    std=test_data.scaler["y_std"])
        trues_rescale = test_data.inverse_transform(trues_restored, mean=test_data.scaler["y_mean"],
                                                    std=test_data.scaler["y_std"])

        # plot the final forecasting
        for idx_feature in range(preds_rescale.shape[-1]):
            visual(trues_rescale[0, :1000, idx_feature], preds_rescale[0, :1000, idx_feature],
                   os.path.join(self.results_dir,
                                'feature_{}.pdf'.format(self.config.target_variables[idx_feature])))

        # write the config
        f.write(self.config_format(self.config))

        # calculate the metrics for each feature
        for idx_feature in range(preds_rescale.shape[-1]):
            metrics_list = ["NSE", "KGE", "Corr"]
            metrics_dict = cal_stations_metrics(trues_rescale[:, :, idx_feature], preds_rescale[:, :, idx_feature],
                                                metrics_list)
            print("{}. NSE: {:.3f}, KGE: {:.3f}, Corr: {:.3f}".format(self.config.target_variables[idx_feature],
                                                                      np.nanmedian(metrics_dict["NSE"]),
                                                                      np.nanmedian(metrics_dict["KGE"]),
                                                                      np.nanmedian(metrics_dict["Corr"])))
            f.write('\n')
            f.write("{}. NSE: {:.3f}, KGE: {:.3f}, Corr: {:.3f}".format(self.config.target_variables[idx_feature],
                                                                        np.nanmedian(metrics_dict["NSE"]),
                                                                        np.nanmedian(metrics_dict["KGE"]),
                                                                        np.nanmedian(metrics_dict["Corr"])))

        print("obs_mean: {:.3f}, obs_median: {:.7f}".format(np.nanmean(trues_rescale), np.nanmedian(trues_rescale)))
        print("pred_mean: {:.3f}, pred_median: {:.7f}".format(np.nanmean(preds_rescale), np.nanmedian(preds_rescale)))

        f.write('\n')
        f.close()

        # np.save(self.results_dir + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(os.path.join(self.results_dir, 'pred.npy'), preds_rescale)
        np.save(os.path.join(self.results_dir, 'true.npy'), trues_rescale)

        return