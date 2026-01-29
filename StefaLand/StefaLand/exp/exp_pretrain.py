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

        if config.ratio_static_variables is None:
            weights_static_variables = torch.ones(len(config.static_variables)) * 1.0
        else:
            assert len(config.ratio_static_variables) == len(config.static_variables)
            weights_static_variables = torch.from_numpy(np.array(config.ratio_static_variables).astype(float))

        self.weights_time_series_variables = torch.nn.Parameter(weights_time_series_variables)
        self.weights_static_variables = torch.nn.Parameter(weights_static_variables)

        self.loss_weights_time_series = torch.nn.Parameter(torch.tensor([config.ratio_time_series_loss]))
        self.loss_weights_static = torch.nn.Parameter(torch.tensor([config.ratio_static_loss]))
        # self.weights_time_series_variables = weights_time_series_variables
        # self.loss_weights_time_series = torch.tensor([1.0])
        # self.loss_weights_static =torch.tensor([0.5])
        initial_ratio_categorical = len(config.static_variables_category) / len(config.static_variables)
        initial_ratio_numerical = 1 - initial_ratio_categorical
        self.loss_weights_static_numerical = torch.nn.Parameter(torch.tensor([initial_ratio_numerical]))
        self.loss_weights_static_categorical = torch.nn.Parameter(torch.tensor([initial_ratio_categorical]))

        # model uncertainty
        self.num_MC_dropout_samples = config.num_MC_dropout_samples
        self.add_input_noise = config.add_input_noise

        self.experiment_verifier()

    def experiment_verifier(self):

        if self.add_input_noise:
            assert self.config.calculate_time_series_each_variable_loss, "add_input_noise must be used with calculate_time_series_each_variable_loss"
            assert self.config.calculate_static_each_variable_loss, "add_input_noise must be used with calculate_static_each_variable_loss"

    def compute_loss(self, batch_data_dict, output_dict):
        batch_y = batch_data_dict["batch_x"]
        batch_c = batch_data_dict["batch_c"]
        outputs_time_series = output_dict["outputs_time_series"]
        outputs_static = output_dict["outputs_static"]
        target_stds = batch_data_dict["batch_target_std"]
        x_stds = batch_data_dict["batch_x_std"]
        masked_time_series_index = output_dict["masked_time_series_index"]
        masked_static_index = output_dict["masked_static_index"]
        masked_missing_time_series_index = output_dict["masked_missing_time_series_index"]
        masked_missing_static_index = output_dict["masked_missing_static_index"]

        # remove masked_missing_index from masked_index using torch
        masked_time_series_index = masked_time_series_index & (~masked_missing_time_series_index)
        masked_static_index = masked_static_index & (~masked_missing_static_index)

        outputs_time_series = outputs_time_series[:, -self.config.pred_len:, :]
        batch_y = batch_y[:, -self.config.pred_len:, :].to(self.device)
        batch_c = batch_c.to(self.device)

        # calculate time series loss
        if self.config.calculate_time_series_each_variable_loss:
            loss_time_series = 0
            for idx_feature in range(batch_y.shape[-1]):

                if ((x_stds is not None) and (0 not in x_stds.size())):
                    masked_x_stds = x_stds.unsqueeze(1).expand_as(batch_y)
                    masked_x_stds = masked_x_stds[..., idx_feature][masked_time_series_index[..., idx_feature]]
                else:
                    masked_x_stds = None

                if self.add_input_noise:
                    assert outputs_time_series.shape[-1] % 2 == 0, "each output must be data and noise"

                    sub_output_time_series = outputs_time_series[..., idx_feature * 2]
                    sub_output_time_series_noise = outputs_time_series[..., idx_feature * 2 + 1]
                    sub_masked_time_series_index = masked_time_series_index[..., idx_feature]  # double check
                    sub_batch_y = batch_y[..., idx_feature]
                    
                    sub_loss = self.criterion(
                        sub_output_time_series[sub_masked_time_series_index],
                        sub_output_time_series_noise[sub_masked_time_series_index],
                        sub_batch_y[sub_masked_time_series_index],)
                else:
                    sub_loss = self.criterion(
                        outputs_time_series[..., idx_feature][masked_time_series_index[..., idx_feature]],
                        batch_y[..., idx_feature][masked_time_series_index[..., idx_feature]],
                        target_stds=masked_x_stds)
                if not torch.isnan(sub_loss):
                    loss_time_series += sub_loss * self.weights_time_series_variables[idx_feature]
                else:
                    continue
        else:
            if ((x_stds is not None) and (0 not in x_stds.size())):
                masked_x_stds = x_stds.unsqueeze(1).expand_as(batch_y)
                masked_x_stds = masked_x_stds[masked_time_series_index]
            else:
                masked_x_stds = None
            weighted_outputs = outputs_time_series * self.weights_time_series_variables.to(outputs_time_series.device)
            loss_time_series = self.criterion(weighted_outputs[masked_time_series_index],
                                              batch_y[masked_time_series_index],
                                              target_stds=masked_x_stds)

        # calculate numerical and categorical loss
        if len(self.config.static_variables_category) > 0:
            raise NotImplementedError("Not implemented yet")
            # ratio_categorical = len(self.config.static_variables_category) / len(self.config.static_variables)
            # ratio_numerical = 1 - ratio_categorical
            #
            # static_variables_dec_index_start = output_dict['static_variables_dec_index_start']
            # static_variables_dec_index_end = output_dict['static_variables_dec_index_end']
            #
            # loss_target_categorical = 0
            # total_categorical_indices_pred = []
            # for ndx_static_variable_category, static_variable_category in enumerate(
            #         self.config.static_variables_category):
            #     indices_static_variable_category = self.config.static_variables.index(static_variable_category)
            #
            #     categorical_indices_pred = range(static_variables_dec_index_start[indices_static_variable_category],
            #                                      static_variables_dec_index_end[indices_static_variable_category])
            #     categorical_indices_obs = [indices_static_variable_category]
            #
            #     categorical_masked_static_index_obs = masked_static_index[:, categorical_indices_obs]  # [batch_size, 1]
            #     # broadcast to [batch_size, num_categorical]
            #     categorical_masked_static_index_pred = categorical_masked_static_index_obs.repeat(1,
            #                                                                                       len(categorical_indices_pred))
            #
            #     sub_loss_target_categorical = self.criterion_category(
            #         outputs_static[:, categorical_indices_pred][categorical_masked_static_index_pred].reshape(-1,
            #                                                                                                   len(categorical_indices_pred)),
            #         batch_c[:, categorical_indices_obs][categorical_masked_static_index_obs].long())
            #
            #     loss_target_categorical += sub_loss_target_categorical
            #     total_categorical_indices_pred += categorical_indices_pred
            #
            # numerical_indices_pred = [x for x in range(outputs_static.shape[-1]) if
            #                           x not in total_categorical_indices_pred]
            # numerical_indices_obs = [self.config.static_variables.index(x) for x in self.config.static_variables if
            #                          x not in self.config.static_variables_category]
            #
            # numerical_masked_static_index_obs = masked_static_index[:, numerical_indices_obs]
            # numerical_masked_static_index_pred = numerical_masked_static_index_obs
            #
            # loss_target_numerical = self.criterion(
            #     outputs_static[:, numerical_indices_pred][numerical_masked_static_index_pred],
            #     batch_c[:, numerical_indices_obs][numerical_masked_static_index_obs],
            #     target_stds=None)
            # # loss_static = loss_target_numerical * ratio_numerical + loss_target_categorical * ratio_categorical
            # loss_static = loss_target_numerical * self.loss_weights_static_numerical.to(batch_c.device) + \
            #               loss_target_categorical * self.loss_weights_static_categorical.to(batch_c.device)
        else:
            if self.config.calculate_static_each_variable_loss:
                loss_static = 0
                for idx_feature in range(batch_c.shape[-1]):

                    if self.add_input_noise:
                        assert outputs_static.shape[-1] % 2 == 0, "each output must be data and noise"
                        sub_output_static = outputs_static[..., idx_feature * 2]
                        sub_output_static_noise = outputs_static[..., idx_feature * 2 + 1]
                        sub_masked_static_index = masked_static_index[..., idx_feature]  # double check
                        sub_batch_c = batch_c[..., idx_feature]
                        sub_loss = self.criterion(
                            sub_output_static[sub_masked_static_index],
                            sub_output_static_noise[sub_masked_static_index],
                            sub_batch_c[sub_masked_static_index], )
                    else:
                        sub_loss = self.criterion(outputs_static[..., idx_feature][masked_static_index[..., idx_feature]],
                                                  batch_c[..., idx_feature][masked_static_index[..., idx_feature]],
                                                  target_stds=None)
                    if not torch.isnan(sub_loss):
                        loss_static += sub_loss * self.weights_static_variables[idx_feature]
                    else:
                        continue
            else:
                weighted_static = outputs_static * self.weights_static_variables.to(outputs_static.device)
                loss_static = self.criterion(weighted_static[masked_static_index],
                                             batch_c[masked_static_index],
                                             target_stds=None)

        loss = loss_time_series * self.loss_weights_time_series.to(loss_time_series.device) + \
               loss_static * self.loss_weights_static.to(loss_static.device)
        return loss

    def train(self):

        # record the start time
        f = open(os.path.join(self.saved_dir, "results.txt"), 'a')
        f.write("\n")
        f.write("train start time: " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + '\n')

        # resume from checkpoint
        self.load_model_resume()

        if self.config.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in np.arange(self.start_epoch, self.config.epochs):
            epoch += 1

            train_loss, val_loss = [], []
            epoch_time = time.time()
            for data_name in self.config.data:
                train_loader = self.dataset_dict[data_name]["train_loader"]
                val_loader = self.dataset_dict[data_name]["val_loader"]
                self.config = self.dataset_dict[data_name]["config"]

                self.model.train()

                for idx_index_file in range(train_loader.dataset.num_index_files):
                    if idx_index_file > 0:
                        train_loader.dataset.load_index(idx_index_file)
                        train_loader.dataset.set_length(len(train_loader.dataset.slice_grid_list))

                    for i, batch_data_dict in enumerate(train_loader):

                        self.model_optim.zero_grad()

                        batch_data_dict = {
                            key: value.float().to(self.device) if torch.is_tensor(value) and not value.dtype == torch.bool
                            else value for key, value in batch_data_dict.items()
                        }
                        batch_data_dict['mode'] = 'train'
                        # encoder - decoder
                        if self.config.use_amp:
                            with torch.cuda.amp.autocast():
                                output_dict = self.model(batch_data_dict)
                                loss = self.compute_loss(batch_data_dict, output_dict)

                                # skip nan loss
                                if torch.isnan(loss):
                                    continue

                                train_loss.append(loss.item())
                            scaler.scale(loss).backward()
                            scaler.step(self.model_optim)
                            scaler.update()

                        else:
                            output_dict = self.model(batch_data_dict)

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

                if self.config.do_eval:
                    val_loss.append(self.val(val_loader))
                else:
                    val_loss = None

                # update the dataset index
                update_time = time.time()
                train_loader.dataset.update_samples_index(True)
                print("update dataset index cost time: {:.2f} s".format(time.time() - update_time))
                f.write("update dataset index cost time: {:.2f} s".format(time.time() - update_time) + '\n')

            self.save_model(self.checkpoints_dir, epoch, self.model, self.model_optim, self.epoch_train_loss_list,
                            self.epoch_val_loss_list)

            adjust_learning_rate(self.model_optim, epoch + 1, self.config)

            train_loss = np.average(train_loss)
            val_loss = np.average(val_loss) if val_loss is not None else None

            print("Epoch: {} train loss: {:.3f} cost time: {:.2f} s".format(epoch, train_loss,
                                                                               time.time() - epoch_time))
            print("Epoch: {} val loss: {:.3f} ".format(epoch, val_loss))
            f = open(os.path.join(self.saved_dir, "results.txt"), 'a')
            f.write("Epoch: {} train loss: {:.3f} cost time: {:.2f} s".format(epoch, train_loss,
                                                                              time.time() - epoch_time) + '\n')
            f.write("Epoch: {} val loss: {:.3f}".format(epoch, val_loss) + '\n')
            self.epoch_train_loss_list.append(train_loss)
            self.epoch_val_loss_list.append(val_loss)

            # save the loss
            loss_csv_file = os.path.join(self.results_dir, "loss_data.csv")
            loss_pic_file = os.path.join(self.saved_dir, "loss_curve.png")
            self.write_loss_to_csv(epoch, train_loss, val_loss, loss_csv_file)
            # plot the loss curve
            self.plot_loss_curve_from_csv(loss_csv_file, loss_pic_file)
        f.close()

        print("results saved in: ", os.path.abspath(self.results_dir))

        return self.model

    def val(self, val_loader):
        total_loss = []
        self.model.eval()
        with torch.no_grad():

            for idx_index_file in range(val_loader.dataset.num_index_files):
                if idx_index_file > 0:
                    val_loader.dataset.load_index(idx_index_file)
                    val_loader.dataset.set_length(len(val_loader.dataset.slice_grid_list))

                for i, batch_data_dict in enumerate(val_loader):

                    batch_data_dict = {
                        key: value.float().to(self.device) if torch.is_tensor(value) and not value.dtype == torch.bool
                        else value for key, value in batch_data_dict.items()
                    }
                    batch_data_dict['mode'] = 'val'

                    # encoder - decoder
                    if self.config.use_amp:
                        with torch.cuda.amp.autocast():
                            output_dict = self.model(batch_data_dict)
                    else:
                        output_dict = self.model(batch_data_dict)

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

            preds, trues, preds_static, trues_static = [], [], [], []
            masked_index_time_series_list, masked_index_static_list = [], []

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

                        # encoder - decoder
                        if self.config.use_amp:
                            with torch.cuda.amp.autocast():
                                output_dict = self.model(batch_data_dict)
                        else:
                            output_dict = self.model(batch_data_dict)

                        outputs_time_series = output_dict["outputs_time_series"]  # [batch_size, pred_len, num_features]
                        outputs_static = output_dict["outputs_static"]  # [batch_size, num_features]
                        masked_time_series_index = output_dict["masked_time_series_index"]  # [batch_size, pred_len]
                        masked_static_index = output_dict["masked_static_index"]  # [batch_size, num_features]

                        pred = outputs_time_series[:, -self.config.pred_len:, :]
                        true = batch_y[:, -self.config.pred_len:, :]

                        preds.append(pred.detach().cpu().numpy())
                        trues.append(true.detach().cpu().numpy())

                        preds_static.append(outputs_static.detach().cpu().numpy()[:, None, :])
                        trues_static.append(batch_data_dict["batch_c"].detach().cpu().numpy()[:, None, :])

                        masked_index_time_series_list.append(masked_time_series_index.detach().cpu().numpy())
                        masked_index_static_list.append(masked_static_index.detach().cpu().numpy()[:, None, :])

            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)

            preds_static = np.concatenate(preds_static, axis=0)
            trues_static = np.concatenate(trues_static, axis=0)

            masked_index_time_series_list = np.concatenate(masked_index_time_series_list, axis=0)
            masked_index_static_list = np.concatenate(masked_index_static_list, axis=0)

            # result save
            preds_restored = test_data.restore_data(data=preds, num_stations=test_data.num_row)
            trues_restored = test_data.restore_data(data=trues, num_stations=test_data.num_row)
            masked_index_time_series_list_restored = test_data.restore_data(data=masked_index_time_series_list,
                                                                            num_stations=test_data.num_row)

            preds_static_restored = test_data.restore_data(data=preds_static, num_stations=test_data.num_row)
            trues_static_restored = test_data.restore_data(data=trues_static, num_stations=test_data.num_row)
            masked_index_static_list_restored = test_data.restore_data(data=masked_index_static_list,
                                                                       num_stations=test_data.num_row)

            raw_date_len = len(pd.date_range(test_data.config_dataset.test_date_list[0],
                                             test_data.config_dataset.test_date_list[-1]))
            assert raw_date_len == preds_restored.shape[1]

            # rescale the data
            preds_rescale = test_data.inverse_transform(preds_restored, mean=test_data.scaler["x_mean"],
                                                        std=test_data.scaler["x_std"])
            trues_rescale = test_data.inverse_transform(trues_restored, mean=test_data.scaler["x_mean"],
                                                        std=test_data.scaler["x_std"])

            preds_static_rescale = test_data.inverse_transform(preds_static_restored, mean=test_data.scaler["c_mean"],
                                                               std=test_data.scaler["c_std"], inverse_categorical=True)
            trues_static_rescale = test_data.inverse_transform(trues_static_restored, mean=test_data.scaler["c_mean"],
                                                               std=test_data.scaler["c_std"], inverse_categorical=True)

            # preds_static_rescale = preds_static_rescale[:, self.config.static_pred_start_point, :]
            # trues_static_rescale = trues_static_rescale[:, self.config.static_pred_start_point, :]
            preds_static_rescale = np.nanmean(preds_static_rescale, axis=1)
            trues_static_rescale = np.nanmean(trues_static_rescale, axis=1)
            masked_index_static_list_restored = masked_index_static_list_restored[:,
                                                self.config.static_pred_start_point,
                                                :]

            unmasked_index_time_series_list_restored = np.logical_not(masked_index_time_series_list_restored)
            unmasked_index_static_list_restored = np.logical_not(masked_index_static_list_restored)

            # replace the masked value with the np.nan to calculate the metrics
            preds_rescale[unmasked_index_time_series_list_restored] = np.nan
            trues_rescale[unmasked_index_time_series_list_restored] = np.nan

            preds_static_rescale[unmasked_index_static_list_restored] = np.nan
            trues_static_rescale[unmasked_index_static_list_restored] = np.nan

            # plot the final forecasting
            for idx_feature in range(preds_rescale.shape[-1]):
                visual(trues_rescale[0, :1000, idx_feature], preds_rescale[0, :1000, idx_feature],
                       os.path.join(self.results_dir,
                                    'feature_{}.pdf'.format(self.config.time_series_variables[idx_feature])))

            time_series_median_metrics_dict = {key: [] for key in ["NSE", "KGE", "Corr"]}
            static_median_metrics_dict = {key: [] for key in ["NSE", "KGE", "Corr"]}
            # write the date and time
            f.write("Test date: {}, time: {}".format(datetime.datetime.now().strftime("%Y-%m-%d"),
                                                     datetime.datetime.now().strftime("%H:%M:%S")))
            f.write('\n')

            # calculate the metrics for each feature
            for idx_feature in range(preds_rescale.shape[-1]):
                metrics_list = ["NSE", "KGE", "Corr"]

                trues_rescale_temp = trues_rescale[..., idx_feature]
                preds_rescale_temp = preds_rescale[..., idx_feature]
                if not self.config.time_series_variables[idx_feature] in self.config.negative_value_variables:
                    trues_rescale_temp[trues_rescale_temp < 0] = 0
                    preds_rescale_temp[preds_rescale_temp < 0] = 0
                metrics_dict = cal_stations_metrics(trues_rescale_temp, preds_rescale_temp, metrics_list)
                print(
                    "{}. NSE: {:.3f}, KGE: {:.3f}, Corr: {:.3f}".format(self.config.time_series_variables[idx_feature],
                                                                        np.nanmedian(metrics_dict["NSE"]),
                                                                        np.nanmedian(metrics_dict["KGE"]),
                                                                        np.nanmedian(metrics_dict["Corr"])))
                f.write(
                    "{}. NSE: {:.3f}, KGE: {:.3f}, Corr: {:.3f}".format(self.config.time_series_variables[idx_feature],
                                                                        np.nanmedian(metrics_dict["NSE"]),
                                                                        np.nanmedian(metrics_dict["KGE"]),
                                                                        np.nanmedian(metrics_dict["Corr"])))
                f.write('\n')
                time_series_median_metrics_dict["NSE"].append(np.nanmedian(metrics_dict["NSE"]))
                time_series_median_metrics_dict["KGE"].append(np.nanmedian(metrics_dict["KGE"]))
                time_series_median_metrics_dict["Corr"].append(np.nanmedian(metrics_dict["Corr"]))
            f.write('\n')

            print("obs_mean: {:.3f}, obs_median: {:.7f}".format(np.nanmean(trues_rescale), np.nanmedian(trues_rescale)))
            print(
                "pred_mean: {:.3f}, pred_median: {:.7f}".format(np.nanmean(preds_rescale), np.nanmedian(preds_rescale)))

            # for idx_feature_static in range(preds_static_rescale.shape[-1]):
            for idx_feature_static, static_variable in enumerate(self.config.static_variables):
                if not static_variable in self.config.static_variables_category:
                    metrics_list = ["NSE", "KGE", "Corr"]

                    trues_static_rescale_temp = trues_static_rescale[:, idx_feature_static][None, :]
                    preds_static_rescale_temp = preds_static_rescale[:, idx_feature_static][None, :]
                    if not self.config.static_variables[idx_feature_static] in self.config.negative_value_variables:
                        trues_static_rescale_temp[trues_static_rescale_temp < 0] = 0
                        preds_static_rescale_temp[preds_static_rescale_temp < 0] = 0
                    metrics_dict = cal_stations_metrics(trues_static_rescale_temp, preds_static_rescale_temp, metrics_list, remove_neg=False)

                    print("{}. NSE: {:.3f}, KGE: {:.3f}, Corr: {:.3f}".format(
                        self.config.static_variables[idx_feature_static],
                        np.nanmedian(metrics_dict["NSE"]),
                        np.nanmedian(metrics_dict["KGE"]),
                        np.nanmedian(metrics_dict["Corr"])))
                    static_median_metrics_dict["NSE"].append(np.nanmedian(metrics_dict["NSE"]))
                    static_median_metrics_dict["KGE"].append(np.nanmedian(metrics_dict["KGE"]))
                    static_median_metrics_dict["Corr"].append(np.nanmedian(metrics_dict["Corr"]))

                    f.write(
                        "{}. NSE: {:.3f}, KGE: {:.3f}, Corr: {:.3f}".format(
                            self.config.static_variables[idx_feature_static],
                            np.nanmedian(metrics_dict["NSE"]),
                            np.nanmedian(metrics_dict["KGE"]),
                            np.nanmedian(metrics_dict["Corr"])))
                    f.write('\n')
                else:
                    metrics_list = ["Accuracy", "Precision", "Recall", "F1"]
                    sub_pred = preds_static_rescale[:, idx_feature_static]
                    sub_true = trues_static_rescale[:, idx_feature_static]

                    # remove the nan value
                    index_nan = np.isnan(sub_pred) | np.isnan(sub_true)
                    sub_pred = sub_pred[~index_nan]
                    sub_true = sub_true[~index_nan]

                    Accuracy = accuracy_score(sub_true, sub_pred)
                    Precision = precision_score(sub_true, sub_pred, average='micro')
                    Recall = recall_score(sub_true, sub_pred, average='micro')
                    F1 = f1_score(sub_true, sub_pred, average='micro')
                    print("{}. Accuracy: {:.3f}, Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}".format(
                        self.config.static_variables[idx_feature_static],
                        Accuracy,
                        Precision,
                        Recall,
                        F1))
                    static_median_metrics_dict["NSE"].append(F1)
                    static_median_metrics_dict["KGE"].append(F1)
                    static_median_metrics_dict["Corr"].append(F1)

                    f.write(
                        "{}. Accuracy: {:.3f}, Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}".format(
                            self.config.static_variables[idx_feature_static],
                            Accuracy,
                            Precision,
                            Recall,
                            F1))
                    f.write('\n')

            f.write('\n')

            # np.save(self.results_dir + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
            np.save(os.path.join(self.results_dir, 'pred_time_series.npy'), preds_rescale)
            np.save(os.path.join(self.results_dir, 'true_time_series.npy'), trues_rescale)

            np.save(os.path.join(self.results_dir, 'pred_static.npy'), preds_static_rescale)
            np.save(os.path.join(self.results_dir, 'true_static.npy'), trues_static_rescale)

            # plot the metrics
            for key in ["NSE", "KGE", "Corr"]:
                saved_file = os.path.join(self.results_dir, "metrics_{}.png".format(key))
                self.plot_time_series_statics_metrics_bar(saved_file,
                                                          key,
                                                          time_series_median_metrics_dict[key],
                                                          self.config.time_series_variables,
                                                          static_median_metrics_dict[key],
                                                          self.config.static_variables)

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
            static_order = self.config.static_variables

            # inference variables
            if self.config.inference_variables is None:
                inference_time_series_variables = self.config.time_series_variables
                inference_static_variables = self.config.static_variables
            else:
                inference_time_series_variables = [var for var in self.config.inference_variables if var in
                                                   self.config.time_series_variables]
                inference_static_variables = [var for var in self.config.inference_variables if var in
                                              self.config.static_variables]

            metrics_list = ["NSE", "KGE", "Corr"]
            time_series_median_metrics_dict = {key: [] for key in metrics_list}
            static_median_metrics_dict = {key: [] for key in metrics_list}

            # inference time series
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
                                # set bool index and size as batch_x and value are all False
                                batch_time_series_mask_index = torch.zeros_like(batch_data_dict["batch_x"],
                                                                                dtype=torch.bool)
                                batch_static_mask_index = torch.zeros_like(batch_data_dict["batch_c"], dtype=torch.bool)
                                batch_time_series_mask_index[:, :, idx_time_series_variable] = True

                                batch_data_dict["batch_time_series_mask_index"] = batch_time_series_mask_index
                                batch_data_dict["batch_static_mask_index"] = batch_static_mask_index

                                batch_y = batch_data_dict["batch_x"]
                                batch_data_dict['mode'] = 'test'
                                output_dict = self.model(batch_data_dict)

                                outputs_time_series = output_dict[
                                    "outputs_time_series"]  # [batch_size, pred_len, num_features]

                                pred = outputs_time_series[:, -self.config.pred_len:, idx_time_series_variable][..., None]
                                true = batch_y[:, -self.config.pred_len:, idx_time_series_variable][..., None]

                                preds.append(pred.detach().cpu().numpy())
                                trues.append(true.detach().cpu().numpy())

                    preds = np.concatenate(preds, axis=0)
                    trues = np.concatenate(trues, axis=0)

                    preds_restored = test_data.restore_data(data=preds, num_stations=test_data.num_row)
                    trues_restored = test_data.restore_data(data=trues, num_stations=test_data.num_row)

                    raw_date_len = len(pd.date_range(test_data.config_dataset.test_date_list[0],
                                                     test_data.config_dataset.test_date_list[-1]))
                    assert raw_date_len == preds_restored.shape[1]

                    # rescale the data
                    preds_rescale = test_data.inverse_transform(preds_restored,
                                                                mean=test_data.scaler["x_mean"][
                                                                    idx_time_series_variable],
                                                                std=test_data.scaler["x_std"][idx_time_series_variable])
                    trues_rescale = test_data.inverse_transform(trues_restored,
                                                                mean=test_data.scaler["x_mean"][
                                                                    idx_time_series_variable],
                                                                std=test_data.scaler["x_std"][idx_time_series_variable])

                    trues_rescale_temp = trues_rescale[..., 0]
                    preds_rescale_temp = preds_rescale[..., 0]
                    if not self.config.time_series_variables[idx_time_series_variable] in self.config.negative_value_variables:
                        trues_rescale_temp[trues_rescale_temp < 0] = 0
                        preds_rescale_temp[preds_rescale_temp < 0] = 0
                    metrics_dict = cal_stations_metrics(trues_rescale_temp, preds_rescale_temp, metrics_list)
                    print(
                        "{}. NSE: {:.3f}, KGE: {:.3f}, Corr: {:.3f}".format(
                            self.config.time_series_variables[idx_time_series_variable],
                            np.nanmedian(metrics_dict["NSE"]),
                            np.nanmedian(metrics_dict["KGE"]),
                            np.nanmedian(metrics_dict["Corr"])))
                    f.write(
                        "{}. NSE: {:.3f}, KGE: {:.3f}, Corr: {:.3f}".format(
                            self.config.time_series_variables[idx_time_series_variable],
                            np.nanmedian(metrics_dict["NSE"]),
                            np.nanmedian(metrics_dict["KGE"]),
                            np.nanmedian(metrics_dict["Corr"])))
                    f.write('\n')

                    time_series_median_metrics_dict["NSE"].append(np.nanmedian(metrics_dict["NSE"]))
                    time_series_median_metrics_dict["KGE"].append(np.nanmedian(metrics_dict["KGE"]))
                    time_series_median_metrics_dict["Corr"].append(np.nanmedian(metrics_dict["Corr"]))

            f.write('\n')

            # inference static
            if len(inference_static_variables) > 0:
                for static_variable in inference_static_variables:

                    idx_static_variable = static_order.index(static_variable)

                    if self.config.group_mask_dict is not None:
                        group = next(
                            (values for key, values in self.config.group_mask_dict.items() if
                             static_variable in values),
                            [static_variable])
                    else:
                        group = [static_variable]
                    mask_idx_static_variable = np.array([static_order.index(item) for item in group])

                    preds_static, trues_static = [], []

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
                                # set bool index and size as batch_x and value are all False
                                batch_time_series_mask_index = torch.zeros_like(batch_data_dict["batch_x"],
                                                                                dtype=torch.bool)
                                batch_static_mask_index = torch.zeros_like(batch_data_dict["batch_c"], dtype=torch.bool)
                                batch_static_mask_index[:, mask_idx_static_variable] = True

                                batch_data_dict["batch_time_series_mask_index"] = batch_time_series_mask_index
                                batch_data_dict["batch_static_mask_index"] = batch_static_mask_index

                                batch_c = batch_data_dict["batch_c"]
                                batch_data_dict['mode'] = 'test'
                                output_dict = self.model(batch_data_dict)

                                outputs_static = output_dict["outputs_static"]

                                preds_static.append(
                                    outputs_static.detach().cpu().numpy()[:, None, idx_static_variable][..., None])
                                trues_static.append(batch_c.detach().cpu().numpy()[:, None, idx_static_variable][..., None])

                    preds_static = np.concatenate(preds_static, axis=0)
                    trues_static = np.concatenate(trues_static, axis=0)

                    preds_static_restored = test_data.restore_data(data=preds_static, num_stations=test_data.num_row)
                    trues_static_restored = test_data.restore_data(data=trues_static, num_stations=test_data.num_row)

                    preds_static_rescale = test_data.inverse_transform(preds_static_restored,
                                                                       mean=test_data.scaler["c_mean"][
                                                                           idx_static_variable],
                                                                       std=test_data.scaler["c_std"][
                                                                           idx_static_variable],
                                                                       inverse_categorical=True,
                                                                       single_variable_name=static_variable)
                    trues_static_rescale = test_data.inverse_transform(trues_static_restored,
                                                                       mean=test_data.scaler["c_mean"][
                                                                           idx_static_variable],
                                                                       std=test_data.scaler["c_std"][
                                                                           idx_static_variable],
                                                                       inverse_categorical=True,
                                                                       single_variable_name=static_variable)

                    # preds_static_rescale = preds_static_rescale[:, self.config.static_pred_start_point, :]
                    # trues_static_rescale = trues_static_rescale[:, self.config.static_pred_start_point, :]
                    preds_static_rescale = np.nanmean(preds_static_rescale, axis=1)
                    trues_static_rescale = np.nanmean(trues_static_rescale, axis=1)

                    # save the results
                    np.save(os.path.join(self.results_dir, 'pred_static_{}.npy'.format(static_variable)),
                            preds_static_rescale)
                    np.save(os.path.join(self.results_dir, 'true_static_{}.npy'.format(static_variable)),
                            trues_static_rescale)

                    if not static_variable in self.config.static_variables_category:

                        # calculate the metrics for each feature
                        metrics_list = ["NSE", "KGE", "Corr"]

                        trues_static_rescale_temp = trues_static_rescale[:, 0][None, :]
                        preds_static_rescale_temp = preds_static_rescale[:, 0][None, :]
                        if not self.config.static_variables[idx_static_variable] in self.config.negative_value_variables:
                            trues_static_rescale_temp[trues_static_rescale_temp < 0] = 0
                            preds_static_rescale_temp[preds_static_rescale_temp < 0] = 0
                        metrics_dict = cal_stations_metrics(trues_static_rescale_temp, preds_static_rescale_temp,
                                                            metrics_list, remove_neg=False)

                        print("{}. NSE: {:.3f}, KGE: {:.3f}, Corr: {:.3f}".format(
                            self.config.static_variables[idx_static_variable],
                            np.nanmedian(metrics_dict["NSE"]),
                            np.nanmedian(metrics_dict["KGE"]),
                            np.nanmedian(metrics_dict["Corr"])))
                        f.write(
                            "{}. NSE: {:.3f}, KGE: {:.3f}, Corr: {:.3f}".format(
                                self.config.static_variables[idx_static_variable],
                                np.nanmedian(metrics_dict["NSE"]),
                                np.nanmedian(metrics_dict["KGE"]),
                                np.nanmedian(metrics_dict["Corr"])))
                        f.write('\n')

                        static_median_metrics_dict["NSE"].append(np.nanmedian(metrics_dict["NSE"]))
                        static_median_metrics_dict["KGE"].append(np.nanmedian(metrics_dict["KGE"]))
                        static_median_metrics_dict["Corr"].append(np.nanmedian(metrics_dict["Corr"]))

                    else:
                        metrics_list = ["Accuracy", "Precision", "Recall", "F1"]
                        sub_pred = preds_static_rescale[:, 0]
                        sub_true = trues_static_rescale[:, 0]

                        # remove the nan value
                        index_nan = np.isnan(sub_pred) | np.isnan(sub_true)
                        sub_pred = sub_pred[~index_nan]
                        sub_true = sub_true[~index_nan]

                        Accuracy = accuracy_score(sub_true, sub_pred)
                        Precision = precision_score(sub_true, sub_pred, average='micro')
                        Recall = recall_score(sub_true, sub_pred, average='micro')
                        F1 = f1_score(sub_true, sub_pred, average='micro')
                        print("{}. Accuracy: {:.3f}, Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}".format(
                            static_variable,
                            Accuracy,
                            Precision,
                            Recall,
                            F1))
                        f.write(
                            "{}. Accuracy: {:.3f}, Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}".format(
                                static_variable,
                                Accuracy,
                                Precision,
                                Recall,
                                F1))
                        f.write('\n')

                        static_median_metrics_dict["NSE"].append(F1)
                        static_median_metrics_dict["KGE"].append(F1)
                        static_median_metrics_dict["Corr"].append(F1)

            f.write('\n')

            # plot the metrics
            for key in ["NSE", "KGE", "Corr"]:
                saved_file = os.path.join(self.results_dir, "inference_metrics_{}.png".format(key))
                self.plot_time_series_statics_metrics_bar(saved_file,
                                                          key,
                                                          time_series_median_metrics_dict[key],
                                                          inference_time_series_variables,
                                                          static_median_metrics_dict[key],
                                                          inference_static_variables)
        f.close()
        return

    def MCD_N_inference(self):

        assert self.num_MC_dropout_samples > 1, "The number of Monte Carlo samples should be larger than 1"

        f = open(os.path.join(self.saved_dir, "results.txt"), 'a')
        f.write("Monte Carlo Dropout date: {}, time: {}".format(datetime.datetime.now().strftime("%Y-%m-%d"),
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
            static_order = self.config.static_variables

            # inference variables
            if self.config.inference_variables is None:
                inference_time_series_variables = self.config.time_series_variables
                inference_static_variables = self.config.static_variables
            else:
                inference_time_series_variables = [var for var in self.config.inference_variables if var in
                                                   self.config.time_series_variables]
                inference_static_variables = [var for var in self.config.inference_variables if var in
                                              self.config.static_variables]

            metrics_list = ["NSE", "KGE", "Corr"]

            for idx_MC in range(self.num_MC_dropout_samples+1):

                f.write(f'MC Dropout sample {idx_MC}: \n')

                time_series_median_metrics_dict = {key: [] for key in metrics_list}
                static_median_metrics_dict = {key: [] for key in metrics_list}

                # inference time series
                if len(inference_time_series_variables) > 0:

                    for time_series_variable in inference_time_series_variables:

                        idx_time_series_variable = time_series_order.index(time_series_variable)

                        preds, trues, noises = [], [], []

                        self.model.eval()
                        if idx_MC > 0:
                            self.model.apply(apply_dropout)
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
                                    # set bool index and size as batch_x and value are all False
                                    batch_time_series_mask_index = torch.zeros_like(batch_data_dict["batch_x"],
                                                                                    dtype=torch.bool)
                                    batch_static_mask_index = torch.zeros_like(batch_data_dict["batch_c"], dtype=torch.bool)
                                    batch_time_series_mask_index[:, :, idx_time_series_variable] = True

                                    batch_data_dict["batch_time_series_mask_index"] = batch_time_series_mask_index
                                    batch_data_dict["batch_static_mask_index"] = batch_static_mask_index

                                    batch_y = batch_data_dict["batch_x"]
                                    batch_data_dict['mode'] = 'test'
                                    output_dict = self.model(batch_data_dict)

                                    outputs_time_series = output_dict[
                                        "outputs_time_series"]  # [batch_size, pred_len, num_features]

                                    pred = outputs_time_series[:, -self.config.pred_len:, idx_time_series_variable * 2][..., None]
                                    noise = outputs_time_series[:, -self.config.pred_len:, idx_time_series_variable * 2 + 1][..., None]
                                    true = batch_y[:, -self.config.pred_len:, idx_time_series_variable][..., None]


                                    preds.append(pred.detach().cpu().numpy())
                                    trues.append(true.detach().cpu().numpy())
                                    noises.append(noise.detach().cpu().numpy())

                        preds = np.concatenate(preds, axis=0)
                        trues = np.concatenate(trues, axis=0)
                        noises = np.concatenate(noises, axis=0)

                        preds_restored = test_data.restore_data(data=preds, num_stations=test_data.num_row)
                        trues_restored = test_data.restore_data(data=trues, num_stations=test_data.num_row)
                        noises_restored = test_data.restore_data(data=noises, num_stations=test_data.num_row)

                        raw_date_len = len(pd.date_range(test_data.config_dataset.test_date_list[0],
                                                         test_data.config_dataset.test_date_list[-1]))
                        assert raw_date_len == preds_restored.shape[1]

                        # rescale the data
                        preds_rescale = test_data.inverse_transform(preds_restored,
                                                                    mean=test_data.scaler["x_mean"][
                                                                        idx_time_series_variable],
                                                                    std=test_data.scaler["x_std"][idx_time_series_variable])
                        trues_rescale = test_data.inverse_transform(trues_restored,
                                                                    mean=test_data.scaler["x_mean"][
                                                                        idx_time_series_variable],
                                                                    std=test_data.scaler["x_std"][idx_time_series_variable])
                        noises_rescale = np.sqrt(np.exp(noises_restored)) * test_data.scaler["x_std"][idx_time_series_variable]

                        trues_rescale_temp = trues_rescale[..., 0]
                        preds_rescale_temp = preds_rescale[..., 0]
                        if not self.config.time_series_variables[idx_time_series_variable] in self.config.negative_value_variables:
                            trues_rescale_temp[trues_rescale_temp < 0] = 0
                            preds_rescale_temp[preds_rescale_temp < 0] = 0
                        metrics_dict = cal_stations_metrics(trues_rescale_temp, preds_rescale_temp, metrics_list)
                        print(
                            "MC_{}: {}. NSE: {:.3f}, KGE: {:.3f}, Corr: {:.3f}".format(
                                idx_MC,
                                self.config.time_series_variables[idx_time_series_variable],
                                np.nanmedian(metrics_dict["NSE"]),
                                np.nanmedian(metrics_dict["KGE"]),
                                np.nanmedian(metrics_dict["Corr"])))
                        f.write(
                            "MC_{}: {}. NSE: {:.3f}, KGE: {:.3f}, Corr: {:.3f}".format(
                                idx_MC,
                                self.config.time_series_variables[idx_time_series_variable],
                                np.nanmedian(metrics_dict["NSE"]),
                                np.nanmedian(metrics_dict["KGE"]),
                                np.nanmedian(metrics_dict["Corr"])))
                        f.write('\n')

                        time_series_median_metrics_dict["NSE"].append(np.nanmedian(metrics_dict["NSE"]))
                        time_series_median_metrics_dict["KGE"].append(np.nanmedian(metrics_dict["KGE"]))
                        time_series_median_metrics_dict["Corr"].append(np.nanmedian(metrics_dict["Corr"]))

                f.write('\n')

                # inference static
                if len(inference_static_variables) > 0:
                    for static_variable in inference_static_variables:

                        idx_static_variable = static_order.index(static_variable)

                        if self.config.group_mask_dict is not None:
                            group = next(
                                (values for key, values in self.config.group_mask_dict.items() if
                                 static_variable in values),
                                [static_variable])
                        else:
                            group = [static_variable]
                        mask_idx_static_variable = np.array([static_order.index(item) for item in group])

                        preds_static, trues_static, noises_static = [], [], []

                        self.model.eval()
                        if idx_MC > 0:
                            self.model.apply(apply_dropout)
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
                                    # set bool index and size as batch_x and value are all False
                                    batch_time_series_mask_index = torch.zeros_like(batch_data_dict["batch_x"],
                                                                                    dtype=torch.bool)
                                    batch_static_mask_index = torch.zeros_like(batch_data_dict["batch_c"], dtype=torch.bool)
                                    batch_static_mask_index[:, mask_idx_static_variable] = True

                                    batch_data_dict["batch_time_series_mask_index"] = batch_time_series_mask_index
                                    batch_data_dict["batch_static_mask_index"] = batch_static_mask_index

                                    batch_c = batch_data_dict["batch_c"]
                                    batch_data_dict['mode'] = 'test'
                                    output_dict = self.model(batch_data_dict)

                                    outputs_static = output_dict["outputs_static"]

                                    preds_static.append(
                                        outputs_static.detach().cpu().numpy()[:, None, idx_static_variable * 2][..., None])
                                    noises_static.append(
                                        outputs_static.detach().cpu().numpy()[:, None, idx_static_variable * 2 + 1][..., None])
                                    trues_static.append(batch_c.detach().cpu().numpy()[:, None, idx_static_variable][..., None])

                        preds_static = np.concatenate(preds_static, axis=0)
                        trues_static = np.concatenate(trues_static, axis=0)
                        noises_static = np.concatenate(noises_static, axis=0)

                        preds_static_restored = test_data.restore_data(data=preds_static, num_stations=test_data.num_row)
                        trues_static_restored = test_data.restore_data(data=trues_static, num_stations=test_data.num_row)
                        noises_static_restored = test_data.restore_data(data=noises_static, num_stations=test_data.num_row)

                        preds_static_rescale = test_data.inverse_transform(preds_static_restored,
                                                                           mean=test_data.scaler["c_mean"][
                                                                               idx_static_variable],
                                                                           std=test_data.scaler["c_std"][
                                                                               idx_static_variable],
                                                                           inverse_categorical=True,
                                                                           single_variable_name=static_variable)
                        trues_static_rescale = test_data.inverse_transform(trues_static_restored,
                                                                           mean=test_data.scaler["c_mean"][
                                                                               idx_static_variable],
                                                                           std=test_data.scaler["c_std"][
                                                                               idx_static_variable],
                                                                           inverse_categorical=True,
                                                                           single_variable_name=static_variable)
                        noises_static_rescale = np.sqrt(np.exp(noises_static_restored)) * test_data.scaler["c_std"][
                            idx_static_variable]

                        # preds_static_rescale = preds_static_rescale[:, self.config.static_pred_start_point, :]
                        # trues_static_rescale = trues_static_rescale[:, self.config.static_pred_start_point, :]
                        preds_static_rescale = np.nanmean(preds_static_rescale, axis=1)
                        trues_static_rescale = np.nanmean(trues_static_rescale, axis=1)
                        noises_static_rescale = noises_static_rescale[:, self.config.static_pred_start_point, :]

                        # save the results
                        np.save(os.path.join(self.results_dir, 'pred_static_{}_MC_{}.npy'.format(static_variable, idx_MC)),
                                preds_static_rescale)
                        np.save(os.path.join(self.results_dir, 'true_static_{}_MC_{}.npy'.format(static_variable, idx_MC)),
                                trues_static_rescale)
                        np.save(os.path.join(self.results_dir, 'noise_static_{}_MC_{}.npy'.format(static_variable, idx_MC)),
                                noises_static_rescale)

                        if not static_variable in self.config.static_variables_category:

                            # calculate the metrics for each feature
                            metrics_list = ["NSE", "KGE", "Corr"]

                            trues_static_rescale_temp = trues_static_rescale[:, 0][None, :]
                            preds_static_rescale_temp = preds_static_rescale[:, 0][None, :]
                            if not self.config.static_variables[idx_static_variable] in self.config.negative_value_variables:
                                trues_static_rescale_temp[trues_static_rescale_temp < 0] = 0
                                preds_static_rescale_temp[preds_static_rescale_temp < 0] = 0
                            metrics_dict = cal_stations_metrics(trues_static_rescale_temp, preds_static_rescale_temp,
                                                                metrics_list, remove_neg=False)

                            print("MC_{}: {}. NSE: {:.3f}, KGE: {:.3f}, Corr: {:.3f}".format(
                                idx_MC,
                                self.config.static_variables[idx_static_variable],
                                np.nanmedian(metrics_dict["NSE"]),
                                np.nanmedian(metrics_dict["KGE"]),
                                np.nanmedian(metrics_dict["Corr"])))
                            f.write(
                                "MC_{}: {}. NSE: {:.3f}, KGE: {:.3f}, Corr: {:.3f}".format(
                                    idx_MC,
                                    self.config.static_variables[idx_static_variable],
                                    np.nanmedian(metrics_dict["NSE"]),
                                    np.nanmedian(metrics_dict["KGE"]),
                                    np.nanmedian(metrics_dict["Corr"])))
                            f.write('\n')

                            static_median_metrics_dict["NSE"].append(np.nanmedian(metrics_dict["NSE"]))
                            static_median_metrics_dict["KGE"].append(np.nanmedian(metrics_dict["KGE"]))
                            static_median_metrics_dict["Corr"].append(np.nanmedian(metrics_dict["Corr"]))

                        else:
                            metrics_list = ["Accuracy", "Precision", "Recall", "F1"]
                            sub_pred = preds_static_rescale[:, 0]
                            sub_true = trues_static_rescale[:, 0]

                            # remove the nan value
                            index_nan = np.isnan(sub_pred) | np.isnan(sub_true)
                            sub_pred = sub_pred[~index_nan]
                            sub_true = sub_true[~index_nan]

                            Accuracy = accuracy_score(sub_true, sub_pred)
                            Precision = precision_score(sub_true, sub_pred, average='micro')
                            Recall = recall_score(sub_true, sub_pred, average='micro')
                            F1 = f1_score(sub_true, sub_pred, average='micro')
                            print("MC_{}: {}. Accuracy: {:.3f}, Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}".format(
                                idx_MC,
                                static_variable,
                                Accuracy,
                                Precision,
                                Recall,
                                F1))
                            f.write(
                                "MC_{}: {}. Accuracy: {:.3f}, Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}".format(
                                    idx_MC,
                                    static_variable,
                                    Accuracy,
                                    Precision,
                                    Recall,
                                    F1))
                            f.write('\n')

                            static_median_metrics_dict["NSE"].append(F1)
                            static_median_metrics_dict["KGE"].append(F1)
                            static_median_metrics_dict["Corr"].append(F1)

                f.write('\n')

                # plot the metrics
                for key in ["NSE", "KGE", "Corr"]:
                    saved_file = os.path.join(self.results_dir, "inference_metrics_{}_MC_{}.png".format(key, idx_MC))
                    self.plot_time_series_statics_metrics_bar(saved_file,
                                                              key,
                                                              time_series_median_metrics_dict[key],
                                                              inference_time_series_variables,
                                                              static_median_metrics_dict[key],
                                                              inference_static_variables)
        f.close()
        return

    @staticmethod
    def plot_time_series_statics_metrics_bar(saved_file, metric_name,
                                             metric_time_series=None,
                                             time_series_variables=None,
                                             metric_static=None,
                                             static_variables=None):
        """
        Plot two types of metrics: time series and statics, in a 2-row subplot.

        metric_time_series: An array of time series metrics [features]
        metric_static: An array of static metrics [features]
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 2]})
        font_size = 14

        # Plotting horizontal bar chart for the time series metrics
        if metric_time_series is not None:
            metric_time_series = np.array(metric_time_series)
            # convert inf, -inf negative and nan to 0
            index = np.isinf(metric_time_series) | np.isnan(metric_time_series) | (metric_time_series < 0)
            metric_time_series[index] = 0

            ax1.barh(np.arange(len(metric_time_series)), metric_time_series, color='skyblue')
            # ax1.set_xlabel(metric_name)

        # set time series variables as yticks
        if time_series_variables is not None:
            ax1.set_yticks(np.arange(len(metric_time_series)))
            ax1.set_yticklabels(time_series_variables, fontsize=font_size)
            ax1.tick_params(axis='y', rotation=45)
            # # set xlim
            ax1.set_xlim([0, 1])

        # set xticks font size
        ax1.tick_params(axis='x', labelsize=font_size)

        # Plotting vertical bar chart for the static metrics
        if metric_static is not None:
            metric_static = np.array(metric_static)
            # convert inf, -inf negative and nan to 0
            index = np.isinf(metric_static) | np.isnan(metric_static) | (metric_static < 0)
            metric_static[index] = 0
            ax2.bar(np.arange(len(metric_static)), metric_static, color='salmon')
            # ax2.set_ylabel(metric_name)

            ax2.set_ylim([0, 1])

        # set static variables as xticks
        if static_variables is not None:
            ax2.set_xticks(np.arange(len(metric_static)))
            ax2.set_xticklabels(static_variables, fontsize=font_size)
            ax2.tick_params(axis='x', rotation=60)

            for label in ax2.get_xticklabels():
                label.set_horizontalalignment('right')

        # set yticks font size
        ax2.tick_params(axis='y', labelsize=font_size)

        # set title for the whole figure
        fig.suptitle(metric_name, fontsize=font_size + 2)

        # Adjusting layout and displaying the plot
        plt.tight_layout()
        # save
        plt.savefig(saved_file)
