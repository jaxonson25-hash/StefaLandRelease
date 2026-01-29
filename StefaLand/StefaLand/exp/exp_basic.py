import os
import re
import json
import glob
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from MFFormer.models import (MFFormer, LSTM, LSTM_mask, MFFormer_fine_tune, MFFormer_LSTM, MFFormer_dec_LSTM, MFFormer_regression, 
MFFormer_dec_embedding_LSTM, MFFormer_dec_LSTM_time_series, MFFormer_dec_embedding_LSTM_Tape, MFFormer_dec_embedding_LSTM_Informer, MFFormer_dec_embedding_LSTM_TFT, StefaLand_PatchTokens_TFT,StefaLand_PatchTokens_TFT_old,StefaLand_PatchTokens_noTFT  )
from MFFormer.utils.losses.NSE import MaskedNSELoss, MaskedNSE_hydroDL
from MFFormer.utils.losses.MSE import MaskedMSELoss
from MFFormer.utils.losses.SIGMA import SigmaLoss
from MFFormer.config.config_basic import save_as_bash_script
from MFFormer.config.config_basic import get_config, get_dataset_config, update_configs

#adding for to speed this stuff up 
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

class Exp_Basic(object):
    def __init__(self, config):
        self.config = config

        config_dataset = get_dataset_config(self.config.data[0])
        self.config, config_dataset = update_configs(self.config, config_dataset)

        self.model_dict = {
            'MFFormer': MFFormer,
            'LSTM': LSTM,
            'LSTM_mask': LSTM_mask,
            'MFFormer_fine_tune': MFFormer_fine_tune,
            'MFFormer_LSTM': MFFormer_LSTM,
            'MFFormer_dec_LSTM':MFFormer_dec_LSTM,
            'MFFormer_regression': MFFormer_regression,
            'MFFormer_dec_embedding_LSTM': MFFormer_dec_embedding_LSTM,
            'MFFormer_dec_LSTM_time_series': MFFormer_dec_LSTM_time_series,
            'MFFormer_dec_LSTM_Tape': MFFormer_dec_embedding_LSTM_Tape,
            'MFFormer_dec_LSTM_Informer':  MFFormer_dec_embedding_LSTM_Informer,
            'MFFormer_dec_LSTM_TFT':  MFFormer_dec_embedding_LSTM_TFT,
            'StefaLand' : StefaLand_PatchTokens_TFT,
            'StefaLandOld' : StefaLand_PatchTokens_TFT_old,
            'StefaLandPatchOnly' : StefaLand_PatchTokens_noTFT,
        }
        self.criterion_dict = {
            'MaskedMSE': MaskedMSELoss,
            'MaskedNSE': MaskedNSELoss,
            'MaskedNSE_hydroDL': MaskedNSE_hydroDL,
            'SigmaLoss': SigmaLoss,
        }
        self.optimizer_dict = {
            'AdamW': optim.AdamW,
            'SGD': optim.SGD,
            'Adadelta': optim.Adadelta,
        }

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        self.model_optim = self._select_optimizer()
        self.criterion = self._select_criterion()
        self.criterion_category = nn.CrossEntropyLoss()

        self.start_epoch = 0
        self.epoch_train_loss_list = []
        self.epoch_val_loss_list = []

        self.saved_folder_name = config.saved_folder_name
        self.saved_dir = config.saved_dir
        self.checkpoints_dir = config.checkpoints_dir
        self.results_dir = config.results_dir
        self.index_dir = config.index_dir

        # save config as .sh file
        save_as_bash_script(config, os.path.join(self.saved_dir, 'configs.sh'))

    def _build_model(self):

        # # only for some experiments
        # from copy import deepcopy
        # config = deepcopy(self.config)
        # static_variables = config.static_variables
        # # remove "_upper" and "_lower" from static_variables
        # static_variables = [var.replace('_upper', '').replace('_lower', '') for var in static_variables]
        # config.static_variables = static_variables
        # model = self.model_dict[self.config.model].Model(config).float()

        model = self.model_dict[self.config.model].Model(self.config).float()

        if self.config.use_multi_gpu and self.config.use_gpu:
            device_ids = np.arange(len(self.config.device_ids)).tolist()
            model = nn.DataParallel(model, device_ids=device_ids)

        return model

    def _select_optimizer(self):
        model_optim = self.optimizer_dict[self.config.optimizer](self.model.parameters(), lr=self.config.learning_rate,
                                                               weight_decay=self.config.weight_decay)
        return model_optim

    def _select_criterion(self):
        criterion = self.criterion_dict[self.config.criterion]()
        return criterion

    def _acquire_device(self):
        if self.config.use_gpu:
            if not self.config.use_multi_gpu:
                torch.cuda.set_device(self.config.gpu)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = self.config.devices

            device = torch.device(f'cuda:{self.config.gpu}')
            print(f'Use GPU: cuda:{self.config.gpu}')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def val(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

    def plot_loss_curve(self, epoch_train_loss, epoch_val_loss, saved_file):
        plt.figure()
        plt.plot(epoch_train_loss, label='train loss')
        if epoch_val_loss is not None:
            plt.plot(epoch_val_loss, label='val loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(saved_file)
        plt.close()

    def plot_loss_curve_from_csv(self, loss_csv_file, loss_pic_file="./loss_curve.png"):
        """
        Plot the loss curve from the csv file
        Args:
            loss_csv_file: the csv file path
            loss_pic_file: the output picture file path
        """
        if not os.path.exists(loss_csv_file):
            print(f"Error: loss csv file {loss_csv_file} does not exist")
            return

        try:
            # Read the loss data
            df = pd.read_csv(loss_csv_file)

            # Find the start of the last sequence where epoch starts with 1
            last_start_indices = df[df['epoch'] == 1].index
            if len(last_start_indices) > 0:
                last_start = last_start_indices[-1]
                df_last = df.iloc[last_start:].copy()  # Use .copy() to avoid SettingWithCopyWarning
            else:
                df_last = df.copy()  # If no epoch 1 is found, use the entire dataframe

            # Convert pandas Series to numpy arrays before plotting
            epochs = df_last['epoch'].values  # Use .values instead of to_numpy() for compatibility
            train_losses = df_last['train_loss'].values
            
            plt.figure()
            plt.plot(epochs, train_losses, label='Train Loss')
            
            # Check if validation loss column exists and has data
            if 'val_loss' in df_last.columns and not df_last['val_loss'].isnull().all():
                val_losses = df_last['val_loss'].values
                plt.plot(epochs, val_losses, label='Validation Loss')
            
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Loss Curve')
            plt.legend()
            plt.grid(True)
            plt.savefig(loss_pic_file)
            plt.close()
        except Exception as e:
            print(f"Error plotting loss curve: {str(e)}")
            # Create a simple error plot with text
            plt.figure()
            plt.text(0.5, 0.5, f"Error plotting loss curve:\n{str(e)}", 
                    horizontalalignment='center', verticalalignment='center')
            plt.savefig(loss_pic_file)
            plt.close()

    def write_loss_to_csv(self, epoch, train_loss, val_loss=None, loss_csv_file='loss_data.csv'):

        df_current = pd.DataFrame({'epoch': [epoch], 'train_loss': [train_loss], 'val_loss': [val_loss]})

        if not os.path.isfile(loss_csv_file):
            # Create a new file if it doesn't exist
            if epoch != 1:
                df_nan = pd.DataFrame({
                    'epoch': range(1, epoch),  # Create epochs from 1 to current epoch-1
                    'train_loss': [float('nan')] * (epoch - 1),  # Fill train_loss with NaN
                    'val_loss': [float('nan')] * (epoch - 1)  # Fill val_loss with NaN
                })
                df_nan.to_csv(loss_csv_file, index=False)
                df_current.to_csv(loss_csv_file, mode='a', header=False, index=False)
            else:
                df_current.to_csv(loss_csv_file, index=False)
        else:
            # Append to the file if it exists
            df_current.to_csv(loss_csv_file, mode='a', header=False, index=False)

    def load_model_resume(self):
        if self.config.resume_from_checkpoint is None:
            return

        if os.path.isdir(self.config.resume_from_checkpoint):
            self.checkpoints_dir = self.config.resume_from_checkpoint

        ckpts = [f for f in os.listdir(self.checkpoints_dir) if re.search(r'^.+_[\d]*.pt$', f)]
        if not ckpts:
            return
        ckpts.sort()
        checkpoint_file = os.path.join(self.checkpoints_dir, ckpts[-1])

        ckpt = torch.load(checkpoint_file, map_location=self.device)
        src = ckpt['model_state_dict']

        is_src_dp = any(k.startswith('module.') for k in src.keys())
        is_dst_dp = isinstance(self.model, nn.DataParallel)

        if is_src_dp and not is_dst_dp:
            src = {k.replace('module.', '', 1): v for k, v in src.items()}
        elif not is_src_dp and is_dst_dp:
            src = {'module.' + k: v for k, v in src.items()}

        dst = self.model.state_dict()
        filtered = {}
        skipped = []

        for k, v in src.items():
            if k in dst and v.shape == dst[k].shape:
                filtered[k] = v
            else:
                skipped.append(k)

        # load what matches; allow missing/unexpected without crashing
        info = self.model.load_state_dict(filtered, strict=False)

        # optional: quick visibility in logs
        print(f"[resume] loaded: {len(filtered)} keys, "
            f"missing: {len(info.missing_keys)}, unexpected: {len(info.unexpected_keys)}, "
            f"skipped_shape_mismatch: {len(skipped)}")

        if self.model_optim is not None and 'optim_state_dict' in ckpt:
            try:
                self.model_optim.load_state_dict(ckpt['optim_state_dict'])
            except Exception as e:
                print(f"[resume] optimizer state skipped: {e}")

        self.start_epoch = ckpt.get('epoch', 0)
        self.epoch_train_loss_list = ckpt.get('epoch_train_loss_list', [])
        self.epoch_val_loss_list = ckpt.get('epoch_val_loss_list', [])

    @staticmethod
    def save_model(checkpoint_dir, epoch, model, optim, epoch_train_loss_list, epoch_val_loss_list=None):

        checkpoint_file = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        last_checkpoint_file = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch - 1}.pt')

        # model = model.module if isinstance(model, DDP) else model
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optim.state_dict(),
            'epoch_train_loss_list': epoch_train_loss_list,
            'epoch_val_loss_list': epoch_val_loss_list,
        }

        # backup last epoch
        if epoch > 0:
            try:
                os.rename(last_checkpoint_file, last_checkpoint_file + '.bak')
            except:
                pass

        # save ckpt
        torch.save(checkpoint_dict, checkpoint_file)

        # delete last two epoch
        if epoch > 1:
            try:
                last_last_checkpoint_file = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch - 2}.pt.bak')
                os.remove(last_last_checkpoint_file)
            except:
                pass

    def config_format(self, config):

        basic_config = f"--task_name {config.task_name} --model {config.model} --do_eval {config.do_eval} --output_dir {config.output_dir} --resume_from_checkpoint {config.resume_from_checkpoint} --des {config.des}"
        data_config = f"--data {config.data}"
        train_config = f"--seq_len {config.seq_len} --label_len {config.label_len} --pred_len {config.pred_len} --epochs {config.epochs} --batch_size {config.batch_size} --patience {config.patience} --learning_rate {config.learning_rate} --criterion {config.criterion} --optimizer {config.optimizer} --lradj {config.lradj} --use_amp {config.use_amp} --clip_grad {config.clip_grad} --use_target_stds {config.use_target_stds}"
        model_config = f"--enc_in {config.enc_in} --dec_in {config.dec_in} --c_out {config.c_out} --d_model {config.d_model} --num_heads {config.num_heads} --num_enc_layers {config.num_enc_layers} --num_dec_layers {config.num_dec_layers} --d_ffd {config.d_ffd} --dropout {config.dropout} --activation {config.activation} --output_attention {config.output_attention} --initial_forget_bias {config.initial_forget_bias}"
        gpu_config = f"--use_gpu {config.use_gpu} --gpu {config.gpu} --use_multi_gpu {config.use_multi_gpu} --devices {config.devices}"
        pretrained_config = f"--mask_ratio_time_series {config.mask_ratio_time_series} --mask_ratio_static {config.mask_ratio_static} --min_window_size {config.min_window_size} --max_window_size {config.max_window_size} --time_series_variables {config.time_series_variables} --static_variables {config.static_variables}"

        config_text = f"{basic_config}\n{data_config}\n{train_config}\n{model_config}\n{pretrained_config}\n{gpu_config}\n"

        return config_text

    def del_train_val_test_index(self):

        pattern = os.path.join(self.index_dir, 'sampling_index_*.npz')
        # Find all files matching the pattern
        files_to_delete = glob.glob(pattern)
        # Delete each file
        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except OSError as e:
                print(f"Error deleting {file_path}: {e}")