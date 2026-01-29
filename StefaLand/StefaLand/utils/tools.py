import math
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj == None:
        return

    if args.lradj == "cosine":
        lr = args.learning_rate / 2 * (1 + math.cos(epoch / args.epochs * math.pi))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))
        return

    elif args.lradj == "warmup_cosine":
        min_lr = args.learning_rate * 0.1
        if epoch < args.warmup_epochs+1:
            lr = min_lr + (args.learning_rate - min_lr) * epoch / args.warmup_epochs
        elif epoch >= 100:
            lr = min_lr
        else:
            lr = min_lr + (args.learning_rate - min_lr) * 0.5 * \
                 (1 + math.cos((epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs) * math.pi))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))
        return

    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == "LSTM":
        lr_adjust = {
            0: 0.001, 10: 0.0005, 25: 0.0001
        }
    elif args.lradj == "Transformer":
        lr_adjust = {
            0: 0.001, 7: 0.0005, 15: 0.0001, 25: 0.00005
        }
    # elif args.lradj == "cosine":
    #     lr_adjust = {epoch: args.learning_rate / 2 * (1 + math.cos(epoch / args.epochs * math.pi))}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure(figsize=(20, 5))
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

def get_kfold_index(num_pixel, num_kfold, seed=42):
    """
    Generate index from 0 to num_pixel-1, and divide it into num_fold sub index (train index list and test index list).
    All ids have the chance to participate in the test index, and each sub index are not duplicated.

    e.g.
    num_pixel=10
    num_kfold=3
    train_ids_list, test_ids_list = get_kfold_index(num_pixel=10, num_kfold=3, seed=1)

    output:
    train_ids_list:
        [[0, 1, 3, 5, 7, 8], [2, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 6, 9]]
    test_ids_list:
        [[2, 4, 6, 9], [0, 1, 3], [5, 7, 8]]

    """
    index_all = np.arange(num_pixel)
    csv_kfolder = KFold(n_splits=num_kfold, shuffle=True, random_state=seed)
    index_generator = csv_kfolder.split(index_all)
    index_list_train, index_list_test = [], []
    for train_ids, test_ids in index_generator:
        index_list_train.append(train_ids.tolist())
        index_list_test.append(test_ids.tolist())

    return index_list_train, index_list_test

def read_results_txt(file_path):
    metrics = {}  # Dictionary to store the metrics for each variable
    include_inference = False

    with open(file_path, 'r') as file:
        lines = file.readlines()

        # Iterate through each line in the file
        for line in lines:
            # If a new "Inference date" section is found, reset the metrics dictionary
            if "Inference" in line:
                metrics = {}
                include_inference = True

            # If the line contains metrics data (indicated by the presence of 'NSE')
            if 'NSE:' in line:
                # Split the string to isolate the variable name and the metrics
                parts = line.split('. ')
                variable_name = parts[0]  # 'prcp_daymet'
                metrics_part = parts[1]  # 'NSE: 0.447, KGE: 0.472, Corr: 0.677\n'

                # Further split the metrics part to extract individual metric values
                metrics_values = metrics_part.split(', ')
                nse_value = float(metrics_values[0].split(': ')[1])  # 0.447
                kge_value = float(metrics_values[1].split(': ')[1])  # 0.472
                corr_value = float(metrics_values[2].split(': ')[1])  # 0.677

                # Store the extracted values in a dictionary
                metrics[variable_name] = {
                    'NSE': nse_value,
                    'KGE': kge_value,
                    'Corr': corr_value
                }

    df = pd.DataFrame.from_dict(metrics, orient='index')
    assert include_inference, f"No 'Inference date' section found in {file_path}"

    return df