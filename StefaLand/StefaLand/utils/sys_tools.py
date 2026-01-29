import os
import random
import shutil
import numpy as np
import pandas as pd
from distutils.dir_util import copy_tree


def re_folder_rec(path_s):
    b = os.path.normpath(path_s).split(os.sep)  # e.g. ['da','jxl','wp']
    b = [x + "/" for x in b]  # e.g. ['da/','jxl/','wp/']
    fn_rec = [
        "".join(b[0: x[0]]) for x in list(enumerate(b, 1))
    ]  # e.g. ['da/','da/jxl/','da/jxl/wp/']
    fn_None = [os.mkdir(x) for x in fn_rec if not os.path.exists(x)]


def re_folder(path_s, del_old_path=False):
    """
    delete old folder and recreate new one
        Use "try" to avoid errors. When kfold runs, it will create the n process and sometimes it will generate the folder
    even if this folder exists.

    Parameters
        path_s: str, path to folder
        del_old_path: bool, delete old folder or not

    Returns
        None

    Syntax
        re_folder("/mnthh/sdbhh/consthh/demo_data", del_old_path=True)

        or using makedirs(path_s, exist_ok=True)
    """
    if os.path.exists(path_s):
        if del_old_path:
            try:
                shutil.rmtree(path_s)
            except:
                pass
            try:
                re_folder_rec(path_s)
            except:
                pass
        else:
            pass
    else:
        try:
            re_folder_rec(path_s)
        except:
            pass


def mkdir(path_s, del_old_path=False):
    re_folder(path_s=path_s, del_old_path=del_old_path)


def cp(src, dst, forced=True):
    """
    Copy all the files in directory A to directory B

    e.g.
    src: /data/file1.py
              /file2.py
              .empty
    dst: /mnt/sdb/

    cp("/data", "/mnt/sdb")

    output:
        /mnt/sdb/file1.py
                /file2.py
                .empty
    """
    copy_tree(src=src, dst=dst, )


def check_folder_within_dir(inp_path, target_folder, max_depth=1):
    """
    |--mnt
    |   |--data
    |   |   |--forcing
    |   |   |   |--2010
    |   |   |   |--2011
    |   |   |--const
    |   |   |   |--soil.csv
    ...

    inp_path = "/mnt"
    target_folder = "const"

    syntax:
        check_folder_in_dir(inp_path, target_foldr)
        output: False
        check_folder_in_dir(inp_path, target_foldr, max_depth=2)
        output: True
    """

    abs_path = os.path.abspath(inp_path)
    for root, dirs, files in os.walk(abs_path):
        if root[len(abs_path):].count(os.sep) < max_depth:
            if target_folder in dirs:
                return True
        else:
            break
    return False


def check_folder_on_dir(inp_path, target_folder):
    """
    inp_path = "/mnt/sdb/const"
    target_folder = "const"
    check_folder_on_dir(inp_path, target_folder)
    output: True
    """
    inp_path = inp_path.replace("\\", "/")
    inp_path = os.path.normpath(inp_path)
    path_list = inp_path.split(os.sep)
    if target_folder in path_list:
        return True
    else:
        return False


def fix_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except:
        pass


def split_path(inp_file):
    """
    inp_file = "/data/average/basin.shp"

    output,
    inp_path = "/data/average"
    basename = "basin"
    suffix = ".shp"
    """
    inp_path, file_name = os.path.split(inp_file)
    basename, suffix = os.path.splitext(file_name)
    return inp_path, basename, suffix


def print_args(args):
    print("\033[1m" + "Basic Config" + "\033[0m")
    print(f'  {"Task Name:":<20}{args.task_name:<20}')
    print(f'  {"Model:":<20}{args.model:<20}')
    print()

    print("\033[1m" + "Data Loader" + "\033[0m")
    data_name = ', '.join(args.data)
    print(f'  {"Data:":<20}{data_name:<20}')
    # print(f'  {"Checkpoints:":<20}{args.checkpoints:<20}')
    print()

    if args.task_name in ['forecast', ]:
        print("\033[1m" + "Forecasting Task" + "\033[0m")
        print(f'  {"Seq Len:":<20}{args.seq_len:<20}{"Label Len:":<20}{args.label_len:<20}')
        print(f'  {"Pred Len:":<20}{args.pred_len:<20}')  #
        print()

    print("\033[1m" + "Model Parameters" + "\033[0m")
    print(f'  {"Enc In:":<20}{args.enc_in:<20}{"Dec In:":<20}{args.dec_in:<20}')
    print(f'  {"C Out:":<20}{args.c_out:<20}{"d model:":<20}{args.d_model:<20}')
    print(f'  {"n heads:":<20}{args.num_heads:<20}{"e layers:":<20}{args.num_enc_layers:<20}')
    print(f'  {"d layers:":<20}{args.num_dec_layers:<20}{"d FF:":<20}{args.d_ffd:<20}')
    print(f'  {"Output Attention:":<20}{args.output_attention:<20}')
    print()

    print("\033[1m" + "Run Parameters" + "\033[0m")
    print(f'  {"Train Epochs:":<20}{args.epochs:<20}{"Batch Size:":<20}{args.batch_size:<20}')
    print(f'  {"Patience:":<20}{args.patience:<20}{"Learning Rate:":<20}{args.learning_rate:<20}')
    print(f'  {"Des:":<20}{args.des:<20}{"Criterion:":<20}{args.criterion:<20}')
    print(f'  {"Optimizer:":<20}{args.optimizer:<20}{"Use Amp:":<20}{args.use_amp:<20}')
    print()

    print("\033[1m" + "GPU" + "\033[0m")
    print(f'  {"Use GPU:":<20}{args.use_gpu:<20}{"GPU:":<20}{args.gpu:<20}')
    print(f'  {"Use Multi GPU:":<20}{args.use_multi_gpu:<20}{"Devices:":<20}{args.devices:<20}')
    print()

    print("\033[1m" + "De-stationary Projector Params" + "\033[0m")
    print()
