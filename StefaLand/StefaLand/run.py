
from MFFormer.config.config_basic import get_config
from MFFormer.exp.exp_pretrain import ExpPretrain
from MFFormer.exp.exp_forecasting import ExpForecast
from MFFormer.utils.sys_tools import fix_seed
import torch
import time

if __name__ == '__main__':
    start_time = time.time()
    configs = get_config() # configurations for the framework

    fix_seed(seed=configs.seed)

    if configs.task_name == 'pretrain':
        trainer = ExpPretrain(configs)
    elif configs.task_name in ['regression', 'forecast', 'fine_tune']:
        trainer = ExpForecast(configs)
    elif configs.task_name in ['inference']:
        trainer = ExpPretrain(configs)
        trainer.inference()
        exit()
    elif configs.task_name in ['MCD_N_pretrain', 'MCD_N_inference']:
        # Monte Carlo Dropout with input noise
        trainer = ExpPretrain(configs)
        if configs.task_name == 'MCD_N_pretrain':
            trainer.train()
        trainer.MCD_N_inference()
        trainer.del_train_val_test_index()
        exit()

    print("Total time for processing configs: {:.2f}s".format(time.time() - start_time))
    start_time = time.time()
    print('>>>>>>>start training >>>>>>>>>>>>>>>>>>>>>>>>>>')
    if not configs.do_test:
        trainer.train()
    print("Total time for training: {:.2f}s".format(time.time() - start_time))
    start_time = time.time()
    print('>>>>>>>start testing >>>>>>>>>>>>>>>>>>>>>>>>>>')
    trainer.test()
    print("Total time for testing: {:.2f}s".format(time.time() - start_time))

    if configs.task_name in ['pretrain', 'inference']:
        print('>>>>>>>start inference >>>>>>>>>>>>>>>>>>>>>>>>>>')
        trainer.inference()

    trainer.del_train_val_test_index()
    torch.cuda.empty_cache()
    exit()