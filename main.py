import logging
import sys
import random
import argparse
import torch
import numpy as np
from model.CTCP import CTCP
from utils.data_processing import get_data
from train.train import train_model
from utils.my_utils import EarlyStopMonitor, set_config, Metric
from collections import defaultdict

parser = argparse.ArgumentParser('hyper parameters of CTCP')
parser.add_argument('--dataset', type=str, help='dataset name ',
                    default='twitter', choices=['aps', 'twitter', 'weibo'])
parser.add_argument('--bs', type=int, default=128, help='batch size')
parser.add_argument('--prefix', type=str, default='test', help='prefix to name a trial')
parser.add_argument('--epoch', type=int, default=5, help='number of epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--run', type=int, default=1, help='number of runs')
parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=64, help='dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=16, help='dimensions of the time embedding')
parser.add_argument('--patience', type=int, default=15, help='patience for the early stopping strategy')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout probability')
parser.add_argument('--predictor', type=str, default="linear", choices=["linear", "merge"], help="type of predictor")
parser.add_argument('--embedding_module', type=str, default="aggregate", choices=["identity", "aggregate"],
                    help="type of embedding module")
parser.add_argument('--single', action='store_true',
                    help='whether to use different state updaters and message functions for users and cascades')
parser.add_argument('--use_static', action='store_true', help='whether use static embedding for users')
parser.add_argument('--use_dynamic', action='store_true', help='whether use dynamic embedding for users and cascades')
parser.add_argument('--use_structural', action='store_true',
                    help='whether to adopt structural learning in the cascade embedding module')
parser.add_argument('--use_temporal', action='store_true',
                    help='whether to adopt temporal learning in the cascade embedding module')
parser.add_argument('--lambda', type=float, default=0.5,
                    help='the weight to balance the static result and dynamic result')
parser.add_argument('--use_incremental', action='store_true', 
                    help='whether to use incremental learning during testing')
parser.add_argument('--batch_incremental', action='store_true',
                    help='enable batch-level incremental learning (sets update_interval = bs)')
parser.add_argument('--update_interval', type=int, default=50,
                    help='frequency of model updates in incremental learning (number of samples)')
parser.add_argument('', type=float, default=1.6,
                    help='error threshold for triggering model update in incremental learning (log domain absolute error)')
parser.add_argument('--incremental_lr', type=float, default=1e-5, 
                    help='learning rate for incremental learning updates')
parser.add_argument('--save_incremental', action='store_true',
                    help='whether to save the incremental model after updates')
try:
    args = parser.parse_args()
    # 如果启用了batch级增量学习，自动设置update_interval=bs
    if args.batch_incremental:
        args.use_incremental = True  # 自动启用增量学习
        args.update_interval = args.bs  # 设置update_interval等于batch_size
except:
    parser.print_help()
    sys.exit(0)
param = set_config(args)
logging.getLogger('matplotlib.font_manager').disabled = True
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(f"{param['log_path']}", mode='w')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
dataset = get_data(dataset=param['dataset'], observe_time=param['observe_time'],
                   predict_time=param['predict_time'],
                   train_time=param['train_time'], val_time=param['val_time'],
                   test_time=param['test_time'], time_unit=param['time_unit'],
                   log=logger, param=param)
logger.info(param)
result = defaultdict(lambda: 0)
incremental_result = defaultdict(lambda: 0)
torch.set_num_threads(5)
for num in range(param['run']):
    logger.info(f'begin runs:{num}')
    my_seed = num
    random.seed(my_seed)
    np.random.seed(my_seed)
    torch.manual_seed(my_seed)
    device_string = 'cuda:{}'.format(param['gpu']) if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_string)
    model = CTCP(device=device, node_dim=param['node_dim'], embedding_module_type=param['embedding_module'],
                 state_updater_type='gru', predictor=param['predictor'], time_enc_dim=param['time_dim'],
                 single=param['single'], ntypes={'user', 'cas'}, dropout=param['dropout'],
                 n_nodes=param['node_num'], max_time=param['max_time'], use_static=param['use_static'],
                 merge_prob=param['lambda'], max_global_time=param['max_global_time'], use_dynamic=param['use_dynamic'],
                 use_temporal=param['use_temporal'], use_structural=param['use_structural'])
    metric = Metric(path=f"{param['result_path']}_{num}.pkl", logger=logger, fig_path=f"fig/{param['prefix']}")
    early_stopper = EarlyStopMonitor(max_round=param['patience'], higher_better=False, tolerance=1e-3,
                                     save_path=param['model_path'],
                                     logger=logger, model=model, run=num)
    train_model(num, dataset, model.to(device), logger, early_stopper, device, param, metric, result, incremental_result)

logger.info(
    f"Final: msle:{result['msle']:.4f} male:{result['male']:.4f} "
    f"mape:{result['mape']:.4f} pcc:{result['pcc']:.4f}")

if param['use_incremental']:
    logger.info(
        f"Incremental Final: msle:{incremental_result['msle']:.4f} male:{incremental_result['male']:.4f} "
        f"mape:{incremental_result['mape']:.4f} pcc:{incremental_result['pcc']:.4f}")
