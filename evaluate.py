import logging
import sys
import random
import argparse
import torch
import numpy as np
from model.CTCP import CTCP
from utils.data_processing import get_data
from train.train import train_model, eval_model # We'll likely need the eval function from here
from utils.my_utils import set_config, Metric # Metric might be useful
from collections import defaultdict
import torch.nn as nn # For loss function

# Argument Parsing (similar to main.py but add path for pretrained model)
parser = argparse.ArgumentParser('Evaluation Script for CTCP')
parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., twitter, weibo, aps)',
                    choices=['aps', 'twitter', 'weibo'])
parser.add_argument('--pretrained_model', type=str, required=True, help='Path to the pre-trained model .pth file')
parser.add_argument('--bs', type=int, default=50, help='Batch size for evaluation')
parser.add_argument('--gpu', type=int, default=0, help='GPU index to use')
parser.add_argument('--node_dim', type=int, default=64, help='Node embedding dimension (must match pretrained model)')
parser.add_argument('--time_dim', type=int, default=16, help='Time embedding dimension (must match pretrained model)')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout (usually set to 0 for eval, but keep consistent)')
parser.add_argument('--predictor', type=str, default="linear", choices=["linear", "merge"], help="Predictor type (must match pretrained model)")
parser.add_argument('--embedding_module', type=str, default="aggregate", choices=["identity", "aggregate"],
                    help="Embedding module type (must match pretrained model)")
parser.add_argument('--single', action='store_true', help='Single architecture flag (must match pretrained model)')
parser.add_argument('--use_static', action='store_true', help='Use static embedding (must match pretrained model)')
parser.add_argument('--use_dynamic', action='store_true', help='Use dynamic embedding (must match pretrained model)')
parser.add_argument('--use_structural', action='store_true', help='Use structural learning (must match pretrained model)')
parser.add_argument('--use_temporal', action='store_true', help='Use temporal learning (must match pretrained model)')
parser.add_argument('--lambda', type=float, default=0.5, help='Lambda weight (must match pretrained model)')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

# Use a simplified config, mainly for data loading and model structure
param = {
    'dataset': args.dataset,
    'bs': args.bs,
    'gpu': args.gpu,
    'node_dim': args.node_dim,
    'time_dim': args.time_dim,
    'dropout': args.dropout,
    'predictor': args.predictor,
    'embedding_module': args.embedding_module,
    'single': args.single,
    'use_static': args.use_static,
    'use_dynamic': args.use_dynamic,
    'use_structural': args.use_structural,
    'use_temporal': args.use_temporal,
    'lambda': getattr(args, 'lambda'), # Access lambda correctly
    # Add default data params (these might need adjustment based on dataset specifics)
    'observe_time': 2,
    'predict_time': 86400,
    'train_time': 8,
    'val_time': 12,
    'test_time': 16,
    'time_unit': 86400
}

# Logging Setup
logging.getLogger('matplotlib.font_manager').disabled = True
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Evaluation')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.info(f"Evaluation parameters: {param}")
logger.info(f"Pretrained model path: {args.pretrained_model}")


# Set Seed and Device
my_seed = 0 # Fixed seed for evaluation consistency
random.seed(my_seed)
np.random.seed(my_seed)
torch.manual_seed(my_seed)
device_string = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
device = torch.device(device_string)
logger.info(f"Using device: {device}")

# --- Data Loading, Model Init, Load Pretrained, Eval Function, and Main Eval Loop will go here ---

# 1. Load Data
try:
    dataset = get_data(dataset=param['dataset'], observe_time=param['observe_time'],
                       predict_time=param['predict_time'],
                       train_time=param['train_time'], val_time=param['val_time'],
                       test_time=param['test_time'], time_unit=param['time_unit'],
                       log=logger, param=param)
    logger.info(f"Successfully loaded dataset: {args.dataset}")
except Exception as e:
    logger.error(f"Failed to load dataset: {e}")
    sys.exit(1)

# 2. Initialize Model (ensure parameters match the pre-trained model)
model = CTCP(device=device, node_dim=param['node_dim'], embedding_module_type=param['embedding_module'],
             state_updater_type='gru', predictor=param['predictor'], time_enc_dim=param['time_dim'],
             single=param['single'], ntypes={'user', 'cas'}, dropout=param['dropout'],
             n_nodes=param['node_num'], max_time=param['max_time'], use_static=param['use_static'],
             merge_prob=param['lambda'], max_global_time=param['max_global_time'], use_dynamic=param['use_dynamic'],
             use_temporal=param['use_temporal'], use_structural=param['use_structural'])
model = model.to(device)
logger.info("Model initialized.")

# 3. Load Pretrained Weights
try:
    model.load_state_dict(torch.load(args.pretrained_model, map_location=device))
    logger.info(f"Successfully loaded pre-trained model weights from {args.pretrained_model}")
except Exception as e:
    logger.error(f"Failed to load pre-trained model from {args.pretrained_model}: {e}")
    sys.exit(1)

# 4. Setup for Evaluation
loss_criterion = nn.MSELoss()
# Note: The Metric class saves results, adjust path if needed or disable saving
metric = Metric(path=f"results/eval_{args.dataset}_{args.pretrained_model.split('/')[-1]}.pkl", # Example path
                logger=logger,
                fig_path=None) # No figures needed for just eval

logger.info("Starting evaluation on the test set...")

# 5. Run Evaluation
test_metrics = eval_model(model=model,
                          eval=dataset, # Pass the whole dataset object, eval_model selects test data internally
                          device=device,
                          param=param,
                          metric=metric,
                          loss_criteria=loss_criterion,
                          move_final=True) # Ensure final test metrics are calculated

# 6. Report Metrics
logger.info("Evaluation complete.")
logger.info("--- Test Set Performance ---")
if 'test' in test_metrics:
    final_results = test_metrics['test']
    logger.info(f"  MSLE: {final_results.get('msle', 'N/A'):.4f}")
    logger.info(f"  MALE: {final_results.get('male', 'N/A'):.4f}")
    logger.info(f"  MAPE: {final_results.get('mape', 'N/A'):.4f}")
    logger.info(f"  PCC:  {final_results.get('pcc', 'N/A'):.4f}")
else:
    logger.warning("Test metrics not found in the results.")

# Optional: Save metrics if needed (Metric class might do this automatically)
# metric.save()

logger.info("Evaluation script finished.")

# logger.info("Evaluation script setup complete. Next steps: Load data and model.") # Remove placeholder 