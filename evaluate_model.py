#!/usr/bin/env python
"""
Script to evaluate pre-trained CTCP models with both standard and incremental learning approaches
"""

import argparse
import logging
import os
import sys
import torch
import numpy as np
from model.CTCP import CTCP
from utils.data_processing import get_data
from utils.my_utils import load_model, Metric
from train.train import eval_model, eval_model_online_incremental
import time
from collections import defaultdict
from utils.deepcopy_safe import safe_deepcopy_ctcp


def setup_logging(log_path):
    """设置日志记录"""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger('CTCP-Evaluation')
    logger.setLevel(logging.INFO)

    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    # 文件输出
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def create_model(param, device):
    """创建CTCP模型实例"""
    model = CTCP(device=device,
                 node_dim=param['node_dim'],
                 embedding_module_type=param['embedding_module'],
                 state_updater_type="gru",
                 predictor=param['predictor'],
                 time_enc_dim=param['time_dim'],
                 single=param['single'],
                 ntypes={'user', 'cas'},
                 dropout=param['dropout'],
                 n_nodes=param['node_num'],
                 max_time=param['max_time'],
                 max_global_time=param['max_global_time'],
                 use_dynamic=True,
                 use_static=param.get('use_static', False),
                 use_temporal=param.get('use_temporal', False),
                 use_structural=param.get('use_structural', False),
                 merge_prob=param.get('merge_prob', 0.5))
    return model


def main():
    parser = argparse.ArgumentParser('CTCP 预训练模型评估')
    parser.add_argument('--dataset', type=str, default='twitter',
                        choices=['aps', 'twitter', 'weibo'],
                        help='要使用的数据集')
    parser.add_argument('--model_path', type=str, required=True,
                        help='预训练模型路径 (不包含 _1.pth 等后缀)')
    parser.add_argument('--run', type=int, default=1,
                        help='要加载的模型运行编号')
    parser.add_argument('--bs', type=int, default=64,
                        help='评估的批量大小')
    parser.add_argument('--gpu', type=int, default=0,
                        help='要使用的GPU ID')
    parser.add_argument('--update_threshold', type=float, default=1.15,
                        help='增量更新的误差阈值 (日志域绝对误差)')
    parser.add_argument('--update_interval', type=int, default=256,
                        help='增量更新的频率 (样本数)')
    parser.add_argument('--incremental_lr', type=float, default=3e-4,
                        help='增量更新的学习率')
    parser.add_argument('--skip_standard', action='store_true',
                        help='跳过标准评估')
    parser.add_argument('--skip_incremental', action='store_true',
                        help='跳过增量学习评估')
    parser.add_argument('--save_incremental', action='store_true',
                        help='保存增量学习后的模型')
    parser.add_argument('--use_static', action='store_true',
                        help='使用静态嵌入')
    parser.add_argument('--use_temporal', action='store_true',
                        help='使用时间嵌入')
    parser.add_argument('--use_structural', action='store_true',
                        help='使用结构嵌入')
    parser.add_argument('--single', action='store_true',
                        help='是否对用户和级联使用相同的状态更新器和消息函数')
    parser.add_argument('--node_dim', type=int, default=64,
                        help='节点嵌入维度')
    parser.add_argument('--time_dim', type=int, default=16,
                        help='时间嵌入维度')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout概率')
    parser.add_argument('--merge_prob', type=float, default=0.5,
                        help='Merge预测器的合并概率')
    parser.add_argument('--lambda_param', type=float, dest='lambda', default=0.1,
                        help='Lambda正则化参数')
    parser.add_argument('--predictor', type=str, default="linear",
                        choices=["linear", "merge"],
                        help='预测器类型')
    parser.add_argument('--embedding_module', type=str, default="aggregate",
                        choices=["identity", "aggregate"],
                        help='嵌入模块类型')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='结果保存目录')
    parser.add_argument('--log_dir', type=str, default='log',
                        help='日志保存目录')
    parser.add_argument('--exact_match_size', action='store_true',
                        help='尝试精确匹配模型的大小（预处理数据的同样子集）')

    args = parser.parse_args()

    # 创建目录
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("fig", exist_ok=True)

    # 设置日志
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(args.log_dir, f"evaluation_{args.dataset}_{timestamp}.log")
    logger = setup_logging(log_path)

    # 设置设备
    device_name = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    logger.info(f"Using device: {device}")

    # 数据集参数
    if args.dataset == 'twitter':
        param = {'lr': 1e-4, 'bs': args.bs, 'observe_time': 2, 'predict_time': 500,
                 'time_unit': 86400, 'train_time': 8, 'val_time': 12,
                 'test_time': 16, 'epoch': 50, 'run': args.run}
    elif args.dataset == 'weibo':
        param = {'lr': 1e-4, 'bs': args.bs, 'observe_time': 1, 'predict_time': 500,
                 'time_unit': 3600, 'train_time': 10, 'val_time': 13,
                 'test_time': 16, 'epoch': 35, 'run': args.run}
    elif args.dataset == 'aps':
        param = {'lr': 1e-4, 'bs': args.bs, 'observe_time': 5, 'predict_time': 500,
                 'time_unit': 365, 'train_time': 84, 'val_time': 94, 'test_time': 104,
                 'epoch': 50, 'run': args.run}
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # 添加增量学习参数
    param.update({
        'node_dim': args.node_dim,
        'time_dim': args.time_dim,
        'single': args.single,
        'dropout': args.dropout,
        'predictor': args.predictor,
        'embedding_module': args.embedding_module,
        'use_static': args.use_static,
        'use_temporal': args.use_temporal,
        'use_structural': args.use_structural,
        'merge_prob': args.merge_prob,
        'lambda': getattr(args, 'lambda', 0.1),
        'use_incremental': not args.skip_incremental,
        'update_threshold': args.update_threshold,
        'update_interval': args.update_interval,
        'incremental_lr': args.incremental_lr,
        'save_incremental': args.save_incremental,
        'model_path': args.model_path
    })

    # 日志记录模型配置
    logger.info(f"Model configuration:")
    logger.info(f"  Node dimension: {param['node_dim']}")
    logger.info(f"  Embedding module: {param['embedding_module']}")
    logger.info(f"  Predictor: {param['predictor']}")
    logger.info(f"  Dropout: {param['dropout']}")
    logger.info(f"  Use static: {param['use_static']}")
    logger.info(f"  Use temporal: {param.get('use_temporal', False)}")
    logger.info(f"  Use structural: {param.get('use_structural', False)}")
    logger.info(f"  Merge probability: {param.get('merge_prob', 0.5)}")
    logger.info(f"  Lambda: {param.get('lambda', 0.1)}")

    # 加载数据集
    logger.info(f"Loading dataset: {args.dataset}")
    dataset = get_data(args.dataset, param['observe_time'], param['predict_time'],
                       param['train_time'], param['val_time'], param['test_time'],
                       param['time_unit'], logger, param)

    # 创建指标记录器
    result_path = os.path.join(args.result_dir, f"{args.dataset}_result_{timestamp}.pkl")
    fig_path = os.path.join("fig", f"{args.dataset}_fig_{timestamp}")
    metric = Metric(result_path, logger, fig_path)

    # 创建模型
    model = create_model(param, device)

    # 加载预训练模型
    try:
        load_model(model, args.model_path, args.run)
        logger.info(f"Successfully loaded model from {args.model_path}_{args.run}.pth")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        sys.exit(1)

    # 损失函数
    loss_criterion = torch.nn.MSELoss()

    # 评估结果
    standard_result = defaultdict(float)
    incremental_result = defaultdict(float)

    # # 进行标准评估
    # if not args.skip_standard:
    #     logger.info("Starting standard evaluation...")
    #     try:
    #         standard_metric = eval_model(model, dataset, device, param, metric, loss_criterion, move_final=True)
    #
    #         logger.info("\nStandard Evaluation Results:")
    #         for dtype in ['train', 'val', 'test']:
    #             logger.info(f"- {dtype.upper()} dataset:")
    #             logger.info(f"  MSLE={standard_metric[dtype]['msle']:.4f}, "
    #                        f"MALE={standard_metric[dtype]['male']:.4f}, "
    #                        f"MAPE={standard_metric[dtype]['mape']:.4f}, "
    #                        f"PCC={standard_metric[dtype]['pcc']:.4f}")
    #
    #         # 保存测试结果
    #         standard_result['msle'] = standard_metric['test']['msle']
    #         standard_result['mape'] = standard_metric['test']['mape']
    #         standard_result['male'] = standard_metric['test']['male']
    #         standard_result['pcc'] = standard_metric['test']['pcc']
    #
    #     except Exception as e:
    #         logger.error(f"Error during standard evaluation: {e}")
    #         import traceback
    #         logger.error(traceback.format_exc())

    # 进行增量学习评估
    if not args.skip_incremental:
        logger.info("\nStarting incremental learning evaluation...")
        try:
            # 创建一个新的模型副本用于增量学习
            # incremental_model = safe_deepcopy_ctcp(model)
            # incremental_model.to(device)
            model.to(device)
            # 运行增量学习评估
            standard_metric, incremental_metric = eval_model_online_incremental(
                model, dataset, device, param, metric,
                loss_criterion, move_final=True, logger=logger,
            )

            # 保存测试结果
            if incremental_metric and 'test' in incremental_metric:
                incremental_result['msle'] = incremental_metric['test']['msle']
                incremental_result['mape'] = incremental_metric['test']['mape']
                incremental_result['male'] = incremental_metric['test']['male']
                incremental_result['pcc'] = incremental_metric['test']['pcc']

        except Exception as e:
            logger.error(f"Error during incremental evaluation: {e}")
            import traceback
            logger.error(traceback.format_exc())

    # 保存结果
    try:
        metric.save()
        logger.info(f"Results saved to {result_path}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")

    # 打印最终结果对比
    if not args.skip_standard and not args.skip_incremental:
        logger.info("\n=== Final Comparison: Standard vs Incremental ===")
        logger.info("Test dataset metrics:")
        logger.info(
            f"Standard:    MSLE={standard_result['msle']:.4f}, MALE={standard_result['male']:.4f}, MAPE={standard_result['mape']:.4f}, PCC={standard_result['pcc']:.4f}")
        logger.info(
            f"Incremental: MSLE={incremental_result['msle']:.4f}, MALE={incremental_result['male']:.4f}, MAPE={incremental_result['mape']:.4f}, PCC={incremental_result['pcc']:.4f}")

        # 计算改进比例
        msle_improve = (standard_result['msle'] - incremental_result['msle']) / standard_result['msle'] * 100 if \
        standard_result['msle'] > 0 else 0
        male_improve = (standard_result['male'] - incremental_result['male']) / standard_result['male'] * 100 if \
        standard_result['male'] > 0 else 0
        mape_improve = (standard_result['mape'] - incremental_result['mape']) / standard_result['mape'] * 100 if \
        standard_result['mape'] > 0 else 0
        pcc_improve = (incremental_result['pcc'] - standard_result['pcc']) / standard_result['pcc'] * 100 if \
        standard_result['pcc'] > 0 else 0

        logger.info(
            f"Improvement: MSLE={msle_improve:.2f}%, MALE={male_improve:.2f}%, MAPE={mape_improve:.2f}%, PCC={pcc_improve:.2f}%")


if __name__ == "__main__":
    main() 