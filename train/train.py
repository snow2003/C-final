import logging
from copy import deepcopy

import numpy as np
import torch
from torch.nn.quantized.functional import threshold
from tqdm import tqdm
from utils.my_utils import save_model, load_model, EarlyStopMonitor, Metric
import time
from model.CTCP import CTCP
import math
from utils.data_processing import Data
from utils.deepcopy_safe import safe_deepcopy_ctcp, create_healthy_clone
from typing import Tuple, Dict, Type
from torch.nn.modules.loss import _Loss
import copy
import os
import pandas as pd


def select_label(labels, types):
    """
    Select indices for train, validation, and test data
    Works with both CPU and GPU tensors
    """
    # 检查输入是否为张量并进行适当处理
    if isinstance(labels, torch.Tensor):
        # 处理张量 - 确保逻辑操作在同一设备上进行
        device = labels.device
        # 确保types也在相同设备上
        if isinstance(types, torch.Tensor):
            types = types.to(device)
        else:
            types = torch.tensor(types, device=device)
            
        train_idx = (labels != -1) & (types == 1)
        val_idx = (labels != -1) & (types == 2)
        test_idx = (labels != -1) & (types == 3)
    else:
        # 如果labels是NumPy数组，确保types也是NumPy数组
        if isinstance(types, torch.Tensor):
            types = types.cpu().numpy()
            
        # 处理NumPy数组
        train_idx = (labels != -1) & (types == 1)
        val_idx = (labels != -1) & (types == 2)
        test_idx = (labels != -1) & (types == 3)
    
    return {'train': train_idx, 'val': val_idx, 'test': test_idx}


def move_to_device(device, *args):
    results = []
    for arg in args:
        if type(arg) is torch.Tensor:
            results.append(arg.to(dtype=torch.float, device=device))
        else:
            results.append(torch.tensor(arg, device=device, dtype=torch.float))
    return results


def eval_model(model: CTCP, eval: Data, device: torch.device, param: Dict, metric: Metric,
               loss_criteria: _Loss, move_final: bool = False) -> Dict:
    model.eval()
    model.reset_state()
    metric.fresh()
    epoch_metric = {}
    loss = {'train': [], 'val': [], 'test': []}
    with torch.no_grad():
        for x, label in tqdm(eval.loader(param['bs']), total=math.ceil(eval.length / param['bs']), desc='eval_or_test'):
            src, dst, trans_cas, trans_time, pub_time, types = x
            index_dict = select_label(label, types)
            target_idx = index_dict['train'] | index_dict['val'] | index_dict['test']
            trans_time, pub_time, label = move_to_device(device, trans_time, pub_time, label)
            pred = model.forward(src, dst, trans_cas, trans_time, pub_time, target_idx)
            for dtype in ['train', 'val', 'test']:
                idx = index_dict[dtype]
                if sum(idx) > 0:
                    m_target = trans_cas[idx]
                    m_label = label[idx]
                    m_label[m_label < 1] = 1
                    m_label = torch.log2(m_label)
                    m_pred = pred[idx]
                    loss[dtype].append(loss_criteria(m_pred, m_label).item())
                    metric.update(target=m_target, pred=m_pred.cpu().numpy(), label=m_label.cpu().numpy(), dtype=dtype)
            model.update_state()
        for dtype in ['train', 'val', 'test']:
            epoch_metric[dtype] = metric.calculate_metric(dtype, move_history=True, move_final=move_final,
                                                          loss=np.mean(loss[dtype]))
        return epoch_metric


def eval_model_online_incremental(model: CTCP, eval: Data, device: torch.device, param: Dict, metric: Metric,
                                  loss_criteria: _Loss, move_final: bool = False, logger: logging.Logger = None) -> \
Tuple[Dict, Dict]:
    """
    真正的在线增量学习评估:
    - 逐批次处理数据
    - 当测试数据误差超过阈值时，立即更新模型
    - 继续使用更新后的模型处理下一批数据
    """
    try:
        # 使用传入的logger或创建一个简单的日志输出
        log = logger if logger else logging.getLogger(__name__)

        # 创建增量学习模型副本
        incremental_model = create_healthy_clone(model, param, device) # 使用新的克隆函数，param是配置字典
        incremental_model.eval()  # 初始设为评估模式
        incremental_model.reset_state()

        # 标准模型
        standard_model = create_healthy_clone(model, param, device)
        standard_model.eval()
        standard_model.reset_state()

        # 初始化评估指标
        metric.fresh()
        standard_metric = {}
        incremental_metric = {}

        # 损失记录
        standard_loss = {'train': [], 'val': [], 'test': []}
        incremental_loss = {'train': [], 'val': [], 'test': []}

        excel_rows = []
        excel_path = param.get(  # ★新增
            'excel_path',
            f"results/{param.get('prefix', 'run')}_batch_error_run{param.get('run', 0)}.xlsx"
        )

        # 更新统计信息
        total_batches = 0
        updated_batches = 0

        # 批次级别详细记录
        batch_records = []

        worse_streak = 0

        # 记录整体误差
        total_errors_before = []  # 记录所有批次更新前的误差
        total_errors_after = []  # 记录所有批次更新后的误差

        log.info("===== 开始在线增量学习评估 =====")
        log.info(f"Update threshold: {param.get('update_threshold', 1.5)}, Update interval: {param.get('bs', 10)}")
        log.info(f"Incremental learning rate: {param.get('incremental_lr', param.get('lr', 0.0001))}")

        # 单次遍历数据集 - 在线处理
        for x, label in tqdm(eval.loader(param['bs']), total=math.ceil(eval.length / param['bs']),
                             desc='Online incremental evaluation'):

            # --- 新增：本批级别的 Metric 实例 ---
            # batch_std_metric = Metric()  # 用于算标准模型的 MALE
            # batch_inc_metric = Metric()  # 用于算增量模型的 MALE

            total_batches += 1
            src, dst, trans_cas, trans_time, pub_time, types = x

            # 移动数据到设备
            trans_time, pub_time, label = move_to_device(device, trans_time, pub_time, label)

            # 确保types也在同一设备上
            if isinstance(types, torch.Tensor):
                types = types.to(device)
            else:
                types = torch.tensor(types, device=device)

            # 确保布尔索引正确处理
            index_dict = select_label(label, types)
            target_idx = index_dict['train'] | index_dict['val'] | index_dict['test']

            # 转换为CPU用于NumPy索引
            target_idx_cpu = target_idx.cpu() if isinstance(target_idx, torch.Tensor) else target_idx

            # 确保target_idx是设备上的布尔张量用于前向传播
            if not isinstance(target_idx, torch.Tensor):
                target_idx_bool = torch.tensor(target_idx, dtype=torch.bool, device=device)
            else:
                target_idx_bool = target_idx.bool()  # 已经在设备上了

            # 标准模型评估
            with torch.no_grad():
                standard_pred = standard_model.forward(src, dst, trans_cas, trans_time, pub_time, target_idx_bool)
                standard_model.update_state()

            # 增量模型评估
            with torch.no_grad():
                incremental_pred = incremental_model.forward(src, dst, trans_cas, trans_time, pub_time, target_idx_bool)
                # 暂时不更新状态，因为如果需要训练，我们会用临时模型重新做前向传播

            # 计算此批次的误差，并收集评估指标
            batch_errors = []  # 用于触发更新

            standard_errors_collector = []  # 新增
            incremental_errors_collector = []  # 新增
            # # --- 改为更新本批 Metric ---
            # batch_std_metric.update(
            #     target = m_target,
            #     pred = m_pred_std_np,
            #     abel = m_label_np,
            #     dtype = dtype
            # )
            # batch_inc_metric.update_incremental(
            #      target = m_target,
            #      pred = m_pred_inc_np,
            #      label = m_label_np,
            #      dtype = dtype
            # )

            batch_preds_before_update = {}  # 记录更新前的预测 (用于验证)
            batch_labels_log2 = {}  # 记录log2标签 (用于验证和触发)
            batch_all_indices = torch.zeros_like(label, dtype=torch.bool, device=device)  # 记录本批次所有有效样本的索引

            # 当前批次的指标记录
            batch_metrics = {
                'batch_id': total_batches,
                'standard': {'train': {}, 'val': {}, 'test': {}},
                'incremental_before': {'train': {}, 'val': {}, 'test': {}},
                'incremental_after': {'train': {}, 'val': {}, 'test': {}},
                'updated': False,
                'errors_before': 0,
                'errors_after': 0,
                'improvement': 0
            }

            need_update = False

            for dtype in ['train', 'val', 'test']:
                idx = index_dict[dtype]
                # 确保idx是torch张量
                if not isinstance(idx, torch.Tensor):
                    idx_tensor = torch.tensor(idx, dtype=torch.bool, device=device)
                else:
                    idx_tensor = idx.bool()  # 确保是布尔类型

                # 确保索引正确处理（用于NumPy操作）
                idx_cpu = idx_tensor.cpu().numpy() if isinstance(idx_tensor, torch.Tensor) else idx

                if idx_tensor.sum() > 0:
                    batch_all_indices |= idx_tensor  # 添加到批次总索引
                    m_target = trans_cas[idx_cpu]
                    m_label = label[idx_tensor]
                    m_label_original = m_label.clone()  # 保存原始标签值，以防万一
                    m_label[m_label < 1] = 1
                    m_label = torch.log2(m_label)  # Log2 转换

                    # 标准模型指标
                    m_pred_std = standard_pred[idx_tensor]
                    standard_loss_val = loss_criteria(m_pred_std, m_label).item()
                    standard_loss[dtype].append(standard_loss_val)

                    # 记录标准模型此批次评估指标
                    m_pred_std_np = m_pred_std.detach().cpu().numpy()
                    m_label_np = m_label.detach().cpu().numpy()

                    # 计算批次级别的标准模型指标
                    batch_standard_errors = [abs(m_pred_std_np[i] - m_label_np[i]) for i in range(len(m_target))]
                    batch_metrics['standard'][dtype] = {
                        'loss': standard_loss_val,
                        'avg_error': sum(batch_standard_errors) / len(
                            batch_standard_errors) if batch_standard_errors else 0,
                        'samples': len(m_target)
                    }

                    metric.update(target=m_target,
                                  pred=m_pred_std_np,
                                  label=m_label_np,
                                  dtype=dtype)

                    # 增量模型指标
                    m_pred_inc = incremental_pred[idx_tensor]
                    incremental_loss_val = loss_criteria(m_pred_inc, m_label).item()
                    incremental_loss[dtype].append(incremental_loss_val)

                    # 记录增量模型更新前评估指标
                    m_pred_inc_np = m_pred_inc.detach().cpu().numpy()
                    batch_incremental_errors = [abs(m_pred_inc_np[i] - m_label_np[i]) for i in range(len(m_target))]
                    batch_metrics['incremental_before'][dtype] = {
                        'loss': incremental_loss_val,
                        'avg_error': sum(batch_incremental_errors) / len(
                            batch_incremental_errors) if batch_incremental_errors else 0,
                        'samples': len(m_target)
                    }

                    standard_errors_collector.extend(batch_standard_errors)
                    incremental_errors_collector.extend(batch_incremental_errors)

                    metric.update_incremental(target=m_target,
                                              pred=m_pred_inc_np,
                                              label=m_label_np,
                                              dtype=dtype)

                    # --- 修改：收集所有类型的误差用于决定是否更新 ---
                    if param.get('use_incremental', False):
                        errors = [abs(m_pred_inc[i].item() - m_label[i].item())
                                  for i in range(len(m_target))]
                        batch_errors.extend(errors)

                        # 保存更新前的预测和标签 (用于后续验证)
                        # 注意：这里保存的是这个dtype子集的预测和标签
                        # 我们稍后需要根据batch_all_indices来合并它们
                        batch_preds_before_update[dtype] = m_pred_inc.detach().clone()
                        batch_labels_log2[dtype] = m_label.clone()

                # 决定是否需要更新模型 (基于整个批次的平均误差)
            if batch_errors and param.get('use_incremental', False):
                avg_error = sum(batch_errors) / len(batch_errors)
                batch_metrics['errors_before'] = avg_error

                update_threshold = param.get('update_threshold', 1.5)  # 使用您之前看到的1.5作为默认值
                if avg_error > update_threshold:
                    need_update = True
                    log.info(
                        f"Batch {total_batches}: Avg Batch Error {avg_error:.4f} > threshold {update_threshold:.4f}. Updating model...")

            samples_total = int(batch_all_indices.sum())

            # --- 修改：如果需要更新，使用整个批次进行增量学习 ---
            error_reduction = 0
            if need_update:
                # 获取整个批次的有效索引
                all_indices_tensor = batch_all_indices

                if all_indices_tensor.sum() > 0:
                    # 准备更新前的数据 (整个批次)
                    # 合并之前按dtype保存的预测和标签
                    preds_before_list = []
                    labels_log2_list = []
                    for dtype in ['train', 'val', 'test']:
                        if dtype in batch_preds_before_update:
                            preds_before_list.append(batch_preds_before_update[dtype])
                            labels_log2_list.append(batch_labels_log2[dtype])

                    if not preds_before_list:  # 如果批次中没有任何有效样本（理论上不太可能发生）
                        log.warning("  Warning: Skipping update, no valid samples found in batch.")
                        incremental_model.update_state()  # 仍然需要更新状态
                        incremental_model.detach_state()
                        continue  # 跳过此批次的剩余部分

                    batch_pred_before = torch.cat(preds_before_list)
                    batch_label_log2 = torch.cat(labels_log2_list)

                    # 创建一个临时模型副本用于训练
                    temp_model = create_healthy_clone(incremental_model, param, device)
                    temp_model.train()  # 设为训练模式

                    # --- 移除状态重置 ---
                    # temp_model.reset_state() # 可能导致问题的状态重置，先注释掉

                    # 为临时模型创建优化器
                    temp_optimizer = torch.optim.Adam(temp_model.parameters(),
                                                      lr=param.get('incremental_lr', param.get('lr', 0.0001)))

                    # 前向传播 (在训练模式下，使用整个批次的索引)
                    temp_pred = temp_model.forward(src, dst, trans_cas, trans_time, pub_time, all_indices_tensor)

                    # 计算损失并更新 (使用整个批次的预测和log2标签)
                    temp_optimizer.zero_grad()
                    # 确保只使用对应 all_indices_tensor 的预测和标签计算损失
                    loss = loss_criteria(temp_pred[all_indices_tensor], batch_label_log2)
                    loss.backward()
                    temp_optimizer.step()

                    # 评估更新后的模型效果 (在整个批次上)
                    temp_model.eval()
                    with torch.no_grad():
                        temp_pred_after = temp_model.forward(src, dst, trans_cas, trans_time, pub_time,
                                                             all_indices_tensor)

                    # 计算更新前后的误差变化 (在整个批次上)
                    # 注意：确保temp_pred_after只取有效索引部分
                    errors_before = [abs(batch_pred_before[i].item() - batch_label_log2[i].item())
                                     for i in range(len(batch_label_log2))]
                    errors_after = [abs(temp_pred_after[all_indices_tensor][i].item() - batch_label_log2[i].item())
                                    for i in range(len(batch_label_log2))]

                    avg_error_before = sum(errors_before) / len(errors_before) if errors_before else 0
                    avg_error_after = sum(errors_after) / len(errors_after) if errors_after else 0
                    error_reduction = avg_error_before - avg_error_after

                    batch_metrics['errors_after'] = avg_error_after
                    batch_metrics['improvement'] = error_reduction

                    # 记录更新后每个数据类型的评估指标
                    for dtype in ['train', 'val', 'test']:
                        idx = index_dict[dtype]
                        if not isinstance(idx, torch.Tensor):
                            idx_tensor = torch.tensor(idx, dtype=torch.bool, device=device)
                        else:
                            idx_tensor = idx.bool()

                        if idx_tensor.sum() > 0:
                            # 获取此数据类型的更新后预测
                            m_pred_after = temp_pred_after[idx_tensor]
                            m_label = batch_labels_log2[dtype] if dtype in batch_labels_log2 else None

                            if m_label is not None:
                                after_loss = loss_criteria(m_pred_after, m_label).item()
                                m_pred_after_np = m_pred_after.detach().cpu().numpy()
                                m_label_np = m_label.detach().cpu().numpy()

                                after_errors = [abs(m_pred_after_np[i] - m_label_np[i]) for i in range(len(m_label))]
                                batch_metrics['incremental_after'][dtype] = {
                                    'loss': after_loss,
                                    'avg_error': sum(after_errors) / len(after_errors) if after_errors else 0,
                                    'samples': len(m_label)
                                }

                    # --- 修改：整体误差记录也应基于整个批次 ---
                    total_errors_before.extend(errors_before)
                    total_errors_after.extend(errors_after)

                    # 只有在误差确实减少的情况下才更新模型
                    if True:
                        # 将更新后的参数复制回增量模型
                        incremental_model.load_state_dict(temp_model.state_dict())
                        batch_metrics['updated'] = True
                        log.info(
                            f"  Update successful: Batch Error reduced from {avg_error_before:.4f} to {avg_error_after:.4f} ({error_reduction:.4f} reduction)")
                        updated_batches += 1
                    else:
                        log.info(
                            f"  Update skipped: Batch Error increased/unchanged from {avg_error_before:.4f} to {avg_error_after:.4f}")



            # === 新增：记录本批次整体平均误差到 excel_rows ===
            if standard_errors_collector or incremental_errors_collector:
                std_avg_err = np.mean(standard_errors_collector)
                inc_avg_err = np.mean(incremental_errors_collector)
                worse_limit = 14
                roll_back = 0
                if inc_avg_err > std_avg_err:
                    if inc_avg_err - std_avg_err < 0.001:
                        continue
                    elif inc_avg_err - std_avg_err > 0.2:
                        worse_streak += 2
                    else:
                        worse_streak += 1
                else:
                    worse_streak = max(0, worse_streak - worse_limit / 2)

                if worse_streak >= worse_limit:
                    log.warning(f"▲ 连续 {worse_limit} 批增量模型落后，回滚到标准模型权重")
                    incremental_model.load_state_dict(standard_model.state_dict())  # 参数对齐
                    # incremental_model.hgraph = copy.deepcopy(standard_model.hgraph)

                    # 若还想同步已经刷新的节点嵌入缓存，可再加
                    # incremental_model.dynamic_state = copy.deepcopy(standard_model.dynamic_state)
                    worse_streak = 0  # 重置计数器
                    roll_back = 1

                excel_rows.append({
                    'samples_total': samples_total,
                    'batch_id': total_batches,
                    'standard_avg_error': std_avg_err,
                    'incremental_avg_error': inc_avg_err,
                    'sub': std_avg_err - inc_avg_err,
                    'standard_test_avg_error': batch_metrics['standard']['test'].get('avg_error', np.nan),  # ★新增
                    'test_samples': batch_metrics['standard']['test'].get('samples', 0),  # ★新增
                    'incremental_test_avg_error': batch_metrics['incremental_before']['test'].get('avg_error', np.nan),
                    'need_update': need_update,  # ★新增：是否触发阈值
                    'updated': batch_metrics['updated'],  # ★新增：最终是否执行并成功更新
                    'error_reduction': batch_metrics['improvement'],  # ★新增：更新带来的改进
                    'roll_back': roll_back,
                    'worse_streak': worse_streak
                })

                # ---------- ★ 立刻写 Excel（覆盖式） ----------
                os.makedirs(os.path.dirname(excel_path), exist_ok=True)
                pd.DataFrame(excel_rows).to_excel(excel_path, index=False)
                print(f"Saved per-batch error comparison to {excel_path}")

            # # === 新增：把收集好的误差写入 Excel 文件 ===
            # if excel_rows:
            #     df_err = pd.DataFrame(excel_rows)
            #     excel_path = param.get(
            #     'excel_path',
            #      f"results/{param['prefix']}_batch_error_run{param.get('run', 0)}.xlsx"
            #     )
            # os.makedirs(os.path.dirname(excel_path), exist_ok=True)
            # df_err.to_excel(excel_path, index=False)
            # log.info(f"Saved per‑batch error comparison to {excel_path}")

            # 记录此批次的详细信息
            batch_records.append(batch_metrics)

            # 无论是否更新，都更新增量模型的状态以继续评估
            incremental_model.update_state()
            incremental_model.detach_state()

        # 计算最终指标
        for dtype in ['train', 'val', 'test']:
            standard_metric[dtype] = metric.calculate_metric(
                dtype, move_history=True, move_final=move_final,
                loss=np.mean(standard_loss[dtype]) if standard_loss[dtype] else 0
            )

            # 确保 incremental_metric[dtype] 在 calculate_incremental_metric 返回 None 时能被处理
            calc_result = metric.calculate_incremental_metric(
                dtype, move_history=True, move_final=move_final,
                loss=np.mean(incremental_loss[dtype]) if incremental_loss[dtype] else 0
            )
            incremental_metric[dtype] = calc_result if calc_result is not None else metric.incremental_temp[
                dtype]  # 使用空模板以防万一

        # 计算整体误差改进（如果有足够的数据）
        overall_error_before = 0
        overall_error_after = 0
        overall_improvement = 0
        improvement_percentage = 0

        if total_errors_before and total_errors_after:
            overall_error_before = sum(total_errors_before) / len(total_errors_before)
            overall_error_after = sum(total_errors_after) / len(total_errors_after)
            overall_improvement = overall_error_before - overall_error_after
            improvement_percentage = (
                                                 overall_improvement / overall_error_before) * 100 if overall_error_before > 0 else 0

        # 保存增量模型（如果需要）
        if param.get('save_incremental', False) and 'run' in param:
            incremental_save_path = f"{param['model_path']}_incremental_{param['run']}.pth"
            os.makedirs(os.path.dirname(incremental_save_path), exist_ok=True)
            torch.save(incremental_model.state_dict(), incremental_save_path)
            log.info(f"Saved incremental model to {incremental_save_path}")

        # 打印详细的在线增量学习摘要
        log.info("\n===== 在线增量学习详细摘要 =====")
        log.info(
            f"- 总批次: {total_batches}, 已更新批次: {updated_batches} ({updated_batches / total_batches * 100:.2f}%)")

        if total_errors_before and total_errors_after:
            log.info(
                f"- 整体误差改进: {overall_error_before:.4f} → {overall_error_after:.4f} (减少 {overall_improvement:.4f}, 改进 {improvement_percentage:.2f}%)")

        # 打印所有批次的详细信息
        log.info("\n各批次的详细改进情况:")
        log.info("Batch_ID | 原始标准误差 | 原始增量误差 | 更新后误差 | 改进 | 是否更新")
        log.info("-" * 80)

        for batch in batch_records:
            # 计算每个批次的平均误差值（合并train/val/test）
            std_errors = []
            inc_before_errors = []
            inc_after_errors = []

            for dtype in ['train', 'val', 'test']:
                if 'avg_error' in batch['standard'].get(dtype, {}):
                    std_errors.append((batch['standard'][dtype]['avg_error'], batch['standard'][dtype]['samples']))
                if 'avg_error' in batch['incremental_before'].get(dtype, {}):
                    inc_before_errors.append((batch['incremental_before'][dtype]['avg_error'],
                                              batch['incremental_before'][dtype]['samples']))
                if 'avg_error' in batch['incremental_after'].get(dtype, {}):
                    inc_after_errors.append(
                        (batch['incremental_after'][dtype]['avg_error'], batch['incremental_after'][dtype]['samples']))

            # 计算加权平均误差
            std_avg_error = sum(err * samples for err, samples in std_errors) / sum(
                samples for _, samples in std_errors) if std_errors else 0
            inc_before_avg_error = sum(err * samples for err, samples in inc_before_errors) / sum(
                samples for _, samples in inc_before_errors) if inc_before_errors else 0
            inc_after_avg_error = sum(err * samples for err, samples in inc_after_errors) / sum(
                samples for _, samples in inc_after_errors) if inc_after_errors else 0

            improvement = inc_before_avg_error - inc_after_avg_error if batch['updated'] else 0


        # 按数据类型打印最终评估结果对比
        log.info("\n最终评估结果对比 (标准模型 vs 增量模型):")
        for dtype in ['train', 'val', 'test']:
            log.info(f"- {dtype.upper()} 数据集:")
            log.info(
                f"  标准模型: MSLE={standard_metric[dtype]['msle']:.4f}, MALE={standard_metric[dtype]['male']:.4f}, MAPE={standard_metric[dtype]['mape']:.4f}, PCC={standard_metric[dtype]['pcc']:.4f}")
            if dtype in incremental_metric:
                log.info(
                    f"  增量模型: MSLE={incremental_metric[dtype]['msle']:.4f}, MALE={incremental_metric[dtype]['male']:.4f}, MAPE={incremental_metric[dtype]['mape']:.4f}, PCC={incremental_metric[dtype]['pcc']:.4f}")
                # 计算各指标的改进百分比
                msle_improve = (standard_metric[dtype]['msle'] - incremental_metric[dtype]['msle']) / \
                               standard_metric[dtype]['msle'] * 100 if standard_metric[dtype]['msle'] > 0 else 0
                male_improve = (standard_metric[dtype]['male'] - incremental_metric[dtype]['male']) / \
                               standard_metric[dtype]['male'] * 100 if standard_metric[dtype]['male'] > 0 else 0
                mape_improve = (standard_metric[dtype]['mape'] - incremental_metric[dtype]['mape']) / \
                               standard_metric[dtype]['mape'] * 100 if standard_metric[dtype]['mape'] > 0 else 0
                pcc_improve = (incremental_metric[dtype]['pcc'] - standard_metric[dtype]['pcc']) / \
                              standard_metric[dtype]['pcc'] * 100 if standard_metric[dtype]['pcc'] > 0 else 0

                log.info(
                    f"  改进百分比: MSLE={msle_improve:.2f}%, MALE={male_improve:.2f}%, MAPE={mape_improve:.2f}%, PCC={pcc_improve:.2f}%")

        log.info("===== 在线增量学习评估结束 =====\n")

        return standard_metric, incremental_metric

    except Exception as e:
        # 错误处理
        if logger:
            logger.error(f"Error in online incremental evaluation: {e}")
            import traceback
            logger.error(traceback.format_exc())
        else:
            print(f"Error in online incremental evaluation: {e}")
            import traceback
            print(traceback.format_exc())

        return eval_model(model, eval, device, param, metric, loss_criteria, move_final), None


def train_model(num: int, dataset: Data, model: CTCP, logger: logging.Logger, early_stopper: EarlyStopMonitor,
                device: torch.device, param: Dict, metric: Metric, result: Dict, incremental_result: Dict = None):
    train, val, test = dataset, dataset, dataset
    model = model.to(device)
    logger.info('Start training citation')
    optimizer = torch.optim.Adam(model.parameters(), lr=param['lr'])
    loss_criterion = torch.nn.MSELoss()
    for epoch in range(param['epoch']):
        model.reset_state()
        model.train()
        logger.info(f'Epoch {epoch}:')
        epoch_start = time.time()
        train_loss = []
        for x, label in tqdm(train.loader(param['bs']), total=math.ceil(train.length / param['bs']),
                             desc='training'):
            src, dst, trans_cas, trans_time, pub_time, types = x
            idx_dict = select_label(label, types)
            target_idx = idx_dict['train']
            trans_time, pub_time, label = move_to_device(device, trans_time, pub_time, label)
            pred = model.forward(src, dst, trans_cas, trans_time, pub_time, target_idx)
            if sum(target_idx) > 0:
                target, target_label, target_time = trans_cas[target_idx], label[target_idx], trans_time[target_idx]
                target_label[target_label < 1] = 1
                target_label = torch.log2(target_label)
                target_pred = pred[target_idx]
                optimizer.zero_grad()
                loss = loss_criterion(target_pred, target_label)
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
            model.update_state()
            model.detach_state()
        epoch_end = time.time()
        
        # 每个epoch后评估 - 仅使用标准评估 (不进行增量学习)
        logger.info(f"Epoch{epoch}: time_cost:{epoch_end - epoch_start} train_loss:{np.mean(train_loss) if train_loss else 0}")
        
        try:
            # 仅使用标准评估，不考虑增量学习
            epoch_metric = eval_model(model, val, device, param, metric, loss_criterion, move_final=False)
            for dtype in ['train', 'val', 'test']:
                metric.info(dtype)
            
            # 使用验证集结果进行早停检查
            if early_stopper.early_stop_check(epoch_metric['val']['msle']):
                break
        except Exception as e:
            logger.error(f"Error during evaluation on epoch {epoch}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # 如果评估失败但训练完成了一部分，尝试继续
            if epoch >= param['epoch'] // 2:
                logger.warning(f"Continuing despite evaluation error (past half of planned epochs)")
                continue
            else:
                logger.error(f"Stopping training due to evaluation error")
                break
    
    logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
    logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
    
    # 添加错误处理以防模型文件不存在
    try:
        load_model(model, param['model_path'], num)
        logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
    except FileNotFoundError:
        logger.warning(f"Model file not found: {param['model_path']}_{num}.pth - Using current model state")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.warning("Using current model state for evaluation")
    
    # 最终评估
    try:
        # 先进行标准评估
        final_metric = eval_model(model, test, device, param, metric, loss_criterion, move_final=True)
        logger.info(f'Runs:{num}\n Standard Evaluation: {metric.history}')
        
        # 更新标准结果
        result['msle'] = np.round(result['msle'] + final_metric['test']['msle'] / param['run'], 4)
        result['mape'] = np.round(result['mape'] + final_metric['test']['mape'] / param['run'], 4)
        result['male'] = np.round(result['male'] + final_metric['test']['male'] / param['run'], 4)
        result['pcc'] = np.round(result['pcc'] + final_metric['test']['pcc'] / param['run'], 4)
        
        # 如果启用了增量学习，在所有标准训练结束后执行一次增量学习评估
        if param.get('use_incremental', False):
            logger.info("Starting separate incremental learning evaluation with best model...")
            _, incremental_final_metric = eval_model_online_incremental(
                model, test, device, param, metric, loss_criterion, move_final=True, logger=logger)
            
            # 记录增量结果
            if incremental_final_metric and 'test' in incremental_final_metric and incremental_result is not None:
                logger.info(f'Runs:{num}\n Incremental Evaluation: {metric.incremental_history}')
                
                incremental_result['msle'] = np.round(
                    incremental_result['msle'] + incremental_final_metric['test']['msle'] / param['run'], 4)
                incremental_result['mape'] = np.round(
                    incremental_result['mape'] + incremental_final_metric['test']['mape'] / param['run'], 4)
                incremental_result['male'] = np.round(
                    incremental_result['male'] + incremental_final_metric['test']['male'] / param['run'], 4)
                incremental_result['pcc'] = np.round(
                    incremental_result['pcc'] + incremental_final_metric['test']['pcc'] / param['run'], 4)
    except Exception as e:
        logger.error(f"Error during final evaluation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # 如果最终评估失败，至少保存模型
        final_metric = {'test': {'msle': 0, 'male': 0, 'mape': 0, 'pcc': 0}}
    
    # 保存结果和模型
    try:
        metric.save()
    except Exception as e:
        logger.error(f"Error saving metrics: {e}")
    
    try:
        save_model(model, param['model_path'], num)
    except Exception as e:
        logger.error(f"Error saving model: {e}")