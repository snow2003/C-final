# utils/deepcopy_safe.py
from utils.hgraph import HGraph # 确保导入 HGraph
from copy import deepcopy
import torch

def safe_deepcopy_ctcp(model):
    hgraph_bak = model.hgraph
    model.hgraph = None
    clone = deepcopy(model)
    model.hgraph = hgraph_bak
    # 为克隆体创建一个全新的 HGraph 实例
    # 你可能需要从 model 或 hgraph_bak 获取初始化 HGraph 所需的参数
    if hgraph_bak is not None: # 确保原始 hgraph 存在
        clone.hgraph = HGraph(num_user=hgraph_bak.num_user, num_cas=hgraph_bak.num_cas)
        # 如果 HGraph 还有其他需要从原始 hgraph 复制的初始状态（且这些状态是只读的），
        # 你可能需要更复杂的复制逻辑，或者接受克隆的 hgraph 从完全空白开始。
        # 对于在线评估场景，一个空白的 hgraph 给 incremental_model 可能是合适的。
    else:
        clone.hgraph = None # 或者进行相应的错误处理/日志记录
    return clone


from copy import deepcopy as std_deepcopy # 以免与你的 safe_deepcopy_ctcp 重名
from model.CTCP import CTCP # 确保导入
from utils.hgraph import HGraph # 确保导入

def create_healthy_clone(original_model: CTCP, param: dict, device: torch.device) -> CTCP:
    """
    创建一个功能上等价的 CTCP 模型克隆。
    这个克隆将拥有全新的 hgraph 和 dynamic_state 实例，但参数与原始模型相同。
    """
    # 1. 使用与原始模型相同的配置参数创建一个全新的 CTCP 实例
    #    你需要确保 param 字典包含了所有 CTCP 初始化所需的参数
    #    这些参数通常在 main.py 或 evaluate_model.py 中定义并传递给 CTCP 的构造函数
    cloned_model = CTCP(
        device=device,
        node_dim=param['node_dim'],
        embedding_module_type=param['embedding_module'],
        state_updater_type='gru', # 假设这个是固定的或从param获取
        predictor=param['predictor'],
        time_enc_dim=param['time_dim'],
        single=param['single'],
        ntypes={'user', 'cas'}, # 假设固定
        dropout=param['dropout'],
        n_nodes=param['node_num'], # 这个 node_num 需要从加载数据后确定的param中获取
        max_time=param['max_time'],
        use_static=param.get('use_static', False), # 使用 .get 提供默认值以增加稳健性
        merge_prob=param.get('lambda', 0.5), # 注意原始 CTCP 中 merge_prob 参数
        max_global_time=param['max_global_time'],
        use_dynamic=param.get('use_dynamic', True), # 假设默认为 True
        use_temporal=param.get('use_temporal', True),
        use_structural=param.get('use_structural', True)
    ).to(device)

    # 2. 深拷贝原始模型的 state_dict (参数和缓冲区) 并加载到新模型中
    #    使用标准库的 deepcopy 来复制 state_dict，以确保值的独立性
    original_state_dict = std_deepcopy(original_model.state_dict())
    cloned_model.load_state_dict(original_state_dict)

    # 3. 确保 HGraph 是新的 (CTCP 构造函数会创建一个新的 HGraph)
    #    cloned_model.hgraph 已经是新的了

    # # 4. 设置评估模式 (如果原始模型是 eval 模式)
    # if not original_model.training:
    #     cloned_model.eval()
    # else:
    #     cloned_model.train() # 或者保持训练模式，取决于你的需求

    return cloned_model