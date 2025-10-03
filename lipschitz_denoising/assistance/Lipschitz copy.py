"""This module contains the essential Lipschitz estimation functions."""
import warnings
from collections.abc import Callable

import numpy as np
import scipy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet
from .conv_to_matrix import get_conv2d_matrix
from .functions import (check_basic_block_structure,
                             check_bottleneck_structure)
from .math import (compute_matrix_1norm, compute_matrix_1norm_batched,
                        compute_matrix_2norm_power_method,
                        compute_matrix_2norm_power_method_batched)

# 添加调试标志
DEBUG_LIPSCHITZ = True  # 设置为True启用详细调试输出

def debug_print(message, value=None):
    """调试输出函数"""
    if DEBUG_LIPSCHITZ:
        if value is not None:
            print(f"[Lipschitz DEBUG] {message}: {value}")
        else:
            print(f"[Lipschitz DEBUG] {message}")

try:
    from lipschitz_denoising.trainers.hybrid_trainer import SimpleHybridModel
    HYBRID_MODEL_AVAILABLE = True
except ImportError:
    HYBRID_MODEL_AVAILABLE = False
    print("警告: 无法导入 SimpleHybridModel，Lipschitz估计将使用默认值")

# 添加对自定义模型的导入支持
try:
    from lipschitz_denoising.models.data_driven import SimpleDataDrivenBranch
    from lipschitz_denoising.models.dual_branch import DualBranchDenoise
    from lipschitz_denoising.models.model_driven import SimpleModelDriven
    from lipschitz_denoising.models.fusion import SimpleLearnableFusion
    CUSTOM_MODELS_AVAILABLE = True
except ImportError:
    CUSTOM_MODELS_AVAILABLE = False
    print("警告: 无法导入自定义模型模块，Lipschitz估计将使用默认值")

def compute_lower_bound(
    model: torch.nn.Module,
    dataset_dataloader: DataLoader,
    device: torch.device = torch.device("cpu"),
    norm_ord: int | float = 2,
) -> torch.Tensor:
    """Computes the lower bound of the whole model, given the dataset.

    Parameters
    ----------
    model
        The model object.
    dataset_dataloader
        Dataset dataloader, that will be used for computing the lower bound.
    device, optional
        Torch device, by default torch.device("cpu").
    norm_ord, optional
        Matrix or vector norm to use for Jacobian computation.
        Be careful to user the appropriate norm to obtain proper Lipschitz 
        bounds. By default 2.

    Returns
    -------
        Two tensors with one number, representing the lower Lipschitz and the 
        average Lipschitz bounds.
    """
    debug_print("开始计算Lipschitz下界")
    debug_print(f"使用的范数类型: {norm_ord}")
    debug_print(f"设备: {device}")
    
    n_samples = len(dataset_dataloader.dataset)
    debug_print(f"数据集样本数: {n_samples}")

    lower_bound = 0
    avg_bound = 0

    for i, (x_batch, y_batch) in enumerate(dataset_dataloader):
        debug_print(f"处理批次 {i+1}, 批次大小: {x_batch.shape[0]}")
        
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        norms = compute_jac_norm(model, x_batch, ord=norm_ord)
        debug_print(f"批次 {i+1} 的Jacobian范数", norms)

        lower = torch.max(norms)
        debug_print(f"批次 {i+1} 的最大范数", lower)
        
        if lower > lower_bound:
            lower_bound = lower
            debug_print(f"更新下界为", lower_bound)

        avg_bound += torch.sum(norms)
        debug_print(f"批次 {i+1} 的范数总和", torch.sum(norms))

    avg_bound /= n_samples
    debug_print("最终下界", lower_bound)
    debug_print("最终平均界", avg_bound)
    
    return lower_bound, avg_bound


def compute_upper_bound(
    lipschitz_per_layer: float | list | dict | torch.Tensor,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Computes the upper bound of the whole model, given all Lipschitz 
    constants for each layer.

    Parameters
    ----------
    lipschitz_per_layer
        Output of the `compute_lipschitz_upper_bound_per_layer` function, 
        can be a Tensor, float, list or a dict.
    dtype, optional
        dtype to use for Lipschitz computation, by default torch.float32.

    Returns
    -------
        A tensor with one number, representing the answer.

    Raises
    ------
    TypeError
        This exception is raised when `lipschitz_per_layer` type does not match 
        torch.Tensor, dict or list.

    """
    debug_print("开始计算Lipschitz上界")
    debug_print(f"输入类型: {type(lipschitz_per_layer)}")
    debug_print(f"输入值: {lipschitz_per_layer}")

    def _ensure_cpu_tensor(obj, dtype):
        """确保对象转换为CPU上的Tensor"""
        if isinstance(obj, torch.Tensor):
            return obj.cpu().type(dtype)
        elif isinstance(obj, (int, float)):
            return torch.tensor(obj, dtype=dtype, device='cpu')
        else:
            return obj

    # 统一设备处理函数
    def _process_with_device_unification(obj, dtype):
        if isinstance(obj, torch.Tensor):
            result = obj.cpu().type(dtype)
            debug_print("处理Tensor类型，移动到CPU", result)
            return result

        if isinstance(obj, float):
            result = torch.tensor(obj, dtype=dtype, device='cpu')
            debug_print("处理float类型，转换为CPU Tensor", result)
            return result

        if isinstance(obj, dict):
            debug_print("处理dict类型（残差块）")
            # 递归处理字典中的每个值
            processed_dict = {}
            for key, value in obj.items():
                processed_dict[key] = _process_with_device_unification(value, dtype)
            
            residual = compute_upper_bound(processed_dict["residual"], dtype)
            sequential = compute_upper_bound(processed_dict["sequential"], dtype)
            result = residual + sequential
            debug_print(f"残差部分: {residual}, 顺序部分: {sequential}, 总和: {result}")
            return result

        if isinstance(obj, list):
            debug_print(f"处理list类型，包含 {len(obj)} 个元素")
            # 递归处理列表中的每个元素
            elements = []
            for i, elem in enumerate(obj):
                elem_result = _process_with_device_unification(elem, dtype)
                elements.append(elem_result)
                debug_print(f"列表元素 {i}: {elem} -> {elem_result}")
            
            # 确保所有元素都在CPU上
            cpu_elements = [elem.cpu() if isinstance(elem, torch.Tensor) else elem for elem in elements]
            tensor_elements = []
            for elem in cpu_elements:
                if isinstance(elem, torch.Tensor):
                    tensor_elements.append(elem)
                else:
                    tensor_elements.append(torch.tensor(elem, dtype=dtype, device='cpu'))
            
            result = torch.stack(tensor_elements).prod()
            debug_print(f"列表所有元素的乘积: {result}")
            return result

        debug_print(f"未知类型: {type(obj)}")
        raise TypeError(f"Lipschitz per layer type {type(obj)} is unknown for this function.")

    # 使用设备统一处理
    result = _process_with_device_unification(lipschitz_per_layer, dtype)
    debug_print("最终上界结果", result)
    return result

def compute_lipschitz_upper_bound_per_layer(
    layer: torch.nn.Module,
    layer_input_shape: list[int],
    dtype: torch.dtype = torch.float32,
    fast_approximation: bool = True,
    ord: int | float = 2,
) -> torch.Tensor | list | dict:
    """Returns the Lipschitz constant for each particular layer of the model.
    If a Sequence is given, outputs a list of Lipschitz constants. If a 
    Bottleneck or BasicBlock are given, output a dictionary with a sequence of 
    Lipschitz constants for the sequential and residual parts.

    Parameters
    ----------
    layer
        Layer object.
    layer_input_shape
        Shape(s) of the input for this layer. Must be present to compute Conv 
        layers' Lip. constant.
    dtype
        dtype to use for Lipschitz computation, by default torch.float32.
    ord, optional
        The order of the norm, by default 2. Possible options: 1, 2, torch.inf.

    Returns
    -------
        The Lipschitz constant for this particular layer (not considering 
        previous layers).
    """
    debug_print(f"\n=== 开始处理层: {type(layer).__name__} ===")
    debug_print(f"输入形状: {layer_input_shape}")
    debug_print(f"范数类型: {ord}")
    debug_print(f"快速近似: {fast_approximation}")

    # supress warnings from scipy
    warnings.filterwarnings(action="ignore", module="scipy")

    # 快速近似：对于大尺寸图像，使用权重范数近似
    if fast_approximation and isinstance(layer, nn.Conv2d) and len(layer_input_shape) >= 3:
        img_size = layer_input_shape[-1]
        if img_size >= 128:  # 对于大尺寸图像使用快速近似
            debug_print("使用卷积层快速近似")
            return compute_conv2d_fast_approximation(layer, dtype, ord)

    # 首先检查是否为自定义模型
    if CUSTOM_MODELS_AVAILABLE:
        # 处理 DualBranchDenoise 模型
        if isinstance(layer, DualBranchDenoise):
            debug_print("检测到 DualBranchDenoise 模型")
            return compute_dual_branch_lipschitz(layer, layer_input_shape, dtype, ord)
        
        # 处理 SimpleDataDrivenBranch 模型
        elif isinstance(layer, SimpleDataDrivenBranch):
            debug_print("检测到 SimpleDataDrivenBranch 模型")
            return compute_data_driven_branch_lipschitz(layer, layer_input_shape, dtype, ord)
        
        # 处理 SimpleModelDriven 模型
        elif isinstance(layer, SimpleModelDriven):
            debug_print("检测到 SimpleModelDriven 模型")
            return compute_model_driven_lipschitz(layer, layer_input_shape, dtype, ord)
        
        # 处理 SimpleLearnableFusion 模型
        elif isinstance(layer, SimpleLearnableFusion):
            debug_print("检测到 SimpleLearnableFusion 模型")
            return compute_fusion_lipschitz(layer, layer_input_shape, dtype, ord)
    
    # 新增：处理 SimpleHybridModel
    if HYBRID_MODEL_AVAILABLE and isinstance(layer, SimpleHybridModel):
        debug_print("检测到 SimpleHybridModel 模型")
        return compute_hybrid_model_lipschitz(layer, layer_input_shape, dtype, ord)
    
    # 如果无法导入但类名匹配，也尝试处理
    if not HYBRID_MODEL_AVAILABLE and 'SimpleHybridModel' in str(type(layer)):
        debug_print("检测到 SimpleHybridModel 类名（通过字符串匹配）")
        return compute_hybrid_model_lipschitz(layer, layer_input_shape, dtype, ord)
    
    if isinstance(layer, nn.Sequential):
        debug_print("检测到 Sequential 层")
        # 改进的Sequential处理：支持单个输入形状或形状列表
        if isinstance(layer_input_shape, list) and len(layer_input_shape) == len(layer):
            # 如果提供了每个层的输入形状
            debug_print("使用逐个层的输入形状")
            result = [
                compute_lipschitz_upper_bound_per_layer(
                    layer[i], layer_input_shape[i], dtype, ord)
                for i in range(len(layer))
            ]
        else:
            # 如果只提供了一个输入形状，为所有层使用相同的形状
            debug_print("使用相同的输入形状给所有层")
            result = [
                compute_lipschitz_upper_bound_per_layer(
                    layer[i], layer_input_shape, dtype, ord)
                for i in range(len(layer))
            ]
        debug_print(f"Sequential 层结果: {result}")
        return result

    if isinstance(layer, nn.Linear):
        debug_print("检测到 Linear 层")
        # for linear layers, jacobian is the weight matrix
        w = layer.state_dict()["weight"]
        debug_print(f"权重矩阵形状: {w.shape}")

        if ord == 2:
            result = compute_matrix_2norm_power_method(w).type(dtype)
            debug_print(f"2-范数结果: {result}")
            return result
        if ord == 1:
            result = compute_matrix_1norm(w).type(dtype)
            debug_print(f"1-范数结果: {result}")
            return result
        if ord == torch.inf:
            result = torch.linalg.matrix_norm(w, ord=torch.inf).type(dtype)
            debug_print(f"无穷范数结果: {result}")
            return result

    if isinstance(layer, nn.Conv2d):
        debug_print("检测到 Conv2d 层")
        # Here, we compute the lipschitz constant of the layer as a
        # largest singular value of the identical linear matrix multiplication.

        conv_kernel = layer.weight.cpu().detach().numpy()
        debug_print(f"卷积核形状: {conv_kernel.shape}")
        img_size = layer_input_shape[-1]
        debug_print(f"图像尺寸: {img_size}")
        
        K = get_conv2d_matrix(
            conv_kernel, layer.padding[0], layer.stride[0], img_size)
        debug_print(f"卷积矩阵形状: {K.shape}")

        if ord == 2:
            result = torch.tensor(scipy.sparse.linalg.norm(K, ord=2), dtype=dtype)
            debug_print(f"2-范数结果: {result}")
            return result
        if ord == 1:
            result = np.abs(K).sum(axis=0).max(axis=1)
            debug_print(f"1-范数结果: {result}")
            return result
        if ord == torch.inf:
            result = np.abs(K).sum(axis=1).max(axis=0)
            debug_print(f"无穷范数结果: {result}")
            return result

    if isinstance(layer, nn.BatchNorm2d):
        debug_print("检测到 BatchNorm2d 层")
        state = layer.state_dict()
        var = state["running_var"]
        gamma = state["weight"]
        eps = layer.eps
        
        debug_print(f"方差形状: {var.shape}, gamma形状: {gamma.shape}, eps: {eps}")
        
        result = torch.max(torch.abs(gamma / torch.sqrt(var + eps))).type(dtype)
        debug_print(f"BatchNorm 结果: {result}")
        return result

    if isinstance(layer, resnet.BasicBlock) or isinstance(layer, resnet.Bottleneck):
        debug_print(f"检测到 ResNet 块: {type(layer).__name__}")
        # this is a ResNet residual block
        if isinstance(layer, resnet.BasicBlock):
            assert check_basic_block_structure(layer)
            debug_print("BasicBlock 结构验证通过")
        else:
            assert check_bottleneck_structure(layer)
            debug_print("Bottleneck 结构验证通过")

        # in comments starting with # //, we present a forward pass of the
        # BasicBlock acc. to
        # https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html
        # for clarity

        # // identity = x
        lip_residual = torch.tensor(1.0, dtype=dtype)
        debug_print("残差部分初始化为 1.0")

        # // out = self.conv1(x)
        lip = [compute_lipschitz_upper_bound_per_layer(
            layer.conv1, layer_input_shape[0], dtype, ord)]
        debug_print(f"conv1 Lipschitz: {lip[-1]}")
        
        # // out = self.bn1(out)
        lip += [compute_lipschitz_upper_bound_per_layer(layer.bn1, [], dtype, ord)]
        debug_print(f"bn1 Lipschitz: {lip[-1]}")
        
        # // out = self.relu(out)

        # // out = self.conv2(out)
        lip += [compute_lipschitz_upper_bound_per_layer(
            layer.conv2, layer_input_shape[1], dtype, ord)]
        debug_print(f"conv2 Lipschitz: {lip[-1]}")
        
        # // out = self.bn2(out)
        lip += [compute_lipschitz_upper_bound_per_layer(layer.bn2, [], dtype, ord)]
        debug_print(f"bn2 Lipschitz: {lip[-1]}")

        # in case of "Bottleneck" module
        if isinstance(layer, resnet.Bottleneck):
            debug_print("处理 Bottleneck 额外层")
            # // out = self.relu(out)

            # // out = self.conv3(out)
            lip += [
                compute_lipschitz_upper_bound_per_layer(
                    layer.conv3, layer_input_shape[2], dtype, ord)
            ]
            debug_print(f"conv3 Lipschitz: {lip[-1]}")
            
            # // out = self.bn3(out)
            lip += [compute_lipschitz_upper_bound_per_layer(
                layer.bn3, [], dtype, ord)]
            debug_print(f"bn3 Lipschitz: {lip[-1]}")

        # // if self.downsample is not None:
        # //     identity = self.downsample(x)
        if layer.downsample is not None:
            debug_print("处理下采样层")
            # downsample is a sequential layer with a convolution and batchnorm
            lip_residual = compute_lipschitz_upper_bound_per_layer(
                layer.downsample, [layer_input_shape[0], []], dtype, ord
            )
            debug_print(f"下采样层 Lipschitz: {lip_residual}")
        else:
            debug_print("无下采样层")

        # // out += identity
        # // out = self.relu(out)
        result = {"residual": lip_residual, "sequential": lip}
        debug_print(f"ResNet 块最终结果: {result}")
        return result

    # by default, for other types of layers, output 1
    # (be careful to include all non 1-Lipschitz layers in the check before)
    debug_print(f"未知层类型 {type(layer)}，返回默认值 1.0")
    return torch.tensor(1.0, dtype=dtype)


def compute_jac_norm(model: Callable,
                     x_batch: torch.Tensor,
                     ord: int | float = 2) -> torch.Tensor:
    """Computes the norm of the model jacobian for each input in the batch.
    This method works for all models with any inputs of any dimension. The 
    jacobian matrix is represented as a (num_outputs x flat_num_inputs) matrix.

    Parameters
    ----------
    model
        Model function.
    x_batch
        Batched inputs tensor of shape (batch_size x input_shape).
        Input_shape can be a number or 3 numbers (channels x height x width), 
        making x_batch of shape (batch_size x channels x height x width).
    ord, optional
        The order of the norm, by default 2. Possible options: 1, 2, torch.inf.

    Returns
    -------
        A vector of norms of the jacobian for each input in the batch.
    """
    debug_print("开始计算Jacobian范数")
    debug_print(f"输入批次形状: {x_batch.shape}")
    
    x_batch.requires_grad_(True)
    out = model(x_batch)
    debug_print(f"模型输出形状: {out.shape}")

    # jacobian of dimensions batch_size x output_dim x input_dim
    # input_dim can be a list
    jac = compute_jacobian(out, x_batch)
    debug_print(f"Jacobian矩阵形状: {jac.shape}")

    # to compute the 2-norm, we flatten the jacobian in the dim. of the input
    jac = jac.flatten(start_dim=2, end_dim=-1)
    debug_print(f"展平后Jacobian形状: {jac.shape}")

    if ord == 1:
        jac_norm = compute_matrix_1norm_batched(jac)
        debug_print("使用1-范数计算")
    if ord == 2:
        jac_norm = compute_matrix_2norm_power_method_batched(jac)
        debug_print("使用2-范数计算")
    if ord == torch.inf:
        jac_norm = torch.linalg.matrix_norm(jac, ord=torch.inf)
        debug_print("使用无穷范数计算")

    debug_print(f"Jacobian范数结果: {jac_norm}")
    
    x_batch.requires_grad_(False)

    return jac_norm


# 移除 @torch.jit.script 装饰器，改为普通函数
def compute_jacobian(
    y_batch: torch.Tensor, x_batch: torch.Tensor, create_graph: bool = False
) -> torch.Tensor:
    """Computes the Jacobian of y wrt to x.
    Thanks to https://github.com/magamba/overparameterization/blob/530948c72662b062fcb0c5c084b857a3951efb63/core/metric_helpers.py#L235
    for providing the code to this function.

    Parameters
    ----------
    y_batch
        Output tensor of the model, batched. Dimensions: batch_size x output_dim
    x_batch
        Input tensor of the model, batched. Dimensions: batch_size x input_dim
    create_graph, optional
        Flag that tells torch to create the graph. Useful to compute hessians. 
        By default False.

    Returns
    -------
        Jacobian of y_batch wrt to x_batch. 
        Dimensions: batch_size x output_dim x input_dim.

    Note
    ----
        x_batch has to track gradients before the function is called. If you use
        this function for second order derivatives, RESET THE GRADIENT OF 
        x_batch BY x_batch.grad = None. Otherwise, the second derivative will be
        added to the first derivative. If there are other nodes in the graph 
        that depend on y_batch, their gradients will also be computed.
    """
    # 在函数内部检查调试标志，而不是在函数定义时
    if DEBUG_LIPSCHITZ:
        print("[Lipschitz DEBUG] 计算Jacobian矩阵")
        print(f"[Lipschitz DEBUG] y_batch形状: {y_batch.shape}, x_batch形状: {x_batch.shape}")
    
    nclasses = y_batch.shape[1]
    
    if DEBUG_LIPSCHITZ:
        print(f"[Lipschitz DEBUG] 输出类别数: {nclasses}")

    x_batch.retain_grad()
    # placeholder for the jacobian
    jacobian = torch.zeros(x_batch.shape + (nclasses,),
                           dtype=x_batch.dtype, device=x_batch.device)
    
    if DEBUG_LIPSCHITZ:
        print(f"[Lipschitz DEBUG] Jacobian占位符形状: {jacobian.shape}")
    
    # this mask tells torch to only compute the jacobian wrt. to the 0th index
    # of the output = ∇_x(f_0).
    indexing_mask = torch.zeros_like(y_batch)
    indexing_mask[:, 0] = 1.0

    for dim in range(nclasses):
        if DEBUG_LIPSCHITZ:
            print(f"[Lipschitz DEBUG] 计算第 {dim} 个输出的梯度")
        y_batch.backward(gradient=indexing_mask,
                         retain_graph=True, create_graph=create_graph)
        # fill the ith index with grad data of ∇_x(f_i).
        jacobian[..., dim] = x_batch.grad
        
        if DEBUG_LIPSCHITZ:
            print(f"[Lipschitz DEBUG] 第 {dim} 个输出的梯度范数: {torch.norm(x_batch.grad)}")
            
        x_batch.grad.data.zero_()
        # shift the mask to compute the i+1th index of the output
        indexing_mask = torch.roll(indexing_mask, shifts=1, dims=1)

    # permute jacobian dimensions
    # from batch_size x input_dim x output_dim
    # to batch_size x output_dim x input_dim
    permute_dims = [0] + [len(jacobian.shape) - 1] + \
        list(range(1, len(jacobian.shape) - 1))

    result = torch.permute(jacobian, permute_dims)
    
    if DEBUG_LIPSCHITZ:
        print(f"[Lipschitz DEBUG] 重排后的Jacobian形状: {result.shape}")
    
    return result

def compute_hybrid_model_lipschitz(layer, layer_input_shape, dtype, ord):
    """计算 SimpleHybridModel 的 Lipschitz 常数"""
    debug_print("计算 SimpleHybridModel Lipschitz常数")
    try:
        # 基于 DualBranchDenoise 的结构计算 Lipschitz 常数
        # 假设 SimpleHybridModel 与 DualBranchDenoise 结构相似
        
        lip_constants = []
        
        # 检查并处理数据驱动分支
        if hasattr(layer, 'data_branch') and layer.data_branch is not None:
            debug_print("检测到数据驱动分支")
            lip_data = compute_lipschitz_upper_bound_per_layer(
                layer.data_branch, layer_input_shape, dtype, ord)
            lip_constants.append(lip_data)
            debug_print(f"数据驱动分支 Lipschitz: {lip_data}")
        else:
            debug_print("未找到数据驱动分支")
        
        # 检查并处理模型驱动分支
        if hasattr(layer, 'model_branch') and layer.model_branch is not None:
            debug_print("检测到模型驱动分支")
            lip_model = compute_lipschitz_upper_bound_per_layer(
                layer.model_branch, layer_input_shape, dtype, ord)
            lip_constants.append(lip_model)
            debug_print(f"模型驱动分支 Lipschitz: {lip_model}")
        else:
            debug_print("未找到模型驱动分支")
        
        # 检查并处理融合层
        if hasattr(layer, 'fusion') and layer.fusion is not None:
            debug_print("检测到融合层")
            lip_fusion = compute_lipschitz_upper_bound_per_layer(
                layer.fusion, layer_input_shape, dtype, ord)
            lip_constants.append(lip_fusion)
            debug_print(f"融合层 Lipschitz: {lip_fusion}")
        else:
            debug_print("未找到融合层")
        
        # 处理其他可能的组件
        other_components = []
        for name, module in layer.named_children():
            if name not in ['data_branch', 'model_branch', 'fusion']:
                other_components.append((name, module))
        
        for name, module in other_components:
            debug_print(f"处理其他组件 {name}: {type(module).__name__}")
            lip = compute_lipschitz_upper_bound_per_layer(module, layer_input_shape, dtype, ord)
            lip_constants.append(lip)
            debug_print(f"组件 {name} Lipschitz: {lip}")
        
        if len(lip_constants) == 0:
            debug_print("未找到任何组件，返回默认值1.0")
            return torch.tensor(1.0, dtype=dtype)
        
        # 对于双分支模型，整体 Lipschitz 常数可以保守估计为各分支的最大值乘以融合系数
        if len(lip_constants) >= 3:  # 至少有数据分支、模型分支和融合层
            debug_print("使用双分支模型 Lipschitz 估计策略")
            
            # 计算各分支的上界
            data_bound = compute_upper_bound(lip_constants[0], dtype) if len(lip_constants) > 0 else torch.tensor(1.0, dtype=dtype)
            model_bound = compute_upper_bound(lip_constants[1], dtype) if len(lip_constants) > 1 else torch.tensor(1.0, dtype=dtype)
            fusion_bound = compute_upper_bound(lip_constants[2], dtype) if len(lip_constants) > 2 else torch.tensor(1.0, dtype=dtype)
            
            # 整体 Lipschitz 常数不超过各分支的最大值乘以融合层的常数
            max_branch = torch.max(data_bound, model_bound)
            total_bound = max_branch * fusion_bound
            
            # 如果还有其他组件，乘以它们的 Lipschitz 常数
            if len(lip_constants) > 3:
                for i in range(3, len(lip_constants)):
                    comp_bound = compute_upper_bound(lip_constants[i], dtype)
                    total_bound = total_bound * comp_bound
            
            debug_print(f"SimpleHybridModel 总 Lipschitz: {total_bound}")
            return total_bound
        else:
            # 如果不是标准双分支结构，返回所有组件的乘积
            debug_print("使用顺序模型 Lipschitz 估计策略")
            total_bound = torch.tensor(1.0, dtype=dtype)
            for lip in lip_constants:
                lip_bound = compute_upper_bound(lip, dtype)
                total_bound = total_bound * lip_bound
                debug_print(f"当前总 Lipschitz: {total_bound}")
            
            debug_print(f"SimpleHybridModel 总 Lipschitz: {total_bound}")
            return total_bound
            
    except Exception as e:
        warnings.warn(f"SimpleHybridModel Lipschitz估计失败: {e}, 使用默认值1.0")
        debug_print(f"错误详情: {e}")
        return torch.tensor(1.0, dtype=dtype)

def compute_conv2d_fast_approximation(layer: nn.Conv2d, dtype: torch.dtype, ord: int | float) -> torch.Tensor:
    """快速近似计算卷积层的 Lipschitz 常数"""
    debug_print("使用卷积层快速近似方法")
    
    w = layer.weight
    debug_print(f"卷积核形状: {w.shape}")
    
    if ord == 2:
        # 使用权重矩阵的谱范数乘以卷积核尺寸的平方根作为近似
        # 这是一个保守但快速的估计
        spectral_norm = compute_matrix_2norm_power_method(w.flatten(1))
        kernel_area = w.shape[2] * w.shape[3]
        approx = spectral_norm * torch.sqrt(torch.tensor(kernel_area, dtype=dtype))
        debug_print(f"快速2-范数近似: {approx}")
        return approx.type(dtype)
    
    elif ord == 1:
        # 1-范数近似：最大列和
        w_flat = w.flatten(1)
        approx = torch.max(torch.sum(torch.abs(w_flat), dim=0))
        debug_print(f"快速1-范数近似: {approx}")
        return approx.type(dtype)
    
    elif ord == torch.inf:
        # 无穷范数近似：最大行和
        w_flat = w.flatten(1)
        approx = torch.max(torch.sum(torch.abs(w_flat), dim=1))
        debug_print(f"快速无穷范数近似: {approx}")
        return approx.type(dtype)
    
    else:
        # 默认返回1.0
        debug_print(f"不支持的范数类型 {ord}，返回1.0")
        return torch.tensor(1.0, dtype=dtype)

def compute_dual_branch_lipschitz(layer, layer_input_shape, dtype, ord):
    """计算DualBranchDenoise模型的Lipschitz常数"""
    debug_print("计算 DualBranchDenoise Lipschitz常数")
    try:
        # 递归计算两个分支的Lipschitz常数
        lip_data = compute_lipschitz_upper_bound_per_layer(
            layer.data_branch, layer_input_shape, dtype, ord)
        debug_print(f"数据分支 Lipschitz: {lip_data}")
        
        lip_model = compute_lipschitz_upper_bound_per_layer(
            layer.model_branch, layer_input_shape, dtype, ord)
        debug_print(f"模型分支 Lipschitz: {lip_model}")
        
        lip_fusion = compute_lipschitz_upper_bound_per_layer(
            layer.fusion, layer_input_shape, dtype, ord)
        debug_print(f"融合层 Lipschitz: {lip_fusion}")
        
        # 确保所有张量都在CPU上
        lip_data_cpu = lip_data.cpu() if isinstance(lip_data, torch.Tensor) else torch.tensor(lip_data, dtype=dtype, device='cpu')
        lip_model_cpu = lip_model.cpu() if isinstance(lip_model, torch.Tensor) else torch.tensor(lip_model, dtype=dtype, device='cpu')
        lip_fusion_cpu = lip_fusion.cpu() if isinstance(lip_fusion, torch.Tensor) else torch.tensor(lip_fusion, dtype=dtype, device='cpu')
        
        # 双分支模型的整体Lipschitz常数可以近似为各分支的最大值乘以融合系数
        data_bound = compute_upper_bound(lip_data_cpu, dtype)
        model_bound = compute_upper_bound(lip_model_cpu, dtype)
        fusion_bound = compute_upper_bound(lip_fusion_cpu, dtype)
        
        debug_print(f"数据分支上界: {data_bound}")
        debug_print(f"模型分支上界: {model_bound}")
        debug_print(f"融合层上界: {fusion_bound}")
        
        # 使用最大值策略
        max_branch = torch.max(data_bound, model_bound)
        total_bound = max_branch * fusion_bound
        
        debug_print(f"双分支模型总 Lipschitz: {total_bound}")
        return total_bound.cpu()  # 确保返回CPU张量
        
    except Exception as e:
        warnings.warn(f"DualBranchDenoise Lipschitz估计失败: {e}, 使用默认值1.0")
        return torch.tensor(1.0, dtype=dtype, device='cpu')

def compute_data_driven_branch_lipschitz(layer, layer_input_shape, dtype, ord):
    """计算SimpleDataDrivenBranch模型的Lipschitz常数"""
    debug_print("计算 SimpleDataDrivenBranch Lipschitz常数")
    try:
        # 将数据驱动分支视为Sequential处理
        layers = []
        
        # 初始卷积层
        layers.append(layer.initial_conv)
        debug_print("添加初始卷积层")
        
        # 中间层（需要展开ModuleList）
        for i in range(0, len(layer.middle_layers), 3):  # 每3层一组：Conv2d, BatchNorm2d, ReLU
            if i < len(layer.middle_layers):
                layers.append(layer.middle_layers[i])  # Conv2d
                debug_print(f"添加中间卷积层 {i}")
            if i+1 < len(layer.middle_layers):
                layers.append(layer.middle_layers[i+1])  # BatchNorm2d
                debug_print(f"添加中间批归一化层 {i+1}")
            # ReLU的Lipschitz常数为1，可以跳过
        
        # 最终卷积层
        layers.append(layer.final_conv)
        debug_print("添加最终卷积层")
        
        debug_print(f"总共 {len(layers)} 层需要处理")
        
        # 为每层生成输入形状（简化：假设所有层输入形状相同）
        input_shapes = [layer_input_shape] * len(layers)
        
        # 计算每层的Lipschitz常数
        lip_constants = []
        for i, sublayer in enumerate(layers):
            debug_print(f"处理第 {i} 层: {type(sublayer).__name__}")
            lip = compute_lipschitz_upper_bound_per_layer(
                sublayer, input_shapes[i], dtype, ord)
            lip_constants.append(lip)
            debug_print(f"第 {i} 层 Lipschitz: {lip}")
        
        # 返回层列表，让compute_upper_bound处理乘积
        debug_print(f"数据驱动分支 Lipschitz 列表: {lip_constants}")
        return lip_constants
        
    except Exception as e:
        warnings.warn(f"SimpleDataDrivenBranch Lipschitz估计失败: {e}, 使用默认值1.0")
        return torch.tensor(1.0, dtype=dtype)

def compute_model_driven_lipschitz(layer, layer_input_shape, dtype, ord):
    """计算SimpleModelDriven模型的Lipschitz常数"""
    debug_print("计算 SimpleModelDriven Lipschitz常数")
    try:
        # 根据SimpleModelDriven的实际结构实现
        # 这里需要您提供SimpleModelDriven的详细结构
        
        # 临时方案：基于模型参数估计
        total_params = sum(p.numel() for p in layer.parameters())
        debug_print(f"模型参数总数: {total_params}")
        
        if total_params > 0:
            # 简单启发式：参数越多，Lipschitz常数可能越大
            # 这是一个非常粗略的估计，需要根据实际模型改进
            estimate = 1.0 + 0.01 * (total_params / 1000)  # 每1000个参数增加0.01
            result = torch.tensor(min(estimate, 10.0), dtype=dtype)  # 设置上限
            debug_print(f"基于参数的估计结果: {result}")
            return result
        else:
            debug_print("无参数，返回默认值1.0")
            return torch.tensor(1.0, dtype=dtype)
            
    except Exception as e:
        warnings.warn(f"SimpleModelDriven Lipschitz估计失败: {e}, 使用默认值1.0")
        return torch.tensor(1.0, dtype=dtype)

def compute_fusion_lipschitz(layer, layer_input_shape, dtype, ord):
    """计算SimpleLearnableFusion模型的Lipschitz常数"""
    debug_print("计算 SimpleLearnableFusion Lipschitz常数")
    try:
        # 融合层的Lipschitz常数通常较小
        # 对于加权和融合：lip <= |alpha| + |1-alpha|
        # 由于alpha通过sigmoid输出在0-1之间，所以lip <= 1
        
        if hasattr(layer, 'alpha'):
            # 如果使用可学习参数alpha
            alpha = torch.sigmoid(layer.alpha).item()
            lip_constant = alpha + (1 - alpha)  # 总是等于1
            debug_print(f"使用alpha参数: {alpha}, Lipschitz常数: {lip_constant}")
            return torch.tensor(lip_constant, dtype=dtype)
        else:
            # 默认融合层的Lipschitz常数
            debug_print("无alpha参数，返回默认值1.0")
            return torch.tensor(1.0, dtype=dtype)
            
    except Exception as e:
        warnings.warn(f"SimpleLearnableFusion Lipschitz估计失败: {e}, 使用默认值1.0")
        return torch.tensor(1.0, dtype=dtype)