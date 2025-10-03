以下为项目的大致构造及讲解：python scripts/train_hybrid.py   --dataset bsd68   --experiment_dir experiments/hybrid_v1

/workspace
lipschitz_denoising/
├── lipschitz_denoising/                        # 主项目包
│   ├── __init__.py
│   ├── models/                                 # 模型定义
│   │   ├── __init__.py
│   │   ├── dual_branch.py                      # 双分支网络架构
│   │   ├── data_driven.py                      # 数据驱动分支（ResNet变体 + 动态谱归一化）
│   │   ├── model_driven.py                     # 模型驱动分支（小波+TV正则）
│   │   └── fusion.py                           # 动态门控融合模块
│   ├── functions/                              # 工具函数
│   │   ├── __init__.py
│   │   ├── lipschitz.py                        # Lipschitz估计（谱范数、Jacobian估计等）
│   │   ├── noise_generation.py                 # 混合噪声生成（高斯+泊松+脉冲）
│   │   ├── regularization.py                   # 正则化项（TV、稀疏性、Lipschitz正则）
│   │   └── metrics.py                          # 评估指标（PSNR, SSIM, LSR, MNR, DDD等）
│   ├── trainers/                               # 训练脚本
│   │   ├── __init__.py
│   │   ├── base_trainer.py                     # 基础训练器
│   │   ├── hybrid_trainer.py                   # 混合驱动训练器
│   │   └── adversarial_trainer.py              # 对抗训练器
│   ├── evaluators/                             # 评估脚本
│   │   ├── __init__.py
│   │   ├── robustness.py                       # 鲁棒性评估（白盒/黑盒攻击、跨域泛化）
│   │   ├── lipschitz_analysis.py               # Lipschitz常数估计与分析
│   │   └── visualization.py                    # 结果可视化
│   ├── utils/                                  # 工具脚本
│   │   ├── __init__.py
│   │   ├── logger.py                           # 日志记录
│   │   ├── data_loader.py                      # 数据加载与预处理
│   │   └── checkpoint.py                       # 模型保存与加载
│   │
│   └── configs/                                # 配置文件
│       ├── base.yaml                           # 基础配置（共享参数）
│       ├── hybrid.yaml                         # 混合驱动模型主配置
│       ├── adversarial.yaml                    # 对抗训练配置
│       ├── ablation/                           # 消融实验配置
│       │   ├── no_lipschitz_constraint.yaml
│       │   ├── fixed_fusion_weights.yaml
│       │   ├── target_lipschitz_variants.yaml
│       │   ├── single_branch_data.yaml
│       │   └── single_branch_model.yaml
│       ├── datasets/                           # 数据集特定配置
│       │   ├── bsd68.yaml
│       │   ├── medical_ct.yaml
│       │   ├── transfer_learning.yaml
│       │   └── real_world_noise.yaml
│       ├── datasets/                           # 评估特定配置
│       │   ├── robustness_full.yaml
│       │   ├── lipschitz_analysis.yaml
│       │   └── cross_domain.yaml
│       └── environments/                       # 环境特定配置
│           ├── pt28_env.yaml                   # 主训练环境配置
│           └── lirpa_env.yaml                  # Lipschitz分析环境配置
│
├── experiments/                                # 实验记录与结果
│   ├── logs/                                   # 训练日志（TensorBoard）
│   ├── checkpoints/                            # 模型权重
│   ├── results/                                # 评估结果（表格、图片）
│   └── figures/                                # 论文用图
│
├── data/                                       # 数据集（按类型组织）
│   ├── basic_gray/                             # 灰度数据集（BSD68, Set12）
│   ├── hi-quality_cc/                          # 彩色数据集（Kodak24, McMaster, CBSD68）
│   ├── real_noise/                             # 真实噪声（RNI6, RNI15）
│   ├── cset9/                                  # 特定场景（CSet9）
│   │
│   ├── test_basic/                             # 基准测试与训练 (BSD68, CBSD68)
│   ├── test_cross-domain/                      # 跨域泛化验证 (Kodak24, McMaster)
│   └── test_real_noise/                        # 真实噪声鲁棒性测试 (RNI6, RNI15)
├── scripts/                                    # 运行脚本
│   ├── train_hybrid.py                         # 训练混合模型
│   ├── eval_robustness.py                      # 评估鲁棒性
│   ├── estimate_lipschitz.py                   # 估计Lipschitz常数
│   └── generate_noise.py                       # 生成混合噪声数据集
├── requirements.txt                            # Python依赖（主环境）
├── environment.yml                             # Conda环境配置（双环境）
└── README.md                                   # 项目说明（包含此结构）

### 项目结构详解

#### **1. 根目录 (`/workspace`)**
- **职责**：项目的容器，包含所有源代码、数据、实验记录和配置。
- **关键文件**：
    - `README.md`: 项目总览，环境配置说明，快速开始指南。
    - `requirements.txt`: 主环境（`pt28_env`）的Python依赖列表。
    - `environment.yml`: 用于导出或重建两个Conda环境的配置。
    - `pyproject.toml` (可选): 现代Python项目配置，可用于打包。

#### **2. 核心代码包 (`lipschitz_denoising/`)**
这是项目的核心，所有源代码都组织在这里。

- **`models/` - 模型定义**
    - **职责**：定义神经网络架构，低完成度完成。

1. `data_driven.py` — 数据驱动分支（ResNet变体 + 动态谱归一化）

    - `SpectralNormConv2d`：实现动态谱归一化的卷积层，用于控制 Lipschitz 常数。  
        **初始化参数**：
        - `in_channels`, `out_channels`, `kernel_size`, `stride`, `padding`, `dilation`, `groups`, `bias`：同标准 `nn.Conv2d`
        - `beta`：滑动平均系数，用于估计谱范数（默认 0.99）
        **方法**：
        - `forward(x)`：返回归一化后的卷积结果，并更新运行估计的谱范数

    - `AdaptiveReLU`：具有逐通道可学习缩放因子的 ReLU 激活函数（对应式2.2）。  
        **初始化参数**：
        - `num_channels`：通道数
        - `init_alpha`：初始缩放因子（默认 1.0）
        **方法**：
        - `forward(x)`：返回 `ReLU(x) * alpha`

    - `ResidualBlock`：改进的残差块，集成动态谱归一化和自适应 ReLU。  
        **初始化参数**：
        - `channels`：输入输出通道数
        - `beta`：谱归一化滑动平均系数（默认 0.99）
        - `use_dropout`：是否使用 Dropout（默认 False）
        - `dropout_rate`：Dropout 比例（默认 0.1）
        **方法**：
        - `forward(x)`：残差前向传播

    - `DataDrivenBranch`：基于 ResNet-18 改进的数据驱动分支。  
        **初始化参数**：
        - `config`：配置字典，需包含 `model.data_driven` 下的各项参数
        **方法**：
        - `forward(x)`：返回去噪后的图像
        - `get_lipschitz_estimate()`：返回整个分支的 Lipschitz 常数上界估计（式2.2）

2. `model_driven.py` — 模型驱动分支（小波+TV正则）

    - `WaveletThreshold`：可微分小波阈值去噪模块。  
        **初始化参数**：
        - `wavelet`：小波类型（默认 'haar'）
        - `threshold`：初始阈值（默认 0.1）
        - `learnable`：是否可学习（默认 True）
        **方法**：
        - `forward(x)`：返回小波阈值去噪后的图像

    - `TVRegularization`：计算全变分正则化损失（式2.3）。  
        **初始化参数**：
        - `weight`：TV 正则权重（默认 0.05）
        **方法**：
        - `forward(x)`：返回 TV 正则损失值

    - `LipschitzRegularization`：计算 Lipschitz 正则化损失（式2.3）。  
        **初始化参数**：
        - `weight`：Lipschitz 正则权重（默认 0.01）
        **方法**：
        - `forward(x, model)`：返回 Lipschitz 正则损失值

    - `EnergyMinimization`：实现能量函数最小化过程（式2.3）。  
        **初始化参数**：
        - `lambda1`：稀疏性正则权重（默认 0.1）
        - `lambda2`：TV 正则权重（默认 0.05）
        - `lambda3`：Lipschitz 正则权重（默认 0.01）
        - `wavelet`：小波类型
        - `initial_threshold`：初始小波阈值
        - `learnable_threshold`：阈值是否可学习
        - `num_iterations`：迭代次数（默认 5）
        **方法**：
        - `forward(noisy_img, clean_img)`：返回去噪图像和各损失项

    - `ModelDrivenBranch`：模型驱动分支主类，封装能量最小化与后处理。  
        **初始化参数**：
        - `in_channels`：输入通道数（默认 1）
        - `wavelet`：小波类型
        - `lambda1`, `lambda2`, `lambda3`：正则化权重
        - `initial_threshold`：小波初始阈值
        - `learnable_threshold`：是否可学习
        - `num_iterations`：迭代次数
        **方法**：
        - `forward(x, y=None)`：返回去噪图像和损失字典（若有 y）

3. `fusion.py` — 动态门控融合模块

    - `LipschitzAwareFusion`：基于局部 Lipschitz 常数的动态门控融合（式2.4）。  
        **初始化参数**：
        - `neighborhood_size`：局部邻域大小（默认 9）
        - `activation`：激活函数类型（'sigmoid' 或 'tanh'，默认 'sigmoid'）
        **方法**：
        - `forward(x, model)`：返回门控权重图 `g`，用于融合两个分支输出

4. `dual_branch.py` — 双分支混合驱动网络

    - `DualBranchHybrid`：整合数据驱动与模型驱动分支，实现动态融合。  
        **初始化参数**：
        - `config`：配置字典，需包含 `model.data_driven`、`model.model_driven`、`model.fusion` 等参数
        **方法**：
        - `forward(noisy, clean=None)`：返回去噪图像和损失字典
        - `get_lipschitz_estimate()`：返回整个混合模型的 Lipschitz 常数估计（式4.2）

---

- **`functions/` - 工具函数**

- **职责**：提供通用的、与模型无关的工具函数，涵盖 Lipschitz 估计、噪声生成、正则化项和评估指标等，已实现，接口展示如下。

##### **`lipschitz.py`**
- **作用**：实现 Lipschitz 常数估计相关方法，包括谱归一化、幂迭代法、Jacobian 范数计算等。
- **主要函数**：
  - `spectral_norm_conv2d(weight, beta=0.99, power_iterations=5)`：对卷积层权重进行动态谱归一化。
    - 输入：`weight`（卷积权重），`beta`（滑动平均系数），`power_iterations`（幂迭代次数）
    - 输出：归一化后的权重、谱范数估计值
  - `spectral_norm_linear(weight, beta=0.99, power_iterations=5)`：对全连接层权重进行谱归一化。
  - `power_iteration(weight, iterations=10000, return_vectors=False)`：通用幂迭代法估计最大奇异值。
  - `estimate_lipschitz_bounds(model, dataloader, num_batches=10)`：估计模型的 Lipschitz 常数上下界。
    需要注意的是，目前该代码（包括import的项目的源代码）对非线性模型的估计有些问题，由于时间问题暂时没有改进

##### **`noise_generation.py`**
- **作用**：生成多种噪声（高斯、泊松、脉冲及其混合噪声），用于构建训练和测试数据。
- **主要函数**：
  - `add_gaussian_noise(x, sigma=25.0, mean=0.0)`：添加高斯噪声。
    - 输入：图像张量 `x`，噪声标准差 `sigma`，均值 `mean`
    - 输出：加噪后的图像
  - `add_poisson_noise(x, lam=30.0)`：添加泊松噪声。
  - `add_impulse_noise(x, density=0.1, salt_vs_pepper=0.5)`：添加脉冲噪声（椒盐噪声）。
  - `add_mixed_noise(x, noise_config)`：按配置比例添加混合噪声。
    - `noise_config` 示例：
      ```python
      [{"type": "gaussian", "sigma": 25, "ratio": 0.5},
       {"type": "poisson", "lambda": 30, "ratio": 0.3},
       {"type": "impulse", "density": 0.1, "ratio": 0.2}]
      ```

##### **`regularization.py`**
- **作用**：定义损失函数中的正则化项，包括 TV 正则、稀疏性正则和 Lipschitz 正则。
- **主要函数**：
  - `tv_regularization(x, weight=0.05)`：计算全变分正则化损失。
    - 输入：图像张量 `x`，正则化权重 `weight`
    - 输出：TV 损失值
  - `sparsity_regularization(x, weight=0.1, p=1.0)`：计算稀疏性正则化损失（Lp 范数）。
  - `lipschitz_regularization(model, target_constant=2.5, penalty_lambda=0.1, method="power_iteration")`：计算 Lipschitz 正则化损失。

##### **`metrics.py`**
- **作用**：实现图像去噪任务的评估指标，包括传统指标和本项目提出的新指标。
- **主要函数**：
  - `psnr(denoised, target, data_range=None)`：计算 PSNR。
  - `ssim(denoised, target, data_range=None)`：计算 SSIM。
  - `lipschitz_sensitivity_ratio(model, test_dataset, num_samples=100)`：计算 Lipschitz 敏感度比（LSR）。
  - `mixed_noise_robustness(clean_psnr, noisy_psnr)`：计算混合噪声鲁棒性（MNR）。
  - `domain_difference_decay(source_psnr, target_psnr)`：计算域差异衰减率（DDD）。

---

- **`trainers/` - 训练脚本**
    - **职责**：封装训练循环的逻辑，使训练过程模块化，未完成。
    - **`base_trainer.py`**: 定义抽象基类，包含通用的训练、验证、保存 checkpoint 等方法。
    - **`hybrid_trainer.py`**: 继承基类，实现**混合模型的特定训练逻辑**，例如交替优化策略、Lipschitz正则项的动态加权等。
    - **`adversarial_trainer.py`**: 实现对抗训练（如PGD攻击），用于提升模型鲁棒性。

- **`evaluators/` - 评估脚本**
    - **职责**：封装模型评估的逻辑，用于系统性的性能测试，未完成。
    - **`robustness.py`**: **核心评估文件**。执行白盒/黑盒对抗攻击、混合噪声测试、跨域泛化实验，并输出指标。
    - **`lipschitz_analysis.py`**: 专门用于分析模型或各组件的Lipschitz常数，验证理论推导（式4.1, 4.2）。
    - **`visualization.py`**: 生成可视化结果，如去噪效果对比图、损失曲线、敏感度分布图等，用于论文配图。

- **`configs/` - 配置文件**
    - **职责**：使用YAML文件管理所有超参数，实现**实验配置与代码分离**，保证可复现性。
    - **`base.yaml`**: 通用参数（如随机种子、工作线程数、日志设置）。
    - **`hybrid.yaml`**: 混合模型特有参数（如分支权重、谱归一化系数β、正则化强度λ）。
    - **`adversarial.yaml`**: 对抗训练参数（如攻击步长、强度ϵ）。

---

- **`utils/` - 工具脚本**
    - **职责**：提供辅助功能，包括数据加载、日志记录、模型检查点管理等，已完成，接口展示如下。
    
##### **`data_loader.py`**: 
- **作用**：创建和管理去噪数据集，支持多种噪声类型（高斯、泊松、脉冲、运动模糊）和混合噪声。
- **主要类/函数**：
    - `DenoisingDataset`: 自定义PyTorch数据集类，用于加载图像并添加指定噪声。
        - 初始化参数：`data_dir`, `noise_config`, `transform`, `mode`, `img_extensions`
        - 方法：`__len__`, `__getitem__`, `add_noise`
    - `create_data_loaders(config, dataset_config)`: 创建训练、验证和测试数据加载器。
        - 输入：主配置 `config`，数据集配置 `dataset_config`
        - 输出：`train_loader`, `val_loader`, `test_loader`
    - `get_dataset_stats(data_dir, img_extensions)`: 计算数据集的均值和标准差。
        - 输入：数据目录 `data_dir`，图像扩展名列表 `img_extensions`
        - 输出：均值 `mean`，标准差 `std`

##### **`logger.py`**:
- **作用**：设置日志记录系统和TensorBoard记录器，用于训练过程的可视化和日志记录。
- **主要函数**：
    - `setup_logger(log_dir, name=None, level=logging.INFO)`: 配置日志记录器，输出到文件和控制台。
        - 输入：日志目录 `log_dir`，日志器名称 `name`，日志级别 `level`
        - 输出：配置好的 `logging.Logger` 实例
    - `setup_tensorboard(log_dir, config=None)`: 设置TensorBoard记录器，可记录实验配置。
        - 输入：日志目录 `log_dir`，实验配置字典 `config`
        - 输出：`SummaryWriter` 实例

##### **`checkpoint.py`**:
- **作用**：处理模型检查点的保存、加载和评估，支持恢复训练和模型选择。
- **主要函数**：
    - `save_checkpoint(state, filename, is_best=False, best_filename='best_model.pth')`: 保存模型检查点。
        - 输入：状态字典 `state`，文件名 `filename`，是否最佳模型标志 `is_best`，最佳模型文件名 `best_filename`
    - `load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, logger=None)`: 加载检查点。
        - 输入：检查点路径 `checkpoint_path`，模型 `model`，优化器 `optimizer`，学习率调度器 `scheduler`，日志记录器 `logger`
        - 输出：起始轮次 `start_epoch`，最佳指标 `best_metric`
    - `find_latest_checkpoint(checkpoint_dir, pattern='checkpoint_epoch_*.pth')`: 查找最新检查点文件。
        - 输入：检查点目录 `checkpoint_dir`，文件名模式 `pattern`
        - 输出：最新检查点文件路径（如无则返回 `None`）
    - `evaluate_checkpoint(model, checkpoint_path, test_loader, device, logger=None)`: 评估检查点模型性能。
        - 输入：模型 `model`，检查点路径 `checkpoint_path`，测试数据加载器 `test_loader`，设备 `device`，日志记录器 `logger`
        - 输出：指标字典 `{'psnr': avg_psnr, 'ssim': avg_ssim}`

---

#### **3. 实验记录 (`experiments/`)**
- **职责**：存储所有实验的输出，与代码完全分离，避免混乱。
- **`logs/`**: 存储TensorBoard日志文件，用于可视化训练过程。
- **`checkpoints/`**: 存储训练好的模型权重。可按实验日期和名称组织子文件夹。
- **`results/`**: 存储评估脚本输出的结构化数据（如CSV、JSON文件），包含所有指标的详细结果。
- **`figures/`**: 存储可视化脚本生成的高质量图片，可直接用于论文写作。

#### **4. 数据目录 (`data/`)**
- **职责**：集中存放所有数据集。
- **组织方式**：按数据集类型和用途分门别类，便于管理。你需要将下载的数据集放入对应文件夹。
    - `basic_gray/`: BSD68, Set12 等经典灰度基准数据集。
    - `hi-quality_cc/`: Kodak24, McMaster 等高质量彩色数据集。
    - `real_noise/`: RNI6, RNI15 等真实噪声数据集。
    - `cset9/`: CSet9 等特定场景数据集。

#### **5. 运行脚本 (`scripts/`)**
- **职责**：提供顶层入口脚本，通过命令行参数调用核心包中的功能。
- **`train_hybrid.py`**: 主训练脚本。读取`configs/hybrid.yaml`，启动训练。
- **`eval_robustness.py`**: 主评估脚本。加载指定模型，运行鲁棒性测试套件。
- **`estimate_lipschitz.py`**: 专门用于计算指定模型的Lipschitz常数上/下界。
- **`generate_noise.py`**: 为干净数据集添加噪声，构建训练/测试对。

#### **6. 文档 (`docs/`)**
- **职责**：辅助你进行研究和论文写作。
- **`proposal.md`**: 你的开题报告的精简版，突出核心思想。
- **`theory_notes.md`**: 你的学习笔记，记录泛函分析、Lipschitz理论等与项目相关的数学推导和心得。
- **`references/`**: 存放你收集的重要论文的PDF文件。
