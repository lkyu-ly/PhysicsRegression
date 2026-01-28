# PaddlePaddle 迁移指南

> **项目**: PhysicsRegression PyTorch → PaddlePaddle 框架迁移
> **迁移工具**: PaConvert (百度自动转换工具)
> **迁移日期**: 2026年
> **文档版本**: 1.0

---

## 📋 目录

- [迁移概览](#迁移概览)
- [核心API变化](#核心api变化)
- [paddle_utils.py 兼容层](#paddle_utilspy-兼容层)
- [关键代码对比](#关键代码对比)
- [设备管理变化](#设备管理变化)
- [模型文件格式](#模型文件格式)
- [特殊处理说明](#特殊处理说明)
- [迁移检查清单](#迁移检查清单)
- [已知问题](#已知问题)

---

## 迁移概览

### 迁移方法

本项目使用 **PaConvert** (百度官方工具) 进行自动代码转换:

```bash
# 迁移命令 (已完成)
paconvert --in_dir ./PhysicsRegression --out_dir ./PhysicsRegressionPaddle
```

### 迁移状态

| 组件 | 迁移状态 | 自动转换率 | 备注 |
|------|---------|-----------|------|
| **符号回归模块** | ✅ 完成 | ~95% | Transformer, Embedders, Environment |
| **Oracle模块** | ✅ 完成 | ~98% | SimpleNet网络, Oracle训练 |
| **训练脚本** | ✅ 完成 | ~90% | train.py, trainer.py |
| **评估脚本** | ✅ 完成 | ~90% | evaluate.py |
| **工具函数** | ✅ 完成 | ~95% | utils.py, metrics.py |
| **兼容层** | ✅ 自动生成 | 100% | paddle_utils.py |

### 文件结构对比

```
PhysicsRegression/              PhysicsRegressionPaddle/
├── *.py (PyTorch代码)         ├── *.py (PaddlePaddle代码)
├── symbolicregression/         ├── symbolicregression/
├── Oracle/                     ├── Oracle/
├── physical/                   ├── physical/
├── model.pt (PyTorch模型)     ├── model.pdparams (需转换)
└── CLAUDE.md                   ├── paddle_utils.py (新增兼容层)
                                └── CLAUDE.md (需更新)
```

---

## 核心API变化

### 模块导入变化

```python
# PyTorch → PaddlePaddle

# ❌ PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# ✅ PaddlePaddle
import paddle
import paddle.nn as nn
from paddle.io import DataLoader
from paddle_utils import *  # 导入兼容层
```

### 神经网络模块

#### 基础类继承

```python
# PyTorch → PaddlePaddle

# ❌ PyTorch
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

# ✅ PaddlePaddle
class MyModel(paddle.nn.Module):  # 或 paddle.nn.Layer
    def __init__(self):
        super().__init__()
```

#### 线性层

```python
# PyTorch → PaddlePaddle

# ❌ PyTorch
self.fc = torch.nn.Linear(128, 64)

# ✅ PaddlePaddle (兼容命名空间)
self.fc = paddle.compat.nn.Linear(128, 64)

# ⚠️ 注意: 使用 paddle.compat.nn.Linear 而非 paddle.nn.Linear
# 这是PaConvert工具的处理方式,确保API兼容性
```

#### 激活函数

```python
# PyTorch → PaddlePaddle

# ❌ PyTorch
x = torch.tanh(x)
x = torch.nn.functional.relu(x)

# ✅ PaddlePaddle
x = paddle.tanh(x)
x = paddle.nn.functional.relu(x)
```

#### 嵌入层

```python
# PyTorch → PaddlePaddle

# ❌ PyTorch
self.embed = torch.nn.Embedding(vocab_size, emb_dim, padding_idx=0)

# ✅ PaddlePaddle
self.embed = paddle.nn.Embedding(vocab_size, emb_dim, padding_idx=0)
```

#### 容器模块

```python
# PyTorch → PaddlePaddle

# ❌ PyTorch
self.layers = torch.nn.ModuleList([...])

# ✅ PaddlePaddle
self.layers = paddle.nn.ModuleList([...])
```

### 张量操作

#### 张量创建

```python
# PyTorch → PaddlePaddle

# ❌ PyTorch
x = torch.tensor([1, 2, 3])
x = torch.zeros(3, 4)
x = torch.FloatTensor([1.0, 2.0])

# ✅ PaddlePaddle
x = paddle.to_tensor([1, 2, 3])  # 注意: paddle.tensor也可用
x = paddle.zeros([3, 4])
x = paddle.FloatTensor([1.0, 2.0])
```

#### 数据类型

```python
# PyTorch → PaddlePaddle

# ❌ PyTorch
x = x.long()
x = x.float()

# ✅ PaddlePaddle
x = x.astype(paddle.long)  # 或 paddle.int64
x = x.astype(paddle.float32)
```

#### 张量方法差异

```python
# PyTorch → PaddlePaddle

# ❌ PyTorch: 使用 dim 参数
x.max(dim=1)
x.sum(dim=0)

# ✅ PaddlePaddle: 使用 axis 参数
x.max(axis=1)  # 或使用 paddle_utils 中的兼容方法
x.sum(axis=0)
```

### 优化器

```python
# PyTorch → PaddlePaddle

# ❌ PyTorch
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# ✅ PaddlePaddle
optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=0.001)
optimizer = paddle.optimizer.SGD(parameters=model.parameters(), learning_rate=0.01, momentum=0.9)
```

### 损失函数

```python
# PyTorch → PaddlePaddle

# ❌ PyTorch
loss_fn = torch.nn.MSELoss()
loss = torch.nn.functional.cross_entropy(pred, target)

# ✅ PaddlePaddle
loss_fn = paddle.nn.MSELoss()
loss = paddle.nn.functional.cross_entropy(pred, target)
```

### 数据加载

```python
# PyTorch → PaddlePaddle

# ❌ PyTorch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    pass

loader = DataLoader(dataset, batch_size=32, shuffle=True)

# ✅ PaddlePaddle
from paddle.io import Dataset, DataLoader

class MyDataset(Dataset):
    pass

loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

---

## paddle_utils.py 兼容层

PaConvert 自动生成的兼容层文件,用于处理PyTorch和PaddlePaddle的API差异。

### 文件位置

```
PhysicsRegressionPaddle/
└── paddle_utils.py  # 项目根目录
```

### 核心功能

#### 1. 设备字符串转换

**功能**: 将PyTorch的设备字符串转换为PaddlePaddle格式

```python
def device2int(device):
    """
    转换设备字符串格式

    示例:
        'cuda:0' → 'gpu:0' → 0
        'cuda:1' → 'gpu:1' → 1
    """
    if isinstance(device, str):
        print("Converting device string to int:", device)
        device = device.replace('cuda', 'gpu')
        device = device.replace('gpu:', '')
    return int(device)
```

**使用场景**:
```python
# PyTorch代码: device = 'cuda:0'
# PaddlePaddle转换: device = device2int('cuda:0')  # 返回 0
```

#### 2. Tensor.max() 方法适配

**功能**: 处理 `dim`/`axis` 参数差异

```python
def _Tensor_max(self, *args, **kwargs):
    """
    适配 Tensor.max() 方法

    处理:
    1. PyTorch: tensor.max(dim=1)
    2. PaddlePaddle: tensor.max(axis=1)
    3. 返回值差异: (values, indices)
    """
    if "other" in kwargs:
        kwargs["y"] = kwargs.pop("other")
        ret = paddle.maximum(self, *args, **kwargs)
    elif len(args) == 1 and isinstance(args[0], paddle.Tensor):
        ret = paddle.maximum(self, *args, **kwargs)
    else:
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")  # ← 关键转换

        if "axis" in kwargs or len(args) >= 1:
            ret = paddle.max(self, *args, **kwargs), paddle.argmax(self, *args, **kwargs)
        else:
            ret = paddle.max(self, *args, **kwargs)

    return ret

# 将方法绑定到 Tensor 类
setattr(paddle.Tensor, "_max", _Tensor_max)
```

**使用示例**:
```python
import paddle
from paddle_utils import *

x = paddle.randn([3, 4])

# PyTorch风格 (通过兼容层自动处理)
max_val, max_idx = x._max(dim=1)

# 等价于PaddlePaddle原生写法:
max_val = x.max(axis=1)
max_idx = x.argmax(axis=1)
```

### 使用方法

在每个需要兼容处理的模块顶部添加:

```python
import paddle
from paddle_utils import *
```

**注意事项**:
- ⚠️ `paddle_utils.py` 必须位于Python导入路径中
- ⚠️ 导入顺序: 先 `import paddle`,再 `from paddle_utils import *`
- ⚠️ 某些项目文件通过 `sys.path.append` 添加项目根目录到路径

---

## 关键代码对比

### 示例 1: Transformer MultiHeadAttention

**文件**: `symbolicregression/model/transformer.py:54-74`

```python
# ===== PyTorch版本 =====
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, dim, src_dim, dropout, normalized_attention):
        super().__init__()
        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(src_dim, dim)
        self.v_lin = nn.Linear(src_dim, dim)
        self.out_lin = nn.Linear(dim, dim)
        if self.normalized_attention:
            self.attention_scale = nn.Parameter(
                torch.tensor(1.0 / math.sqrt(dim // n_heads))
            )

# ===== PaddlePaddle版本 =====
import paddle
from paddle_utils import *

class MultiHeadAttention(paddle.nn.Module):
    def __init__(self, n_heads, dim, src_dim, dropout, normalized_attention):
        super().__init__()
        self.q_lin = paddle.compat.nn.Linear(dim, dim)           # ← 使用 compat
        self.k_lin = paddle.compat.nn.Linear(src_dim, dim)       # ← 使用 compat
        self.v_lin = paddle.compat.nn.Linear(src_dim, dim)       # ← 使用 compat
        self.out_lin = paddle.compat.nn.Linear(dim, dim)         # ← 使用 compat
        if self.normalized_attention:
            self.attention_scale = paddle.nn.Parameter(
                paddle.tensor(1.0 / math.sqrt(dim // n_heads))  # ← paddle.tensor
            )
```

**变化要点**:
1. 导入: `torch` → `paddle`
2. 类继承: `nn.Module` → `paddle.nn.Module`
3. 线性层: `nn.Linear` → `paddle.compat.nn.Linear`
4. 参数: `torch.tensor` → `paddle.tensor`

---

### 示例 2: Oracle SimpleNet

**文件**: `Oracle/oracle.py:20-35`

```python
# ===== PyTorch版本 =====
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, _in):
        super().__init__()
        self.linear1 = nn.Linear(_in, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, 64)
        self.linear5 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        x = torch.tanh(self.linear4(x))
        x = self.linear5(x)
        return x

# ===== PaddlePaddle版本 =====
import paddle

class SimpleNet(paddle.nn.Module):
    def __init__(self, _in):
        super().__init__()
        self.linear1 = paddle.compat.nn.Linear(_in, 128)
        self.linear2 = paddle.compat.nn.Linear(128, 128)
        self.linear3 = paddle.compat.nn.Linear(128, 64)
        self.linear4 = paddle.compat.nn.Linear(64, 64)
        self.linear5 = paddle.compat.nn.Linear(64, 1)

    def forward(self, x):
        x = paddle.tanh(self.linear1(x))      # ← paddle.tanh
        x = paddle.tanh(self.linear2(x))
        x = paddle.tanh(self.linear3(x))
        x = paddle.tanh(self.linear4(x))
        x = self.linear5(x)
        return x
```

---

### 示例 3: LinearPointEmbedder

**文件**: `symbolicregression/model/embedders.py:45-73`

```python
# ===== PyTorch版本 =====
import torch
import torch.nn as nn

class LinearPointEmbedder(Embedder):
    def __init__(self, params, env):
        super().__init__()
        self.embeddings = Embedding(
            len(self.env.float_id2word),
            self.input_dim,
            padding_idx=env.float_word2id["<PAD>"],
        )
        self.activation_fn = torch.nn.functional.relu
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(size, hidden_size))
        for i in range(self.params.n_emb_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        self.fc = nn.Linear(hidden_size, self.output_dim)

# ===== PaddlePaddle版本 =====
import paddle
from paddle_utils import *

class LinearPointEmbedder(Embedder):
    def __init__(self, params, env):
        super().__init__()
        self.embeddings = Embedding(
            len(self.env.float_id2word),
            self.input_dim,
            padding_idx=env.float_word2id["<PAD>"],
        )
        self.activation_fn = paddle.nn.functional.relu  # ← paddle
        self.hidden_layers = paddle.nn.ModuleList()     # ← paddle
        self.hidden_layers.append(paddle.compat.nn.Linear(size, hidden_size))
        for i in range(self.params.n_emb_layers - 1):
            self.hidden_layers.append(paddle.compat.nn.Linear(hidden_size, hidden_size))
        self.fc = paddle.compat.nn.Linear(hidden_size, self.output_dim)
```

---

### 示例 4: 训练循环

```python
# ===== PyTorch版本 =====
import torch

def train_step(model, optimizer, x, y):
    model.train()

    # 前向传播
    pred = model(x)
    loss = torch.nn.functional.mse_loss(pred, y)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

# ===== PaddlePaddle版本 =====
import paddle

def train_step(model, optimizer, x, y):
    model.train()

    # 前向传播
    pred = model(x)
    loss = paddle.nn.functional.mse_loss(pred, y)

    # 反向传播
    optimizer.clear_grad()  # ← clear_grad 而非 zero_grad
    loss.backward()
    optimizer.step()

    return loss.item()
```

**关键差异**:
- `optimizer.zero_grad()` → `optimizer.clear_grad()`

---

## 设备管理变化

### 设备字符串格式

```python
# PyTorch → PaddlePaddle

# ❌ PyTorch
device = 'cuda:0'
device = 'cuda:1'
device = 'cpu'

# ✅ PaddlePaddle
device = 'gpu:0'   # 或使用 device2int() 转换为整数 0
device = 'gpu:1'   # 或整数 1
device = 'cpu'
```

### 模型移动到设备

```python
# PyTorch → PaddlePaddle

# ❌ PyTorch
model = model.to('cuda:0')
x = x.to('cuda:0')

# ✅ PaddlePaddle (方式1: 字符串)
model = model.to('gpu:0')
x = x.to('gpu:0')

# ✅ PaddlePaddle (方式2: 设备对象)
device = paddle.CUDAPlace(0)
model = model.to(device)
x = x.to(device)
```

### 检查GPU可用性

```python
# PyTorch → PaddlePaddle

# ❌ PyTorch
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

# ✅ PaddlePaddle
if paddle.is_compiled_with_cuda():
    device = 'gpu:0'
else:
    device = 'cpu'
```

---

## 模型文件格式

### 模型保存

```python
# PyTorch → PaddlePaddle

# ❌ PyTorch: 保存为 .pt 或 .pth
torch.save(model.state_dict(), 'model.pt')

# ✅ PaddlePaddle: 保存为 .pdparams
paddle.save(model.state_dict(), 'model.pdparams')
```

### 模型加载

```python
# PyTorch → PaddlePaddle

# ❌ PyTorch
state_dict = torch.load('model.pt', map_location='cpu')
model.load_state_dict(state_dict)

# ✅ PaddlePaddle
state_dict = paddle.load('model.pdparams')
model.set_state_dict(state_dict)
```

### 模型转换

**从PyTorch迁移到PaddlePaddle时,模型文件需要重新训练或使用转换工具**:

```python
# 方法1: 重新训练 (推荐)
# 使用 train.py 重新训练模型

# 方法2: 手动转换权重 (复杂,仅当必要时)
# 需要编写自定义转换脚本匹配网络结构
```

**注意**: 由于架构差异,直接转换`.pt`到`.pdparams`可能不可行,建议重新训练。

---

## 特殊处理说明

### paddle.compat.nn.Linear

**为什么使用 `paddle.compat.nn.Linear`?**

PaConvert工具使用 `paddle.compat.nn.Linear` 确保API兼容性:

```python
# PyTorch原始代码
fc = torch.nn.Linear(128, 64)

# PaConvert转换后
fc = paddle.compat.nn.Linear(128, 64)

# 而非直接使用
fc = paddle.nn.Linear(128, 64)  # ← 可能存在细微差异
```

**兼容命名空间位置**:
- 文件: `symbolicregression/model/transformer.py`
- 文件: `symbolicregression/model/embedders.py`
- 文件: `Oracle/oracle.py`

**是否可以改为 `paddle.nn.Linear`?**

理论上可以,但需要验证以下内容:
1. 权重初始化方法是否一致
2. bias处理是否相同
3. 前向传播数值精度

### sys.path.append

多个文件包含以下代码以确保导入路径正确:

```python
import sys
sys.path.append("/home/lkyu/baidu/PhysicsRegressionPaddle")
```

**位置**:
- `symbolicregression/model/transformer.py:1-2`
- `symbolicregression/model/embedders.py:1-2`

**作用**: 确保 `paddle_utils.py` 可以被正确导入

**注意**: 如果项目路径改变,需要更新这些路径

---

## 迁移检查清单

### 代码层面

- [x] 所有 `import torch` 已替换为 `import paddle`
- [x] 所有 `torch.nn.Module` 已替换为 `paddle.nn.Module`
- [x] 所有 `torch.nn.Linear` 已替换为 `paddle.compat.nn.Linear`
- [x] 所有 `torch.optim.Adam` 已替换为 `paddle.optimizer.Adam`
- [x] 所有激活函数已更新 (`torch.tanh` → `paddle.tanh`)
- [x] 优化器调用已更新 (`zero_grad()` → `clear_grad()`)
- [x] 设备字符串已更新 (`cuda:0` → `gpu:0`)
- [x] 张量方法已更新 (`dim` → `axis`)
- [x] `paddle_utils.py` 已正确导入

### 功能验证

- [ ] **测试训练流程**: 运行 `train.py` 确认无错误
- [ ] **测试评估流程**: 运行 `evaluate.py` 验证模型推理
- [ ] **测试Oracle模块**: 验证分治策略正常工作
- [ ] **测试MCTS/GP**: 确认优化算法可用
- [ ] **对比数值精度**: PyTorch vs PaddlePaddle 输出差异 < 1e-5
- [ ] **GPU内存测试**: 确认显存使用合理
- [ ] **多卡训练**: 测试分布式训练功能

### 文档更新

- [ ] 更新根目录 `CLAUDE.md`
- [ ] 更新 `symbolicregression/CLAUDE.md`
- [ ] 更新 `Oracle/CLAUDE.md`
- [ ] 更新 `physical/CLAUDE.md`
- [x] 创建 `PADDLE_MIGRATION.md` (本文档)

### 环境配置

- [ ] 创建PaddlePaddle版本的 `environment.yml`
- [ ] 更新 `README.md` 安装说明
- [ ] 准备PaddlePaddle版本的预训练模型

---

## 已知问题

### 问题 1: 模型格式不兼容

**描述**: PyTorch的 `.pt` 模型文件无法直接用于PaddlePaddle

**解决方案**:
1. 使用相同数据重新训练模型
2. 或编写自定义转换脚本 (需要深入理解网络结构)

### 问题 2: paddle.compat 命名空间

**描述**: 代码中使用 `paddle.compat.nn.Linear` 可能让人困惑

**说明**:
- 这是PaConvert工具的标准做法
- 确保API兼容性
- 不影响功能

### 问题 3: 数值精度差异

**描述**: PaddlePaddle和PyTorch在某些操作上可能有细微数值差异

**验证方法**:
```python
import paddle
import torch
import numpy as np

# 相同输入
x_np = np.random.randn(4, 128).astype('float32')

# PyTorch
x_torch = torch.from_numpy(x_np)
out_torch = torch_model(x_torch).detach().numpy()

# PaddlePaddle
x_paddle = paddle.to_tensor(x_np)
out_paddle = paddle_model(x_paddle).numpy()

# 对比
diff = np.abs(out_torch - out_paddle).max()
print(f"最大差异: {diff}")  # 应该 < 1e-5
```

### 问题 4: 硬编码路径

**描述**: 部分文件包含硬编码的绝对路径

**位置**:
```python
sys.path.append("/home/lkyu/baidu/PhysicsRegressionPaddle")
```

**解决方案**: 使用相对路径或环境变量

### 问题 5: 优化器基类初始化签名不兼容 ⚠️

**描述**: PaConvert **无法自动处理** PyTorch 和 PaddlePaddle 优化器基类的构造函数签名差异

**影响文件**: `symbolicregression/optim.py`

**问题根源**:

| 框架 | 优化器基类签名 |
|------|--------------|
| **PyTorch** | `__init__(self, params, defaults)` |
| **PaddlePaddle** | `__init__(self, learning_rate, parameters, weight_decay, ...)` |

**错误代码示例**:
```python
# ❌ 错误: PaConvert自动转换后的代码
class Adam(paddle.optimizer.Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)  # ← 错误: params 被传给了 learning_rate
```

**错误信息**:
```
TypeError: `parameters` argument should not get dict type, if parameter groups is needed,
please set `parameters` as list of dict
```

**手动修复** (已完成):
```python
# ✅ 正确: 使用命名参数调用父类
class Adam(paddle.optimizer.Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0):
        # 参数验证...

        super().__init__(
            learning_rate=lr,      # 明确指定学习率
            parameters=params,     # 明确指定参数列表
            weight_decay=weight_decay if weight_decay != 0 else None
        )

        # 状态初始化...
```

**修复位置**:
- `Adam` (第25行)
- `AdamWithWarmup` (第94-101行)
- `AdamInverseSqrtWithWarmup` (第149-156行)
- `AdamCosineWithWarmup` (第211-218行)

**为什么 PaConvert 无法自动处理**:
1. 参数位置完全不同 (第1个参数: `params` vs `learning_rate`)
2. 参数名称不同 (`params` vs `parameters`)
3. PaddlePaddle 不使用 `defaults` 字典模式
4. 需要根据语义重新映射，超出工具能力

**最佳实践**:
- 迁移后务必测试优化器初始化
- 保持 PyTorch 版本不变（标准实现）
- 在 PaddlePaddle 版本中手动修复

---

### 问题 6: tensor.cuda(device=) 参数不兼容 ⚠️

**描述**: PaddlePaddle 的 `tensor.cuda()` 不接受 `device` 参数，这是与PyTorch的关键差异

**影响文件**:
- `symbolicregression/utils.py` (to_cuda 函数，第140-152行)

**错误信息**:
```
TypeError: monkey_patch_tensor.<locals>.cuda() got an unexpected keyword argument 'device'
```

**根本原因**:

| API类型 | PyTorch | PaddlePaddle |
|---------|---------|--------------|
| **Tensor.cuda()** | `tensor.cuda(device=0)` ✅ 接受device参数 | `tensor.cuda()` ❌ 不接受任何参数 |
| **Module.cuda()** | `module.cuda(device=0)` ✅ 接受device参数 | `module.cuda(device=device_id)` ✅ 接受device参数 |

**关键发现**:
- Module和Tensor的cuda()方法行为不同
- PaddlePaddle的Tensor.cuda()完全不接受参数
- 官方文档说明不准确（文档说有device_id参数，实际不存在）

**手动修复** (已完成):

```python
# ❌ 修复前 (PyTorch风格)
def to_cuda(*args, use_cpu=False, device=None):
    if not CUDA or use_cpu:
        return args
    if device is None:
        device = 0
    return [(None if x is None else x.cuda(device=device)) for x in args]
    #                                       ^^^^^^^^^^^^^ 错误！

# ✅ 修复后 (方案B: 全局设备 + 无参数.cuda())
def to_cuda(*args, use_cpu=False, device=None):
    """
    Move tensors to CUDA (PaddlePaddle version).

    Note: PaddlePaddle's Tensor.cuda() does not accept any parameters.
    We set global device first, then call parameter-less .cuda()
    """
    if not CUDA or use_cpu:
        return args

    # 设置全局默认设备 (如果指定了device)
    if device is not None:
        import paddle
        from paddle_utils import device2int

        if isinstance(device, str):
            device = device2int(device)

        # 设置全局默认GPU设备
        paddle.device.set_device(f'gpu:{device}')

    # 调用无参数的 .cuda() 方法
    return [
        (None if x is None else x.cuda())
        for x in args
    ]
```

**修复策略选择**:
- 方案A: 使用`paddle.to_device() + CUDAPlace()`
- **方案B**: 使用`paddle.device.set_device() + 无参数.cuda()` ← 已采用
- 方案C: 检查张量设备 + 条件移动

选择方案B的原因：
1. 与源代码最相似
2. 实现简单，易于维护
3. 与Module.cuda()的使用方式一致
4. 适用于单GPU场景（项目主要场景）

**为什么 PaConvert 无法自动处理**:
1. 需要区分Module.cuda()和Tensor.cuda()的不同行为
2. 需要插入全局设备设置逻辑
3. 需要理解device参数的语义转换
4. 超出简单API映射范围

**调用位置** (无需修改):
- `symbolicregression/model/embedders.py:101-106`
- `symbolicregression/trainer.py:666, 669`

这些调用位置无需修改，因为to_cuda的接口保持不变。

**最佳实践**:
- 对于Module: 可以使用`.cuda(device=device_id)`
- 对于Tensor: 必须先`set_device()`再调用无参数`.cuda()`
- 建议统一使用`paddle.to_device(tensor, place)`显式指定设备

---

### 问题 7: tensor.new() 方法不存在 ⚠️

**描述**: PaddlePaddle 的 Tensor 没有 `.new()` 方法，这是PyTorch独有的便捷创建张量的方法

**影响文件**:
- `symbolicregression/model/transformer.py` (15处调用)

**错误信息**:
```
AttributeError: 'Tensor' object has no attribute 'new'. Did you mean: 'ne'?
```

**根本原因**:

| 功能 | PyTorch | PaddlePaddle |
|------|---------|--------------|
| **创建同设备张量** | `tensor.new(size)` | 不存在此方法 |
| **创建同类型张量** | `tensor.new([1,2,3])` | 不存在此方法 |
| **便捷方法** | `tensor.new(5).long()` | 需要显式使用paddle API |

**手动修复** (已完成):

修复了transformer.py中所有15处`.new()`调用：

```python
# ❌ 修复前 (PyTorch风格)
positions = x.new(slen).long()
positions = paddle.arange(slen, out=positions).unsqueeze(0)

# ✅ 修复后 (PaddlePaddle风格)
positions = paddle.arange(slen, dtype='int64').unsqueeze(0)
```

**修复模式总结**:

| PyTorch模式 | PaddlePaddle替代 | 说明 |
|-------------|-----------------|------|
| `x.new(size).fill_(val)` | `paddle.full([size], val, dtype=x.dtype)` | 创建填充张量 |
| `x.new(size).long()` | `paddle.arange(size, dtype='int64')` | 创建整数序列 |
| `x.new([list])` | `paddle.to_tensor([list], dtype=x.dtype)` | 从列表创建 |
| `x.new(size).float().fill_(0)` | `paddle.zeros([size], dtype='float32')` | 创建零张量 |

**详细修复位置** (共15处):

1. **第399行** - `fwd()`方法中的位置张量:
```python
# 修复前:
positions = x.new(slen).long()
positions = paddle.arange(slen, out=positions).unsqueeze(0)

# 修复后:
positions = paddle.arange(slen, dtype='int64').unsqueeze(0)
```

2. **第516-520行** - `generate()`方法中的生成张量:
```python
# 修复前:
generated = src_len.new(max_len, bs)
generated.fill_(self.pad_index)
positions = src_len.new(max_len).long()

# 修复后:
generated = paddle.full([max_len, bs], self.pad_index, dtype=src_len.dtype)
generated[0].fill_(self.eos_index)
positions = paddle.arange(max_len, dtype='int64').unsqueeze(1).expand([max_len, bs])
```

3. **第578-584行** - `generate_double_seq()`方法:
```python
# 修复前:
generated1 = src_len.new(max_len, bs)
generated2 = src_len.new(max_len, bs, 5)

# 修复后:
generated1 = paddle.full([max_len, bs], self.pad_index, dtype=src_len.dtype)
generated2 = paddle.full([max_len, bs, 5], self.pad_index, dtype=src_len.dtype)
```

4. **第758-769行** - `generate_beam()`方法的束搜索初始化:
```python
# 修复前:
generated = src_len.new(max_len, bs * beam_size)
beam_scores = src_enc.new(bs, beam_size).float().fill_(0)

# 修复后:
generated = paddle.full([max_len, bs * beam_size], self.pad_index, dtype=src_len.dtype)
beam_scores = paddle.full([bs, beam_size], 0.0, dtype='float32')
beam_scores[:, 1:] = -1000000000.0
```

5. **第778行** - 束搜索循环中的长度张量:
```python
# 修复前:
lengths=src_len.new(bs * beam_size).fill_(cur_len)

# 修复后:
lengths=paddle.full([bs * beam_size], cur_len, dtype=src_len.dtype)
```

6. **第830-833行** - 从列表创建束搜索跟踪张量:
```python
# 修复前:
beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
beam_words = generated.new([x[1] for x in next_batch_beam])
beam_idx = src_len.new([x[2] for x in next_batch_beam])

# 修复后:
beam_scores = paddle.to_tensor([x[0] for x in next_batch_beam], dtype='float32')
beam_words = paddle.to_tensor([x[1] for x in next_batch_beam], dtype=generated.dtype)
beam_idx = paddle.to_tensor([x[2] for x in next_batch_beam], dtype=src_len.dtype)
```

7. **第845-853行** - 最终解码结果:
```python
# 修复前:
tgt_len = src_len.new(bs)
decoded = src_len.new(tgt_len._max().item(), bs).fill_(self.pad_index)

# 修复后:
tgt_len = paddle.zeros([bs], dtype=src_len.dtype)
# ... 填充tgt_len ...
decoded = paddle.full([int(tgt_len._max().item()), bs], self.pad_index, dtype=src_len.dtype)
```

**为什么 PaConvert 无法自动处理**:
1. `.new()`是PyTorch的便捷方法，没有直接对应的PaddlePaddle API
2. 需要根据使用场景选择不同的替代方法（full/zeros/arange/to_tensor）
3. 需要保持dtype一致性，要从原张量推断类型
4. 涉及复杂的方法链（如`.new().long().fill_()`）需要语义理解
5. 超出简单API映射的能力范围

**最佳实践**:
- 使用`paddle.full()`创建填充张量
- 使用`paddle.zeros()`/`paddle.ones()`创建零/一张量
- 使用`paddle.arange()`创建序列
- 使用`paddle.to_tensor()`从Python列表创建
- 始终显式指定`dtype`确保类型一致

**验证结果**: ✅ 训练成功运行120+步，所有.new()调用已正确替换

---

### 问题 8: model.parameters() 返回类型差异 ⚠️

**描述**: PaddlePaddle 的 `model.parameters()` 返回 list 而非 generator，以及相关的类型提升问题

**影响文件**:
- `symbolicregression/model/model_wrapper.py` (第40行)
- `symbolicregression/model/__init__.py` (第66行)
- `Oracle/oracle.py` (第179行)
- `symbolicregression/model/transformer.py` (多处类型提升)

**错误信息**:
```
TypeError: 'list' object is not an iterator
```

**根本原因**:

| API类型 | PyTorch | PaddlePaddle |
|---------|---------|--------------|
| **model.parameters()** | 返回 **generator** | 返回 **list** |
| **named_parameters()** | 返回 **generator** | 返回 **list** |
| **next(model.parameters())** | ✅ 可行 | ❌ TypeError |
| **iter(model.parameters())** | ✅ 返回generator本身 | ✅ 创建list_iterator |

#### 子问题 8.1: parameters() 迭代器问题

**错误位置**: `model_wrapper.py:40`

**手动修复** (已完成):
```python
# ❌ 修复前
class ModelWrapper:
    def __init__(self, ...):
        self.device = next(self.embedder.parameters()).device  # ← 错误！

# ✅ 修复后
class ModelWrapper:
    def __init__(self, ...):
        # PaddlePaddle: parameters() 返回list，需要用iter()包装
        self.device = next(iter(self.embedder.parameters())).device
```

**为什么这样修复**:
- `iter(list)` 创建 list_iterator，开销极小
- 在 PyTorch 中，`iter(generator)` 返回 generator 本身，无额外开销
- 代码兼容两个框架

#### 子问题 8.2: 参数统计方法差异

**错误位置**: `model/__init__.py:66`

**手动修复** (已完成):
```python
# ❌ 修复前
f"Number of parameters ({k}): {sum([p.size for p in v.parameters() if p.requires_grad])}"

# ✅ 修复后
f"Number of parameters ({k}): {sum([p.numel() for p in v.parameters() if p.requires_grad])}"
```

**说明**: `.numel()` (number of elements) 在两个框架中都存在且语义一致

#### 子问题 8.3: 优化器参数传递

**错误位置**: `Oracle/oracle.py:179`

**手动修复** (已完成):
```python
# ❌ 修复前
optimizer = paddle.optimizer.Adam(
    parameters=model.parameters(), ...
)

# ✅ 修复后
optimizer = paddle.optimizer.Adam(
    parameters=list(model.parameters()), ...
)
```

#### 子问题 8.4: 类型提升问题 - float × int

**根本原因**: PaddlePaddle 不允许 float32 和 int64 之间的隐式类型提升

**错误信息**:
```
TypeError: (InvalidType) Type promotion only support calculations between floating-point numbers
and between complex and real numbers. But got different data type x: float32, y: int64.
```

**影响位置**:
- `transformer.py:561, 705, 708` - `paddle.log(perplexity) * unfinished_sents`

**手动修复** (已完成):
```python
# ❌ 修复前
word_perplexity.add_(
    paddle.log(next_words_perplexity.detach()) * unfinished_sents  # int64
)

# ✅ 修复后
word_perplexity.add_(
    # PaddlePaddle: 显式类型转换 float32 * int64 -> float32
    paddle.log(next_words_perplexity.detach()) * unfinished_sents.astype('float32')
)
```

#### 子问题 8.5: .ne() 方法参数类型

**根本原因**: PaddlePaddle 的 `.ne()` 方法要求参数必须是 Tensor

**错误信息**:
```
ValueError: not_equal(): argument 'y' (position 1) must be Tensor, but got int
```

**影响位置**:
- `transformer.py:565, 714` - `next_words.ne(self.eos_index)`

**手动修复** (已完成):
```python
# ❌ 修复前
unfinished_sents.mul_(next_words.ne(self.eos_index).long())

# ✅ 修复后
# PaddlePaddle: .ne() 需要tensor参数，改用 != 运算符
unfinished_sents.mul_((next_words != self.eos_index).astype('int64'))
```

**为什么使用 `!=`**:
- `!=` 运算符在 PaddlePaddle 中可以处理标量
- 更简洁，避免创建不必要的 tensor

#### 子问题 8.6: 除法类型提升

**影响位置**:
- `transformer.py:575, 726, 727` - `word_perplexity / rows`

**手动修复** (已完成):
```python
# ❌ 修复前
rows, cols = paddle.nonzero(generated[1:] == self.eos_index, as_tuple=True)
word_perplexity = paddle.exp(word_perplexity / rows)  # rows 是 int64

# ✅ 修复后
rows, cols = paddle.nonzero(generated[1:] == self.eos_index, as_tuple=True)
# PaddlePaddle: 显式转换 int64 -> float32
word_perplexity = paddle.exp(word_perplexity / rows.astype('float32'))
```

**修复总结**:

| 文件 | 修复点 | 类型 | 数量 |
|------|--------|------|------|
| `model_wrapper.py` | parameters() 迭代 | 迭代器 | 1 |
| `model/__init__.py` | 参数统计方法 | API差异 | 1 |
| `Oracle/oracle.py` | 优化器参数 | 显式list | 1 |
| `transformer.py` | float × int 乘法 | 类型转换 | 3 |
| `transformer.py` | .ne() 方法调用 | API差异 | 2 |
| `transformer.py` | float / int 除法 | 类型转换 | 3 |
| **总计** | | | **11处** |

**为什么 PaConvert 无法自动处理**:
1. 需要识别 `next(model.parameters())` 模式并自动插入 `iter()`
2. 需要理解返回值类型差异（generator vs list）
3. 需要检测所有潜在的类型提升位置
4. 需要理解方法调用语义（`.ne()` 参数要求）
5. 超出简单API映射的能力范围

**最佳实践**:
- 使用 `next(iter(model.parameters()))` 兼容两个框架
- 参数统计使用 `.numel()` 标准方法
- 优化器初始化显式使用 `list(model.parameters())`
- **关键**: PaddlePaddle 中所有 float 和 int 的混合运算都需要显式类型转换
- 使用 `!=` 运算符代替 `.ne()` 方法更简洁
- 除法运算前确保两边类型一致

**验证结果**: ✅ 完整训练-验证循环成功运行（500步训练 + 5样本验证）

---

## 参考资源

### PaddlePaddle 官方文档

- **API映射表**: https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/model_convert/pytorch_api_mapping_cn.html
- **迁移指南**: https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/model_convert/convert_from_pytorch/pytorch_migration_cn.html
- **PaConvert工具**: https://github.com/PaddlePaddle/PaConvert

### 项目相关

- **原项目论文**: Ying et al., Nature Machine Intelligence (2025)
- **GitHub**: PhysicsRegression (原PyTorch版本)
- **Google Drive**: 预训练模型和数据集

---

**最后更新**: 2026-01-28
**维护者**: 迁移项目团队
**问题反馈**: 请在项目Issue中报告迁移相关问题

---

## 附录: 完整API对照表

| 功能类别 | PyTorch | PaddlePaddle | 备注 |
|---------|---------|--------------|------|
| **模块导入** | `import torch` | `import paddle` | |
| **神经网络基类** | `torch.nn.Module` | `paddle.nn.Module` | 或 `paddle.nn.Layer` |
| **线性层** | `torch.nn.Linear` | `paddle.compat.nn.Linear` | ⚠️ 使用compat |
| **嵌入层** | `torch.nn.Embedding` | `paddle.nn.Embedding` | |
| **激活函数** | `torch.tanh` | `paddle.tanh` | |
| | `torch.nn.functional.relu` | `paddle.nn.functional.relu` | |
| **参数** | `torch.nn.Parameter` | `paddle.nn.Parameter` | |
| **容器** | `torch.nn.ModuleList` | `paddle.nn.ModuleList` | |
| **张量创建** | `torch.tensor` | `paddle.to_tensor` | 推荐用法 |
| | `torch.zeros` | `paddle.zeros` | |
| | `torch.FloatTensor` | `paddle.FloatTensor` | |
| **数据类型** | `.long()` | `.astype(paddle.int64)` | |
| **张量操作** | `.max(dim=1)` | `.max(axis=1)` | ⚠️ dim→axis |
| **优化器** | `torch.optim.Adam` | `paddle.optimizer.Adam` | 参数名不同 |
| | `.zero_grad()` | `.clear_grad()` | ⚠️ 方法名不同 |
| **损失函数** | `torch.nn.functional.mse_loss` | `paddle.nn.functional.mse_loss` | |
| **数据加载** | `torch.utils.data.DataLoader` | `paddle.io.DataLoader` | |
| **设备管理** | `cuda:0` | `gpu:0` | ⚠️ 字符串格式 |
| | `torch.cuda.is_available()` | `paddle.is_compiled_with_cuda()` | |
| **模型保存** | `torch.save` | `paddle.save` | |
| **模型加载** | `torch.load` | `paddle.load` | |
| | `.load_state_dict` | `.set_state_dict` | ⚠️ 方法名不同 |
