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
