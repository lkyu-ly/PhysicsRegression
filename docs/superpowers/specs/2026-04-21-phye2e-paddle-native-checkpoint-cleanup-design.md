# PhyE2E Paddle 原生模型 I/O 收敛设计

**目标**

去掉 `PhysicsRegressionPaddle` 里遗留的 Torch/Pickle 兼容加载和旧转换脚本，统一收敛为 Paddle 原生模型读写路径：

- Paddle 推理模型只接受 `paddle.save()` 产出的 `model.pdparams`
- Paddle 训练恢复只接受 `Trainer.save_checkpoint()` 产出的 `checkpoint.pth`
- Torch `model.pt` 只允许由仓库根目录脚本 `convert_torch_to_paddle.py` 消费，不再允许 Paddle 运行时直接加载
- 根目录 `compare_torch_paddle_forward.py` 继续作为双端前向精度验证入口

---

## 背景与现状

已确认的代码现状如下：

1. `PhysicsRegressionPaddle/PhysicsRegression.py`
当前 `_load_model()` 同时支持：
- Paddle 原生 `paddle.load()`
- 旧 `tools/convert_model.py` 生成的 `.pkl`
- 对 Torch `.pt` 的探测与报错提示

2. `PhysicsRegressionPaddle/symbolicregression/model/__init__.py`
`build_modules()` 在 `params.reload_model` 分支里用 `paddle.load()` 加载，但仍沿用迁移期写法 `load_state_dict()`。

3. `PhysicsRegressionPaddle/symbolicregression/trainer.py`
`reload_checkpoint()` 仍保留“checkpoint 文件”和“非 checkpoint 模型文件”双分支，允许把推理模型当作 checkpoint 走恢复路径。

4. `PhysicsRegressionPaddle/bash/eval_bash.py`
存在 `if "model.pt" in params.reload_checkpoint` 的特判，说明评估链路仍把 Torch/Paddle 预训练模型混在同一个入口里处理。

5. `PhysicsRegressionPaddle/bash/eval_synthetic.sh` 与 `bash/eval_feynman.sh`
默认仍传 `--reload_checkpoint ./model.pt`。

6. 文档和 notebook
以下用户入口仍在引导 Paddle 侧使用 `model.pt`：
- `PhysicsRegressionPaddle/README.md`
- `PhysicsRegressionPaddle/CLAUDE.md`
- `PhysicsRegressionPaddle/PADDLE_MIGRATION.md`
- `PhysicsRegressionPaddle/symbolicregression/CLAUDE.md`
- `PhysicsRegressionPaddle/Oracle/CLAUDE.md`
- `PhysicsRegressionPaddle/physical/CLAUDE.md`
- `PhysicsRegressionPaddle/example.ipynb`
- `PhysicsRegressionPaddle/physical/case1_*.ipynb` 至 `case5_*.ipynb`

7. 仓库根目录已经存在并可复用的官方脚本
- `convert_torch_to_paddle.py`
- `compare_torch_paddle_forward.py`

8. 实际模型文件已在仓库根目录存在
- `models/model.pt`
- `models/model.pdparams`

---

## 方案对比

### 方案 1：只删除旧转换脚本

做法：
- 删除 `PhysicsRegressionPaddle/tools/convert_model.py`
- 其余加载逻辑不动

优点：
- 改动最小

缺点：
- `PhyReg` 仍会继续支持 `.pkl`
- 评估入口仍保留 `model.pt` 特判
- 文档和 notebook 仍会把用户带回旧路径
- 不能根除“Paddle 兼容 Torch 模型”的历史包袱

### 方案 2：统一为 Paddle 原生 I/O 路径

做法：
- Paddle 运行时只接受 `model.pdparams` 与 `checkpoint.pth`
- Torch `model.pt` 只通过根目录官方转换脚本进入 Paddle 世界
- 删除旧 `.pkl` 路径与评估 hack
- 统一文档、shell 脚本、notebook

优点：
- 路径单一，边界清晰
- 与 Torch 原生行为最接近
- 维护成本最低
- 方便用单测把行为钉死

缺点：
- 需要联动清理代码、脚本、文档、notebook

### 方案 3：再加一层更大的兼容抽象

做法：
- 抽象一整套多格式模型适配层，继续兼容 `.pt/.pkl/.pdparams/.pth`

优点：
- 理论上兼容面最广

缺点：
- 明显违背本次“统一放弃兼容 Torch 模型”的目标
- 会继续保留复杂度
- 属于过度设计

**结论**

采用 **方案 2**。这是用户已确认的方向，也是唯一能把问题根除的方案。

---

## 最终设计

### 1. 单一事实来源

Paddle 侧只保留两类模型文件：

| 用途 | 文件 | 生产方式 | 消费入口 |
| --- | --- | --- | --- |
| 推理模型 | `models/model.pdparams` | 根目录 `convert_torch_to_paddle.py` 或 `PhyReg.save()` | `PhysicsRegressionPaddle/PhysicsRegression.py` |
| 训练检查点 | `checkpoint.pth` | `Trainer.save_checkpoint()` | `Trainer.reload_checkpoint()` |

额外规则：

- `models/model.pt` 只作为 Torch 原始权重保留在仓库根目录
- Paddle 运行时不得直接接受 `.pt`
- Paddle 运行时不得接受旧 `.pkl`
- `reload_model` 负责加载推理模型
- `reload_checkpoint` 只负责恢复训练 checkpoint
- `checkpoint.pth` 虽然沿用 `.pth` 文件名，但内容是 `paddle.save()` 产物，不是 Torch 序列化文件
- `params` 的保存策略尽量贴近 Torch 现状：
  - `PhyReg.save()` 产出的推理模型 bundle 保存 `Namespace`
  - `Trainer.save_checkpoint()` 产出的训练 checkpoint 保存 `dict`
  - helper 加载时统一规范化为 `Namespace`

### 2. 新增一个小而专一的 checkpoint helper

新增：

- `PhysicsRegressionPaddle/symbolicregression/checkpoint_io.py`

职责只做三件事：

1. 规范化 `params`
2. 校验 Paddle 原生模型/状态字典结构
3. 用 Paddle 原生状态接口恢复 Layer / Module / Optimizer / GradScaler

这个文件是本次唯一新增的共享代码单元，理由明确：

- `PhysicsRegression.py`
- `symbolicregression/model/__init__.py`
- `symbolicregression/trainer.py`

三处都需要相同的“加载 + 校验 + 应用状态”逻辑。单独抽成一个小 helper 比在三处各自复制更符合 DRY，也没有超出本次需求。

### 3. Paddle 原生状态接口策略

基于本地环境确认：

- `Layer.set_state_dict()` 可用
- `Optimizer.set_state_dict()` 可用
- `GradScaler.load_state_dict()` 可用

因此本次统一策略为：

- Layer / Module：`set_state_dict()`
- Optimizer：`set_state_dict()`
- GradScaler：`load_state_dict()`

不再沿用迁移期的 `load_state_dict()` 习惯写法，也不再保留旧格式 fallback。

### 4. `PhysicsRegression.py` 的新职责

`PhysicsRegressionPaddle/PhysicsRegression.py` 只负责：

1. 调用 `load_paddle_model_bundle(path)`
2. 构建 env / modules
3. 应用 `embedder/encoder/decoder` 权重
4. 暴露 `PhyReg.save()`，保存 Paddle 原生 bundle

它不再负责：

- 读取 `.pkl`
- 探测 `.pt`
- 提示 `tools/convert_model.py`
- 兼容多套历史格式

如果传入 `model.pt` 或 `.pkl`，应直接报错，并明确告诉用户：

- Paddle 运行时仅支持 Paddle 原生文件
- Torch 权重请先运行根目录 `convert_torch_to_paddle.py`

### 5. `build_modules()` 只处理 Paddle 预训练模型

`PhysicsRegressionPaddle/symbolicregression/model/__init__.py`

`params.reload_model` 的语义收敛为：

- 只接受 Paddle 推理模型 bundle
- 由 helper 完成结构校验
- 统一用 `set_state_dict()` 恢复模块权重

不再承担 Torch / 旧 pickle 兼容职责。

### 6. `Trainer.reload_checkpoint()` 只处理训练 checkpoint

`PhysicsRegressionPaddle/symbolicregression/trainer.py`

新语义：

- `reload_checkpoint` 只接受训练 checkpoint
- 文件里必须存在训练恢复所需字段：
  - `epoch`
  - `n_total_iter`
  - `best_metrics`
  - `best_stopping_criterion`
  - `params`
  - `embedder`
  - `encoder`
  - `decoder`
  - `optimizer`

如果用户把 `model.pdparams` 传给 `reload_checkpoint`，应直接报错并提示改用 `--reload_model`。

这一步会顺带删除当前的两个历史分支：

- `"checkpoint" in checkpoint_path` 的文件名特判
- 非 checkpoint 模型 bundle 的 fallback 加载

### 7. 评估入口回归正常职责分工

评估链路调整为：

- shell 脚本传 `--reload_model ../models/model.pdparams`
- `build_modules()` 负责加载推理模型
- `Trainer()` 正常初始化，不再通过 `path="model.pt", root="./"` 走 checkpoint hack

这会清理：

- `PhysicsRegressionPaddle/bash/eval_bash.py` 的 `model.pt` 特判
- `bash/eval_synthetic.sh`
- `bash/eval_feynman.sh`

### 8. 删除旧转换脚本

删除：

- `PhysicsRegressionPaddle/tools/convert_model.py`
- `PhysicsRegressionPaddle/tools/__init__.py`（若目录为空则一并移除目录）

根目录 `convert_torch_to_paddle.py` 成为唯一官方转换入口。

### 9. 文档与 notebook 路径统一

统一后的用户路径规则：

- 从仓库根目录运行转换脚本：
  - `python "convert_torch_to_paddle.py"`
- Paddle 代码加载：
  - 在 `PhysicsRegressionPaddle/` 目录下使用 `../models/model.pdparams`
- `physical/*.ipynb`：
  - 使用 `../../models/model.pdparams`

文档修改只覆盖“用户会照抄执行的模型 I/O 指引”，不去重写整个迁移教程。

---

## 错误处理设计

### `PhyReg(path)` 传入非法文件

应报：

- 仅支持 Paddle 原生模型 bundle
- Torch `.pt` 请先运行根目录 `convert_torch_to_paddle.py`
- 旧 `.pkl` 已废弃

### `reload_model` 指向 checkpoint

允许，只要文件中包含 `embedder/encoder/decoder/params` 且能作为模型 bundle 使用。

### `reload_checkpoint` 指向推理模型

直接报错，提示：

- 当前文件是推理模型，不是训练 checkpoint
- 若要加载预训练模型，请使用 `--reload_model`

### 状态字典键或 shape 不匹配

helper 直接抛出带模块名的错误，不再做旧格式兜底修补。

---

## 测试策略

测试分两层。

### 1. 细粒度单测

针对新 helper 做小而快的测试：

- `normalize_params()`
- `load_paddle_model_bundle()`
- `set_layer_state()`
- `set_optimizer_state()` / `set_grad_scaler_state()`
- `Trainer.reload_checkpoint()` 的 checkpoint / inference bundle 语义区分

### 2. 真实模型烟雾测试

使用仓库根目录真实文件：

- `models/model.pt`
- `models/model.pdparams`

验证：

1. `PhyReg("../models/model.pdparams")` 能正常加载
2. `PhyReg("../models/model.pt")` 会被拒绝
3. 根目录 `convert_torch_to_paddle.py` 能从 `models/model.pt` 生成新的 Paddle 原生模型文件
4. 根目录 `compare_torch_paddle_forward.py` 能用 `models/model.pt` 与 `models/model.pdparams` 输出误差报告

这满足用户要求的“实际测试使用 `PhyE2E/models` 中的模型（双端都有）”。

---

## 非目标

本次不做以下工作：

- 不改动根目录 `convert_torch_to_paddle.py` 的总体设计
- 不改动根目录 `compare_torch_paddle_forward.py` 的总体设计
- 不重构 Oracle、MCTS、GA 或实际公式搜索逻辑
- 不追求兼容历史 `.pkl` 文件
- 不兼容 Paddle 直接读取 Torch `.pt`
- 不新增与本任务无关的通用框架层

---

## 验收标准

完成后应满足：

1. `PhysicsRegressionPaddle` 代码中不存在对旧 `tools/convert_model.py` 的依赖
2. `PhyReg` 只能加载 Paddle 原生模型 bundle
3. `reload_model` 与 `reload_checkpoint` 的职责清晰分离
4. `bash/eval_bash.py`、`eval_synthetic.sh`、`eval_feynman.sh` 不再出现 `model.pt` 特判与默认路径
5. 用户文档和 notebook 不再把 Paddle 使用方式写成 `model.pt`
6. 单测覆盖 helper、公共加载入口、Trainer checkpoint 语义、根目录官方脚本
7. 真实模型烟雾测试使用仓库根目录 `models/model.pt` 与 `models/model.pdparams` 跑通
