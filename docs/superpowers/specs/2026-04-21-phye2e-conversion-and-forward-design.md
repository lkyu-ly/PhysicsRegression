# PhyE2E Torch 权重转换与 Paddle 前向精度对比设计

**目标**

在不改动现有核心库代码的前提下，为仓库新增两个仅用于本地测试的根目录脚本：

1. `convert_torch_to_paddle.py`
   从 `models/model.pt` 读取 PyTorch 权重，构建 Paddle 版核心模型，并保存为 `models/model.pdparams`。
2. `compare_torch_paddle_forward.py`
   在固定随机输入下分别执行 Torch / Paddle 核心前向，比较 teacher-forcing logits 的平均/最大绝对误差与相对误差，并打印到终端。

---

## 范围与非目标

**范围**

- 仅覆盖核心模型链路：`embedder -> encoder("fwd") -> decoder("fwd") -> decoder("predict")`
- 仅做 CPU 本地对比验证
- 仅生成和使用 `models/model.pdparams`
- 两个实现脚本都直接放在项目根目录

**非目标**

- 不接入 `PhyReg.fit()`、Oracle、MCTS、GP、公式后处理
- 不实现真实物理公式预测流程
- 不新增共享库模块，不改造现有框架目录结构
- 不复用 `PhysicsRegressionPaddle/tools/convert_model.py` 输出的 `.pkl` 中间格式

---

## 关键发现

### 1. 权重可以直接按同名参数转换

已验证 `models/model.pt` 中的 `embedder / encoder / decoder` 三个 `state_dict` 与 Paddle 侧 `build_modules()` 构建出来的模块：

- 参数键名完全一致
- 顺序一致
- shape 完全一致
- 直接将 `torch.Tensor -> numpy -> paddle.Tensor` 后可成功 `set_state_dict()`

因此转换方案采用最简单且最可靠的实现：

- 不做手工参数映射表
- 直接逐键校验
- 校验通过后直接保存为 Paddle 原生 checkpoint

### 2. 前向对比应以 teacher-forcing logits 为准

对比目标选择为 `decoder("predict")` 返回的 logits，而不是：

- `encoder("fwd")`：覆盖范围不够，无法验证 decoder
- `decoder.generate()`：离散解码结果不适合做连续误差分析，且容易混入 beam/search/后处理差异

teacher-forcing logits 具备以下优点：

- 覆盖完整核心链路
- 输出是稠密浮点张量
- 可稳定计算平均/最大绝对误差与相对误差
- 不依赖真实公式搜索逻辑

### 3. 比较脚本必须规避同名包冲突

`PhysicsRegression/` 与 `PhysicsRegressionPaddle/` 内部都使用：

- `symbolicregression`
- `parsers`
- `Oracle`

这意味着在同一个 Python 进程里直接先后导入两套工程代码，极易出现 `sys.modules` 污染和导入冲突。

最稳妥且实现简单的方案是：

- `compare_torch_paddle_forward.py` 作为父进程
- 父进程通过 `subprocess.run()` 调用自身两次
- 子进程分别以 `--worker torch` / `--worker paddle` 模式运行
- 每个子进程只导入一侧工程代码，并将 logits 存到临时 `.npy`
- 父进程负责读取两份 `.npy` 并输出误差报告

该方案避免：

- 包名冲突
- 全局状态污染
- 手动清理 `sys.modules` 的脆弱实现

---

## 输入与目标序列设计

### 随机输入

比较脚本使用固定种子的 `numpy` 随机数组生成：

- `x`: 形状 `(num_points, input_dim)`
- `y`: 形状 `(num_points, 1)`

然后按现有核心模型要求组装为：

```python
inputs = [[[x[i], y[i]] for i in range(num_points)]]
```

这里的 `y` 不要求与目标公式语义一致，因为本任务只验证“同一输入下 Torch / Paddle 的数值前向是否一致”，不是验证模型语义精度。

### 最小合法目标序列

为触发 decoder 的 teacher-forcing 路径，脚本手工构造最小合法公式树：

```python
root = Node("x_0", params)
root.unit = np.array([0, 0, 0, 0, 0])
```

再通过环境编码器得到 `tree_encoded`，作为 decoder 目标序列。

原因：

- `x_0` 一定在词表中
- 单节点树最短、最稳定
- 与随机输入解耦，便于重复验证

### 最小 hints

当前预训练模型的 `params.use_hints` 为：

```text
units,complexity,unarys,consts
```

因此比较脚本需要构造最小 hints：

- `units`: 全零单位，长度为 `input_dim + 1`
- `complexity`: `["simple"]`
- `unarys`: `[]`
- `consts`: `[]`

如果未来遇到脚本尚未覆盖的 hint 类型，脚本应直接报错并指出未支持项，而不是静默跳过。

---

## 脚本职责

### `convert_torch_to_paddle.py`

职责：

1. 解析输入输出路径，默认：
   - 输入：`models/model.pt`
   - 输出：`models/model.pdparams`
2. 用 `torch.load(..., map_location="cpu")` 读取 Torch checkpoint
3. 规范化 `params`
   - `Namespace` 保持不变
   - `dict` 转 `Namespace`
4. 进入 `PhysicsRegressionPaddle/` 目录上下文，构建 Paddle 环境和模块
5. 校验三组模块的：
   - 参数键集合
   - 参数顺序
   - 参数 shape
6. 将 Torch 权重转换为 Paddle Tensor
7. 调用 `set_state_dict()` 再次验证可装载性
8. 用 `paddle.save()` 保存：
   - `embedder`
   - `encoder`
   - `decoder`
   - `params`（建议保存为普通 `dict`，降低反序列化耦合）
9. 在终端打印输出路径和校验结果

### `compare_torch_paddle_forward.py`

职责：

1. 父进程解析：
   - `--torch-model`
   - `--paddle-model`
   - `--seed`
   - `--num-points`
   - `--input-dim`
   - `--eps`
2. 父进程创建临时目录，分别启动：
   - `--worker torch`
   - `--worker paddle`
3. 子进程内部：
   - 导入对应一侧工程代码
   - 构建最小随机输入、最小 hints、最小目标树
   - 执行 core forward
   - 保存 logits 到 `.npy`
4. 父进程读取两份 logits，计算：
   - 平均绝对误差
   - 最大绝对误差
   - 平均相对误差
   - 最大相对误差
5. 父进程打印：
   - logits shape
   - Torch / Paddle 输出 dtype
   - 四个误差指标

---

## 误差定义

以 Torch 输出为基准：

```python
abs_err = np.abs(paddle_logits - torch_logits)
rel_err = abs_err / np.maximum(np.abs(torch_logits), eps)
```

默认 `eps = 1e-12`，用于避免基准值接近零时的除零问题。

终端输出以下四项：

- `mean_abs_error`
- `max_abs_error`
- `mean_rel_error`
- `max_rel_error`

---

## 路径与上下文处理

两套工程都依赖各自目录下的相对路径数据文件，例如：

- `PhysicsRegression/data/FeynmanEquations.xlsx`
- `PhysicsRegressionPaddle/data/FeynmanEquations.xlsx`

因此两个脚本都需要一个最小上下文管理器，用于在导入和构建环境前临时：

- `os.chdir(project_dir)`
- `sys.path.insert(0, project_dir)`

退出后恢复原始工作目录和 `sys.path`。

这比修改现有库代码更安全，也更符合本任务“只做本地测试脚本”的要求。

---

## 验证标准

实现完成后应满足：

1. `python "convert_torch_to_paddle.py"` 可以在根目录生成 `models/model.pdparams`
2. `paddle.load("models/model.pdparams")` 包含：
   - `embedder`
   - `encoder`
   - `decoder`
   - `params`
3. `python "compare_torch_paddle_forward.py"` 可在根目录直接运行成功
4. 终端能输出 logits shape 和四项误差指标
5. 不修改现有核心库代码，不依赖 `.pkl` 中间格式

---

## 最终决策

- 采用 **同名直转 + 严格校验** 的权重转换方案
- 采用 **teacher-forcing logits** 作为前向精度对比对象
- 采用 **父进程 + 自调用子进程 worker** 规避 Torch / Paddle 工程包名冲突
- 保持实现只落在两个根目录脚本，符合最简本地测试目标
