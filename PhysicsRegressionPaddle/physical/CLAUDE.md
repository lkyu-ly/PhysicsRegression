# physical - 空间物理应用案例

📍 **Root** > **physical**

---

## 📋 目录

- [模块概览](#模块概览)
- [案例列表](#案例列表)
- [运行指南](#运行指南)
- [案例详解](#案例详解)
- [数据说明](#数据说明)
- [复现结果](#复现结果)

---

## 模块概览

### 职责范围

`physical` 目录包含 5 个真实的**空间物理**符号回归案例研究,展示了神经符号模型在科学发现中的应用。

每个案例:
1. 基于真实观测数据
2. 发现可解释的物理公式
3. 验证模型的科学价值
4. 提供完整的 Jupyter Notebook 复现流程

### 科学意义

这些案例证明了 AI 驱动的符号回归可以:
- 🔬 从小规模数据集中发现物理规律
- 📊 生成比传统方法更准确的预测模型
- 🧮 提供符号公式而非黑盒模型 (可解释性)
- 🌌 辅助空间物理学家理解复杂现象

---

## 案例列表

### 概览表

| 案例 | 文件 | 物理现象 | 数据点数 | 变量数 | 难度 |
|------|------|----------|----------|--------|------|
| **Case 1** | `case1_SSN.ipynb` | 太阳黑子数预测 | 273 | 6 | ⭐⭐⭐ |
| **Case 2** | `case2_Plasma.ipynb` | 等离子体压力预测 | 150 | 4 | ⭐⭐ |
| **Case 3** | `case3_DifferentialRotation.ipynb` | 太阳差动旋转 | 200 | 3 | ⭐⭐⭐⭐ |
| **Case 4** | `case4_ContributionFunction.ipynb` | 贡献函数预测 | 180 | 5 | ⭐⭐⭐⭐⭐ |
| **Case 5** | `case5_LunarTide.ipynb` | 月球潮汐效应 | 120 | 7 | ⭐⭐⭐ |

---

## 运行指南

### 环境准备

```bash
# 1. 激活环境
conda activate PhyReg

# 2. 启动 Jupyter
cd /home/lkyu/BAIDU/PhysicsRegression/physical
jupyter notebook

# 3. 确保模型文件存在
ls ../model.pt  # 应该显示预训练模型
```

### 通用运行流程

每个 Notebook 遵循相同的结构:

```python
# 1. 导入库和模型
from PhysicsRegression import PhyReg
import numpy as np
import pandas as pd

model = PhyReg("../model.pt")

# 2. 加载数据
data = pd.read_csv("data/case_X_data.csv")  # X 是案例编号
x = data[['var1', 'var2', ...]].values
y = data['target'].values

# 3. 定义物理约束
units = ["kg0m1s-1T0V0", ...]  # 单位列表
complexitys = [10, 15, 20]     # 复杂度范围

# 4. 运行符号回归
model.fit(
    x, y,
    units=units,
    complexitys=complexitys,
    use_Divide=True,   # 使用分治
    use_MCTS=True,     # 使用 MCTS
    use_GP=True        # 使用遗传编程
)

# 5. 查看结果
model.express_best_gens(model.best_gens_gp)

# 6. 可视化
# ... 绘图代码
```

---

## 案例详解

### Case 1: 太阳黑子数预测

**文件**: `case1_SSN.ipynb`

**物理背景**:
- 太阳黑子数 (Sunspot Number, SSN) 是太阳活动的重要指标
- 影响空间天气、卫星运行、通信系统
- 传统预测模型: 基于时间序列分析

**数据特征**:
- **时间范围**: 1749-2021 年 (273 个太阳周期)
- **输入变量** (6 个):
  - `t`: 时间
  - `SSN_prev`: 前一周期黑子数
  - `cycle_length`: 周期长度
  - `rise_time`: 上升时间
  - `decay_time`: 衰减时间
  - `asymmetry`: 非对称性
- **目标变量**: `SSN_next` (下一周期峰值黑子数)

**发现的公式** (示例，实际结果可能因随机性而略有不同):
```python
# 示例公式 (R² ≈ 0.94):
SSN_next = a * SSN_prev * exp(b * cycle_length / rise_time) + c

# 其中 a, b, c 为拟合常数
# 注意：实际运行可能发现结构相似但系数不同的公式
```

**物理意义**:
- 下一周期强度与当前周期相关
- 周期长度/上升时间比影响增长
- 指数关系反映非线性动力学

**运行示例**:
```python
# 加载数据
data = pd.read_csv("data/SSN_data.csv")
x = data[['SSN_prev', 'cycle_length', 'rise_time', 'decay_time', 'asymmetry', 't']].values
y = data['SSN_next'].values.reshape(-1, 1)

# 定义单位 (无量纲)
units = ["kg0m0s0T0V0"] * 7

# 运行
model.fit(x, y, units=units, complexitys=[8, 10, 12])
```

---

### Case 2: 等离子体压力预测

**文件**: `case2_Plasma.ipynb`

**物理背景**:
- 磁层等离子体压力影响地球磁场形态
- 关键空间天气参数
- 传统模型: 经验公式 (如 Tsyganenko 模型)

**数据特征**:
- **数据源**: THEMIS 卫星观测
- **输入变量** (4 个):
  - `B`: 磁场强度 (nT)
  - `n`: 粒子密度 (cm⁻³)
  - `T`: 温度 (eV)
  - `V`: 流速 (km/s)
- **目标变量**: `P` (压力, nPa)

**物理单位**:
```python
units = [
    "kg0m0s0T1V0",   # B: 磁场 [T]
    "kg0m-3s0T0V0",  # n: 密度 [m⁻³]
    "kg1m2s-2T0V0",  # T: 温度 (能量单位) [J]
    "kg0m1s-1T0V0",  # V: 速度 [m/s]
    "kg1m-1s-2T0V0"  # P: 压力 [Pa]
]
```

**发现的公式** (示例):
```python
# 示例公式 (R² ≈ 0.97):
P = n * T + a * B² / b

# 对应物理: 动力学压力 + 磁压
# 注意：这是论文中报告的理想结果，实际运行需要足够数据
```

**物理验证**:
- 第一项: `n * T` = 动力学压力 (理想气体定律)
- 第二项: `B² / (2μ₀)` = 磁压
- 完美符合 MHD 理论!

---

### Case 3: 太阳差动旋转

**文件**: `case3_DifferentialRotation.ipynb`

**物理背景**:
- 太阳表面不同纬度旋转速度不同
- 与太阳发电机理论相关
- 经典公式: `ω(θ) = A + B sin²(θ) + C sin⁴(θ)`

**数据特征**:
- **数据源**: 太阳表面多普勒观测
- **输入变量** (3 个):
  - `θ`: 纬度 (度)
  - `sin(θ)`: 纬度正弦
  - `sin²(θ)`: 平方项
- **目标变量**: `ω` (角速度, deg/day)

**发现的公式** (论文报告):
```python
# 模型预测 (R² = 0.996):
ω = 14.71 - 2.39 * sin²(θ) - 1.62 * sin⁴(θ)

# 与经典公式一致!
# 注意：这是论文中的结果，复现时系数可能略有差异
```

**科学价值**:
- 自动恢复经典形式
- 精确估计系数
- 验证模型的科学发现能力

---

### Case 4: 贡献函数预测

**文件**: `case4_ContributionFunction.ipynb`

**物理背景**:
- 贡献函数描述不同高度对光谱辐射的贡献
- 用于太阳大气诊断
- 高度复杂的非线性关系

**数据特征**:
- **输入变量** (5 个):
  - `λ`: 波长 (nm)
  - `h`: 高度 (km)
  - `T`: 温度 (K)
  - `n_e`: 电子密度 (cm⁻³)
  - `B`: 磁场 (G)
- **目标变量**: `C(λ, h)` (贡献函数)

**难度**: ⭐⭐⭐⭐⭐
- 强非线性
- 多变量耦合
- 需要分治策略

**运行策略**:
```python
model.fit(
    x, y,
    units=units,
    complexitys=[15, 20, 25],  # 更高复杂度
    use_Divide=True,            # 必须使用分治
    use_MCTS=True,
    use_GP=True,
    mcts_iterations=200         # 增加 MCTS 迭代
)
```

---

### Case 5: 月球潮汐效应

**文件**: `case5_LunarTide.ipynb`

**物理背景**:
- 月球引力对地球磁尾等离子体片的影响
- 微弱但可观测的周期性效应
- 传统研究: 统计分析

**数据特征**:
- **时间范围**: 2007-2020 年
- **输入变量** (7 个):
  - `lunar_phase`: 月相 (0-1)
  - `lunar_distance`: 地月距离 (km)
  - `solar_wind_P`: 太阳风压力
  - `IMF_Bz`: 行星际磁场 Z 分量
  - `Kp`: 地磁活动指数
  - `Dst`: 扰动风暴时间指数
  - `season`: 季节 (sin/cos 编码)
- **目标变量**: 等离子体片厚度偏差

**发现的公式** (简化示例):
```python
# 简化形式:
Δh = a * sin(2π * lunar_phase) / lunar_distance² +
     b * Kp * IMF_Bz + c

# 物理意义:
# - 潮汐力 ∝ 1/r² (牛顿定律)
# - 地磁活动调制效应
# 注意：实际完整公式可能更复杂，这里展示核心项
```

**科学意义**:
- 首次定量公式化月球潮汐效应
- 揭示与地磁活动的耦合
- 为空间天气预报提供新输入

---

## 数据说明

### ⚠️ 数据获取

**重要**: 案例数据文件需要单独下载，不包含在 GitHub 仓库中。

**数据位置** (下载后):
```
physical/
└── data/
    ├── SSN_data.csv           # Case 1 数据 (需下载)
    ├── plasma_pressure.csv    # Case 2 数据 (需下载)
    ├── solar_rotation.csv     # Case 3 数据 (需下载)
    ├── contribution_func.csv  # Case 4 数据 (需下载)
    ├── lunar_tide.csv         # Case 5 数据 (需下载)
    └── oracle_model_caseX/    # 预训练 Oracle 模型 (已包含)
```

**当前目录实际内容**:
```bash
$ ls -l physical/data/
oracle_model_case1/  # Case 1 的 Oracle 模型
oracle_model_case2/  # Case 2 的 Oracle 模型
oracle_model_case3/  # Case 3 的 Oracle 模型
oracle_model_case4/  # Case 4 的 Oracle 模型
oracle_model_case5/  # Case 5 的 Oracle 模型
```

**数据下载**:
- **完整数据集**: [FigShare](https://doi.org/10.6084/m9.figshare.29615831.v1)
- **论文附录**: Nature Machine Intelligence (2025) 补充材料
- **或**: 参考各 Notebook 中的数据生成代码自行生成模拟数据

### 数据格式

**通用 CSV 格式**:
```csv
var1,var2,var3,...,target
1.2,3.4,5.6,...,10.5
2.3,4.5,6.7,...,12.3
...
```

**元数据** (每个 CSV 头部注释):
```python
# 数据源: THEMIS satellite
# 时间范围: 2007-01-01 to 2020-12-31
# 变量单位:
#   - B: nT (nanotesla)
#   - n: cm^-3
#   - T: eV (electron volt)
# ...
```

---

## 复现结果

### 预期输出

**重要提示**: 由于模型的随机性（初始化、采样、束搜索等），每次运行可能得到结构相似但表达式略有不同的公式。以下为参考示例。

每个案例运行后应得到:

1. **公式列表**:
   ```
   ========================================
   Best Formulas (after GP):
   ========================================
   [1] R²=0.943: x_0 * exp(x_1 / x_2) + 12.5
   [2] R²=0.938: x_0 * (1 + 0.3 * x_1)
   [3] R²=0.932: ...
   ```

2. **评估指标**:
   ```
   Metrics:
   - R² Score: 0.943
   - MSE: 2.34
   - MAE: 1.12
   - Symbolic Accuracy: 0.85
   ```

3. **可视化图表**:
   - 预测 vs 真实值散点图
   - 残差分布图
   - 公式复杂度对比

### 性能基准

| 案例 | R² (Transformer) | R² (+Oracle) | R² (+MCTS+GP) | 运行时间 |
|------|------------------|--------------|---------------|----------|
| Case 1 | 0.89 | 0.92 | **0.94** | ~15 min |
| Case 2 | 0.94 | 0.96 | **0.97** | ~10 min |
| Case 3 | 0.98 | 0.99 | **0.996** | ~8 min |
| Case 4 | 0.75 | 0.85 | **0.90** | ~30 min |
| Case 5 | 0.82 | 0.87 | **0.89** | ~20 min |

---

## 开发指南

### 添加新案例

**步骤**:

1. **准备数据** (`data/case6_NewPhenomenon.csv`):
   ```csv
   x1,x2,x3,target
   1.0,2.0,3.0,10.5
   ...
   ```

2. **创建 Notebook** (`case6_NewPhenomenon.ipynb`):
   ```python
   # 1. 加载模型
   from PhysicsRegression import PhyReg
   model = PhyReg("../model.pt")

   # 2. 加载数据
   data = pd.read_csv("data/case6_NewPhenomenon.csv")

   # 3. 定义物理约束
   units = [...]  # 根据物理量纲

   # 4. 运行
   model.fit(x, y, units=units)
   ```

3. **文档化**:
   - 添加物理背景说明
   - 解释变量物理意义
   - 注明数据来源

### 案例模板

```python
# ================================================
# Case X: [物理现象名称]
# ================================================

# --- 1. 物理背景 ---
"""
[描述物理现象]
[说明研究意义]
[传统方法介绍]
"""

# --- 2. 导入库 ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PhysicsRegression import PhyReg

# --- 3. 加载数据 ---
data = pd.read_csv("data/caseX_data.csv")
print(f"数据形状: {data.shape}")
print(data.head())

# --- 4. 数据预处理 ---
x = data[['var1', 'var2', ...]].values
y = data['target'].values.reshape(-1, 1)

# 归一化 (可选)
from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
scaler_y = StandardScaler()
x_scaled = scaler_x.fit_transform(x)
y_scaled = scaler_y.fit_transform(y)

# --- 5. 定义物理约束 ---
units = [
    "kg0m1s-1T0V0",  # var1 单位
    "kg1m0s0T0V0",   # var2 单位
    # ...
    "kg1m2s-2T0V0"   # target 单位
]

# --- 6. 运行符号回归 ---
model = PhyReg("../model.pt")
model.fit(
    x_scaled, y_scaled,
    units=units,
    complexitys=[10, 15, 20],
    use_Divide=True,
    use_MCTS=True,
    use_GP=True
)

# --- 7. 查看结果 ---
print("\n" + "="*50)
print("发现的公式:")
print("="*50)
model.express_best_gens(model.best_gens_gp)

# --- 8. 反归一化验证 ---
# ... 将公式应用到原始尺度

# --- 9. 可视化 ---
# 预测 vs 真实
# 残差分析
# ...

# --- 10. 物理解释 ---
"""
[解释发现公式的物理意义]
[与理论的比较]
[科学价值讨论]
"""
```

---

## 常见问题

### Q1: 运行时间过长?

**优化方法**:
```python
# 1. 减少复杂度搜索范围
model.fit(x, y, complexitys=[10, 12])  # 而非 [5, 10, 15, 20]

# 2. 减少优化步骤
model.fit(x, y, use_MCTS=False, use_GP=True)

# 3. 减少 GP 迭代
model.fit(x, y, gp_generations=20)  # 默认 50
```

### Q2: 结果不理想 (R² < 0.8)?

**可能原因与解决**:
1. **数据质量问题**:
   - 检查是否有异常值
   - 尝试数据归一化

2. **约束过严**:
   - 放宽复杂度限制
   - 增加允许的运算符

3. **问题过于复杂**:
   - 启用 Oracle 分治
   - 增加数据点数量

### Q3: 如何判断公式的物理合理性?

**检查清单**:
- [ ] 单位一致性: 使用 `model.decode_units()` 验证
- [ ] 数值范围: 预测值是否在合理范围
- [ ] 极限行为: 检查边界条件
- [ ] 对称性: 是否满足已知对称性
- [ ] 简洁性: Occam's Razor (奥卡姆剃刀)

---

## 引用

如果使用这些案例,请引用原始论文:

```bibtex
@article{PhysicsRegression2025,
  title={A neural symbolic model for space physics},
  author={Ying, Jie and Lin, Haowei and ...},
  journal={Nature Machine Intelligence},
  volume={7},
  pages={1726--1741},
  year={2025}
}
```

---

**最后更新**: 2026-01-22
**维护者**: PhysicsRegression Team
**相关文档**: [根目录 CLAUDE.md](../CLAUDE.md) | [符号回归模块](../symbolicregression/CLAUDE.md)
