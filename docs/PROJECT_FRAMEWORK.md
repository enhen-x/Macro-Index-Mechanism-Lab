# Project Framework

## 1. 目录结构

```text
.
├─ core/                 # 动力系统方程与仿真核心
├─ data_loader/          # 数据加载、清洗、对齐
├─ identification/       # 参数辨识与滤波
├─ backtest/             # 评估与回测
├─ scripts/              # 可直接运行的入口脚本
├─ tests/                # 单元与冒烟测试
├─ notebooks/            # 研究实验 notebook
├─ data/                 # 本地数据（默认不提交大文件）
└─ docs/                 # 路线图、设计说明、实验记录
```

## 2. 模块职责

- `core/`
  - 定义动力学方程、离散化与仿真器。
  - 输出状态轨迹，不关心数据来源。

- `data_loader/`
  - 提供统一数据接口，屏蔽数据源差异。
  - 做频率对齐、缺失处理、特征拼接。

- `identification/`
  - 输入状态序列与外部驱动，输出模型参数。
  - 保持“算法实现”与“评估逻辑”解耦。

- `backtest/`
  - 统一评估指标与切分方式。
  - 输出对比结果，支持不同辨识器横向比较。

- `scripts/`
  - 将上述模块串成最小可运行 pipeline。
  - 不放核心业务逻辑，仅做流程编排。

## 3. 最小开发闭环

1. `data_loader` 提供 `x(t), u(t)`
2. `identification` 估计参数
3. `core` 用参数回放系统响应
4. `backtest` 计算 out-of-sample 指标
5. 在 `scripts/` 输出结果与日志

## 4. 接口约定（MVP）

- 输入数据字段：
  - `timestamp`
  - `x`（指数序列或变换值）
  - `u`（宏观驱动，支持单变量先跑通）

- 辨识结果字段：
  - `m`, `c`, `k`
  - `beta_u`（驱动系数）
  - `x0`（平衡点）

- 评估结果字段：
  - `mse`
  - `mae`
  - `directional_accuracy`

## 5. 工程规范

- Python 版本：`>=3.10`
- 随机种子：统一在脚本入口设置
- 配置：后续集中到 `configs/`
- 测试：每个核心模块至少一个冒烟测试
- 文档：接口变更必须更新 README/docs
