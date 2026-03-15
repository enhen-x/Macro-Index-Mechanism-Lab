# Macro-Index-Mechanism-Lab

将“宏观变量如何影响大盘指数”作为核心研究目标的可解释量化研究工作区。

中文 | [English](./README.en.md)

## 1. 项目定位

`Macro-Index-Mechanism-Lab` 关注的是**作用机制解释**，而不是单一建模范式。

项目目标：
- 识别宏观变量对大盘指数（如 S&P500）的传导路径；
- 区分不同宏观/市场状态下的机制差异；
- 形成可复现、可检验、可扩展的解释框架。

## 2. 方法范围（不设单一路径）

本项目支持多种可解释方法并行研究，包括但不限于：
- 动力系统建模（如振动方程/受迫阻尼类模型）；
- 状态依赖局部投影（State-Dependent Local Projections, SD-LP）；
- Discount-Rate vs Cash-Flow 机制分解；
- 其他结构化计量方法（状态切换、事件窗、稳健回归等）。

说明：
- “振动方程建模”已作为一个阶段性探索完成并归档；
- 当前方向是机制解释优先，而非绑定某个单一方程。

## 3. 当前阶段与归档阶段

当前活跃阶段：
- `stages/us_discount-rate vs cash-flow/`

已归档阶段（示例）：
- `stages/us_vibration_equation_exploration_2026-03-15/`

## 4. 仓库结构（建议理解顺序）

- `stages/`：按研究阶段组织的脚本、数据、文档（推荐入口）。
- `data_loader/`：原始数据拉取与预处理工具。
- `identification/`：参数估计与识别模块。
- `core/`：通用建模与计算组件。
- `tests/`：测试用例。
- `docs/`：全局文档与阶段索引说明。

## 5. 快速开始

1. 创建并激活 Python 环境。
2. 安装依赖：`pip install -r requirements.txt`。
3. 进入目标阶段目录（优先 `stages/us_discount-rate vs cash-flow/`）。
4. 按该阶段 `docs/` 与 `scripts/README.md` 执行。

## 6. 贡献方式

欢迎提交：
- Issue：问题、假设、机制解释疑问；
- PR：脚本实现、实验复现、文档完善、测试补充。

建议 PR 自检：
- 目标与假设明确；
- 数据与参数可复现；
- 有前后对比或图表证据。

## 7. 许可证

MIT License.

