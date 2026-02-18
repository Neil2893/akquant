# AKQuant 设计与开发指南

本文档详细介绍了 `AKQuant` 的内部设计原理、核心组件架构以及扩展开发指南。旨在帮助开发者深入理解项目结构，以便进行二次开发和功能扩展。

## 1. 项目概览

### 1.1 设计理念

`AKQuant` 遵循以下核心设计原则：

1.  **核心计算下沉 (Rust Core)**: 所有的计算密集型任务（事件循环、订单撮合、风控检查、数据管理、绩效计算、历史数据维护）都在 Rust 层实现，以确保高性能和内存安全。
2.  **策略逻辑上浮 (Python API)**: 策略编写、参数配置、数据分析、机器学习模型定义等用户交互层保留在 Python 中，利用其动态特性和丰富的生态系统 (Pandas, Scikit-learn, PyTorch 等)。
3.  **模块化与解耦 (Modularity)**: 借鉴成熟的量化框架设计，将数据、执行、策略、风控、机器学习等模块严格分离，通过清晰的接口（Traits）交互。

### 1.2 项目目录结构

```text
akquant/
├── Cargo.toml              # Rust 项目依赖与配置
├── pyproject.toml          # Python 项目构建配置 (Maturin)
├── Makefile                # 项目管理命令
├── src/                    # Rust 核心源码 (底层实现)
│   ├── lib.rs              # PyO3 模块入口，注册 Python 模块
│   ├── engine/             # 回测引擎模块
│   │   ├── core.rs         # 核心引擎：驱动时间轴与事件循环
│   │   ├── state.rs        # 引擎状态管理
│   │   └── python.rs       # Python 绑定
│   ├── execution/          # 执行层：模拟交易所撮合逻辑
│   │   ├── matcher.rs      # 撮合器 Trait 定义
│   │   ├── simulated.rs    # 模拟执行客户端
│   │   ├── realtime.rs     # 实盘执行客户端
│   │   └── slippage.rs     # 滑点模型
│   ├── market/             # 市场层：定义费率、T+1 规则、交易时段
│   │   ├── china.rs        # 中国市场规则 (A股/期货)
│   │   ├── simple.rs       # 简单市场规则 (7x24, T+0)
│   │   └── stock.rs/futures.rs # 具体资产逻辑
│   ├── portfolio.rs        # 账户层：管理资金、持仓与可用头寸
│   ├── data/               # 数据层：管理 Bar/Tick 数据流
│   │   ├── feed.rs         # 数据源
│   │   └── aggregator.rs   # K线合成器
│   ├── analysis/           # 分析层：计算绩效指标 (Sharpe, Drawdown)
│   ├── context.rs          # 上下文：用于 Python 回调的数据快照
│   ├── clock.rs            # 时钟模块：统一时间管理
│   ├── event.rs            # 事件系统：定义系统内部事件
│   ├── history.rs          # 历史数据：高效的环形缓冲区管理
│   ├── indicators.rs       # 技术指标：Rust 原生指标实现 (如 SMA)
│   └── model/              # 数据模型：定义基础数据结构
│       ├── mod.rs          # 模型模块定义
│       ├── order.rs        # 订单 (Order) 与成交 (Trade)
│       ├── instrument.rs   # 标的物信息 (Instrument)
│       ├── market_data.rs  # 市场数据 (Bar, Tick)
│       ├── timer.rs        # 定时器事件
│       └── types.rs        # 基础枚举 (Side, Type, ExecutionMode)
├── python/
│   └── akquant/            # Python 包源码 (用户接口)
│       ├── __init__.py     # 导出公共 API
│       ├── strategy.py     # Strategy 基类：封装上下文，提供 ML 训练与交易接口
│       ├── backtest/       # 回测结果处理
│       │   ├── engine.py   # 高级入口 run_backtest
│       │   └── result.py   # BacktestResult 分析与 DataFrame 转换
│       ├── config.py       # 配置定义：BacktestConfig, StrategyConfig, RiskConfig
│       ├── risk.py         # 风控配置适配层
│       ├── data.py         # 数据加载与目录服务 (DataCatalog)
│       ├── sizer.py        # Sizer 基类：提供多种仓位管理实现
│       ├── indicator.py    # Python 指标接口
│       ├── optimize.py     # 参数优化工具
│       ├── plot/           # 绘图工具包
│       ├── log.py          # 日志模块
│       ├── utils/          # 工具函数
│       ├── ml/             # 机器学习框架
│       │   ├── __init__.py
│       │   └── model.py    # QuantModel, SklearnAdapter, ValidationConfig
│       └── akquant.pyi     # 类型提示文件 (IDE 补全支持)
├── tests/                  # 测试用例
└── examples/               # 示例代码
    ├── 01_quickstart.py    # 快速上手
    ├── 07_ml_framework.py  # ML 框架基础示例
    └── ...                 # 其他示例
```

## 2. 核心组件架构详解

### 2.1 数据模型层 (`src/model/`)

为了保证跨语言交互的性能与类型安全，核心数据结构均在 Rust 中定义并导出。

*   **`types.rs`**:
    *   `ExecutionMode`:
        *   `CurrentClose`: 当前 Bar 收盘价成交 (Cheat-on-Close)。
        *   `NextOpen`: 次日开盘价成交 (更真实)。
        *   `NextAverage`: 次日均价成交 (TWAP/VWAP 模拟)。
    *   `OrderSide`: `Buy` / `Sell`。
    *   `OrderType`: `Market` (市价), `Limit` (限价), `StopMarket` (止损市价), `StopLimit` (止损限价)。
    *   `TimeInForce`: `Day` (当日有效), `GTC` (撤前有效), `IOC`/`FOK`。
    *   `AssetType`: `Stock`, `Fund`, `Futures`, `Option`。
*   **`instrument.rs`**: `Instrument` 包含 `multiplier` (合约乘数), `tick_size`, `margin_ratio` 等。
*   **`market_data.rs`**: `Bar` (OHLCV) 和 `Tick` (最新价/量)。

### 2.2 执行层 (`src/execution/`)

`ExecutionClient` Trait 定义了执行层的标准接口，支持模拟和实盘的无缝切换。

*   **`SimulatedExecutionClient` (`simulated.rs`)**:
    *   负责回测时的订单撮合。
    *   内置 `ExecutionMatcher`，支持多种撮合逻辑。
    *   **滑点模型 (`slippage.rs`)**: 支持 `FixedSlippage` (固定金额) 和 `PercentSlippage` (百分比)。
*   **撮合机制 (`matcher.rs`)**:
    *   **限价单 (Limit)**: 买入需 `Low <= Price`，卖出需 `High >= Price`。
    *   **市价单 (Market)**: 根据 `ExecutionMode` 决定按 `Close` 或 `Open` 成交。
    *   **触发机制**: 支持 `trigger_price` (止损/止盈单)。

### 2.3 市场规则层 (`src/market/`)

通过 `MarketModel` Trait 实现不同市场的规则隔离。

*   **`ChinaMarket` (`china.rs`)**:
    *   实现 A 股和国内期货的市场规则。
    *   **交易时段**: 支持集合竞价 (CallAuction)、连续竞价 (Continuous)、休市 (Break) 等状态管理。
    *   **费率计算**: 支持股票 (印花税、过户费、佣金) 和期货 (按手或按金额)。
    *   **T+1/T+0**: 严格的可用持仓管理 (昨仓/今仓)。
*   **`SimpleMarket` (`simple.rs`)**:
    *   7x24 小时交易，T+0，无税，适用于加密货币或外汇回测。

### 2.4 风控层 (`src/risk/`)

`RiskManager` 独立于执行层，拦截每一笔订单：

*   **检查规则**: 限制名单、最大单笔数量/金额、最大持仓比例。
*   **配置**: Python 端 `RiskConfig` 自动注入 Rust 引擎。
*   支持针对不同资产类型 (Stock/Futures/Option) 的特定风控规则。

### 2.5 账户层 (`src/portfolio.rs`)

`Portfolio` 结构体维护账户状态：

*   `cash`: 可用资金。
*   `positions`: 总持仓。
*   `available_positions`: 可卖持仓 (T+1 逻辑的核心)。
*   **权益计算**: 实时 Mark-to-Market 计算。

### 2.6 引擎层 (`src/engine/` & `src/history.rs`)

`Engine` 是系统的驱动器：

*   **事件循环**: 消费 `Bar` 或 `Tick` 事件。
*   **历史数据管理**: `Engine` 内部维护 `History` 模块，这是一个高效的环形缓冲区，用于存储最近 N 个 Bar 的数据，供策略通过 `get_history` 快速访问，无需在 Python 端累积数据。
*   **日切处理**: 触发 T+1 解锁、过期订单清理。

### 2.7 分析层 (`src/analysis/`)

遵循标准 PnL 计算：`Gross PnL` (毛利), `Net PnL` (净利), `Commission` (佣金)。
`BacktestResult` 包含详细的绩效指标 (`PerformanceMetrics`) 和交易记录 (`ClosedTrade`)。

### 2.8 Python 抽象层 (`python/akquant/`)

*   **`Strategy` (`strategy.py`)**:
    *   **历史数据访问**:
        *   `set_history_depth(depth)`: 开启 Rust 端历史数据记录。
        *   `get_history(count)` / `get_history_df(count)`: 获取最近 N 个 Bar 的 OHLCV 数据 (Numpy/DataFrame)。
    *   **ML 集成**:
        *   `set_rolling_window(train, step)`: 配置滚动训练参数。
        *   `on_train_signal(context)`: 周期性触发模型训练。
        *   `prepare_features(df)`: 特征工程接口。
*   **`run_backtest` (`backtest/engine.py`)**: 高级入口函数，统一处理配置注入、策略实例化和引擎启动。
*   **`BacktestResult` (`backtest/result.py`)**:
    *   封装 Rust 返回的 `BacktestResult`。
    *   提供 `metrics_df`, `positions` 等便捷属性。
    *   支持 `plot()` 方法直接生成分析图表。

### 2.9 机器学习框架 (`python/akquant/ml/`)

`AKQuant` 提供了一套标准化的 ML 接口，旨在简化“滚动训练-预测”流程。

*   **`QuantModel` (`model.py`)**:
    *   所有模型的抽象基类。
    *   接口: `fit(X, y)`, `predict(X)`, `save(path)`, `load(path)`。
    *   **`set_validation`**: 配置 Walk-forward Validation 参数 (训练窗口、测试窗口、滚动步长)。
*   **`SklearnAdapter`**:
    *   封装 Scikit-learn 风格的模型 (如 RandomForest, LinearRegression)，使其适配 `QuantModel` 接口。

## 3. 关键工作流详解

### 3.1 回测主循环与执行模式

`Engine::run` 的流程依赖 `ExecutionMode`：

*   **NextOpen**: 推荐模式。Bar Close 生成信号 -> Next Bar Open 成交。
*   **CurrentClose**: 简化模式。Bar Close 生成信号 -> Current Bar Close 成交 (Cheat-on-Close)。
*   **NextAverage**: Bar Close 生成信号 -> Next Bar VWAP/Average 成交。

### 3.2 订单全生命周期

Signal -> Creation -> Submission -> Risk Check (Rust) -> Matching (Rust) -> Settlement (Rust) -> Reporting。

## 4. 扩展开发指南

### 4.1 如何添加新的订单类型

1.  `src/model/types.rs`: 添加枚举。
2.  `src/model/order.rs`: 更新结构体。
3.  `src/execution/matcher.rs`: 实现撮合逻辑。
4.  `akquant.pyi`: 更新类型提示。

### 4.2 如何自定义指标

1.  **Python 侧 (快速原型)**: 继承 `akquant.Indicator`，在 `on_bar` 中计算。
2.  **Rust 侧 (高性能)**:
    *   在 `src/indicators.rs` 中实现 `Indicator` Trait。
    *   通过 `#[pyclass]` 导出到 Python。

### 4.3 如何接入新的数据源

将数据转换为 pandas DataFrame，构造 `akquant.Bar` 对象列表，调用 `engine.add_feed` 即可。

## 5. Rust 与 Python 交互注意事项

*   **GIL**: Rust 仅在回调 Python 时获取 GIL，计算密集型任务释放 GIL (视实现而定，目前主要是单线程模型)。
*   **数据拷贝**: 尽量减少 Python 与 Rust 之间的大规模数据传输。`get_history` 返回的是 Numpy 视图或拷贝，效率远高于 Python list。
