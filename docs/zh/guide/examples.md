# 示例集合

## 1. 基础示例 (Basic Examples)

*   [快速开始 (Quickstart)](../start/quickstart.md): 包含手动数据回测和 AKShare 数据回测的完整流程。
*   [简单的均线策略 (SMA Strategy)](strategy.md#class-based): 展示了如何使用类风格编写策略，并在 `on_bar` 中进行简单的交易逻辑。

> 数据源约定：除特别标注需要模拟数据外，本页示例默认使用 AKShare 获取真实市场数据。

## 2. 进阶示例 (Advanced Examples)

*   **Zipline 风格策略**: 展示了如何使用函数式 API (`initialize`, `on_bar`) 编写策略，适合从 Zipline 迁移的用户。
    *   参考 [策略指南](strategy.md#style-selection)。

*   **多品种回测 (Multi-Asset)**:
    *   **期货策略**: 展示期货回测配置（保证金、乘数）。参考 [策略指南](strategy.md#multi-asset)。
    *   **期权策略**: 展示期权回测配置（权利金、按张收费）。参考 [策略指南](strategy.md#multi-asset)。

*   **向量化指标 (Vectorized Indicators)**:
    *   展示如何使用 `IndicatorSet` 预计算指标以提高回测速度。参考 [策略指南](strategy.md#indicatorset)。

### 使用 AKShare 获取 A 股日线数据 (stock_zh_a_daily)

```python
import akshare as ak
import pandas as pd
from akquant import run_backtest

df = ak.stock_zh_a_daily(symbol="sz000001", adjust="qfq")
if "date" not in df.columns:
    df = df.reset_index().rename(columns={"index": "date"})
df.columns = [c.lower() for c in df.columns]
if "time" in df.columns and "date" not in df.columns:
    df = df.rename(columns={"time": "date"})
df["date"] = pd.to_datetime(df["date"]).dt.tz_localize("Asia/Shanghai")
df["symbol"] = "000001"
cols = ["date", "open", "high", "low", "close", "volume", "symbol"]
df = df[cols].sort_values("date").reset_index(drop=True)

# result = run_backtest(data=df, strategy=DualSMAStrategy, lot_size=100)
```

## 3. 常用策略示例 (Common Strategies)

以下是一些常用量化策略的实现代码，可以直接在您的项目中使用。我们为每个策略提供了详细的逻辑说明，帮助您理解其核心思想。

### 使用后复权序列作信号，真实价格撮合

当数据包含 `adj_close` 或 `adj_factor` 时，可直接通过 `get_history(symbol=..., field="adj_close", n)` 获取后复权序列用于信号计算，而撮合与估值仍使用真实收盘价 `close`。示例见仓库 `examples/13_adj_returns_signal.py`。

```python
import akquant as aq
from akquant import Strategy

class AdjSignal(Strategy):
    warmup_period = 5
    def on_bar(self, bar):
        try:
            x = self.get_history(2, bar.symbol, "adj_close")
        except Exception:
            return
        if x is None or len(x) < 2:
            return
        r = x[-1] / x[-2] - 1.0
        pos = self.get_position(bar.symbol)
        if pos == 0 and r > 0:
            self.buy(bar.symbol, 100)
        elif pos > 0 and r < 0:
            self.close_position(bar.symbol)
```

运行该示例（AKShare 数据）：

```python
import akshare as ak
import pandas as pd
from akquant import run_backtest

# 未复权收盘价 (用于撮合/估值)
df_raw = ak.stock_zh_a_daily(symbol="sz000001")
if "date" not in df_raw.columns:
    df_raw = df_raw.reset_index().rename(columns={"index": "date"})
df_raw.columns = [c.lower() for c in df_raw.columns]
if "time" in df_raw.columns and "date" not in df_raw.columns:
    df_raw = df_raw.rename(columns={"time": "date"})
df_raw["date"] = pd.to_datetime(df_raw["date"]).dt.tz_localize("Asia/Shanghai")
df_raw["symbol"] = "000001"
df_raw = df_raw[["date", "open", "high", "low", "close", "volume", "symbol"]]

# 前复权收盘价 (用于信号): 作为 adj_close 列
df_adj = ak.stock_zh_a_daily(symbol="sz000001", adjust="qfq")
if "date" not in df_adj.columns:
    df_adj = df_adj.reset_index().rename(columns={"index": "date"})
df_adj.columns = [c.lower() for c in df_adj.columns]
if "time" in df_adj.columns and "date" not in df_adj.columns:
    df_adj = df_adj.rename(columns={"time": "date"})
df_adj["date"] = pd.to_datetime(df_adj["date"]).dt.tz_localize("Asia/Shanghai")
df_adj = df_adj[["date", "close"]].rename(columns={"close": "adj_close"})

# 合并两者，既有真实收盘 close，又有后复权 adj_close
df = pd.merge(df_raw, df_adj, on="date", how="inner").sort_values("date").reset_index(drop=True)

result = run_backtest(
    data=df,
    strategy=AdjSignal,
    lot_size=100
)
```

### 3.1 双均线策略 (Dual Moving Average)

**核心思想**：
双均线策略利用两条不同周期的移动平均线（SMA）来判断市场趋势。
*   **短期均线**（如 5 日）：反应灵敏，紧跟价格波动。
*   **长期均线**（如 20 日）：反应迟钝，代表长期趋势。

**交易信号**：
*   **金叉 (Golden Cross)**：当短期均线 **上穿** 长期均线时，表明短期趋势走强，是 **买入** 信号。
*   **死叉 (Death Cross)**：当短期均线 **下穿** 长期均线时，表明短期趋势走弱，是 **卖出** 信号。

本示例使用了 Rust 实现的高性能增量指标 `aq.SMA`，计算速度极快。

```python
import akquant as aq

class DualSMAStrategy(aq.Strategy):
    def __init__(self, short_window=5, long_window=20):
        self.sma_short = aq.SMA(short_window)
        self.sma_long = aq.SMA(long_window)

    def on_bar(self, bar: aq.Bar):
        short_val = self.sma_short.update(bar.close)
        long_val = self.sma_long.update(bar.close)

        if short_val is None or long_val is None:
            return

        position = self.get_position(bar.symbol)

        if short_val > long_val and position == 0:
            self.buy(bar.symbol, 100)

        elif short_val < long_val and position > 0:
            self.sell(bar.symbol, 100)
```

运行该策略（AKShare 数据）：

```python
import akshare as ak
import pandas as pd
from akquant import run_backtest

df = ak.stock_zh_a_daily(symbol="sz000001", adjust="qfq")
if "date" not in df.columns:
    df = df.reset_index().rename(columns={"index": "date"})
df.columns = [c.lower() for c in df.columns]
if "time" in df.columns and "date" not in df.columns:
    df = df.rename(columns={"time": "date"})
df["date"] = pd.to_datetime(df["date"]).dt.tz_localize("Asia/Shanghai")
df["symbol"] = "000001"
df = df[["date", "open", "high", "low", "close", "volume", "symbol"]].sort_values("date")

result = run_backtest(data=df, strategy=DualSMAStrategy, lot_size=100)
```

### 3.2 RSI 均值回归策略 (RSI Mean Reversion)

**核心思想**：
RSI (相对强弱指标) 是一种动量指标，数值范围在 0 到 100 之间，用于衡量近期价格变化的幅度。
*   **均值回归 (Mean Reversion)**：该策略假设价格不会一直涨或一直跌，过度偏离后会回归正常水平。
*   **超卖 (Oversold)**：RSI 低于某个阈值（如 30），意味着近期跌幅过大，可能反弹 -> **买入**。
*   **超买 (Overbought)**：RSI 高于某个阈值（如 70），意味着近期涨幅过大，可能回调 -> **卖出**。

本示例展示了如何利用 `get_history_df` 获取历史数据，并结合 `pandas` 计算复杂指标。

```python
import akquant as aq
import pandas as pd
import numpy as np

class RSIStrategy(aq.Strategy):
    def __init__(self, period=14, buy_threshold=30, sell_threshold=70):
        self.period = period
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.set_history_depth(period + 20)

    def calculate_rsi(self, prices: pd.Series) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def on_bar(self, bar: aq.Bar):
        history = self.get_history_df(self.period + 20, bar.symbol)

        if len(history) < self.period + 1:
            return

        rsi_series = self.calculate_rsi(history['close'])
        current_rsi = rsi_series.iloc[-1]

        if np.isnan(current_rsi):
            return

        position = self.get_position(bar.symbol)

        if current_rsi < self.buy_threshold and position == 0:
            self.buy(bar.symbol, 100)

        elif current_rsi > self.sell_threshold and position > 0:
            self.sell(bar.symbol, 100)
```

运行该策略（AKShare 数据）：

```python
import akshare as ak
import pandas as pd
from akquant import run_backtest

df = ak.stock_zh_a_daily(symbol="sz000001", adjust="qfq")
if "date" not in df.columns:
    df = df.reset_index().rename(columns={"index": "date"})
df.columns = [c.lower() for c in df.columns]
if "time" in df.columns and "date" not in df.columns:
    df = df.rename(columns={"time": "date"})
df["date"] = pd.to_datetime(df["date"]).dt.tz_localize("Asia/Shanghai")
df["symbol"] = "000001"
df = df[["date", "open", "high", "low", "close", "volume", "symbol"]].sort_values("date")

result = run_backtest(data=df, strategy=RSIStrategy, lot_size=100)
```

### 3.3 布林带策略 (Bollinger Bands)

**核心思想**：
布林带由三条轨道线组成：
*   **中轨**：N 日移动平均线。
*   **上轨**：中轨 + K 倍标准差。
*   **下轨**：中轨 - K 倍标准差。

根据统计学原理，价格有很大（如 95%）的概率落在上下轨之间。
*   当价格**跌破下轨**时，通常被视为非理性的**超卖**状态，价格可能会回归中轨 -> **买入**。
*   当价格**突破上轨**时，通常被视为非理性的**超买**状态，价格可能会回调 -> **卖出**。

```python
import akquant as aq
import pandas as pd

class BollingerStrategy(aq.Strategy):
    def __init__(self, window=20, num_std=2):
        self.window = window
        self.num_std = num_std
        self.set_history_depth(window + 5)

    def on_bar(self, bar: aq.Bar):
        history = self.get_history_df(self.window, bar.symbol)
        if len(history) < self.window:
            return

        close_prices = history['close']
        ma = close_prices.mean()
        std = close_prices.std()
        upper_band = ma + self.num_std * std
        lower_band = ma - self.num_std * std

        position = self.get_position(bar.symbol)
        current_price = bar.close

        if current_price < lower_band and position == 0:
            self.buy(bar.symbol, 100)

        elif current_price > upper_band and position > 0:
            self.sell(bar.symbol, 100)
```

运行该策略（AKShare 数据）：

```python
import akshare as ak
import pandas as pd
from akquant import run_backtest

df = ak.stock_zh_a_daily(symbol="sz000001", adjust="qfq")
if "date" not in df.columns:
    df = df.reset_index().rename(columns={"index": "date"})
df.columns = [c.lower() for c in df.columns]
if "time" in df.columns and "date" not in df.columns:
    df = df.rename(columns={"time": "date"})
df["date"] = pd.to_datetime(df["date"]).dt.tz_localize("Asia/Shanghai")
df["symbol"] = "000001"
df = df[["date", "open", "high", "low", "close", "volume", "symbol"]].sort_values("date")

result = run_backtest(data=df, strategy=BollingerStrategy, lot_size=100)
```

### 3.4 混合资产回测 (Mixed Asset Backtest) {: #mixed-asset }

**核心思想**：
在实际交易中，策略可能同时涉及股票、期货、期权等多种资产。不同资产的交易属性不同：
*   **股票**：通常 1 手 = 100 股，全额交易。
*   **期货**：有**合约乘数**（如 1 点 = 300 元）和**保证金比例**（如 10% 资金即可买入合约）。

本示例展示了如何使用 `InstrumentConfig` 来配置期货的特殊属性，并在同一个策略中混合交易股票和期货（此示例为演示期货参数，保留模拟数据）。

```python
import akquant as aq
from akquant import InstrumentConfig
import pandas as pd
import numpy as np

# 1. 准备数据 (模拟数据)
def create_dummy_data(symbol, start_time, n_bars, price=100.0):
    dates = pd.date_range(start_time, periods=n_bars, freq="B")
    np.random.seed(42)
    changes = np.random.randn(n_bars)
    prices = price + np.cumsum(changes)

    df = pd.DataFrame({
        "open": prices, "high": prices + 1, "low": prices - 1, "close": prices,
        "volume": 1000, "symbol": symbol
    }, index=dates)
    return df

class TestStrategy(aq.Strategy):
    def __init__(self):
        self.count = 0

    def on_bar(self, bar: aq.Bar):
        # 简单逻辑: 前两根 Bar 分别买入股票和期货
        if self.count < 2:
            print(f"[{bar.timestamp}] Buying {bar.symbol}")
            self.buy(bar.symbol, 1)
        self.count += 1

# 2. 生成数据
df_stock = create_dummy_data("STOCK_A", "2023-01-01", 100, 100.0)
df_future = create_dummy_data("FUTURE_B", "2023-01-01", 100, 3500.0)
data = {"STOCK_A": df_stock, "FUTURE_B": df_future}

# 3. 配置期货参数
# 这里告诉回测引擎：FUTURE_B 是一个期货，乘数 300，保证金 10%
future_config = InstrumentConfig(
    symbol="FUTURE_B",
    asset_type="FUTURES",
    multiplier=300, # 股指期货乘数
    margin_ratio=0.1 # 10% 保证金
)

# 4. 运行
run_backtest(
    data=data,
    strategy=TestStrategy,
    instruments_config=[future_config]
)
```

可选：使用 AKShare 替换股票数据

```python
import akshare as ak
import pandas as pd

df_stock = ak.stock_zh_a_daily(symbol="sz000001", adjust="qfq")
if "date" not in df_stock.columns:
    df_stock = df_stock.reset_index().rename(columns={"index": "date"})
df_stock.columns = [c.lower() for c in df_stock.columns]
if "time" in df_stock.columns and "date" not in df_stock.columns:
    df_stock = df_stock.rename(columns={"time": "date"})
df_stock["date"] = pd.to_datetime(df_stock["date"]).dt.tz_localize("Asia/Shanghai")
df_stock["symbol"] = "000001"
df_stock = df_stock[["date", "open", "high", "low", "close", "volume", "symbol"]]

# 与模拟的期货数据组合
data = {"000001": df_stock, "FUTURE_B": df_future}
```

## 4. 复杂订单与风控 (Complex Orders) {: #complex-orders }

**核心思想**：
高级交易通常需要精细的订单管理。
*   **Bracket Order (括号订单)**：这是一种“进场 + 止损 + 止盈”的组合单。当你开仓（进场）后，立刻为这笔持仓设置“止损单”和“止盈单”，像括号一样把价格包在中间。
*   **OCO (One-Cancels-Other)**：指“止损单”和“止盈单”之间的关系。如果价格上涨触发了止盈，那么止损单就应该自动取消（因为仓位已经平了，不需要再止损了），反之亦然。

虽然 AKQuant 的核心撮合引擎尚未原生内置 OCO 订单类型，但你可以通过策略层的回调函数 (`on_trade`, `on_order`) 轻松实现这些高级逻辑。

### 4.1 OCO 与 Bracket Order 实现

**逻辑流程**：
1.  **进场**：策略发出开仓信号。
2.  **成交回调 (`on_trade`)**：一旦开仓单成交，立即发出两笔平仓单：
    *   **止损单 (Stop Loss)**：价格跌到 X 时卖出（保护本金）。
    *   **止盈单 (Take Profit)**：价格涨到 Y 时卖出（锁定利润）。
3.  **后续成交**：
    *   如果止损单成交 -> 立即取消止盈单。
    *   如果止盈单成交 -> 立即取消止损单。

```python
def on_trade(self, trade):
    # 1. 进场单成交 -> 立即挂止损和止盈
    if trade.order_id == self.entry_order_id:
        # 下达止损单 (Stop Market: 达到触发价后以市价卖出)
        self.stop_loss_id = self.sell(
            trade.symbol, trade.quantity,
            trigger_price=trade.price * 0.98, # 止损价 (成本价 - 2%)
            price=None # None 表示触发后市价卖出
        )

        # 下达止盈单 (Limit Sell: 达到指定价格卖出)
        self.take_profit_id = self.sell(
            trade.symbol, trade.quantity,
            price=trade.price * 1.05 # 止盈价 (成本价 + 5%)
        )

    # 2. 止损成交 -> 取消止盈
    # 说明我们已经止损离场了，之前挂的止盈单需要撤销
    elif trade.order_id == self.stop_loss_id:
        self.cancel_order(self.take_profit_id)

    # 3. 止盈成交 -> 取消止损
    # 说明我们已经止盈离场了，之前挂的止损单需要撤销
    elif trade.order_id == self.take_profit_id:
        self.cancel_order(self.stop_loss_id)
```

!!! tip "参数优化"
    该策略的 `stop_loss_pct` 和 `take_profit_pct` 参数可以通过 `akquant.run_optimization` 进行网格搜索优化。

    ```python
    from akquant import run_optimization
    from examples.complex_orders import BracketStrategy

    param_grid = {
        "stop_loss_pct": [0.01, 0.02, 0.03],
        "take_profit_pct": [0.03, 0.05, 0.08]
    }

    results = run_optimization(BracketStrategy, param_grid, data=df)
    ```

完整代码请参考 [examples/05_complex_orders.py](file:///examples/05_complex_orders.py)。

> **注意**: `buy` / `sell` / `stop_buy` / `stop_sell` 方法都会返回唯一的 `order_id` (str)，你可以利用这个 ID 在 `on_trade` 和 `on_order` 回调中精确追踪每个订单的状态。
