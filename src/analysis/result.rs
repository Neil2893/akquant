use super::types::*;
use crate::model::{Order, Trade};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use serde::{Deserialize, Serialize};

#[gen_stub_pyclass]
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
/// 回测结果.
///
/// :ivar equity_curve: 权益曲线 [(timestamp, equity)]
/// :ivar cash_curve: 现金曲线 [(timestamp, cash)]
/// :ivar metrics: 绩效指标对象
/// :ivar trade_metrics: 交易统计对象
/// :ivar trades: 平仓交易列表
/// :ivar snapshots: 每日持仓快照 [(timestamp, [snapshot])]
/// :ivar orders: 订单列表
/// :ivar executions: 成交列表
pub struct BacktestResult {
    #[pyo3(get)]
    pub equity_curve: Vec<(i64, f64)>,
    #[pyo3(get)]
    pub cash_curve: Vec<(i64, f64)>,
    #[pyo3(get)]
    pub metrics: PerformanceMetrics,
    #[pyo3(get)]
    pub trade_metrics: TradePnL,
    #[pyo3(get)]
    pub trades: Vec<ClosedTrade>,

    #[pyo3(get)]
    pub snapshots: Vec<(i64, Vec<PositionSnapshot>)>,
    #[pyo3(get)]
    pub orders: Vec<Order>,
    #[pyo3(get)]
    pub executions: Vec<Trade>,
}
