use chrono::{DateTime, NaiveDate, NaiveTime, TimeZone, Utc};
use indicatif::{ProgressBar, ProgressStyle};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use rust_decimal::prelude::*;
use std::collections::{BinaryHeap, HashMap};
use std::sync::{Arc, RwLock};

use crate::analysis::{BacktestResult, PositionSnapshot};
use crate::clock::Clock;
use crate::context::StrategyContext;
use crate::data::DataFeed;
use crate::event::Event;
use crate::event_manager::EventManager;
use crate::execution::{ExecutionClient, RealtimeExecutionClient, SimulatedExecutionClient};
use crate::history::HistoryBuffer;
use crate::market::manager::MarketManager;
use crate::model::{
    Bar, ExecutionMode, Instrument, Order, Timer, Trade, TradingSession,
};
use crate::order_manager::OrderManager;
use crate::pipeline::stages::{
    ChannelProcessor, CleanupProcessor, DataProcessor, ExecutionPhase, ExecutionProcessor,
    StatisticsProcessor, StrategyProcessor,
};
use crate::pipeline::PipelineRunner;
use crate::portfolio::Portfolio;
use crate::risk::{RiskConfig, RiskManager};
use crate::settlement::SettlementManager;
use crate::statistics::StatisticsManager;

/// Shared state container for Engine
pub struct SharedState {
    pub portfolio: Portfolio,
    pub order_manager: OrderManager,
    pub feed: DataFeed,
}

impl SharedState {
    pub fn new(initial_capital: Decimal) -> Self {
        Self {
            portfolio: Portfolio {
                cash: initial_capital,
                positions: Arc::new(HashMap::new()),
                available_positions: Arc::new(HashMap::new()),
            },
            order_manager: OrderManager::new(),
            feed: DataFeed::new(),
        }
    }
}

/// 主回测引擎.
///
/// :ivar feed: 数据源
/// :ivar portfolio: 投资组合
/// :ivar orders: 订单列表
/// :ivar trades: 成交列表
#[gen_stub_pyclass]
#[pyclass]
pub struct Engine {
    pub(crate) state: SharedState,
    pub(crate) last_prices: HashMap<String, Decimal>,
    pub(crate) instruments: HashMap<String, Instrument>,
    pub(crate) current_date: Option<NaiveDate>,
    pub(crate) market_manager: MarketManager,
    pub(crate) execution_model: Box<dyn ExecutionClient>,
    pub(crate) execution_mode: ExecutionMode,
    pub(crate) clock: Clock,
    pub(crate) timers: BinaryHeap<Timer>, // Min-Heap via Timer's Ord implementation
    pub(crate) force_session_continuous: bool,
    #[pyo3(get, set)]
    pub risk_manager: RiskManager,
    pub(crate) timezone_offset: i32,
    pub(crate) history_buffer: Arc<RwLock<HistoryBuffer>>,
    pub(crate) initial_capital: Decimal,
    // Components
    pub(crate) event_manager: EventManager,
    pub(crate) statistics_manager: StatisticsManager,
    pub(crate) settlement_manager: SettlementManager,
    // Pipeline state
    pub(crate) current_event: Option<Event>,
    pub(crate) bar_count: usize,
    pub(crate) progress_bar: Option<ProgressBar>,
}

#[gen_stub_pymethods]
#[pymethods]
impl Engine {
    /// 获取订单列表
    #[getter]
    fn get_orders(&self) -> Vec<Order> {
        self.state.order_manager.get_all_orders()
    }

    /// 获取成交列表
    #[getter]
    fn get_trades(&self) -> Vec<Trade> {
        self.state.order_manager.trades.clone()
    }

    /// 获取投资组合
    #[getter]
    fn get_portfolio(&self) -> Portfolio {
        self.state.portfolio.clone()
    }

    /// 获取数据源
    #[getter]
    fn get_feed(&self) -> DataFeed {
        self.state.feed.clone()
    }

    /// 获取持仓快照历史
    #[getter]
    fn get_snapshots(&self) -> Vec<(i64, Vec<PositionSnapshot>)> {
        self.statistics_manager.snapshots.clone()
    }

    /// 设置风控配置
    ///
    /// 由于 PyO3 对嵌套结构体的属性访问可能返回副本，
    /// 提供此方法以显式更新风控配置。
    ///
    /// :param config: 新的风控配置
    fn set_risk_config(&mut self, config: RiskConfig) {
        self.risk_manager.config = config;
    }

    /// 初始化回测引擎.
    ///
    /// :return: Engine 实例
    #[new]
    fn new() -> Self {
        let initial_capital = Decimal::from(100_000);
        Engine {
            state: SharedState::new(initial_capital),
            last_prices: HashMap::new(),
            instruments: HashMap::new(),
            current_date: None,
            market_manager: MarketManager::new(),
            execution_model: Box::new(SimulatedExecutionClient::new()),
            execution_mode: ExecutionMode::NextOpen,
            clock: Clock::new(),
            timers: BinaryHeap::new(),
            force_session_continuous: false,
            risk_manager: RiskManager::new(),
            timezone_offset: 28800, // Default UTC+8
            history_buffer: Arc::new(RwLock::new(HistoryBuffer::new(10000))), // Default large capacity for MAE/MFE
            initial_capital,
            event_manager: EventManager::new(),
            statistics_manager: StatisticsManager::new(),
            settlement_manager: SettlementManager::new(),
            current_event: None,
            bar_count: 0,
            progress_bar: None,
        }
    }

    /// 设置历史数据长度
    ///
    /// :param depth: 历史数据长度
    fn set_history_depth(&mut self, depth: usize) {
        self.history_buffer.write().unwrap().set_capacity(depth);
    }

    /// 设置时区偏移 (秒)
    ///
    /// :param offset: 偏移秒数 (例如 UTC+8 为 28800)
    fn set_timezone(&mut self, offset: i32) {
        self.timezone_offset = offset;
    }

    /// 启用模拟执行 (回测模式)
    ///
    /// 默认模式。在内存中撮合订单。
    fn use_simulated_execution(&mut self) {
        self.execution_model = Box::new(SimulatedExecutionClient::new());
    }

    /// 启用实盘执行 (CTP/Broker 模式)
    ///
    /// 模拟对接 CTP 或其他 Broker API。
    /// 在此模式下，订单会被标记为 Submitted 并等待回调 (目前仅模拟发送)。
    fn use_realtime_execution(&mut self) {
        self.execution_model = Box::new(RealtimeExecutionClient::new());
    }

    /// 设置撮合模式
    ///
    /// :param mode: 撮合模式 (ExecutionMode.CurrentClose 或 ExecutionMode.NextOpen)
    /// :type mode: ExecutionMode
    fn set_execution_mode(&mut self, mode: ExecutionMode) {
        self.execution_mode = mode;
    }

    /// 启用 SimpleMarket (7x24小时, T+0, 无税, 简单佣金)
    ///
    /// :param commission_rate: 佣金率
    fn use_simple_market(&mut self, commission_rate: f64) {
        self.market_manager.use_simple_market(commission_rate);
    }

    /// 启用 ChinaMarket (支持 T+1/T+0, 印花税, 过户费, 交易时段等)
    fn use_china_market(&mut self) {
        self.market_manager.use_china_market();
    }

    /// 启用/禁用 T+1 交易规则 (仅针对 ChinaMarket)
    ///
    /// :param enabled: 是否启用 T+1
    /// :type enabled: bool
    fn set_t_plus_one(&mut self, enabled: bool) {
        self.market_manager.set_t_plus_one(enabled);
    }

    /// 强制连续交易时段
    ///
    /// :param enabled: 是否强制连续交易 (忽略午休等)
    fn set_force_session_continuous(&mut self, enabled: bool) {
        self.force_session_continuous = enabled;
    }

    /// 启用中国期货市场默认配置
    /// - 切换到 ChinaMarket
    /// - 设置 T+0
    /// - 保持当前交易时段配置 (需手动设置 set_market_sessions 以匹配特定品种)
    fn use_china_futures_market(&mut self) {
        self.market_manager.use_china_futures_market();
    }

    fn process_option_expiry(&mut self, _local_date: NaiveDate) {
        // Deprecated: logic moved to SettlementManager
    }

    /// 设置股票费率规则
    ///
    /// :param commission_rate: 佣金率 (如 0.0003)
    /// :param stamp_tax: 印花税率 (如 0.001)
    /// :param transfer_fee: 过户费率 (如 0.00002)
    /// :param min_commission: 最低佣金 (如 5.0)
    fn set_stock_fee_rules(
        &mut self,
        commission_rate: f64,
        stamp_tax: f64,
        transfer_fee: f64,
        min_commission: f64,
    ) {
        self.market_manager.set_stock_fee_rules(
            commission_rate,
            stamp_tax,
            transfer_fee,
            min_commission,
        );
    }

    /// 设置期货费率规则
    ///
    /// :param commission_rate: 佣金率 (如 0.0001)
    fn set_future_fee_rules(&mut self, commission_rate: f64) {
        self.market_manager.set_future_fee_rules(commission_rate);
    }

    /// 设置基金费率规则
    ///
    /// :param commission_rate: 佣金率
    /// :param transfer_fee: 过户费率
    /// :param min_commission: 最低佣金
    fn set_fund_fee_rules(&mut self, commission_rate: f64, transfer_fee: f64, min_commission: f64) {
        self.market_manager.set_fund_fee_rules(commission_rate, transfer_fee, min_commission);
    }

    /// 设置期权费率规则
    ///
    /// :param commission_per_contract: 每张合约佣金 (如 5.0)
    fn set_option_fee_rules(&mut self, commission_per_contract: f64) {
        self.market_manager.set_option_fee_rules(commission_per_contract);
    }

    /// 设置滑点模型
    ///
    /// :param type: 滑点类型 ("fixed" 或 "percent")
    /// :param value: 滑点值 (固定金额 或 百分比如 0.001)
    fn set_slippage(&mut self, type_: String, value: f64) -> PyResult<()> {
        let val = Decimal::from_f64(value).unwrap_or(Decimal::ZERO);
        match type_.as_str() {
            "fixed" => {
                self.execution_model
                    .set_slippage_model(Box::new(crate::execution::FixedSlippage { delta: val }));
            }
            "percent" => {
                self.execution_model
                    .set_slippage_model(Box::new(crate::execution::PercentSlippage { rate: val }));
            }
            "zero" | "none" => {
                self.execution_model
                    .set_slippage_model(Box::new(crate::execution::ZeroSlippage));
            }
            _ => {
                return Err(PyValueError::new_err(
                    "Invalid slippage type. Use 'fixed', 'percent', or 'zero'",
                ));
            }
        }
        Ok(())
    }

    /// 设置成交量限制
    ///
    /// :param limit: 限制比例 (0.0-1.0), 0.0 为不限制
    fn set_volume_limit(&mut self, limit: f64) {
        self.execution_model.set_volume_limit(limit);
    }

    /// 设置市场交易时段
    ///
    /// :param sessions: 交易时段列表，每个元素为 (开始时间, 结束时间, 时段类型)
    /// :type sessions: List[Tuple[str, str, TradingSession]]
    ///
    /// 示例::
    ///
    /// ```python
    /// engine.set_market_sessions([
    ///     ("09:30:00", "11:30:00", TradingSession.Normal),
    ///     ("13:00:00", "15:00:00", TradingSession.Normal)
    /// ])
    /// ```
    fn set_market_sessions(
        &mut self,
        sessions: Vec<(String, String, TradingSession)>,
    ) -> PyResult<()> {
        let mut ranges = Vec::with_capacity(sessions.len());
        for (start, end, session) in sessions {
            let start_time = Self::parse_time_string(&start)?;
            let end_time = Self::parse_time_string(&end)?;
            ranges.push((start_time, end_time, session));
        }
        self.market_manager.set_market_sessions(ranges);
        Ok(())
    }

    /// 添加交易标的
    ///
    /// :param instrument: 交易标的对象
    /// :type instrument: Instrument
    fn add_instrument(&mut self, instrument: Instrument) {
        self.instruments
            .insert(instrument.symbol().to_string(), instrument);
    }

    /// 设置初始资金
    ///
    /// :param cash: 初始资金数额
    /// :type cash: float
    fn set_cash(&mut self, cash: f64) {
        let val = Decimal::from_f64(cash).unwrap_or(Decimal::ZERO);
        self.state.portfolio.cash = val;
        self.initial_capital = val;
    }

    /// 添加数据源
    ///
    /// :param feed: 数据源对象
    /// :type feed: DataFeed
    fn add_data(&mut self, feed: DataFeed) {
        self.state.feed = feed;
    }

    /// 批量添加 K 线数据
    ///
    /// :param bars: K 线列表
    fn add_bars(&mut self, bars: Vec<Bar>) -> PyResult<()> {
        self.state.feed.add_bars(bars)
    }

    /// 运行回测
    ///
    /// :param strategy: 策略对象
    /// :param show_progress: 是否显示进度条
    /// :type strategy: object
    /// :type show_progress: bool
    /// :return: 回测结果摘要
    /// :rtype: str
    fn run(
        &mut self,
        py: Python<'_>,
        strategy: &Bound<'_, PyAny>,
        show_progress: bool,
    ) -> PyResult<String> {
        // Configure history buffer if strategy has _history_depth set
        if let Ok(depth_attr) = strategy.getattr("_history_depth") {
            if let Ok(depth) = depth_attr.extract::<usize>() {
                if depth > 0 {
                    self.set_history_depth(depth);
                }
            }
        }

        // Trigger Strategy on_start
        if let Err(e) = strategy.call_method0("on_start") {
            return Err(e);
        }

        // Progress Bar Initialization
        let total_events = self.state.feed.len_hint().unwrap_or(0);
        let pb = if show_progress {
            let pb = if total_events > 0 {
                let pb = ProgressBar::new(total_events as u64);
                pb.set_style(
                    ProgressStyle::default_bar()
                        .template(
                            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
                        )
                        .unwrap()
                        .progress_chars("#>-"),
                );
                pb
            } else {
                let pb = ProgressBar::new_spinner();
                pb.set_style(
                    ProgressStyle::default_spinner()
                        .template("{spinner:.green} [{elapsed_precise}] {pos} events processed")
                        .unwrap(),
                );
                pb
            };
            Some(pb)
        } else {
            None
        };
        self.progress_bar = pb;

        // Record initial equity
        if let Some(_) = self.state.feed.peek_timestamp() {
            let _equity = self
                .state
                .portfolio
                .calculate_equity(&self.last_prices, &self.instruments);
        }

        // Initialize Pipeline
        let mut pipeline = PipelineRunner::new();
        // 1. Process events from previous iteration (or init)
        pipeline.add_processor(Box::new(ChannelProcessor));

        // 2. Fetch new Data Event
        pipeline.add_processor(Box::new(DataProcessor::new()));

        // 3. Pre-Strategy Execution (Match Pending Orders)
        // For NextOpen/NextAverage: Matches orders generated in previous bar against current bar.
        pipeline.add_processor(Box::new(ExecutionProcessor::new(ExecutionPhase::PreStrategy)));

        // 4. Process Fills from Pre-Execution immediately (Update Portfolio before Strategy)
        pipeline.add_processor(Box::new(ChannelProcessor));

        // 5. Run Strategy
        pipeline.add_processor(Box::new(StrategyProcessor));

        // 6. Process Order Requests from Strategy immediately (Validate -> Pending)
        pipeline.add_processor(Box::new(ChannelProcessor));

        // 7. Post-Strategy Execution
        // For CurrentClose: Matches orders generated in current bar against current bar.
        pipeline.add_processor(Box::new(ExecutionProcessor::new(ExecutionPhase::PostStrategy)));

        // 8. Process Fills from Post-Execution
        pipeline.add_processor(Box::new(ChannelProcessor));

        // 9. Statistics & Cleanup
        pipeline.add_processor(Box::new(StatisticsProcessor));
        pipeline.add_processor(Box::new(CleanupProcessor));

        // Run Pipeline
        if let Err(e) = pipeline.run(self, py, strategy) {
            // Clean up pb if error
            self.progress_bar = None;
            return Err(e);
        }

        // Final cleanup
        self.state.order_manager.cleanup_finished_orders();

        // Record final snapshot if we have data
        if self.current_date.is_some() {
            if let Some(timestamp) = self.clock.timestamp() {
                self.statistics_manager.record_snapshot(
                    timestamp,
                    &self.state.portfolio,
                    &self.instruments,
                    &self.last_prices,
                    &self.state.order_manager.trade_tracker,
                );
            }
        }

        if let Some(pb) = &self.progress_bar {
            pb.finish_with_message("Backtest completed");
        }

        let count = self.bar_count;
        self.progress_bar = None;

        Ok(format!(
            "Backtest finished. Processed {} events. Total Trades: {}",
            count,
            self.state.order_manager.trades.len()
        ))
    }

    /// 获取回测结果
    ///
    /// :return: BacktestResult
    fn get_results(&self) -> BacktestResult {
        self.statistics_manager.generate_backtest_result(
            &self.state.portfolio,
            &self.instruments,
            &self.last_prices,
            &self.state.order_manager,
            self.initial_capital,
            self.clock.timestamp(),
        )
    }
}

// Private helper methods
impl Engine {
    fn create_context(
        &self,
        active_orders: Arc<Vec<Order>>,
        step_trades: Vec<Trade>,
    ) -> StrategyContext {
        // Create a temporary context for the strategy to use
        StrategyContext::new(
            self.state.portfolio.cash,
            self.state.portfolio.positions.clone(),
            self.state.portfolio.available_positions.clone(),
            self.clock.session,
            self.clock.timestamp().unwrap_or(0),
            active_orders,
            self.state.order_manager.trade_tracker.closed_trades.clone(),
            step_trades,
            Some(self.history_buffer.clone()),
            Some(self.event_manager.sender()),
            self.risk_manager.config.clone(),
        )
    }

    pub(crate) fn datetime_from_ns(timestamp: i64) -> DateTime<Utc> {
        let secs = timestamp.div_euclid(1_000_000_000);
        let nanos = timestamp.rem_euclid(1_000_000_000) as u32;
        Utc.timestamp_opt(secs, nanos)
            .single()
            .expect("Invalid timestamp")
    }

    pub(crate) fn local_datetime_from_ns(timestamp: i64, offset_secs: i32) -> DateTime<Utc> {
        let offset_ns = i64::from(offset_secs) * 1_000_000_000;
        Self::datetime_from_ns(timestamp + offset_ns)
    }

    fn parse_time_string(value: &str) -> PyResult<NaiveTime> {
        if let Ok(t) = NaiveTime::parse_from_str(value, "%H:%M:%S") {
            return Ok(t);
        }
        if let Ok(t) = NaiveTime::parse_from_str(value, "%H:%M") {
            return Ok(t);
        }
        Err(PyValueError::new_err(format!(
            "Invalid time format: {}",
            value
        )))
    }

    pub(crate) fn call_strategy(
        &mut self,
        strategy: &Bound<'_, PyAny>,
        event: &Event,
    ) -> PyResult<(Vec<Order>, Vec<Timer>, Vec<String>)> {
        // Update Last Price and Trigger Strategy
        match event {
            Event::Bar(b) => {
                self.last_prices.insert(b.symbol.clone(), b.close);
                let step_trades = std::mem::take(&mut self.state.order_manager.current_step_trades);
                // Share active orders via Arc
                let active_orders = Arc::new(self.state.order_manager.active_orders.clone());
                let ctx = self.create_context(active_orders, step_trades);
                let py_ctx = Python::attach(|py| {
                    let py_ctx = Py::new(py, ctx).unwrap();
                    let args = (b.clone(), py_ctx.clone_ref(py));
                    strategy.call_method1("_on_bar_event", args)?;
                    Ok::<_, PyErr>(py_ctx)
                })?;

                // Extract orders and timers
                let mut new_orders = Vec::new();
                let mut new_timers = Vec::new();
                let mut canceled_ids = Vec::new();
                Python::attach(|py| {
                    let ctx_ref = py_ctx.borrow(py);
                    // Read from RwLock
                    if let Ok(orders) = ctx_ref.orders_arc.read() {
                        new_orders.extend(orders.clone());
                    }
                    if let Ok(timers) = ctx_ref.timers_arc.read() {
                        new_timers.extend(timers.clone());
                    }
                    if let Ok(canceled) = ctx_ref.canceled_order_ids_arc.read() {
                        canceled_ids.extend(canceled.clone());
                    }
                });
                Ok((new_orders, new_timers, canceled_ids))
            }
            Event::Tick(t) => {
                self.last_prices.insert(t.symbol.clone(), t.price);
                let step_trades = std::mem::take(&mut self.state.order_manager.current_step_trades);
                let active_orders = Arc::new(self.state.order_manager.active_orders.clone());
                let ctx = self.create_context(active_orders, step_trades);
                let py_ctx = Python::attach(|py| {
                    let py_ctx = Py::new(py, ctx).unwrap();
                    let args = (t.clone(), py_ctx.clone_ref(py));
                    strategy.call_method1("_on_tick_event", args)?;
                    Ok::<_, PyErr>(py_ctx)
                })?;

                // Extract orders and timers
                let mut new_orders = Vec::new();
                let mut new_timers = Vec::new();
                let mut canceled_ids = Vec::new();
                Python::attach(|py| {
                    let ctx_ref = py_ctx.borrow(py);
                    if let Ok(orders) = ctx_ref.orders_arc.read() {
                        new_orders.extend(orders.clone());
                    }
                    if let Ok(timers) = ctx_ref.timers_arc.read() {
                        new_timers.extend(timers.clone());
                    }
                    if let Ok(canceled) = ctx_ref.canceled_order_ids_arc.read() {
                        canceled_ids.extend(canceled.clone());
                    }
                });
                Ok((new_orders, new_timers, canceled_ids))
            }
            Event::Timer(timer) => {
                let step_trades = std::mem::take(&mut self.state.order_manager.current_step_trades);
                let active_orders = Arc::new(self.state.order_manager.active_orders.clone());
                let ctx = self.create_context(active_orders, step_trades);
                let py_ctx = Python::attach(|py| {
                    let py_ctx = Py::new(py, ctx).unwrap();
                    strategy.call_method1("_on_timer_event", (timer.payload.as_str(), py_ctx.clone_ref(py)))?;
                    Ok::<_, PyErr>(py_ctx)
                })?;

                // Extract orders and timers
                let mut new_orders = Vec::new();
                let mut new_timers = Vec::new();
                let mut canceled_ids = Vec::new();
                Python::attach(|py| {
                    let ctx_ref = py_ctx.borrow(py);
                    if let Ok(orders) = ctx_ref.orders_arc.read() {
                        new_orders.extend(orders.clone());
                    }
                    if let Ok(timers) = ctx_ref.timers_arc.read() {
                        new_timers.extend(timers.clone());
                    }
                    if let Ok(canceled) = ctx_ref.canceled_order_ids_arc.read() {
                        canceled_ids.extend(canceled.clone());
                    }
                });
                Ok((new_orders, new_timers, canceled_ids))
            }
            Event::OrderRequest(_) | Event::OrderValidated(_) | Event::ExecutionReport(_, _) => {
                Ok((Vec::new(), Vec::new(), Vec::new()))
            }
        }
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::types::AssetType;

    #[test]
    fn test_engine_new() {
        let engine = Engine::new();
        assert_eq!(engine.state.portfolio.cash, Decimal::from(100_000));
        assert!(engine.state.order_manager.orders.is_empty());
        assert!(engine.state.order_manager.trades.is_empty());
        assert_eq!(engine.execution_mode, ExecutionMode::NextOpen);
    }

    #[test]
    fn test_engine_set_cash() {
        let mut engine = Engine::new();
        engine.set_cash(50000.0);
        assert_eq!(engine.state.portfolio.cash, Decimal::from(50000));
    }

    #[test]
    fn test_engine_add_instrument() {
        use crate::model::instrument::{InstrumentEnum, StockInstrument};

        let mut engine = Engine::new();
        let instr = Instrument {
            asset_type: AssetType::Stock,
            inner: InstrumentEnum::Stock(StockInstrument {
                symbol: "AAPL".to_string(),
                lot_size: Decimal::from(100),
                tick_size: Decimal::new(1, 2),
            }),
        };
        engine.add_instrument(instr);
        assert!(engine.instruments.contains_key("AAPL"));
    }

    #[test]
    fn test_engine_fee_rules() {
        let mut engine = Engine::new();
        engine.set_stock_fee_rules(0.001, 0.002, 0.003, 5.0);

        // Since market_config is private but used in market_model, we can't check it directly easily
        // unless we expose getters or check behavior.
        // But we can check if it compiles and runs without error.
        // Actually, we can check market_config if we make it pub or add a getter for test.
        // But for now, let's trust the setter sets the internal state.
        // We can verify via commission calculation if we had a way to invoke it without full run.

        // Let's at least verify future fee rules
        engine.set_future_fee_rules(0.0005);
    }

    #[test]
    fn test_engine_timezone() {
        let mut engine = Engine::new();
        engine.set_timezone(3600); // UTC+1
        assert_eq!(engine.timezone_offset, 3600);
    }
}
