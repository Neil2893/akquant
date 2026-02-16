use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use rust_decimal::prelude::*;
use std::collections::HashMap;

use super::feed::DataFeed;

#[derive(Debug, Clone)]
struct ActiveBar {
    open: Decimal,
    high: Decimal,
    low: Decimal,
    close: Decimal,
    #[allow(dead_code)]
    volume_base: Decimal,
    volume_curr: Decimal,
    timestamp_min: i64,
}

#[gen_stub_pyclass]
#[pyclass]
/// K 线合成器.
///
/// 用于将 Tick 数据合成为 K 线 (Bar) 数据。
pub struct BarAggregator {
    feed: DataFeed,
    active_bars: HashMap<String, ActiveBar>,
    last_cumulative_volumes: HashMap<String, Decimal>,
    interval_min: i64,
}

#[gen_stub_pymethods]
#[pymethods]
impl BarAggregator {
    /// 创建 K 线合成器.
    ///
    /// :param feed: 数据源 (合成的 Bar 将添加到此数据源)
    /// :param interval_min: K 线周期 (分钟, 默认 1 分钟)
    #[new]
    #[pyo3(signature = (feed, interval_min=None))]
    pub fn new(feed: DataFeed, interval_min: Option<i64>) -> Self {
        Self {
            feed,
            active_bars: HashMap::new(),
            last_cumulative_volumes: HashMap::new(),
            interval_min: interval_min.unwrap_or(1),
        }
    }

    /// 处理新的 Tick 数据
    ///
    /// :param symbol: 标的代码
    /// :param price: 最新价
    /// :param volume: 累计成交量 (TotalVolume)
    /// :param timestamp_ns: 时间戳 (纳秒)
    pub fn on_tick(
        &mut self,
        symbol: String,
        price: f64,
        volume: f64,
        timestamp_ns: i64,
    ) -> PyResult<()> {
        let price = Decimal::from_f64(price).unwrap_or(Decimal::ZERO);
        let volume = Decimal::from_f64(volume).unwrap_or(Decimal::ZERO);

        // Calculate current interval key
        let current_key = timestamp_ns / 1_000_000_000 / 60 / self.interval_min;

        // Initialize last cumulative volume
        if !self.last_cumulative_volumes.contains_key(&symbol) {
            self.last_cumulative_volumes.insert(symbol.clone(), volume);
        }

        let last_cum_vol = *self.last_cumulative_volumes.get(&symbol).unwrap();
        let volume_delta = if volume >= last_cum_vol {
            volume - last_cum_vol
        } else {
            // Volume reset (e.g., new day or issue)
            volume
        };
        self.last_cumulative_volumes.insert(symbol.clone(), volume);

        if let Some(bar) = self.active_bars.get_mut(&symbol) {
            if bar.timestamp_min == current_key {
                // Update current bar
                if price > bar.high {
                    bar.high = price;
                }
                if price < bar.low {
                    bar.low = price;
                }
                bar.close = price;
                bar.volume_curr += volume_delta;
            } else {
                // Close current bar and start new one
                let finished_bar = crate::model::Bar {
                    timestamp: bar.timestamp_min * 60 * 1_000_000_000 * self.interval_min,
                    open: bar.open,
                    high: bar.high,
                    low: bar.low,
                    close: bar.close,
                    volume: bar.volume_curr,
                    symbol: symbol.clone(),
                    extra: HashMap::new(),
                };
                self.feed.add_bar(finished_bar)?;

                // Start new bar
                let new_bar = ActiveBar {
                    open: price,
                    high: price,
                    low: price,
                    close: price,
                    volume_base: volume,
                    volume_curr: volume_delta,
                    timestamp_min: current_key,
                };
                self.active_bars.insert(symbol.clone(), new_bar);
            }
        } else {
            // First bar
            let new_bar = ActiveBar {
                open: price,
                high: price,
                low: price,
                close: price,
                volume_base: volume,
                volume_curr: volume_delta,
                timestamp_min: current_key,
            };
            self.active_bars.insert(symbol.clone(), new_bar);
        }

        Ok(())
    }
}
