use crate::event::Event;
use crate::execution::matcher::ExecutionMatcher;
use crate::execution::slippage::{SlippageModel, ZeroSlippage};
use crate::execution::{futures, option, stock, ExecutionClient};
use crate::model::{AssetType, Order, OrderStatus, TimeInForce, TradingSession};
use rust_decimal::prelude::*;
use rust_decimal::Decimal;

/// 模拟交易所执行器 (Simulated Execution Client)
/// 负责在内存中撮合订单 (回测模式)
pub struct SimulatedExecutionClient {
    slippage_model: Box<dyn SlippageModel>,
    volume_limit_pct: Decimal, // 成交量限制比例 (0.0 = 不限制)
    pending_orders: Vec<Order>,
    // Matchers
    stock_matcher: stock::StockMatcher,
    futures_matcher: futures::FuturesMatcher,
    option_matcher: option::OptionMatcher,
}

impl SimulatedExecutionClient {
    pub fn new() -> Self {
        SimulatedExecutionClient {
            slippage_model: Box::new(ZeroSlippage),
            volume_limit_pct: Decimal::ZERO,
            pending_orders: Vec::new(),
            stock_matcher: stock::StockMatcher,
            futures_matcher: futures::FuturesMatcher,
            option_matcher: option::OptionMatcher,
        }
    }
}

impl Default for SimulatedExecutionClient {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutionClient for SimulatedExecutionClient {
    fn set_slippage_model(&mut self, model: Box<dyn SlippageModel>) {
        self.slippage_model = model;
    }

    fn set_volume_limit(&mut self, limit: f64) {
        self.volume_limit_pct = Decimal::from_f64(limit).unwrap_or(Decimal::ZERO);
    }

    fn on_order(&mut self, order: Order) {
        // 模拟交易所接收订单
        let mut order = order;
        if order.status == OrderStatus::New {
            order.status = OrderStatus::Submitted;
        }
        self.pending_orders.push(order);
    }

    fn on_cancel(&mut self, order_id: &str) {
        if let Some(order) = self.pending_orders.iter_mut().find(|o| o.id == order_id) {
            if order.status == OrderStatus::Submitted || order.status == OrderStatus::New {
                order.status = OrderStatus::Cancelled;
            }
        }
    }

    fn on_event(
        &mut self,
        event: &Event,
        instruments: &std::collections::HashMap<String, crate::model::Instrument>,
        portfolio: &crate::portfolio::Portfolio,
        last_prices: &std::collections::HashMap<String, rust_decimal::Decimal>,
        // risk_manager: &crate::risk::RiskManager,
        market_model: &dyn crate::market::MarketModel,
        execution_mode: crate::model::ExecutionMode,
        bar_index: usize,
        session: TradingSession,
    ) -> Vec<Event> {
        let mut reports = Vec::new();

        // Skip matching during non-trading sessions
        if session == TradingSession::Break
            || session == TradingSession::Closed
            || session == TradingSession::PreOpen
            || session == TradingSession::PostClose
        {
            // Also check for Day orders expiry if Closed
            if session == TradingSession::Closed {
                let timestamp = match event {
                    Event::Bar(b) => b.timestamp,
                    Event::Tick(t) => t.timestamp,
                    _ => 0,
                };

                for order in self.pending_orders.iter_mut() {
                    if order.status != OrderStatus::Cancelled
                        && order.status != OrderStatus::Filled
                        && order.status != OrderStatus::Rejected
                        && order.status != OrderStatus::Expired
                    {
                         if order.time_in_force == TimeInForce::Day {
                             order.status = OrderStatus::Expired; // Use Expired status
                             order.updated_at = timestamp;
                             reports.push(Event::ExecutionReport(order.clone(), None));
                         }
                    }
                }
            }
            return reports;
        }

        // Track available margin for this step (snapshot of portfolio + changes in this loop)
        let mut current_free_margin = portfolio.calculate_free_margin(last_prices, instruments);

        // 实际撮合逻辑：遍历所有挂单，看当前 Event 是否满足成交条件
        for order in self.pending_orders.iter_mut() {
            // Check for cancellation first
            if order.status == OrderStatus::Cancelled {
                reports.push(Event::ExecutionReport(order.clone(), None));
                continue;
            }

            if order.status != OrderStatus::New && order.status != OrderStatus::Submitted {
                continue;
            }

            // Find Instrument
            if let Some(instrument) = instruments.get(&order.symbol) {
                // Dispatch to specific matcher
                let report_opt = match instrument.asset_type {
                    AssetType::Stock | AssetType::Fund => self.stock_matcher.match_order(
                        order,
                        event,
                        instrument,
                        execution_mode,
                        self.slippage_model.as_ref(),
                        self.volume_limit_pct,
                        bar_index,
                    ),
                    AssetType::Futures => self.futures_matcher.match_order(
                        order,
                        event,
                        instrument,
                        execution_mode,
                        self.slippage_model.as_ref(),
                        self.volume_limit_pct,
                        bar_index,
                    ),
                    AssetType::Option => self.option_matcher.match_order(
                        order,
                        event,
                        instrument,
                        execution_mode,
                        self.slippage_model.as_ref(),
                        self.volume_limit_pct,
                        bar_index,
                    ),
                };

                if let Some(mut report) = report_opt {
                    // Check for dynamic position sizing (margin check)
                    if let Event::ExecutionReport(ref mut _r_order, Some(ref mut trade)) = report {
                        if trade.side == crate::model::OrderSide::Buy {
                            // Calculate estimated commission
                            let commission = market_model.calculate_commission(
                                instrument,
                                trade.side,
                                trade.price,
                                trade.quantity,
                            );

                            let multiplier = instrument.multiplier();
                            let margin_ratio = instrument.margin_ratio();

                            // Margin Required = Price * Qty * Multiplier * MarginRatio
                            let margin_required = trade.price * trade.quantity * multiplier * margin_ratio;
                            let total_required = margin_required + commission;

                            if total_required > current_free_margin {
                                // Insufficient margin, reduce quantity
                                let unit_margin = trade.price * multiplier * margin_ratio;
                                let mut max_qty = if unit_margin > Decimal::ZERO {
                                    if current_free_margin <= Decimal::ZERO {
                                        Decimal::ZERO
                                    } else {
                                        current_free_margin / unit_margin
                                    }
                                } else {
                                    Decimal::ZERO
                                };

                                // Apply safety factor (Default 0.9999 if risk manager not available)
                                let safety_margin = 0.0001; // risk_manager.config.safety_margin;
                                let safety_factor = Decimal::from_f64(1.0 - safety_margin)
                                    .unwrap_or(Decimal::from_f64(0.9999).unwrap());
                                max_qty = max_qty * safety_factor;

                                let lot_size = instrument.lot_size();
                                let mut new_qty = max_qty.floor();
                                if lot_size > Decimal::ZERO {
                                    new_qty = new_qty - (new_qty % lot_size);
                                }

                                // Recalculate to ensure it fits
                                let new_comm = market_model.calculate_commission(
                                    instrument,
                                    trade.side,
                                    trade.price,
                                    new_qty,
                                );
                                let new_margin = trade.price * new_qty * multiplier * margin_ratio;

                                if new_margin + new_comm > current_free_margin {
                                    if new_qty >= lot_size {
                                        new_qty -= lot_size;
                                    } else {
                                        new_qty = Decimal::ZERO;
                                    }
                                }

                                // Update trade quantity
                                trade.quantity = new_qty;
                            }

                            // Deduct cost from current_free_margin for next orders in this loop
                            if trade.quantity > Decimal::ZERO {
                                let final_comm = market_model.calculate_commission(
                                    instrument,
                                    trade.side,
                                    trade.price,
                                    trade.quantity,
                                );
                                let final_margin = trade.price * trade.quantity * multiplier * margin_ratio;
                                current_free_margin -= final_margin + final_comm;
                            }
                        }
                    }

                    // Filter out zero quantity trades
                    let mut keep_report = true;
                    if let Event::ExecutionReport(_, Some(ref trade)) = report {
                        if trade.quantity <= Decimal::ZERO {
                            keep_report = false;
                        }
                    }

                    if keep_report {
                        reports.push(report);
                    }
                }
            } else {
                 // No instrument info, skip
            }
        }

        // Cleanup filled/cancelled/rejected orders from pending list
        // Note: We need to do this carefully because `reports` contains the updated order state.
        // We should mark orders as Filled in `self.pending_orders` if they are reported as Filled.
        for report in &reports {
            if let Event::ExecutionReport(updated_order, _) = report {
                if let Some(existing) = self.pending_orders.iter_mut().find(|o| o.id == updated_order.id) {
                    existing.status = updated_order.status;
                    existing.filled_quantity = updated_order.filled_quantity;
                }
            }
        }

        self.pending_orders.retain(|o| {
            o.status != OrderStatus::Filled
                && o.status != OrderStatus::Cancelled
                && o.status != OrderStatus::Rejected
                && o.status != OrderStatus::Expired
        });

        reports
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{AssetType, Bar, Instrument, TimeInForce, ExecutionMode};
    use std::collections::HashMap;

    fn create_test_instruments() -> HashMap<String, Instrument> {
        use crate::model::instrument::{InstrumentEnum, StockInstrument};

        let mut map = HashMap::new();
        let aapl = Instrument {
            asset_type: AssetType::Stock,
            inner: InstrumentEnum::Stock(StockInstrument {
                symbol: "AAPL".to_string(),
                lot_size: Decimal::from(100),
                tick_size: Decimal::new(1, 2),
            }),
        };
        map.insert("AAPL".to_string(), aapl);
        map
    }

    fn create_test_order(
        symbol: &str,
        side: crate::model::OrderSide,
        order_type: crate::model::OrderType,
        quantity: Decimal,
        price: Option<Decimal>,
    ) -> Order {
        use uuid::Uuid;
        Order {
            id: Uuid::new_v4().to_string(),
            symbol: symbol.to_string(),
            side,
            order_type,
            quantity,
            price,
            time_in_force: TimeInForce::Day,
            trigger_price: None,
            status: OrderStatus::New,
            filled_quantity: Decimal::ZERO,
            average_filled_price: None,
            created_at: 0,
            updated_at: 0,
            commission: Decimal::ZERO,
            tag: String::new(),
            reject_reason: String::new(),
        }
    }

    fn create_test_bar(
        symbol: &str,
        open: Decimal,
        high: Decimal,
        low: Decimal,
        close: Decimal,
    ) -> Bar {
        Bar {
            symbol: symbol.to_string(),
            timestamp: 1000,
            open,
            high,
            low,
            close,
            volume: Decimal::from(1000),
            extra: HashMap::new(),
        }
    }

    #[test]
    fn test_execution_market_order() {
        let mut sim = SimulatedExecutionClient::new();
        let instruments = create_test_instruments();
        let order = create_test_order(
            "AAPL",
            crate::model::OrderSide::Buy,
            crate::model::OrderType::Market,
            Decimal::from(100),
            None,
        );
        sim.on_order(order);

        let bar = create_test_bar(
            "AAPL",
            Decimal::from(100),
            Decimal::from(105),
            Decimal::from(95),
            Decimal::from(102),
        );
        let event = Event::Bar(bar);

        let portfolio = crate::portfolio::Portfolio {
             cash: Decimal::from(100000),
             positions: HashMap::new().into(),
             available_positions: HashMap::new().into(),
        };
        let last_prices = HashMap::new();
        let _risk_manager = crate::risk::RiskManager::new();
        let market_config = crate::market::MarketConfig::default();
        let market_model = market_config.create_model();

        let events = sim.on_event(
            &event,
            &instruments,
            &portfolio,
            &last_prices,
            market_model.as_ref(),
            ExecutionMode::NextOpen,
            0,
            TradingSession::Continuous
        );

        let trades: Vec<crate::model::Trade> = events
            .iter()
            .filter_map(|e| {
                if let Event::ExecutionReport(_, Some(trade)) = e {
                    Some(trade.clone())
                } else {
                    None
                }
            })
            .collect();

        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].price, Decimal::from(100)); // Open price

        let _last_order = events
            .iter()
            .filter_map(|e| {
                if let Event::ExecutionReport(o, _) = e {
                    Some(o)
                } else {
                    None
                }
            })
            .last()
            .unwrap();
    }

    #[test]
    fn test_execution_limit_buy() {
        let mut sim = SimulatedExecutionClient::new();
        let instruments = create_test_instruments();
        // Buy Limit @ 99.0
        let order = create_test_order(
            "AAPL",
            crate::model::OrderSide::Buy,
            crate::model::OrderType::Limit,
            Decimal::from(100),
            Some(Decimal::from(99)),
        );
        sim.on_order(order);

        let bar = create_test_bar(
            "AAPL",
            Decimal::from(100),
            Decimal::from(105),
            Decimal::from(95),
            Decimal::from(102),
        );
        let event = Event::Bar(bar);

        let portfolio = crate::portfolio::Portfolio {
             cash: Decimal::from(100000),
             positions: HashMap::new().into(),
             available_positions: HashMap::new().into(),
        };
        let last_prices = HashMap::new();
        let _risk_manager = crate::risk::RiskManager::new();
        let market_config = crate::market::MarketConfig::default();
        let market_model = market_config.create_model();

        let events = sim.on_event(
            &event,
            &instruments,
            &portfolio,
            &last_prices,
            market_model.as_ref(),
            ExecutionMode::NextOpen,
            0,
            TradingSession::Continuous
        );

        // Should be filled at Limit Price (99)
        let trades: Vec<_> = events.iter().filter_map(|e| if let Event::ExecutionReport(_, Some(t)) = e { Some(t) } else { None }).collect();
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].price, Decimal::from(99));
    }

    #[test]
    fn test_execution_limit_buy_no_fill() {
        let mut sim = SimulatedExecutionClient::new();
        let instruments = create_test_instruments();
        // Buy Limit @ 90.0 (Below Low 95)
        let order = create_test_order(
            "AAPL",
            crate::model::OrderSide::Buy,
            crate::model::OrderType::Limit,
            Decimal::from(100),
            Some(Decimal::from(90)),
        );
        sim.on_order(order);

        let bar = create_test_bar(
            "AAPL",
            Decimal::from(100),
            Decimal::from(105),
            Decimal::from(95),
            Decimal::from(102),
        );
        let event = Event::Bar(bar);

        let portfolio = crate::portfolio::Portfolio {
             cash: Decimal::from(100000),
             positions: HashMap::new().into(),
             available_positions: HashMap::new().into(),
        };
        let last_prices = HashMap::new();
        let _risk_manager = crate::risk::RiskManager::new();
        let market_config = crate::market::MarketConfig::default();
        let market_model = market_config.create_model();

        let events = sim.on_event(
            &event,
            &instruments,
            &portfolio,
            &last_prices,
            market_model.as_ref(),
            ExecutionMode::NextOpen,
            0,
            TradingSession::Continuous
        );

        let trades: Vec<_> = events.iter().filter_map(|e| if let Event::ExecutionReport(_, Some(t)) = e { Some(t) } else { None }).collect();
        assert_eq!(trades.len(), 0);
    }

    #[test]
    fn test_dynamic_position_sizing() {
        let mut sim = SimulatedExecutionClient::new();
        let instruments = create_test_instruments();

        // Buy 1000 shares @ 100 = 100,000 value.
        // But cash is only 50,000.
        let order = create_test_order(
            "AAPL",
            crate::model::OrderSide::Buy,
            crate::model::OrderType::Market,
            Decimal::from(1000),
            None,
        );
        sim.on_order(order);

        let bar = create_test_bar(
            "AAPL",
            Decimal::from(100),
            Decimal::from(105),
            Decimal::from(95),
            Decimal::from(102),
        );
        let event = Event::Bar(bar);

        let portfolio = crate::portfolio::Portfolio {
             cash: Decimal::from(50000), // Only 50k
             positions: HashMap::new().into(),
             available_positions: HashMap::new().into(),
        };
        let last_prices = HashMap::new();
        let _risk_manager = crate::risk::RiskManager::new();
        let market_config = crate::market::MarketConfig::default();
        let market_model = market_config.create_model();

        let events = sim.on_event(
            &event,
            &instruments,
            &portfolio,
            &last_prices,
            market_model.as_ref(),
            ExecutionMode::NextOpen,
            0,
            TradingSession::Continuous
        );

        let trades: Vec<_> = events.iter().filter_map(|e| if let Event::ExecutionReport(_, Some(t)) = e { Some(t) } else { None }).collect();
        assert_eq!(trades.len(), 1);

        // Should be reduced to approx 400 shares (due to lot size 100 and safety margin)
        // 50000 / 100 = 500. 500 * 0.9999 = 499.95 -> floor to 400 (lot size 100)
        assert_eq!(trades[0].quantity, Decimal::from(400));
    }
}
