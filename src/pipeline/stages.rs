use crate::data::FeedAction;
use crate::engine::Engine;
use crate::event::Event;
use crate::model::{OrderStatus, TradingSession};
use crate::pipeline::processor::{Processor, ProcessorResult};
use pyo3::prelude::*;

pub struct ChannelProcessor;

impl Processor for ChannelProcessor {
    fn process(&mut self, engine: &mut Engine, _py: Python<'_>, _strategy: &Bound<'_, PyAny>) -> PyResult<ProcessorResult> {
        let mut trades_to_process = Vec::new();
        while let Some(event) = engine.event_manager.try_recv() {
            match event {
                Event::OrderRequest(mut order) => {
                    // 1. Risk Check & Adjustment
                    if let Err(err) = engine.risk_manager.check_and_adjust(
                        &mut order,
                        &engine.portfolio,
                        &engine.instruments,
                        &engine.order_manager.active_orders,
                        &engine.last_prices,
                    ) {
                        // Rejected
                        order.status = OrderStatus::Rejected;
                        order.reject_reason = err.to_string();
                        order.updated_at = engine.clock.timestamp().unwrap_or(0);

                        // Send ExecutionReport (Rejected)
                        let _ = engine.event_manager.send(Event::ExecutionReport(order, None));
                    } else {
                        // Validated -> Send OrderValidated
                        let _ = engine.event_manager.send(Event::OrderValidated(order));
                    }
                }
                Event::OrderValidated(order) => {
                    // 2. Send to Execution Client
                    engine.execution_model.on_order(order.clone());
                    // Add to local active (Strategy View)
                    engine.order_manager.add_active_order(order);
                }
                Event::ExecutionReport(order, trade) => {
                    // 3. Update Order State
                    engine.order_manager.on_execution_report(order);

                    // 4. Process Trade (if any)
                    if let Some(t) = trade {
                        trades_to_process.push(t);
                    }
                }
                _ => {}
            }
        }

        if !trades_to_process.is_empty() {
            engine.order_manager.process_trades(
                trades_to_process,
                &mut engine.portfolio,
                &engine.instruments,
                engine.market_manager.model.as_ref(),
                &engine.history_buffer,
                &engine.last_prices,
            );
        }

        Ok(ProcessorResult::Next)
    }
}

pub struct DataProcessor {
    last_timestamp: i64,
}

impl DataProcessor {
    pub fn new() -> Self {
        Self { last_timestamp: 0 }
    }
}

impl Processor for DataProcessor {
    fn process(&mut self, engine: &mut Engine, py: Python<'_>, _strategy: &Bound<'_, PyAny>) -> PyResult<ProcessorResult> {
        let next_timer_time = engine.timers.peek().map(|t| t.timestamp);
        let action = engine.feed.next_action(next_timer_time, py);

        match action {
            FeedAction::Wait => Ok(ProcessorResult::Loop),
            FeedAction::End => Ok(ProcessorResult::Break),
            FeedAction::Timer(_timestamp) => {
                if let Some(timer) = engine.timers.pop() {
                    let local_dt = Engine::local_datetime_from_ns(timer.timestamp, engine.timezone_offset);
                    let session = engine.market_manager.get_session_status(local_dt.time());
                    engine.clock.update(timer.timestamp, session);
                    if engine.force_session_continuous {
                        engine.clock.session = TradingSession::Continuous;
                    }
                    engine.current_event = Some(Event::Timer(timer));
                }
                Ok(ProcessorResult::Next)
            }
            FeedAction::Event(event) => {
                let timestamp = match &event {
                    Event::Bar(b) => b.timestamp,
                    Event::Tick(t) => t.timestamp,
                    _ => 0,
                };

                if self.last_timestamp != 0 && timestamp > self.last_timestamp {
                    engine.bar_count += 1;
                    if let Some(pb) = &engine.progress_bar {
                        pb.inc(1);
                    }
                }
                self.last_timestamp = timestamp;

                let local_dt = Engine::local_datetime_from_ns(timestamp, engine.timezone_offset);

                // Update Market Manager (Session)
                let session = engine.market_manager.get_session_status(local_dt.time());

                engine.clock.update(timestamp, session);
                if engine.force_session_continuous {
                    engine.clock.session = TradingSession::Continuous;
                }

                // Daily Snapshot & Settlement
                let local_date = local_dt.date_naive();
                if engine.current_date != Some(local_date) {
                    if engine.current_date.is_some() {
                        engine.statistics_manager.record_snapshot(
                            timestamp,
                            &engine.portfolio,
                            &engine.instruments,
                            &engine.last_prices,
                            &engine.order_manager.trade_tracker,
                        );
                    }
                    engine.current_date = Some(local_date);

                    // Settlement Manager (T+1, Option Expiry, Day Order Expiry)
                    let mut expired_orders = Vec::new();
                    engine.settlement_manager.process_daily_settlement(
                        local_date,
                        &mut engine.portfolio,
                        &engine.instruments,
                        &engine.last_prices,
                        &engine.market_manager,
                        &mut engine.order_manager.active_orders,
                        &mut expired_orders,
                    );

                    for o in expired_orders {
                        engine.order_manager.orders.push(o);
                    }
                }

                engine.current_event = Some(event);
                Ok(ProcessorResult::Next)
            }
        }
    }
}

pub struct StrategyProcessor;

impl Processor for StrategyProcessor {
    fn process(&mut self, engine: &mut Engine, _py: Python<'_>, strategy: &Bound<'_, PyAny>) -> PyResult<ProcessorResult> {
        if let Some(event) = engine.current_event.clone() {
            let (new_orders, new_timers, canceled_ids) = engine.call_strategy(strategy, &event)?;

            for id in canceled_ids {
                engine.execution_model.on_cancel(&id);
            }
            for order in new_orders {
                let _ = engine.event_manager.send(Event::OrderRequest(order));
            }
            for t in new_timers {
                engine.timers.push(t);
            }
        }
        Ok(ProcessorResult::Next)
    }
}

pub struct ExecutionProcessor;

impl Processor for ExecutionProcessor {
    fn process(&mut self, engine: &mut Engine, _py: Python<'_>, _strategy: &Bound<'_, PyAny>) -> PyResult<ProcessorResult> {
        if let Some(event) = engine.current_event.clone() {
            match event {
                Event::Bar(_) | Event::Tick(_) => {
                    let reports = engine.execution_model.on_event(
                        &event,
                        &engine.instruments,
                        &engine.portfolio,
                        &engine.last_prices,
                        engine.market_manager.model.as_ref(),
                        engine.execution_mode,
                        engine.bar_count,
                        engine.clock.session,
                    );
                    for report in reports {
                        let _ = engine.event_manager.send(report);
                    }
                }
                _ => {}
            }
        }
        Ok(ProcessorResult::Next)
    }
}

pub struct CleanupProcessor;

impl Processor for CleanupProcessor {
    fn process(&mut self, engine: &mut Engine, _py: Python<'_>, _strategy: &Bound<'_, PyAny>) -> PyResult<ProcessorResult> {
        engine.order_manager.cleanup_finished_orders();
        Ok(ProcessorResult::Next)
    }
}

pub struct StatisticsProcessor;

impl Processor for StatisticsProcessor {
    fn process(&mut self, engine: &mut Engine, _py: Python<'_>, _strategy: &Bound<'_, PyAny>) -> PyResult<ProcessorResult> {
        if let Some(event) = &engine.current_event {
            match event {
                 Event::Bar(_) | Event::Tick(_) => {
                    if let Some(timestamp) = engine.clock.timestamp() {
                         let equity = engine.portfolio.calculate_equity(&engine.last_prices, &engine.instruments);
                         engine.statistics_manager.update(timestamp, equity, engine.portfolio.cash);
                    }
                 }
                 _ => {}
            }
        }
        Ok(ProcessorResult::Next)
    }
}
