use std::collections::HashMap;

use crate::error::AkQuantError;
use crate::model::market_data::extract_decimal;
use crate::model::{AssetType, Instrument, Order};
use crate::portfolio::Portfolio;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use rust_decimal::Decimal;
use rust_decimal::prelude::*;

pub mod common;
pub mod futures;
pub mod option;
pub mod rule;
pub mod stock;

use common::*;
use futures::*;
use option::*;
use rule::RiskRule;
use stock::*;

#[gen_stub_pyclass]
#[pyclass]
#[derive(Debug, Clone, Default)]
/// 风控配置.
pub struct RiskConfig {
    pub max_order_size: Option<Decimal>,
    pub max_order_value: Option<Decimal>,
    pub max_position_size: Option<Decimal>,
    #[pyo3(get, set)]
    pub restricted_list: Vec<String>,
    #[pyo3(get, set)]
    pub active: bool,
    #[pyo3(get, set)]
    pub check_cash: bool,
    #[pyo3(get, set)]
    pub safety_margin: f64,
}

#[pymethods]
impl RiskConfig {
    #[new]
    pub fn new() -> Self {
        Self {
            max_order_size: None,
            max_order_value: None,
            max_position_size: None,
            restricted_list: Vec::new(),
            active: true,
            check_cash: true,
            safety_margin: 0.0001,
        }
    }

    #[getter]
    pub fn get_max_order_size(&self) -> Option<f64> {
        self.max_order_size.map(|d| d.to_f64().unwrap_or_default())
    }

    #[setter]
    pub fn set_max_order_size(&mut self, value: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        self.max_order_size = match value {
            Some(v) => Some(extract_decimal(v)?),
            None => None,
        };
        Ok(())
    }

    #[getter]
    pub fn get_max_order_value(&self) -> Option<f64> {
        self.max_order_value.map(|d| d.to_f64().unwrap_or_default())
    }

    #[setter]
    pub fn set_max_order_value(&mut self, value: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        self.max_order_value = match value {
            Some(v) => Some(extract_decimal(v)?),
            None => None,
        };
        Ok(())
    }

    #[getter]
    pub fn get_max_position_size(&self) -> Option<f64> {
        self.max_position_size
            .map(|d| d.to_f64().unwrap_or_default())
    }

    #[setter]
    pub fn set_max_position_size(&mut self, value: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        self.max_position_size = match value {
            Some(v) => Some(extract_decimal(v)?),
            None => None,
        };
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::instrument::{InstrumentEnum, StockInstrument, FuturesInstrument};
    use crate::model::{AssetType, Instrument, Order, OrderSide, OrderStatus, OrderType, TimeInForce};
    use crate::portfolio::Portfolio;
    use std::collections::HashMap;

    fn create_test_order(symbol: &str, quantity: Decimal, price: Option<Decimal>) -> Order {
        Order {
            id: "test_order".to_string(),
            symbol: symbol.to_string(),
            side: OrderSide::Buy,
            order_type: OrderType::Limit,
            quantity,
            price,
            status: OrderStatus::New,
            time_in_force: TimeInForce::Day,
            trigger_price: None,
            filled_quantity: Decimal::ZERO,
            average_filled_price: None,
            created_at: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
            updated_at: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
            commission: Decimal::ZERO,
            tag: "".to_string(),
            reject_reason: "".to_string(),
        }
    }

    fn create_test_instrument(symbol: &str, asset_type: AssetType) -> Instrument {
        let inner = match asset_type {
            AssetType::Stock => InstrumentEnum::Stock(StockInstrument {
                symbol: symbol.to_string(),
                lot_size: Decimal::from(100),
                tick_size: Decimal::from_str("0.01").unwrap(),
            }),
            AssetType::Futures => InstrumentEnum::Futures(FuturesInstrument {
                symbol: symbol.to_string(),
                multiplier: Decimal::from(10),
                margin_ratio: Decimal::from_str("0.1").unwrap(),
                tick_size: Decimal::from_str("0.2").unwrap(),
                expiry_date: None,
                settlement_type: None,
            }),
            _ => panic!("Unsupported asset type for test"),
        };
        Instrument {
            asset_type,
            inner,
        }
    }

    #[test]
    fn test_risk_manager_basic() {
        let mut manager = RiskManager::default();
        manager.config.max_order_size = Some(Decimal::from(100));
        manager.config.restricted_list = vec!["BAD_STOCK".to_string()];

        let portfolio = Portfolio {
            cash: Decimal::from(100000),
            positions: HashMap::new().into(),
            available_positions: HashMap::new().into(),
        };
        let mut instruments = HashMap::new();
        instruments.insert("AAPL".to_string(), create_test_instrument("AAPL", AssetType::Stock));
        instruments.insert("BAD_STOCK".to_string(), create_test_instrument("BAD_STOCK", AssetType::Stock));

        // Test valid order
        let order = create_test_order("AAPL", Decimal::from(50), Some(Decimal::from(150)));
        let result = manager.check(&order, &portfolio, instruments.clone(), vec![], None);
        assert!(result.is_none());

        // Test max order size
        let order_large = create_test_order("AAPL", Decimal::from(150), Some(Decimal::from(150)));
        let result = manager.check(&order_large, &portfolio, instruments.clone(), vec![], None);
        assert!(result.is_some());
        assert!(result.unwrap().contains("exceeds limit"));

        // Test restricted list
        let order_restricted = create_test_order("BAD_STOCK", Decimal::from(50), Some(Decimal::from(10)));
        let result = manager.check(&order_restricted, &portfolio, instruments.clone(), vec![], None);
        assert!(result.is_some());
        assert!(result.unwrap().contains("restricted"));
    }
}

#[gen_stub_pyclass]
#[pyclass]
#[derive(Debug, Clone)]
pub struct RiskManager {
    #[pyo3(get, set)]
    pub config: RiskConfig,

    // Internal fields, not exposed to Python directly (unless we add getters)
    // No #[pyo3(skip)] needed as fields are private by default in #[pyclass]
    common_rules: Vec<Box<dyn RiskRule>>,
    asset_rules: HashMap<AssetType, Vec<Box<dyn RiskRule>>>,
}

impl Default for RiskManager {
    fn default() -> Self {
        let mut manager = Self {
            config: RiskConfig::new(),
            common_rules: Vec::new(),
            asset_rules: HashMap::new(),
        };
        manager.init_rules();
        manager
    }
}

#[pymethods]
impl RiskManager {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn check(
        &self,
        order: &Order,
        portfolio: &Portfolio,
        instruments: HashMap<String, Instrument>,
        active_orders: Vec<Order>,
        current_prices: Option<HashMap<String, f64>>,
    ) -> Option<String> {
        let prices_dec: HashMap<String, Decimal> = if let Some(cp) = current_prices {
            cp.into_iter()
                .map(|(k, v)| (k, Decimal::from_f64(v).unwrap_or(Decimal::ZERO)))
                .collect()
            } else {
            HashMap::new()
        };

        // Create a dummy market model for context
        use crate::market::{SimpleMarket, SimpleMarketConfig};
        let market_model = SimpleMarket::from_config(SimpleMarketConfig::default());

        let ctx = crate::context::EngineContext {
            instruments: &instruments,
            portfolio,
            last_prices: &prices_dec,
            market_model: &market_model,
            execution_mode: crate::model::ExecutionMode::NextOpen,
            bar_index: 0,
            session: crate::model::TradingSession::Continuous,
            active_orders: &active_orders,
        };

        match self.check_internal(
            order,
            &ctx,
        ) {
            Ok(_) => None,
            Err(e) => Some(e.to_string()),
        }
    }
}

impl RiskManager {
    fn init_rules(&mut self) {
        // Common rules
        self.common_rules.push(Box::new(RestrictedListRule));
        self.common_rules.push(Box::new(MaxOrderSizeRule));
        self.common_rules.push(Box::new(MaxOrderValueRule));
        self.common_rules.push(Box::new(MaxPositionSizeRule));
        self.common_rules.push(Box::new(CashMarginRule));

        // Stock rules
        self.asset_rules.entry(AssetType::Stock).or_default().push(Box::new(StockAvailablePositionRule));
        self.asset_rules.entry(AssetType::Fund).or_default().push(Box::new(StockAvailablePositionRule));

        // Futures rules
        self.asset_rules.entry(AssetType::Futures).or_default().push(Box::new(FuturesMarginRule));

        // Option rules
        self.asset_rules.entry(AssetType::Option).or_default().push(Box::new(OptionGreekRiskRule));
    }

    pub fn check_and_adjust(
        &self,
        order: &mut Order,
        ctx: &crate::context::EngineContext,
    ) -> Result<(), AkQuantError> {
        // 1. Initial Check
        if let Err(err) = self.check_internal(order, ctx) {
            let err_msg = err.to_string();
            // Check for insufficient cash/margin to attempt auto-reduction
            // This logic was moved from OrderManager
            if (err_msg.contains("Insufficient cash") || err_msg.contains("Insufficient margin"))
                && order.side == crate::model::OrderSide::Buy
            {
                if let Some(instr) = ctx.instruments.get(&order.symbol) {
                    // Get price (Limit or Last)
                    let price = if let Some(p) = order.price {
                        p
                    } else {
                        *ctx.last_prices
                            .get(&order.symbol)
                            .unwrap_or(&Decimal::ZERO)
                    };

                    if price > Decimal::ZERO {
                        let multiplier = instr.multiplier();
                        let margin_ratio = instr.margin_ratio();

                        // Cost per unit = Price * Multiplier * MarginRatio
                        // For Stock, MarginRatio is usually 1.0 (or 100% cash)
                        let cost_per_unit = price * multiplier * margin_ratio;

                        if cost_per_unit > Decimal::ZERO {
                            // Calculate max quantity based on available cash/margin
                            // Note: Portfolio::cash is used here. Ideally should use Free Margin for futures.
                            // But for simple "Insufficient cash" check, let's use cash.
                            // If we want to support margin trading correctly here, we should check what check_internal failed on.

                            // For now, let's use a simplified calculation similar to old OrderManager
                            let max_qty_raw = ctx.portfolio.cash / cost_per_unit;

                            // Buffer for commission (e.g. 1% buffer -> 0.9999 safety factor from config)
                            let safety_margin = self.config.safety_margin;
                            let safety_factor = Decimal::from_f64(1.0 - safety_margin)
                                .unwrap_or(Decimal::from_f64(0.9999).unwrap());

                            let max_qty_raw = max_qty_raw * safety_factor;

                            let lot_size = instr.lot_size();
                            let mut new_qty = max_qty_raw.floor();
                            if lot_size > Decimal::ZERO {
                                new_qty = new_qty - (new_qty % lot_size);
                            }

                            if new_qty > Decimal::ZERO && new_qty < order.quantity {
                                order.quantity = new_qty;
                                // Re-check with new quantity
                                return self.check_internal(order, ctx);
                            }
                        }
                    }
                }
            }
            return Err(err);
        }
        Ok(())
    }

    pub fn check_internal(
        &self,
        order: &Order,
        ctx: &crate::context::EngineContext,
    ) -> Result<(), AkQuantError> {
        if !self.config.active {
            return Ok(());
        }

        let instrument = ctx.instruments.get(&order.symbol).ok_or_else(|| {
            AkQuantError::OrderError(format!("Instrument not found for {}", order.symbol))
        })?;

        // Check common rules
        for rule in &self.common_rules {
            rule.check(
                order,
                ctx.portfolio,
                instrument,
                ctx.instruments,
                ctx.active_orders,
                ctx.last_prices,
                &self.config,
            )?;
        }

        // Check asset-specific rules
        if let Some(rules) = self.asset_rules.get(&instrument.asset_type) {
            for rule in rules {
                rule.check(
                    order,
                    ctx.portfolio,
                    instrument,
                    ctx.instruments,
                    ctx.active_orders,
                    ctx.last_prices,
                    &self.config,
                )?;
            }
        }

        Ok(())
    }
}
