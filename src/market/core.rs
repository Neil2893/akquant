use crate::model::{Instrument, OrderSide, TradingSession};
use chrono::NaiveTime;
use rust_decimal::Decimal;
use std::collections::HashMap;

use super::china::{ChinaMarket, ChinaMarketConfig};
use super::simple::{SimpleMarket, SimpleMarketConfig};

#[derive(Clone, Debug)]
pub enum MarketConfig {
    China(ChinaMarketConfig),
    Simple(SimpleMarketConfig),
}

impl Default for MarketConfig {
    fn default() -> Self {
        MarketConfig::China(ChinaMarketConfig::default())
    }
}

impl MarketConfig {
    pub fn create_model(&self) -> Box<dyn MarketModel> {
        match self {
            MarketConfig::China(c) => Box::new(ChinaMarket::from_config(c.clone())),
            MarketConfig::Simple(c) => Box::new(SimpleMarket::from_config(c.clone())),
        }
    }
}

pub trait MarketModel: Send + Sync {
    fn calculate_commission(
        &self,
        instrument: &Instrument,
        side: OrderSide,
        price: Decimal,
        quantity: Decimal,
    ) -> Decimal;

    fn update_available_position(
        &self,
        available_positions: &mut HashMap<String, Decimal>,
        instrument: &Instrument,
        quantity: Decimal,
        side: OrderSide,
    );

    fn on_day_close(
        &self,
        positions: &HashMap<String, Decimal>,
        available_positions: &mut HashMap<String, Decimal>,
        instruments: &HashMap<String, Instrument>,
    );

    fn get_session_status(&self, time: NaiveTime) -> TradingSession;
}
