use crate::model::Bar;
use rust_decimal::prelude::ToPrimitive;
use std::collections::{HashMap, VecDeque};

#[derive(Debug, Clone)]
pub struct SymbolHistory {
    pub timestamps: VecDeque<i64>,
    pub opens: VecDeque<f64>,
    pub highs: VecDeque<f64>,
    pub lows: VecDeque<f64>,
    pub closes: VecDeque<f64>,
    pub volumes: VecDeque<f64>,
    pub extras: HashMap<String, VecDeque<f64>>,
    pub capacity: usize,
}

impl SymbolHistory {
    pub fn new(capacity: usize) -> Self {
        SymbolHistory {
            timestamps: VecDeque::with_capacity(capacity),
            opens: VecDeque::with_capacity(capacity),
            highs: VecDeque::with_capacity(capacity),
            lows: VecDeque::with_capacity(capacity),
            closes: VecDeque::with_capacity(capacity),
            volumes: VecDeque::with_capacity(capacity),
            extras: HashMap::new(),
            capacity,
        }
    }

    pub fn push(&mut self, bar: &Bar) {
        if self.capacity == 0 {
            return;
        }

        if self.timestamps.len() >= self.capacity {
            self.timestamps.pop_front();
            self.opens.pop_front();
            self.highs.pop_front();
            self.lows.pop_front();
            self.closes.pop_front();
            self.volumes.pop_front();
            for v in self.extras.values_mut() {
                v.pop_front();
            }
        }

        let cur_len = self.timestamps.len();
        for (k, _) in bar.extra.iter() {
            if !self.extras.contains_key(k) {
                let mut dq = VecDeque::with_capacity(self.capacity);
                for _ in 0..cur_len {
                    dq.push_back(f64::NAN);
                }
                self.extras.insert(k.clone(), dq);
            }
        }
        for (k, dq) in self.extras.iter_mut() {
            let val = bar.extra.get(k).cloned().unwrap_or(f64::NAN);
            dq.push_back(val);
        }
        self.timestamps.push_back(bar.timestamp);
        self.opens.push_back(bar.open.to_f64().unwrap_or(0.0));
        self.highs.push_back(bar.high.to_f64().unwrap_or(0.0));
        self.lows.push_back(bar.low.to_f64().unwrap_or(0.0));
        self.closes.push_back(bar.close.to_f64().unwrap_or(0.0));
        self.volumes.push_back(bar.volume.to_f64().unwrap_or(0.0));
    }
}

#[derive(Debug)]
pub struct HistoryBuffer {
    pub data: HashMap<String, SymbolHistory>,
    pub default_capacity: usize,
}

impl HistoryBuffer {
    pub fn new(default_capacity: usize) -> Self {
        HistoryBuffer {
            data: HashMap::new(),
            default_capacity,
        }
    }

    pub fn set_capacity(&mut self, capacity: usize) {
        self.default_capacity = capacity;
        self.data.clear();
    }

    pub fn update(&mut self, bar: &Bar) {
        if self.default_capacity == 0 {
            return;
        }

        let history = self
            .data
            .entry(bar.symbol.clone())
            .or_insert_with(|| SymbolHistory::new(self.default_capacity));

        history.push(bar);
    }

    pub fn get_history(&self, symbol: &str) -> Option<&SymbolHistory> {
        self.data.get(symbol)
    }
}
