use super::result::BacktestResult;
use super::types::{ClosedTrade, PerformanceMetrics, PositionSnapshot, TradePnL};
use crate::model::{Order, Trade};
use rust_decimal::prelude::*;
use rust_decimal::Decimal;
use std::collections::BTreeMap;

impl BacktestResult {
    pub fn calculate(
        equity_curve_decimal: Vec<(i64, Decimal)>,
        cash_curve_decimal: Vec<(i64, Decimal)>,
        snapshots: Vec<(i64, Vec<PositionSnapshot>)>,
        trade_pnl: TradePnL,
        trades: Vec<ClosedTrade>,
        initial_cash: Decimal,
        orders: Vec<Order>,
        executions: Vec<Trade>,
    ) -> Self {
        // Convert equity_curve to f64 for storage/python
        let equity_curve: Vec<(i64, f64)> = equity_curve_decimal
            .iter()
            .map(|(t, d)| (*t, d.to_f64().unwrap_or_default()))
            .collect();
        let cash_curve: Vec<(i64, f64)> = cash_curve_decimal
            .iter()
            .map(|(t, d)| (*t, d.to_f64().unwrap_or_default()))
            .collect();

        if equity_curve_decimal.is_empty() {
            return BacktestResult {
                equity_curve,
                cash_curve,
                metrics: PerformanceMetrics {
                    total_return: 0.0,
                    annualized_return: 0.0,
                    max_drawdown: 0.0,
                    max_drawdown_value: 0.0,
                    max_drawdown_pct: 0.0,
                    sharpe_ratio: 0.0,
                    sortino_ratio: 0.0,
                    calmar_ratio: 0.0,
                    volatility: 0.0,
                    ulcer_index: 0.0,
                    upi: 0.0,
                    equity_r2: 0.0,
                    std_error: 0.0,
                    win_rate: 0.0,
                    initial_market_value: initial_cash.to_f64().unwrap_or_default(),
                    end_market_value: initial_cash.to_f64().unwrap_or_default(),
                    total_return_pct: 0.0,
                    start_time: 0,
                    end_time: 0,
                    duration: 0,
                    total_bars: 0,
                    exposure_time_pct: 0.0,
                    var_95: 0.0,
                    var_99: 0.0,
                    cvar_95: 0.0,
                    cvar_99: 0.0,
                },
                trade_metrics: trade_pnl,
                trades,
                snapshots,
                orders,
                executions,
            };
        }

        let initial_equity = initial_cash;
        let final_equity = equity_curve_decimal.last().unwrap().1;

        // 1. Total Return (Decimal)
        let total_return_dec = if !initial_equity.is_zero() {
            (final_equity - initial_equity) / initial_equity
        } else {
            Decimal::ZERO
        };

        // 2. Max Drawdown & Ulcer Index
        let mut max_drawdown_dec = Decimal::ZERO;
        let mut max_drawdown_val = Decimal::ZERO;
        let mut peak = initial_equity;
        let mut sum_sq_drawdown = 0.0;

        for (_, equity) in &equity_curve_decimal {
            if *equity > peak {
                peak = *equity;
            }
            let drawdown_val = peak - *equity;
            if drawdown_val > max_drawdown_val {
                max_drawdown_val = drawdown_val;
            }

            let drawdown_dec = if !peak.is_zero() {
                drawdown_val / peak
            } else {
                Decimal::ZERO
            };
            if drawdown_dec > max_drawdown_dec {
                max_drawdown_dec = drawdown_dec;
            }

            let dd_f64 = drawdown_dec.to_f64().unwrap_or_default();
            sum_sq_drawdown += dd_f64 * dd_f64;
        }

        let ulcer_index = (sum_sq_drawdown / equity_curve.len() as f64).sqrt();

        // 3. Returns Series for Volatility & Sharpe (Resampled to Daily)
        let mut daily_equity_map: BTreeMap<i64, Decimal> = BTreeMap::new();

        // Use 24h buckets (86400 * 1e9 ns) to group by day.
        // Note: This assumes local time is consistent or UTC.
        // A simple approximation is integer division by 86400_000_000_000 (1 day in ns).
        // This works for UTC timestamps.

        for (ts, eq) in &equity_curve_decimal {
            let day_key = ts / 86_400_000_000_000;
            daily_equity_map.insert(day_key, *eq);
        }

        let daily_equities: Vec<Decimal> = daily_equity_map.values().cloned().collect();

        let mut returns = Vec::new();
        let mut downside_returns = Vec::new();

        if daily_equities.len() > 1 {
            for i in 1..daily_equities.len() {
                let prev = daily_equities[i - 1];
                let curr = daily_equities[i];
                if !prev.is_zero() {
                    let r_dec = (curr - prev) / prev;
                    let r = r_dec.to_f64().unwrap_or_default();
                    returns.push(r);
                    if r < 0.0 {
                        downside_returns.push(r);
                    } else {
                        downside_returns.push(0.0);
                    }
                }
            }
        }

        // 4. Annualized Return & Volatility
        let start_ts = equity_curve.first().unwrap().0;
        let end_ts = equity_curve.last().unwrap().0;
        // Use i128 to prevent overflow when calculating duration for very long backtests (e.g. > 292 years)
        let duration_ns = (end_ts as i128) - (start_ts as i128);
        let duration_seconds = duration_ns as f64 / 1_000_000_000.0;
        let years = duration_seconds / (365.0 * 24.0 * 3600.0);

        let total_return_f64 = total_return_dec.to_f64().unwrap_or_default();

        let annualized_return = if years > 0.0 {
            (1.0 + total_return_f64).powf(1.0 / years) - 1.0
        } else {
            0.0
        };

        let mean_return = if !returns.is_empty() {
            returns.iter().sum::<f64>() / returns.len() as f64
        } else {
            0.0
        };

        let variance = if returns.len() > 1 {
            returns
                .iter()
                .map(|r| (r - mean_return).powi(2))
                .sum::<f64>()
                / (returns.len() - 1) as f64
        } else {
            0.0
        };

        let std_dev = variance.sqrt();
        let annualized_volatility = std_dev * (252.0f64).sqrt();

        // 5. Sharpe Ratio
        let risk_free_rate = 0.0; // Assume 0 for simplicity or pass config
        let sharpe_ratio = if annualized_volatility != 0.0 {
            (annualized_return - risk_free_rate) / annualized_volatility
        } else {
            0.0
        };

        // 6. Sortino Ratio
        // Downside deviation
        let downside_variance = if !downside_returns.is_empty() {
            downside_returns.iter().map(|r| r.powi(2)).sum::<f64>() / downside_returns.len() as f64
        } else {
            0.0
        };
        let downside_std_dev = downside_variance.sqrt();
        let annualized_downside_volatility = downside_std_dev * (252.0f64).sqrt();

        let sortino_ratio = if annualized_downside_volatility != 0.0 {
            (annualized_return - risk_free_rate) / annualized_downside_volatility
        } else {
            0.0
        };

        // 7. UPI
        let upi = if ulcer_index != 0.0 {
            (annualized_return - risk_free_rate) / ulcer_index
        } else {
            0.0
        };

        // 8. Calmar Ratio
        let max_dd_f64 = max_drawdown_dec.to_f64().unwrap_or_default();
        let calmar_ratio = if max_dd_f64 != 0.0 {
            annualized_return / max_dd_f64
        } else {
            0.0
        };

        // 9. R2 and Std Error (Linear Regression of Equity)
        // X = index, Y = Equity
        let n = equity_curve.len() as f64;
        let sum_x = (0..equity_curve.len()).map(|i| i as f64).sum::<f64>();
        let sum_y = equity_curve.iter().map(|(_, y)| *y).sum::<f64>();
        let sum_xy = equity_curve
            .iter()
            .enumerate()
            .map(|(i, (_, y))| i as f64 * *y)
            .sum::<f64>();
        let sum_xx = (0..equity_curve.len()).map(|i| (i * i) as f64).sum::<f64>();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
        let intercept = (sum_y - slope * sum_x) / n;

        // R2
        let ss_tot = equity_curve
            .iter()
            .map(|(_, y)| (y - (sum_y / n)).powi(2))
            .sum::<f64>();
        let ss_res = equity_curve
            .iter()
            .enumerate()
            .map(|(i, (_, y))| (y - (slope * i as f64 + intercept)).powi(2))
            .sum::<f64>();

        let equity_r2 = if ss_tot != 0.0 {
            1.0 - (ss_res / ss_tot)
        } else {
            0.0
        };

        // Standard Error of Estimate
        let std_error = if n > 2.0 {
            (ss_res / (n - 2.0)).sqrt()
        } else {
            0.0
        };

        let total_bars = equity_curve.len();

        // 10. Exposure Time %
        let exposure_count = snapshots.iter().filter(|(_, positions)| {
             positions.iter().any(|p| p.quantity != 0.0)
        }).count();
        let exposure_time_pct = if !snapshots.is_empty() {
             (exposure_count as f64 / snapshots.len() as f64) * 100.0
        } else {
             0.0
        };

        // 11. VaR and CVaR
        let mut sorted_returns = returns.clone();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let calculate_risk_metrics = |alpha: f64, sorted_rets: &Vec<f64>| -> (f64, f64) {
             if sorted_rets.is_empty() {
                 return (0.0, 0.0);
             }
             let idx = ((sorted_rets.len() as f64) * alpha).floor() as usize;
             let idx = idx.min(sorted_rets.len() - 1);
             let var = sorted_rets[idx];

             let cvar_sum: f64 = sorted_rets[0..=idx].iter().sum();
             let cvar = cvar_sum / (idx + 1) as f64;
             (var, cvar)
        };

        let (var_95, cvar_95) = calculate_risk_metrics(0.05, &sorted_returns);
        let (var_99, cvar_99) = calculate_risk_metrics(0.01, &sorted_returns);

        BacktestResult {
            equity_curve,
            cash_curve,
            metrics: PerformanceMetrics {
                total_return: total_return_f64,
                annualized_return,
                max_drawdown: max_drawdown_dec.to_f64().unwrap_or_default(),
                max_drawdown_value: max_drawdown_val.to_f64().unwrap_or_default(),
                max_drawdown_pct: max_drawdown_dec.to_f64().unwrap_or_default() * 100.0,
                sharpe_ratio,
                sortino_ratio,
                calmar_ratio,
                volatility: annualized_volatility,
                ulcer_index,
                upi,
                equity_r2,
                std_error,
                win_rate: trade_pnl.win_rate,
                initial_market_value: initial_equity.to_f64().unwrap_or_default(),
                end_market_value: final_equity.to_f64().unwrap_or_default(),
                total_return_pct: total_return_f64 * 100.0,
                start_time: start_ts,
                end_time: end_ts,
                duration: (end_ts as i128 - start_ts as i128) as u64,
                total_bars,
                exposure_time_pct,
                var_95,
                var_99,
                cvar_95,
                cvar_99,
            },
            trade_metrics: trade_pnl,
            trades,
            snapshots,
            orders,
            executions,
        }
    }
}
