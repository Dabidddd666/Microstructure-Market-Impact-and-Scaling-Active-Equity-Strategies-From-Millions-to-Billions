"""
Neutralizing Module

This module contains functions for portfolio construction, execution modeling,
and performance evaluation for quantitative trading strategies.
"""

import os
import glob
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from utils import (
    read_wide, clean_cols, align_panels, winsorize_z, load_factors_from_directory,
    compute_advisory_volume, compute_ic_daily, get_rebalance_dates, 
    compute_portfolio_return, compute_max_drawdown, compute_turnover,
    update_positions_after_returns, apply_no_trade_band, create_output_directory,
    save_parquet_safe, print_performance_summary
)

# Default configuration parameters
DEFAULT_CONFIG = {
    # # Paths
    # 'FACTOR_DIR': "/Users/luchenshi/Desktop/Data for Alphathon/factors_investable",
    # 'RET_PARQUET': "/Users/luchenshi/Desktop/Data for Alphathon/neutralize_out/neutralized_returns.parquet",
    # 'OPEN_PARQUET': "/Users/luchenshi/Desktop/Data for Alphathon/Open.parquet",
    # 'CLOSE_PARQUET': "/Users/luchenshi/Desktop/Data for Alphathon/Close.parquet",
    # 'VOLUME_PARQUET': "/Users/luchenshi/Desktop/Data for Alphathon/Volume.parquet",
    #
    # Portfolio parameters
    'REBAL_FREQ': "W-FRI",
    'COST_BPS': 20,
    'WINSOR_Q': (0.01, 0.99),
    'LONG_WEIGHT': 0.5,
    'SHORT_WEIGHT': -0.5,
    'MIN_STOCKS': 200,
    'TOPN_BY_ADV': 800,
    'PRICE_MIN': 2.0,
    'PRICE_MAX': 2000.0,
    'NO_TRADE_BAND': 0.0025,
    'MAX_DAILY_TURNOVER': 0.10,
    
    # Factor aggregation
    'IC_WIN': 252,
    'MIN_IC_DAYS': 60,
    
    # Capacity-aware parameters
    'AUM': 20_000_000_000.0,
    'COMMISSION_BPS': 2.0,
    'SPREAD_BPS': 1.0,
    'GROSS_TARGET': 0.70,
    'NAME_WEIGHT_CAP': 0.0020,
    'GAMMA_SLOW': 0.30,
    'EXEC_HORIZON_DAYS': 7,
    'PRATE_MAX': 0.10,
    'IMPACT_K_SQRT_BPS': 30.0,
    
    # Robustness
    'MIN_CS_N': 150,
    'MIN_FACTORS_PER_DAY': 3,
    'RANK_FALLBACK': True,
}


def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names."""
    return clean_cols(df)


def compute_ADV60(close: pd.DataFrame, vol: pd.DataFrame) -> pd.DataFrame:
    """Compute 60-day average daily dollar volume."""
    return compute_advisory_volume(close, vol)


def rolling_icir_weights(factors: Dict[str, pd.DataFrame],
                         ret_cur: pd.DataFrame,
                         ic_win: int = None, min_days: int = None) -> Dict[str, pd.Series]:
    """Compute rolling ICIR weights per factor."""
    ic_win = ic_win or DEFAULT_CONFIG['IC_WIN']
    min_days = min_days or DEFAULT_CONFIG['MIN_IC_DAYS']
    
    weights = {}
    for name, df in factors.items():
        ic_series = compute_ic_daily(df.shift(1), ret_cur)
        from utils import compute_rolling_icir
        weights[name] = compute_rolling_icir(ic_series, ic_win, min_days)
    return weights


def rank_fallback_row(df_row: pd.DataFrame) -> pd.Series:
    """Rank fallback for composite factor construction."""
    ranks = []
    for c in df_row.columns:
        s = df_row[c]
        if s.notna().sum() >= 10:
            ranks.append(s.rank(pct=True) - 0.5)
    if not ranks:
        return pd.Series(index=df_row.index, dtype=float)
    R = pd.concat(ranks, axis=1)
    return R.mean(axis=1)


def composite_weighted(factors: Dict[str, pd.DataFrame],
                       weights: Dict[str, pd.Series]) -> pd.DataFrame:
    """Create composite factor from weighted individual factors."""
    fac_list = align_panels(list(factors.values()))
    idx, cols = fac_list[0].index, fac_list[0].columns
    names = list(factors.keys())
    Zs = [df.apply(winsorize_z, axis=1) for df in fac_list]
    Zpanel = {n: Z for n, Z in zip(names, Zs)}
    comp = pd.DataFrame(np.nan, index=idx, columns=cols)
    
    for d in idx:
        usable = []
        for n in names:
            if Zpanel[n].loc[d].notna().sum() >= DEFAULT_CONFIG['MIN_CS_N']:
                usable.append(n)
        if len(usable) < DEFAULT_CONFIG['MIN_FACTORS_PER_DAY']:
            continue
        num = pd.Series(0.0, index=cols)
        denom: float = 0.0
        for n in usable:
            wd = weights[n].reindex([d]).iloc[0] if d in weights[n].index else np.nan
            if pd.isna(wd) or not np.isfinite(wd) or wd == 0: 
                continue
            wd = float(wd)
            zrow = Zpanel[n].loc[d]
            num = num.add(wd * zrow, fill_value=0.0)
            denom += abs(wd)
        if denom > 0.0:
            comp.loc[d] = num / denom
        else:
            row_df = pd.concat([Zpanel[n].loc[d] for n in usable], axis=1)
            comp.loc[d] = rank_fallback_row(row_df)
    return comp


def rebalance_dates(all_dates: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """Get rebalancing dates based on frequency."""
    return get_rebalance_dates(all_dates, freq)


def dollar_neutral(score_row: pd.Series,
                   long_w: float = None, short_w: float = None) -> pd.Series:
    """Convert scores to dollar-neutral weights."""
    long_w = long_w or DEFAULT_CONFIG['LONG_WEIGHT']
    short_w = short_w or DEFAULT_CONFIG['SHORT_WEIGHT']
    
    s = score_row.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty: 
        return pd.Series(dtype=float)
    pos, neg = s[s > 0], s[s < 0]
    w = pd.Series(0.0, index=s.index)
    if not pos.empty: 
        w.loc[pos.index] = pos / pos.sum() * float(long_w)
    if not neg.empty: 
        w.loc[neg.index] = neg / neg.sum() * float(short_w)
    return w


def position_after_return(prev_w: pd.Series, r: pd.Series) -> pd.Series:
    """Update positions after returns."""
    return update_positions_after_returns(prev_w, r)


def apply_no_trade_band_local(prev_w: pd.Series,
                        target_w: pd.Series,
                        base_band: float = None,
                        frac: float = 0.25) -> pd.Series:
    """Apply no-trade band to suppress small trades."""
    base_band = base_band or DEFAULT_CONFIG['NO_TRADE_BAND']
    return apply_no_trade_band(prev_w, target_w, base_band, frac)


def cap_by_participation(delta_w: pd.Series, adv: pd.Series,
                         aum: float = None, prate: float = None) -> pd.Series:
    """Cap weight changes by participation rate vs ADV."""
    aum = aum or DEFAULT_CONFIG['AUM']
    prate = prate or DEFAULT_CONFIG['PRATE_MAX']
    
    adv_eff = adv.clip(lower=1_000_000.0)
    cap_w = (prate * adv_eff) / max(aum, 1e-12)
    cap_w = cap_w.reindex(delta_w.index).fillna(0.0).clip(lower=0.0)
    exec_w = delta_w.copy()
    over = delta_w.abs() > cap_w
    exec_w.loc[over] = np.sign(delta_w.loc[over]) * cap_w.loc[over]
    return exec_w


def slow_target(prev_target: pd.Series, alpha_target: pd.Series, 
                gamma: float = None) -> pd.Series:
    """Blend old and new targets (slow target)."""
    gamma = gamma or DEFAULT_CONFIG['GAMMA_SLOW']
    
    idx = prev_target.index.union(alpha_target.index)
    pt = prev_target.reindex(idx, fill_value=0.0)
    at = alpha_target.reindex(idx, fill_value=0.0)
    return (1.0 - gamma) * pt + gamma * at


def apply_name_cap_and_rescale(w: pd.Series,
                               name_cap: float = None,
                               long_w: float = None,
                               short_w: float = None) -> pd.Series:
    """Apply name cap and rescale long/short sides."""
    name_cap = name_cap or DEFAULT_CONFIG['NAME_WEIGHT_CAP']
    long_w = long_w or DEFAULT_CONFIG['LONG_WEIGHT']
    short_w = short_w or DEFAULT_CONFIG['SHORT_WEIGHT']
    
    w = w.copy()
    w = w.clip(lower=-name_cap, upper=name_cap)
    pos = w[w > 0]
    neg = w[w < 0]
    if pos.sum() != 0:
        w.loc[pos.index] = pos / pos.sum() * float(long_w)
    if neg.sum() != 0:
        w.loc[neg.index] = neg / neg.sum() * float(short_w)
    return w


def impact_cost_bps_per_name(exec_dollar: pd.Series, adv: pd.Series,
                             k_sqrt: float = None,
                             spread_bps: float = None, 
                             commission_bps: float = None) -> pd.Series:
    """Compute cost per name including market impact."""
    k_sqrt = k_sqrt or DEFAULT_CONFIG['IMPACT_K_SQRT_BPS']
    spread_bps = spread_bps or DEFAULT_CONFIG['SPREAD_BPS']
    commission_bps = commission_bps or DEFAULT_CONFIG['COMMISSION_BPS']
    
    adv_eff = adv.replace(0.0, np.nan)
    ratio = (exec_dollar / adv_eff).clip(lower=0.0)
    sqrt_term = np.sqrt(ratio).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    impact_bps = k_sqrt * sqrt_term
    return commission_bps + spread_bps + impact_bps


def portfolio_return(w: pd.Series, r: pd.Series) -> float:
    """Compute portfolio return for one day."""
    return compute_portfolio_return(w, r)


def max_drawdown(nav: pd.Series) -> float:
    """Compute maximum drawdown."""
    return compute_max_drawdown(nav)


def simple_backtest(factors_dir: str, returns_path: str, config: Dict = None) -> Dict:
    """
    Simple equal-weighted backtest.
    
    Args:
        factors_dir: Directory containing factor files
        returns_path: Path to returns data
        config: Configuration dictionary
        
    Returns:
        Dictionary with performance results
    """
    config = config or DEFAULT_CONFIG.copy()
    
    # Load data
    factors = load_factors_from_directory(factors_dir)
    ret = read_wide(returns_path)

    # Create composite factor
    weights = rolling_icir_weights(factors, ret)
    comp_raw = composite_weighted(factors, weights)

    # Align data
    aligned = align_panels([comp_raw, ret])
    composite, ret = aligned[0], aligned[1]

    # Get rebalancing dates
    dates = composite.index.intersection(ret.index)
    dates = dates[~ret.loc[dates].isna().all(axis=1)]
    rb_dates = rebalance_dates(dates, config['REBAL_FREQ'])
    
    if len(rb_dates) == 0:
        raise ValueError("No rebalancing dates found.")

    # Initialize tracking variables
    target_w_daily = pd.DataFrame(0.0, index=dates, columns=composite.columns)
    w_prev = pd.Series(0.0, index=composite.columns)

    # Generate target weights
    for t in rb_dates:
        score = composite.loc[t]
        valid = score.dropna()
        if valid.size < config['MIN_STOCKS']:
            target_w = w_prev.copy()
        else:
            target_w = dollar_neutral(valid, config['LONG_WEIGHT'], config['SHORT_WEIGHT'])
            target_w = target_w.reindex(composite.columns).fillna(0.0)

        # Fill target weights for the rebalancing period
        t_pos = dates.get_loc(t)
        if t != rb_dates[-1]:
            t_next = rb_dates[rb_dates.get_loc(t) + 1]
            win = dates[(dates >= t) & (dates < t_next)]
        else:
            win = dates[dates >= t]

        target_w_daily.loc[win] = target_w.values
        w_prev = target_w

    # Compute performance
    port_ret = []
    port_gross_ret = []
    costs = []
    gross_exposure = []
    w_prev = pd.Series(0.0, index=composite.columns)

    for i, d in enumerate(dates):
        w_target = target_w_daily.loc[d]

        # Update positions for returns
        if i == 0:
            w_pretrade = w_prev
        else:
            w_pretrade = position_after_return(w_prev, ret.loc[dates[i-1]])

        # Compute turnover and costs
        trn = compute_turnover(w_pretrade, w_target)
        cost = (config['COST_BPS'] / 1e4) * trn
        costs.append(cost)

        w_today = w_target.copy()

        # Portfolio returns
        r_gross = portfolio_return(w_today, ret.loc[d])
        r_net = r_gross - cost

        port_gross_ret.append(r_gross)
        port_ret.append(r_net)
        gross_exposure.append(np.abs(w_today).sum())

        w_prev = w_today

    # Create performance DataFrame
    res = pd.DataFrame({
        "ret_net": port_ret,
        "ret_gross": port_gross_ret,
        "cost": costs,
        "gross_expo": gross_exposure,
    }, index=dates)

    nav = (1 + res["ret_net"]).cumprod()
    nav_gross = (1 + res["ret_gross"]).cumprod()

    # Compute statistics
    ann_mu = res["ret_net"].mean() * 252
    ann_sd = res["ret_net"].std(ddof=0) * np.sqrt(252)
    sharpe = ann_mu / (ann_sd + 1e-12)
    mdd = max_drawdown(nav)

    daily_turnover = (np.array(costs) / (config['COST_BPS'] / 1e4)) if config['COST_BPS'] > 0 else np.zeros(len(costs))
    res["turnover"] = daily_turnover

    stats = pd.Series({
        "AnnReturn_net": ann_mu,
        "AnnVol": ann_sd,
        "Sharpe": sharpe,
        "MaxDrawdown": mdd,
        "AvgDailyTurnover": res["turnover"].mean(),
    })

    return {
        "perf": res,
        "nav": pd.DataFrame({"NAV_net": nav, "NAV_gross": nav_gross}),
        "composite": composite,
        "stats": stats,
    }


def capacity_aware_backtest(factors_dir: str, returns_path: str, 
                           open_path: str, close_path: str, volume_path: str,
                           config: Dict = None, plot: bool = True) -> Dict:
    """
    Capacity-aware backtest with execution modeling and market impact.
    
    Args:
        factors_dir: Directory containing factor files
        returns_path: Path to returns data
        open_path: Path to open prices
        close_path: Path to close prices
        volume_path: Path to volume data
        config: Configuration dictionary
        plot: Whether to generate plots
        
    Returns:
        Dictionary with performance results and diagnostics
    """
    config = config or DEFAULT_CONFIG.copy()
    
    # Load data
    factors = load_factors_from_directory(factors_dir)
    ret = _clean_cols(read_wide(returns_path))
    open_ = _clean_cols(read_wide(open_path))
    close = _clean_cols(read_wide(close_path))
    vol = _clean_cols(read_wide(volume_path))
    
    for k in list(factors.keys()):
        factors[k] = _clean_cols(factors[k])

    # Align panels
    aligned = align_panels([ret, open_, close, vol] + list(factors.values()))
    ret, open_, close, vol = aligned[0], aligned[1], aligned[2], aligned[3]
    for i, k in enumerate(list(factors.keys())):
        factors[k] = aligned[4 + i]

    # Compute liquidity measures
    adv60 = compute_ADV60(close, vol)

    # Factor weights and composite
    weights = rolling_icir_weights(factors, ret)
    for k in list(weights.keys()):
        w = weights[k]
        if w.index.duplicated().any():
            w = w[~w.index.duplicated(keep="last")]
        w = w.ffill()
        if w.isna().all():
            w = pd.Series(1.0, index=w.index)
        else:
            m = w.mean(skipna=True)
            w = w.fillna(float(m) if np.isfinite(m) else 1.0)
        weights[k] = w

    comp_raw = -composite_weighted(factors, weights)
    valid_mask = comp_raw.notna().sum(axis=1) >= config['MIN_CS_N']
    comp_raw = comp_raw.loc[valid_mask]
    ret = ret.loc[comp_raw.index]
    open_ = open_.loc[comp_raw.index]
    close = close.loc[comp_raw.index]
    vol = vol.loc[comp_raw.index]
    adv60 = adv60.loc[comp_raw.index]

    # Get dates and rebalancing schedule
    dates = comp_raw.index.intersection(ret.index)
    dates = dates[~ret.loc[dates].isna().all(axis=1)]
    rb_dates = rebalance_dates(dates, config['REBAL_FREQ'])
    if len(rb_dates) == 0: 
        raise ValueError("No rebalancing dates found.")
    rb_set = set(rb_dates)

    # Initialize state variables
    prev_target = pd.Series(0.0, index=comp_raw.columns)
    current_w = pd.Series(0.0, index=comp_raw.columns)
    remaining_order = pd.Series(0.0, index=comp_raw.columns)
    days_left = 0

    # Tracking lists
    ret_net, ret_gross, cost_list, turnover_list, gross_expo = [], [], [], [], []
    cost_commission_spread, cost_extra_linear, cost_impact = [], [], []
    prate_median, prate_p95, fill_ratio = [], [], []
    n_nonzero_positions, ideal_slice_store, exec_today_store = [], [], []
    weights_hist, target_hist = [], []

    # Main backtest loop
    for i, d in enumerate(dates):
        # (1) Rebalance: generate new target
        if d in rb_set:
            score = comp_raw.loc[d]
            # Apply liquidity and price filters
            liq_names = adv60.shift(1).loc[d].nlargest(config['TOPN_BY_ADV']).index
            mask_liq = score.index.isin(liq_names)
            price_ok = open_.loc[d].between(config['PRICE_MIN'], config['PRICE_MAX'])
            score = score[mask_liq & price_ok]

            if score.dropna().size >= config['MIN_STOCKS']:
                alpha_target = dollar_neutral(score, config['LONG_WEIGHT'], config['SHORT_WEIGHT'])
                alpha_target = apply_name_cap_and_rescale(alpha_target)
                target = slow_target(prev_target, alpha_target)
                target = apply_name_cap_and_rescale(target)
                target = apply_no_trade_band(prev_target, target)
                prev_target = target.copy()
            else:
                target = prev_target.copy()

            # Update execution plan
            remaining_order = target.reindex(current_w.index).fillna(0.0) - current_w
            days_left = config['EXEC_HORIZON_DAYS']
            target_hist.append(target.rename(d))

        # (2) Natural drift to open
        if i > 0:
            current_w = position_after_return(current_w, ret.loc[dates[i-1]])

        # (3) Execute slice today
        exec_today = pd.Series(0.0, index=current_w.index)
        ideal_slice = None
        if days_left > 0 and remaining_order.abs().sum() > 0:
            ideal_slice = (remaining_order / max(days_left, 1))
            adv_t1 = adv60.shift(1).loc[d].reindex(ideal_slice.index).fillna(0.0)
            exec_tmp = cap_by_participation(ideal_slice, adv=adv_t1)
            ss_turn = 0.5 * exec_tmp.abs().sum()
            if ss_turn > config['MAX_DAILY_TURNOVER']:
                exec_tmp *= (config['MAX_DAILY_TURNOVER'] / (ss_turn + 1e-12))
            exec_today = exec_tmp
            remaining_order = remaining_order - exec_today
            days_left -= 1

        # (4) Post-trade position
        w_new = current_w + exec_today
        w_new = apply_name_cap_and_rescale(w_new)
        weights_hist.append(w_new.rename(d))

        # (5) Costs calculation
        exec_dollar = (exec_today.abs() * config['AUM'])
        adv_t1 = adv60.shift(1).loc[d].reindex(exec_today.index).fillna(0.0)

        # Commission + spread
        lin_bps = config['COMMISSION_BPS'] + config['SPREAD_BPS']
        day_cost_comm_spread = ((lin_bps / 1e4) * (exec_dollar / config['AUM'])).sum()

        # Impact
        adv_eff = adv_t1.replace(0.0, np.nan)
        ratio = (exec_dollar / adv_eff).clip(lower=0.0)
        sqrt_term = np.sqrt(ratio).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        impact_bps = config['IMPACT_K_SQRT_BPS'] * sqrt_term
        day_cost_impact = ((impact_bps / 1e4) * (exec_dollar / config['AUM'])).sum()

        # Extra linear cost
        day_turnover_single_side = 0.5 * exec_today.abs().sum()
        day_cost_extra_linear = (config['COST_BPS'] / 1e4) * day_turnover_single_side

        day_cost = day_cost_comm_spread + day_cost_impact + day_cost_extra_linear

        # (6) Daily PnL
        r_gross = (w_new.reindex(ret.columns).fillna(0.0) * ret.loc[d].fillna(0.0)).sum()
        r_net = r_gross - day_cost

        # Record metrics
        turnover_list.append(day_turnover_single_side)
        cost_list.append(day_cost)
        ret_gross.append(r_gross)
        ret_net.append(r_net)
        gross_expo.append(np.abs(w_new).sum())
        cost_commission_spread.append(day_cost_comm_spread)
        cost_extra_linear.append(day_cost_extra_linear)
        cost_impact.append(day_cost_impact)

        # Participation stats
        pr_series = ratio.replace([np.inf, -np.inf], np.nan).dropna()
        if pr_series.empty:
            prate_median.append(0.0)
            prate_p95.append(0.0)
        else:
            prate_median.append(pr_series.median())
            prate_p95.append(pr_series.quantile(0.95))

        # Fill ratio
        try:
            ideal_slice_amt = float(np.abs(ideal_slice).sum()) if ideal_slice is not None else np.nan
        except Exception:
            ideal_slice_amt = np.nan
        exec_amt = float(np.abs(exec_today).sum())
        exec_today_store.append(exec_amt)
        ideal_slice_store.append(ideal_slice_amt)
        if ideal_slice_amt and np.isfinite(ideal_slice_amt) and ideal_slice_amt > 1e-12:
            fill_ratio.append(exec_amt / ideal_slice_amt)
        else:
            fill_ratio.append(np.nan)

        n_nonzero_positions.append(int((w_new.abs() > 1e-9).sum()))
        current_w = w_new

    # Create performance DataFrame
    perf = pd.DataFrame({
        "ret_net": ret_net,
        "ret_gross": ret_gross,
        "cost": cost_list,
        "turnover": turnover_list,
        "gross_expo": gross_expo
    }, index=dates)

    nav = (1 + perf["ret_net"]).cumprod()
    nav_g = (1 + perf["ret_gross"]).cumprod()

    # Compute statistics
    ann_mu = perf["ret_net"].mean() * 252
    ann_sd = perf["ret_net"].std(ddof=0) * np.sqrt(252)
    sharpe = ann_mu / (ann_sd + 1e-12)
    mdd = (nav / nav.cummax() - 1).min()

    print("Summary (capacity-aware):")
    print(f"AnnReturn(net): {ann_mu:.2%}, AnnVol: {ann_sd:.2%}, Sharpe: {sharpe:.2f}, MaxDD: {mdd:.2%}")
    print(f"Avg single-side turnover: {np.mean(turnover_list):.2%}")
    print(f"Avg daily cost (incl. impact): {np.mean(cost_list):.3%}")
    print(f"Avg gross exposure: {np.mean(gross_expo):.2f}")

    # Save weights
    weights_df = pd.DataFrame(weights_hist)
    weights_path = os.path.join("./backtest_out", "strategy_weights.parquet")
    create_output_directory("./backtest_out")
    save_parquet_safe(weights_df, weights_path)
    
    if target_hist:
        targets_df = pd.DataFrame(target_hist)
        targets_path = os.path.join("./backtest_out", "rebalance_targets.parquet")
        save_parquet_safe(targets_df, targets_path)
        print(f"Saved rebalance targets to: {targets_path}")
    print(f"Saved weights to: {weights_path}")

    # Diagnostics DataFrame
    diag = pd.DataFrame({
        "turnover": turnover_list,
        "gross_expo": gross_expo,
        "cost_total": cost_list,
        "cost_commission_spread": cost_commission_spread,
        "cost_extra_linear": cost_extra_linear,
        "cost_impact": cost_impact,
        "prate_median": prate_median,
        "prate_p95": prate_p95,
        "fill_ratio": fill_ratio,
        "n_positions": n_nonzero_positions,
        "exec_abs_sum": exec_today_store,
        "ideal_abs_sum": ideal_slice_store,
    }, index=dates)

    # Plot results
    if plot:
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(nav.index, nav.values, label="NAV (net)", lw=2)
            plt.plot(nav_g.index, nav_g.values, label="NAV (gross)", lw=1.3, ls="--")
            plt.title("PnL / NAV (Capacity-aware)")
            plt.xlabel("Date")
            plt.ylabel("Cumulative NAV")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print("Plot skipped:", e)

    return {
        "perf": perf,
        "nav": nav,
        "nav_gross": nav_g,
        "comp": comp_raw,
        "diag": diag,
    }


def plot_diagnostics(results: Dict, out_dir: str = "./backtest_out"):
    """Generate diagnostic plots for backtest results."""
    perf = results["perf"].copy()
    nav = results["nav"].copy()
    nav_g = results["nav_gross"].copy()
    diag = results["diag"].copy()

    create_output_directory(out_dir)

    # Cost decomposition cumulative
    fig = plt.figure(figsize=(12, 6))
    (diag["cost_commission_spread"].cumsum()).plot(label="Cum Comm+Spread")
    (diag["cost_extra_linear"].cumsum()).plot(label="Cum Extra Linear")
    (diag["cost_impact"].cumsum()).plot(label="Cum Impact (sqrt)")
    (diag["cost_total"].cumsum()).plot(label="Cum Total", linestyle="--", linewidth=2)
    plt.title("Cumulative Costs (decomposed)")
    plt.ylabel("Cumulative return drag")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cost_decomposition_cum.png"), dpi=160)
    plt.close(fig)

    # Monthly cost breakdown
    monthly = diag[["cost_commission_spread", "cost_extra_linear", "cost_impact"]].resample("ME").sum()
    fig = plt.figure(figsize=(12, 6))
    bottom = np.zeros(len(monthly))
    for col in ["cost_commission_spread", "cost_extra_linear", "cost_impact"]:
        plt.bar(monthly.index, monthly[col].values, bottom=bottom, label=col.replace("_", " "))
        bottom += monthly[col].values
    plt.title("Monthly Cost Breakdown (stacked)")
    plt.ylabel("Monthly cost (return drag)")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cost_breakdown_monthly.png"), dpi=160)
    plt.close(fig)

    # Turnover time series
    fig = plt.figure(figsize=(12, 4))
    diag["turnover"].plot()
    plt.title("Single-side Turnover (daily)")
    plt.ylabel("Turnover")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "turnover_timeseries.png"), dpi=160)
    plt.close(fig)

    # Gross exposure
    fig = plt.figure(figsize=(12, 4))
    diag["gross_expo"].plot()
    plt.axhline(0.70, color="k", linestyle="--", alpha=0.5, label="Target Gross 0.70")
    plt.title("Gross Exposure")
    plt.ylabel("Sum |w|")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gross_exposure.png"), dpi=160)
    plt.close(fig)

    # NAV and drawdown
    dd = nav / nav.cummax() - 1.0
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax[0].plot(nav.index, nav.values, label="NAV (net)", linewidth=2)
    ax[0].plot(nav_g.index, nav_g.values, label="NAV (gross)", linestyle="--", alpha=0.8)
    ax[0].set_title("NAV (Net vs Gross)")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[1].plot(dd.index, dd.values)
    ax[1].set_title("Drawdown")
    ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "nav_and_drawdown.png"), dpi=160)
    plt.close(fig)

    print(f"Diagnostics charts saved to: {out_dir}")


if __name__ == "__main__":
    # Example usage
    config = DEFAULT_CONFIG.copy()
    
    # Simple backtest
    print("Running simple backtest...")
    simple_results = simple_backtest(
        config['FACTOR_DIR'], 
        config['RET_PARQUET'], 
        config
    )
    print("Simple backtest completed.")
    
    # Capacity-aware backtest
    print("Running capacity-aware backtest...")
    capacity_results = capacity_aware_backtest(
        config['FACTOR_DIR'],
        config['RET_PARQUET'],
        config['OPEN_PARQUET'],
        config['CLOSE_PARQUET'],
        config['VOLUME_PARQUET'],
        config
    )
    print("Capacity-aware backtest completed.")
    
    # Generate diagnostics
    plot_diagnostics(capacity_results)
