"""
Strategy Scaling Module

This module contains functions for factor construction, risk neutralization,
and factor aggregation for quantitative trading strategies.
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, Dict, List
from pathlib import Path
from utils import (
    read_wide, read_panel, align_panels, winsorize_and_zscore, standardize_cross_section,
    compute_returns, compute_rolling_volatility, safe_log, safe_pct_change,
    pivot_to_wide, create_industry_dummies, compute_ic_daily, compute_rolling_icir,
    load_factors_from_directory, create_output_directory, save_parquet_safe
)

# Configuration
# CLOSE_PARQUET = "/Users/luchenshi/Desktop/Data for Alphathon/Close.parquet"
# PANEL_PARQUET = "/Users/luchenshi/Desktop/Data for Alphathon/cleaned_train_data.parquet"
# VOLUME_PARQUET = None
# OUT_DIR = "./neutralize_out"

# Neutralization parameters
RET_LAG = 1
VOL_WIN_SHORT = 20
VOL_WIN_LONG = 60
WLS_WIN = 20
WINSOR_Q = (0.01, 0.99)
MIN_STOCKS = 50

# Factor construction parameters
DATA_DIR = Path(".")
FACTORS_OUT_DIR = Path("./factors_investable")   
SAVE_UNSHIFTED = False
RAW_DIR = Path("./factors_unshifted")

# Factor aggregation parameters
IC_WIN = 252
MIN_IC_DAYS = 60
MIN_CS_N = 150
MIN_FACTORS_PER_DAY = 3
RANK_FALLBACK = True


def compute_c2c_return(close: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """Compute close-to-close returns."""
    return compute_returns(close, lag)


def rolling_vol(ret: pd.DataFrame, win: int) -> pd.DataFrame:
    """Compute rolling volatility."""
    return compute_rolling_volatility(ret, win)


def amihud_illiq(ret: pd.DataFrame, close: pd.DataFrame, vol: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Compute Amihud illiquidity measure."""
    if vol is None: 
        return None
    dollar = close * vol
    illiq = (ret.abs() / dollar).replace([np.inf, -np.inf], np.nan)
    return illiq.rolling(20).mean()


def winsorize_and_z(x: pd.Series, q=WINSOR_Q) -> pd.Series:
    """Winsorize and z-score a series."""
    return winsorize_and_zscore(x, q)


def standardize_cs(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cross-sectional standardization."""
    return standardize_cross_section(df, WINSOR_Q)


def pivot_wide(panel: pd.DataFrame, value_col: str,
               dates: pd.DatetimeIndex, tickers: pd.Index) -> Optional[pd.DataFrame]:
    """Pivot panel data to wide format."""
    return pivot_to_wide(panel, value_col, dates, tickers)


def industry_dummies_row(ind_row: pd.Series) -> pd.DataFrame:
    """Create industry dummy variables for one day."""
    return create_industry_dummies(ind_row)


def neutralize_one_day(
    r_row: pd.Series,              # Series[ticker] float
    style_row_df: Optional[pd.DataFrame],  # DataFrame[ticker x factors] float
    ind_row: Optional[pd.Series],  # Series[ticker] category string
    w_row: Optional[pd.Series],    # Series[ticker] float (>=0)
    add_const: bool = True,
    min_stocks: int = MIN_STOCKS
) -> pd.Series:
    """Neutralize returns for one day using OLS regression."""
    tickers = r_row.index
    r_row = pd.to_numeric(r_row, errors='coerce')

    X_parts = []
    if add_const:
        X_parts.append(pd.Series(1.0, index=tickers, name="const"))
    if ind_row is not None:
        X_parts.append(industry_dummies_row(ind_row.reindex(tickers)))
    if style_row_df is not None:
        F = style_row_df.reindex(index=tickers).apply(pd.to_numeric, errors='coerce')
        F = F.loc[:, ~F.isna().all(axis=0)]  # Remove all-NaN columns
        X_parts.append(F)

    if not X_parts: 
        return r_row

    X = pd.concat(X_parts, axis=1)
    valid = (~r_row.isna()) & (~X.isna().any(axis=1))
    if valid.sum() < min_stocks:
        return pd.Series(np.nan, index=tickers, name=r_row.name)

    X = X.loc[valid]
    X = X.loc[:, X.nunique(dropna=True) > 1]  # Remove constant columns
    if X.shape[1] == 0:
        out = pd.Series(np.nan, index=tickers, name=r_row.name)
        out.loc[valid.index[valid]] = r_row.loc[valid]
        return out

    y = r_row.loc[valid].astype(float).to_numpy()
    Xv = X.to_numpy(dtype=float)

    if w_row is not None:
        w = pd.to_numeric(w_row.reindex(tickers), errors='coerce').fillna(0.0)
        w = np.clip(w.loc[valid].to_numpy(), 1e-12, np.inf).astype(float)
        W = np.sqrt(w)[:, None]
        Xw, yw = Xv * W, y * W.squeeze()
    else:
        Xw, yw = Xv, y

    b, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    resid = y - (Xv @ b)

    out = pd.Series(np.nan, index=tickers, name=r_row.name)
    out.loc[X.index] = resid
    return out


def neutralize_returns(close_path: str, panel_path: str, volume_path: Optional[str] = None):
    """
    Main function to neutralize returns with respect to risk factors.
    
    Args:
        close_path: Path to close prices parquet file
        panel_path: Path to panel data with risk factors
        volume_path: Optional path to volume data
        
    Returns:
        Tuple of (raw_returns, neutralized_returns, style_exposures, industry_wide)
    """
    create_output_directory(OUT_DIR)
    
    close = read_wide(close_path)      
    panel = read_panel(panel_path, ["DATE", "TICKER", "SECTOR"])     
    vol_df = read_wide(volume_path) if volume_path else None

    # Filter panel to only include stocks in close data
    tickers = close.columns
    panel = panel[panel["TICKER"].isin(tickers)].copy()

    ret = compute_c2c_return(close, lag=RET_LAG)
    dates = close.index

    # Create wide format risk factor exposures
    industry_wide = pivot_wide(panel[["DATE", "TICKER", "SECTOR"]], "SECTOR", dates, tickers)
    mom_wide = pivot_wide(panel[["DATE", "TICKER", "MOMENTUM"]], "MOMENTUM", dates, tickers)
    size_wide = pivot_wide(panel[["DATE", "TICKER", "SIZE"]], "SIZE", dates, tickers)
    beta_wide = pivot_wide(panel[["DATE", "TICKER", "BETA"]], "BETA", dates, tickers)

    # Compute additional style factors
    vol20 = rolling_vol(ret, VOL_WIN_SHORT)
    vol60 = rolling_vol(ret, VOL_WIN_LONG)
    amihud = amihud_illiq(ret, close, vol_df)

    # Standardize style exposures
    style_exposures: Dict[str, pd.DataFrame] = {}
    if size_wide is not None: 
        style_exposures["size"] = standardize_cs(size_wide)
    if beta_wide is not None: 
        style_exposures["beta"] = standardize_cs(beta_wide)
    if mom_wide is not None: 
        style_exposures["momentum"] = standardize_cs(mom_wide)
    style_exposures["vol20"] = standardize_cs(vol20)
    style_exposures["vol60"] = standardize_cs(vol60)
    if amihud is not None: 
        style_exposures["amihud"] = standardize_cs(amihud)

    # Compute inverse variance weights for WLS
    sigma2 = ret.rolling(WLS_WIN).var(ddof=0).replace(0, np.nan)
    w_invvar = (1.0 / sigma2).apply(pd.to_numeric, errors='coerce')

    # Neutralize returns day by day
    factor_names = list(style_exposures.keys())
    neutral_rows = []

    for dt in dates:
        r_row = ret.loc[dt]

        # Prepare style factor exposures for this day
        parts = []
        for k in factor_names:
            dfk = style_exposures[k]
            s = dfk.loc[dt] if dt in dfk.index else pd.Series(index=tickers, dtype=float, name=k)
            parts.append(s.rename(k))
        style_row_df = pd.concat(parts, axis=1)

        ind_row = industry_wide.loc[dt] if (industry_wide is not None and dt in industry_wide.index) else None
        w_row = w_invvar.loc[dt] if dt in w_invvar.index else None

        # Neutralize
        resid = neutralize_one_day(
            r_row=r_row,
            style_row_df=style_row_df,
            ind_row=ind_row,
            w_row=w_row,
            add_const=True,
            min_stocks=MIN_STOCKS
        )
        neutral_rows.append(resid)

    neutral = pd.DataFrame(neutral_rows, index=dates, columns=tickers)

    # Save results
    out_neu = os.path.join(OUT_DIR, "neutralized_returns.parquet")
    save_parquet_safe(neutral, out_neu)
    
    fac_dir = os.path.join(OUT_DIR, "factor_exposures")
    create_output_directory(fac_dir)
    for k, dfk in style_exposures.items():
        save_parquet_safe(dfk, os.path.join(fac_dir, f"{k}.parquet"))
    if industry_wide is not None:
        save_parquet_safe(industry_wide, os.path.join(fac_dir, "industry.parquet"))
    
    print("Saved:", out_neu, neutral.shape, " | exposures:", list(style_exposures))

    # Print correlation diagnostics
    last = neutral.index.max()
    if last in neutral.index:
        for k in ["beta", "momentum", "size"]:
            if k in style_exposures:
                xy = pd.concat([neutral.loc[last], style_exposures[k].loc[last]], axis=1).dropna()
                corr = xy.iloc[:, 0].corr(xy.iloc[:, 1]) if len(xy) > 5 else np.nan
                print(f"[{last.date()}] corr(neutral, {k}) = {corr:.3f}")

    return ret, neutral, style_exposures, industry_wide


def construct_alpha_factors():
    """
    Construct alpha factors from price and volume data.
    
    Returns:
        Dictionary mapping factor names to factor DataFrames
    """
    FACTORS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(exist_ok=True)

    def read_pq(name: str) -> pd.DataFrame:
        return read_wide(str(DATA_DIR / f"{name}.parquet"))

    def align(frames: dict) -> dict:
        idx = None
        cols = None
        for _, df in frames.items():
            idx = df.index if idx is None else idx.intersection(df.index)
            cols = df.columns if cols is None else cols.intersection(df.columns)
        return {k: v.loc[idx, cols] for k, v in frames.items()}

    # Load data
    O = read_pq("Open")
    H = read_pq("High")
    L = read_pq("Low")
    C = read_pq("Close")
    V = read_pq("Volume")
    O, H, L, C, V = align({"O": O, "H": H, "L": L, "C": C, "V": V}).values()

    ret_cc = safe_pct_change(C)
    dollarvol = C * V

    # Compute technical indicators
    C_shift = C.shift(1)
    TR = pd.concat([
        (H - L).abs(),
        (H - C_shift).abs(),
        (L - C_shift).abs()
    ], axis=0).groupby(level=0).max()
    ATR14 = TR.rolling(14, min_periods=7).mean()

    # Define alpha factors
    eps = 1e-12
    range_log = np.log(H.div(L)).replace([np.inf, -np.inf], np.nan)
    logV = safe_log(V)
    ADV60 = dollarvol.rolling(60, min_periods=20).mean()

    factors_raw = {
        # No shift needed
        "alpha_overnight": O.div(C.shift(1)) - 1,                    # O_t/C_{t-1}-1
        "alpha_gap_rev": O.sub(C.shift(1)).div(C.shift(1)),         # (O_t - C_{t-1})/C_{t-1}

        # Need shift
        "alpha_intraday_rev": C.sub(O).div(O),                       # (C_t - O_t)/O_t
        "alpha_rangevol_rev": -0.5 * (range_log ** 2),              # -0.5 [ln(H/L)]^2
        "alpha_volsurprise": (logV - logV.rolling(60, min_periods=20).mean()) /
                            logV.rolling(60, min_periods=20).std(),
        "alpha_turnover_jump": (dollarvol / ADV60).replace([np.inf, -np.inf], np.nan) - 1,
        "alpha_amihud_rev": -(ret_cc.abs() / dollarvol.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
                            .rolling(20, min_periods=10).mean(),
        "alpha_clv": ((C - L) - (H - C)) / (H - L).replace(0, np.nan),
        "alpha_st_trend_5_20": (C.rolling(5, min_periods=3).mean() /
                               C.rolling(20, min_periods=10).mean() - 1),
        "alpha_range_to_atr_rev": -(range_log / ATR14.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan),
    }

    NEED_SHIFT = {
        "alpha_intraday_rev",
        "alpha_rangevol_rev",
        "alpha_volsurprise",
        "alpha_turnover_jump",
        "alpha_amihud_rev",
        "alpha_clv",
        "alpha_st_trend_5_20",
        "alpha_range_to_atr_rev",
    }

    # Save investable factors
    for name, df in factors_raw.items():
        if SAVE_UNSHIFTED:
            df.to_parquet(RAW_DIR / f"{name}.parquet")

        investable = df.shift(1) if name in NEED_SHIFT else df
        investable = investable.replace([np.inf, -np.inf], np.nan)
        save_parquet_safe(investable, str(FACTORS_OUT_DIR / f"{name}.parquet"))

    print("All investable factors saved.")
    return factors_raw


def rolling_icir_weights(factors: Dict[str, pd.DataFrame],
                         ret_cur: pd.DataFrame,
                         ic_win=IC_WIN, min_days=MIN_IC_DAYS) -> Dict[str, pd.Series]:
    """Compute rolling ICIR weights per factor, shifted to avoid lookahead."""
    weights = {}
    for name, df in factors.items():
        ic_series = compute_ic_daily(df.shift(1), ret_cur)
        weights[name] = compute_rolling_icir(ic_series, ic_win, min_days)
    return weights


def winsorize_z(x: pd.Series, q=(0.01, 0.99)) -> pd.Series:
    """Winsorize then z-score with fallback to ranks."""
    return winsorize_z(x, q, MIN_CS_N, RANK_FALLBACK)


def rank_fallback_row(df_row: pd.DataFrame) -> pd.Series:
    """Rank fallback: average percentile ranks across columns for one day."""
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
    """Create composite score from weighted factors."""
    fac_list = align_panels(list(factors.values()))
    idx, cols = fac_list[0].index, fac_list[0].columns
    names = list(factors.keys())
    Zs = [df.apply(winsorize_z, axis=1) for df in fac_list]
    Zpanel = {n: Z for n, Z in zip(names, Zs)}
    comp = pd.DataFrame(np.nan, index=idx, columns=cols)
    
    for d in idx:
        usable = []
        for n in names:
            if Zpanel[n].loc[d].notna().sum() >= MIN_CS_N:
                usable.append(n)
        if len(usable) < MIN_FACTORS_PER_DAY:
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


def create_composite_factor(factors_dir: str, returns_path: str) -> pd.DataFrame:
    """
    Create composite factor from individual factors using ICIR weighting.
    
    Args:
        factors_dir: Directory containing factor parquet files
        returns_path: Path to returns data for IC calculation
        
    Returns:
        Composite factor DataFrame
    """
    # Load data
    factors = load_factors_from_directory(factors_dir)
    ret = read_wide(returns_path)

    # Compute factor weights
    weights = rolling_icir_weights(factors, ret, ic_win=IC_WIN, min_days=MIN_IC_DAYS)
    
    # Clean up weights (handle duplicates, fill NaN)
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

    # Create composite factor
    comp_raw = composite_weighted(factors, weights)
    valid_mask = comp_raw.notna().sum(axis=1) >= MIN_CS_N
    comp_raw = comp_raw.loc[valid_mask]
    
    return comp_raw


if __name__ == "__main__":
    # Example usage
    ret, neutral, style_exposures, industry_wide = neutralize_returns(
        CLOSE_PARQUET, PANEL_PARQUET, VOLUME_PARQUET
    )
    construct_alpha_factors()
