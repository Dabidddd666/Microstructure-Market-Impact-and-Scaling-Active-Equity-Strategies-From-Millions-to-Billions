"""
Utility Functions Module

This module contains common utility functions shared between strategy scaling
and backtesting modules.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


def read_wide(path: str) -> pd.DataFrame:
    """
    Read parquet file into wide format DataFrame.
    
    Args:
        path: Path to parquet file
        
    Returns:
        DataFrame with datetime index and standardized columns
    """
    df = pd.read_parquet(path)
    if not np.issubdtype(df.index.dtype, np.datetime64):
        df.index = pd.to_datetime(df.index)
    df.columns = df.columns.astype(str)
    return df.sort_index()


def read_panel(path: str, required_cols: List[str] = None) -> pd.DataFrame:
    """
    Read panel data with required columns.
    
    Args:
        path: Path to panel data file
        required_cols: List of required column names
        
    Returns:
        DataFrame with standardized panel format
    """
    df = pd.read_parquet(path)
    
    if required_cols is None:
        required_cols = ["DATE", "TICKER"]
    
    miss = set(required_cols) - set(df.columns)
    if miss: 
        raise ValueError(f"panel missing columns: {miss}")
    
    df = df.copy()
    df["DATE"] = pd.to_datetime(df["DATE"])
    df["TICKER"] = df["TICKER"].astype(str).str.strip()
    
    # Handle additional required columns
    for col in required_cols:
        if col not in ["DATE", "TICKER"] and col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    
    return df


def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with cleaned column names
    """
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.upper()
    return df


def align_panels(panels: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """
    Align multiple panels to common index and columns.
    
    Args:
        panels: List of DataFrames to align
        
    Returns:
        List of aligned DataFrames
    """
    if not panels:
        return []
    
    idx = panels[0].index
    cols = panels[0].columns
    
    for df in panels[1:]:
        idx = idx.intersection(df.index)
        cols = cols.intersection(df.columns)
    
    return [df.loc[idx, cols].copy() for df in panels]


def winsorize_and_zscore(x: pd.Series, q: Tuple[float, float] = (0.01, 0.99)) -> pd.Series:
    """
    Winsorize and z-score a series.
    
    Args:
        x: Input series
        q: Quantile bounds for winsorization
        
    Returns:
        Winsorized and z-scored series
    """
    s = x.replace([np.inf, -np.inf], np.nan)
    if s.dropna().empty: 
        return s
    
    lo, hi = s.quantile(q)
    s = s.clip(lo, hi)
    mu, sd = s.mean(), s.std(ddof=0)
    
    if not np.isfinite(sd) or sd == 0: 
        return s * 0.0
    
    return (s - mu) / sd


def winsorize_z(x: pd.Series, q: Tuple[float, float] = (0.01, 0.99), 
                min_samples: int = 50, use_ranks: bool = True) -> pd.Series:
    """
    Winsorize then z-score with fallback to ranks.
    
    Args:
        x: Input series
        q: Quantile bounds for winsorization
        min_samples: Minimum samples required for z-scoring
        use_ranks: Whether to fall back to ranks if insufficient data
        
    Returns:
        Processed series
    """
    s = x.replace([np.inf, -np.inf], np.nan)
    valid = s.notna()
    n = valid.sum()
    
    if n == 0: 
        return s
    
    sv = s[valid]
    lo, hi = sv.quantile(q[0]), sv.quantile(q[1])
    sv = sv.clip(lo, hi)
    mu = sv.mean()
    sd = sv.std(ddof=0)
    out = s.copy()
    
    if (not np.isfinite(sd)) or (sd < 1e-12) or (n < min_samples):
        if use_ranks and n >= max(10, min_samples // 3):
            out.loc[valid] = sv.rank(pct=True) - 0.5
        else:
            out.loc[valid] = np.nan
        return out
    
    out.loc[valid] = (sv - mu) / sd
    return out


def standardize_cross_section(df: pd.DataFrame, 
                            q: Tuple[float, float] = (0.01, 0.99)) -> pd.DataFrame:
    """
    Apply cross-sectional standardization.
    
    Args:
        df: Input DataFrame
        q: Quantile bounds for winsorization
        
    Returns:
        Cross-sectionally standardized DataFrame
    """
    return df.apply(lambda x: winsorize_and_zscore(x, q), axis=1)


def compute_returns(close: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """
    Compute close-to-close returns.
    
    Args:
        close: Close prices DataFrame
        lag: Lag for return computation
        
    Returns:
        Returns DataFrame
    """
    ret = close.pct_change(lag, fill_method=None)
    ret = ret.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    return ret


def compute_rolling_volatility(ret: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Compute rolling volatility.
    
    Args:
        ret: Returns DataFrame
        window: Rolling window size
        
    Returns:
        Rolling volatility DataFrame
    """
    return ret.rolling(window).std(ddof=0)


def compute_advisory_volume(close: pd.DataFrame, volume: pd.DataFrame, 
                          window: int = 60, min_periods: int = 20) -> pd.DataFrame:
    """
    Compute average daily volume in dollars.
    
    Args:
        close: Close prices DataFrame
        volume: Volume DataFrame
        window: Rolling window size
        min_periods: Minimum periods for rolling calculation
        
    Returns:
        Average daily volume DataFrame
    """
    dollar_volume = close * volume
    return dollar_volume.rolling(window, min_periods=min_periods).mean()


def safe_log(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log with safe handling of zeros.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Log-transformed DataFrame
    """
    return np.log(df.replace(0, np.nan))


def safe_pct_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute percentage change with safe handling of inf values.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Percentage change DataFrame
    """
    out = df.pct_change()
    return out.replace([np.inf, -np.inf], np.nan)


def pivot_to_wide(panel: pd.DataFrame, value_col: str,
                 dates: pd.DatetimeIndex, tickers: pd.Index) -> Optional[pd.DataFrame]:
    """
    Pivot panel data to wide format.
    
    Args:
        panel: Panel DataFrame
        value_col: Column to use as values
        dates: Target date index
        tickers: Target ticker columns
        
    Returns:
        Wide format DataFrame or None if column not found
    """
    if value_col not in panel.columns: 
        return None
    
    wide = panel.pivot(index="DATE", columns="TICKER", values=value_col)
    wide = wide.reindex(index=dates, columns=tickers)
    
    if value_col.upper() in ("SECTOR", "INDUSTRY"):
        return wide.astype("string").ffill()
    
    return wide.apply(pd.to_numeric, errors='coerce')


def create_industry_dummies(ind_row: pd.Series) -> pd.DataFrame:
    """
    Create industry dummy variables for one day.
    
    Args:
        ind_row: Industry classifications for one day
        
    Returns:
        DataFrame with industry dummy variables
    """
    d = pd.get_dummies(ind_row.astype("string"), prefix="ind")
    if d.shape[1] > 0: 
        d = d.iloc[:, 1:]  # Drop first dummy to avoid multicollinearity
    return d.astype(float)


def compute_ic_daily(factors_lag: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    """
    Compute daily Information Coefficient.
    
    Args:
        factors_lag: Lagged factor values
        returns: Returns data
        
    Returns:
        Series of daily IC values
    """
    f, r = align_panels([factors_lag, returns])
    ics = []
    
    for d in f.index:
        a = f.loc[d].rank()
        b = r.loc[d].rank()
        m = a.notna() & b.notna()
        ics.append(a[m].corr(b[m]) if m.sum() >= 10 else np.nan)
    
    return pd.Series(ics, index=f.index)


def compute_rolling_icir(ic_series: pd.Series, window: int = 252, 
                        min_periods: int = 60) -> pd.Series:
    """
    Compute rolling IC Information Ratio.
    
    Args:
        ic_series: Series of IC values
        window: Rolling window size
        min_periods: Minimum periods for rolling calculation
        
    Returns:
        Series of rolling ICIR values
    """
    roll_mu = ic_series.rolling(window, min_periods=min_periods).mean()
    roll_sd = ic_series.rolling(window, min_periods=min_periods).std(ddof=0)
    icir = roll_mu / (roll_sd + 1e-12)
    sign_hist = np.sign(roll_mu).replace(0, 1.0)
    return (sign_hist * icir).shift(1)


def get_rebalance_dates(all_dates: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """
    Get rebalancing dates based on frequency.
    
    Args:
        all_dates: All available trading dates
        freq: Rebalancing frequency
        
    Returns:
        Rebalancing dates
    """
    if freq == "D": 
        return all_dates.copy()
    
    s = pd.Series(1, index=all_dates)
    picked = s.resample(freq).last().index
    return all_dates.intersection(picked)


def compute_portfolio_return(weights: pd.Series, returns: pd.Series) -> float:
    """
    Compute portfolio return for one day.
    
    Args:
        weights: Portfolio weights
        returns: Asset returns
        
    Returns:
        Portfolio return
    """
    return (weights.reindex(returns.index).fillna(0.0) * returns.fillna(0.0)).sum()


def compute_max_drawdown(nav: pd.Series) -> float:
    """
    Compute maximum drawdown.
    
    Args:
        nav: Net asset value series
        
    Returns:
        Maximum drawdown
    """
    return (nav / nav.cummax() - 1.0).min()


def compute_turnover(prev_weights: pd.Series, target_weights: pd.Series) -> float:
    """
    Compute portfolio turnover.
    
    Args:
        prev_weights: Previous period weights
        target_weights: Target weights
        
    Returns:
        Single-side turnover
    """
    idx = prev_weights.index.union(target_weights.index)
    dw = (target_weights.reindex(idx, fill_value=0.0) - 
          prev_weights.reindex(idx, fill_value=0.0))
    return 0.5 * np.abs(dw).sum()


def update_positions_after_returns(prev_weights: pd.Series, returns: pd.Series) -> pd.Series:
    """
    Update positions after returns, maintaining gross exposure.
    
    Args:
        prev_weights: Previous period weights
        returns: Asset returns
        
    Returns:
        Updated weights
    """
    w = prev_weights.reindex(returns.index).fillna(0.0)
    gross = np.abs(w).sum()
    
    if gross == 0: 
        return w
    
    w_new = w * (1.0 + returns.fillna(0.0))
    denom = np.abs(w_new).sum()
    
    return w_new * (gross / denom if denom > 0 else 1.0)


def apply_no_trade_band(prev_weights: pd.Series, target_weights: pd.Series,
                       base_band: float = 0.0025, frac: float = 0.25) -> pd.Series:
    """
    Apply no-trade band to suppress small trades.
    
    Args:
        prev_weights: Previous weights
        target_weights: Target weights
        base_band: Base no-trade band
        frac: Fraction for dynamic band
        
    Returns:
        Adjusted target weights
    """
    idx = prev_weights.index.union(target_weights.index)
    pw = prev_weights.reindex(idx, fill_value=0.0)
    tw = target_weights.reindex(idx, fill_value=0.0)
    thresh = np.minimum(base_band, frac * np.maximum(pw.abs(), tw.abs()))
    keep_prev = (tw - pw).abs() < thresh
    tw.loc[keep_prev] = pw.loc[keep_prev]
    return tw


def load_factors_from_directory(dir_path: str) -> Dict[str, pd.DataFrame]:
    """
    Load factor parquet files from directory.
    
    Args:
        dir_path: Directory containing factor files
        
    Returns:
        Dictionary mapping factor names to DataFrames
        
    Raises:
        FileNotFoundError: If no parquet files found in directory
    """
    import glob
    
    out = {}
    for p in glob.glob(os.path.join(dir_path, "*.parquet")):
        name = os.path.splitext(os.path.basename(p))[0]
        out[name] = read_wide(p)
    
    if not out:
        raise FileNotFoundError(f"No parquet files found in {dir_path}")
    
    return out


def create_output_directory(path: str) -> None:
    """
    Create output directory if it doesn't exist.
    
    Args:
        path: Directory path to create
    """
    os.makedirs(path, exist_ok=True)


def save_parquet_safe(df: pd.DataFrame, path: str) -> None:
    """
    Save DataFrame to parquet with error handling.
    
    Args:
        df: DataFrame to save
        path: Output path
    """
    try:
        df.to_parquet(path)
        print(f"Saved: {path}")
    except Exception as e:
        print(f"Error saving {path}: {e}")


def print_performance_summary(perf_df: pd.DataFrame, nav: pd.Series) -> None:
    """
    Print performance summary statistics.
    
    Args:
        perf_df: Performance DataFrame
        nav: Net asset value series
    """
    ann_mu = perf_df["ret_net"].mean() * 252
    ann_sd = perf_df["ret_net"].std(ddof=0) * np.sqrt(252)
    sharpe = ann_mu / (ann_sd + 1e-12)
    mdd = compute_max_drawdown(nav)
    
    print(f"AnnReturn(net): {ann_mu:.2%}")
    print(f"AnnVol: {ann_sd:.2%}")
    print(f"Sharpe: {sharpe:.2f}")
    print(f"MaxDD: {mdd:.2%}")
    print(f"Avg daily turnover: {perf_df['turnover'].mean():.2%}")


def validate_data_alignment(data_dict: Dict[str, pd.DataFrame]) -> bool:
    """
    Validate that all DataFrames in dictionary have aligned indices and columns.
    
    Args:
        data_dict: Dictionary of DataFrames
        
    Returns:
        True if all DataFrames are aligned, False otherwise
    """
    if not data_dict:
        return True
    
    dfs = list(data_dict.values())
    aligned_dfs = align_panels(dfs)
    
    for original, aligned in zip(dfs, aligned_dfs):
        if not original.index.equals(aligned.index) or not original.columns.equals(aligned.columns):
            return False
    
    return True
