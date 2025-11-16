"""
Backtesting Helper Library
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, List, Tuple, Optional, Callable
from scipy import linalg, optimize
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings('ignore')


class BacktestingHelper:
    """
    Main class containing all backtesting helper functions organized by module.
    """
    
    def __init__(self):
        """Initialize the backtesting helper."""
        pass


def estimate_forecast_autocorr(signals: pd.DataFrame, lag: int = 1) -> pd.Series:
    """
    Estimate cross-sectional forecast autocorrelation ρ_f for each date.
    
    Purpose: Qian shows forecast-induced turnover depends on (1 - ρ_f).
    Higher persistence → lower turnover.
    
    Args:
        signals: DataFrame with dates as index, assets as columns (z-scored signals)
        lag: Number of periods to lag (default: 1)
        
    Returns:
        Series with autocorrelation for each date
            """
    if signals.empty:
        return pd.Series(dtype=float)
    
    # Calculate cross-sectional correlation between current and lagged signals
    autocorr_series = []
    dates = signals.index[lag:]  # Skip first 'lag' dates
    
    for date in dates:
        current_signals = signals.loc[date]
        lagged_signals = signals.loc[date - pd.Timedelta(days=lag)]
        
        # Remove NaN values for correlation calculation
        valid_mask = ~(current_signals.isna() | lagged_signals.isna())
        
        if valid_mask.sum() < 2:  # Need at least 2 points for correlation
            autocorr_series.append(np.nan)
        else:
            current_clean = current_signals[valid_mask]
            lagged_clean = lagged_signals[valid_mask]
            
            if len(current_clean) > 1:
                corr, _ = pearsonr(current_clean, lagged_clean)
                autocorr_series.append(corr)
            else:
                autocorr_series.append(np.nan)
    
    return pd.Series(autocorr_series, index=dates, name='forecast_autocorr')


def implied_turnover_qian(rho_f: pd.Series, N: int, TE: float, sigma_idio: float) -> pd.Series:
    """
    Calculate analytical one-way turnover forecast from persistence.
    
    Formula (unconstrained case):
    T_t = (N/π) * (σ_model/σ_0) * sqrt(1 - ρ_f(t))
    
    where:
    - σ_model is target tracking error
    - σ_0 is average idiosyncratic volatility
    
    Args:
        rho_f: Forecast autocorrelation series
        N: Number of assets
        TE: Target tracking error (σ_model)
        sigma_idio: Average idiosyncratic volatility (σ_0)
        
    Returns:
        Series with implied turnover for each date
        
    Reference: Qian (2007) Eq. (8)/(9)
    """
    if rho_f.empty:
        return pd.Series(dtype=float)
    
    # Calculate turnover using Qian's formula
    turnover = (N / np.pi) * (TE / sigma_idio) * np.sqrt(1 - rho_f)
    
    # Handle edge cases
    turnover = turnover.clip(lower=0)  # Turnover can't be negative
    turnover = turnover.replace([np.inf, -np.inf], np.nan)
    
    return turnover.rename('implied_turnover')


def smooth_signal(signal: pd.Series, method: str = "EWMA", span: int = 60, 
                 clip: float = None, alpha: float = None) -> pd.Series:
    """
    Smooth signal to increase ρ_f (make alpha "slower") and lower turnover.
    
    Args:
        signal: Input signal series
        method: Smoothing method ("EWMA", "Kalman", "lowpass")
        span: Window span for smoothing
        clip: Optional clipping threshold (tanh compression)
        alpha: Smoothing parameter (for EWMA)
        
    Returns:
        Smoothed signal series
    """
    if signal.empty:
        return signal
    
    if method == "EWMA":
        # Exponential Weighted Moving Average
        if alpha is None:
            alpha = 2 / (span + 1)
        smoothed = signal.ewm(alpha=alpha, adjust=False).mean()
        
    elif method == "Kalman":
        # Simple Kalman filter implementation
        smoothed = _kalman_smooth(signal, span)
        
    elif method == "lowpass":
        # Low-pass filter using rolling mean
        smoothed = signal.rolling(window=span, min_periods=1).mean()
        
    else:
        raise ValueError(f"Unknown smoothing method: {method}")
    
    # Optional clipping to damp spikes
    if clip is not None:
        smoothed = np.tanh(smoothed / clip) * clip
    
    return smoothed.rename(f"{signal.name}_smoothed" if signal.name else "smoothed_signal")


def _kalman_smooth(signal: pd.Series, span: int) -> pd.Series:
    """
    Simple Kalman filter for signal smoothing.
    
    Args:
        signal: Input signal
        span: Smoothing parameter
        
    Returns:
        Kalman-smoothed signal
    """
    if len(signal) < 2:
        return signal
    
    # Simple Kalman filter parameters
    Q = 1e-5  # Process noise
    R = 1.0   # Measurement noise
    P = 1.0   # Initial covariance
    
    x = signal.iloc[0]  # Initial state
    smoothed = [x]
    
    for i in range(1, len(signal)):
        # Prediction step
        P = P + Q
        
        # Update step
        K = P / (P + R)  # Kalman gain
        x = x + K * (signal.iloc[i] - x)
        P = (1 - K) * P
        
        smoothed.append(x)
    
    return pd.Series(smoothed, index=signal.index)


def compose_multispeed_alpha(fast: pd.Series, slow: pd.Series, 
                           w_fast: float, w_slow: float) -> pd.Series:
    """
    Combine fast and slow alphas for optimal performance under frictions.
    
    In frictions, the one-alpha optimum is no longer true; mixing speeds is beneficial.
    
    Args:
        fast: Fast alpha signal
        slow: Slow alpha signal  
        w_fast: Weight for fast alpha
        w_slow: Weight for slow alpha
        
    Returns:
        Combined alpha signal
    """
    if fast.empty or slow.empty:
        return pd.Series(dtype=float)
    
    # Align indices
    common_index = fast.index.intersection(slow.index)
    fast_aligned = fast.reindex(common_index)
    slow_aligned = slow.reindex(common_index)
    
    # Combine signals
    combined = w_fast * fast_aligned + w_slow * slow_aligned
    
    return combined.rename('multispeed_alpha')


def portfolio_drift(prev_w: pd.Series, realized_returns: pd.Series, 
                   rf: float = 0.0) -> pd.Series:
    """
    Compute inherited weights g_t * w_{t-1} (positions drift with returns).
    
    Core equations:
    τ_t = w_t - g_t * w_{t-1}
    g_t = diag(1 + r_f + r_t)
    
    Args:
        prev_w: Previous period weights
        realized_returns: Realized returns for the period
        rf: Risk-free rate
        
    Returns:
        Series with inherited weights after drift
    """
    if prev_w.empty or realized_returns.empty:
        return pd.Series(dtype=float)
    
    # Align indices
    common_assets = prev_w.index.intersection(realized_returns.index)
    w_aligned = prev_w.reindex(common_assets, fill_value=0)
    r_aligned = realized_returns.reindex(common_assets, fill_value=0)
    
    # Calculate growth factors: g_t = diag(1 + r_f + r_t)
    growth_factors = 1 + rf + r_aligned
    
    # Inherited weights: g_t * w_{t-1}
    inherited_w = growth_factors * w_aligned
    
    # Renormalize to ensure weights sum to 1
    total_weight = inherited_w.sum()
    if total_weight != 0:
        inherited_w = inherited_w / total_weight
    
    return inherited_w.rename('inherited_weights')


def trade_vector(current_w: pd.Series, target_w: pd.Series, 
                growth_diag: pd.Series) -> pd.Series:
    """
    Calculate trade vector in weight space: τ_t = w_t - g_t * w_{t-1}.
    
    Args:
        current_w: Current target weights
        target_w: Previous period weights  
        growth_diag: Growth factors g_t
        
    Returns:
        Series with trade vector
    """
    if current_w.empty or target_w.empty:
        return pd.Series(dtype=float)
    
    # Align all indices
    common_assets = current_w.index.intersection(target_w.index)
    if not growth_diag.empty:
        common_assets = common_assets.intersection(growth_diag.index)
    
    w_current = current_w.reindex(common_assets, fill_value=0)
    w_prev = target_w.reindex(common_assets, fill_value=0)
    
    if not growth_diag.empty:
        g = growth_diag.reindex(common_assets, fill_value=1)
        w_prev_drifted = g * w_prev
    else:
        w_prev_drifted = w_prev
    
    # Trade vector: τ_t = w_t - g_t * w_{t-1}
    tau = w_current - w_prev_drifted
    
    return tau.rename('trade_vector')


def quadratic_cost_tau(tau: pd.Series, Lambda: np.ndarray) -> float:
    """
    Calculate trading cost: TC_t = (1/2) * τ_t^T * Λ_t * τ_t.
    
    Args:
        tau: Trade vector
        Lambda: Cost matrix Λ (can be diagonal or full matrix)
        
    Returns:
        Trading cost as float
    """
    if tau.empty or Lambda.size == 0:
        return 0.0
    
    # Convert to numpy arrays, handling alignment
    tau_vec = tau.values
    if len(tau_vec) != Lambda.shape[0]:
        # Handle dimension mismatch
        min_dim = min(len(tau_vec), Lambda.shape[0])
        tau_vec = tau_vec[:min_dim]
        Lambda = Lambda[:min_dim, :min_dim]
    
    # Quadratic cost: (1/2) * τ^T * Λ * τ
    cost = 0.5 * tau_vec.T @ Lambda @ tau_vec
    
    return float(cost)


def net_return_with_costs(w: pd.Series, r_next: pd.Series, tau: pd.Series, 
                         Lambda: np.ndarray, wealth: float = 1.0) -> float:
    """
    Calculate portfolio return net of trading costs.
    
    Formula: r_{t+1}^{net} = w_t^T * r_{t+1} - (1/2) * τ_t^T * Λ_t * τ_t
    
    Args:
        w: Portfolio weights
        r_next: Next period returns
        tau: Trade vector
        Lambda: Cost matrix
        wealth: Portfolio wealth (for scaling)
        
    Returns:
        Net return after costs
    """
    if w.empty or r_next.empty:
        return 0.0
    
    # Align indices
    common_assets = w.index.intersection(r_next.index)
    w_aligned = w.reindex(common_assets, fill_value=0)
    r_aligned = r_next.reindex(common_assets, fill_value=0)
    
    # Gross return: w^T * r
    gross_return = w_aligned.dot(r_aligned)
    
    # Trading cost
    trading_cost = quadratic_cost_tau(tau, Lambda)
    
    # Net return
    net_return = gross_return - trading_cost / wealth
    
    return float(net_return)


def optimal_quadratic_step(target_w: pd.Series, cur_w: pd.Series, 
                          Lambda: np.ndarray, kappa: float) -> pd.Series:
    """
    One-step closed-form optimization towards target with quadratic costs.
    
    Formula: w_t = (I + κ*Λ)^{-1} * (target_t + κ*Λ*g_t*w_{t-1})
    
    This is the discrete-time analog of tracking a moving target with speed Λ^{-1}.
    
    Args:
        target_w: Target weights
        cur_w: Current weights
        Lambda: Cost matrix
        kappa: Speed/penalty parameter
        
    Returns:
        Optimal weights for next period
    """
    if target_w.empty or cur_w.empty:
        return pd.Series(dtype=float)
    
    # Align indices
    common_assets = target_w.index.intersection(cur_w.index)
    target_aligned = target_w.reindex(common_assets, fill_value=0)
    cur_aligned = cur_w.reindex(common_assets, fill_value=0)
    
    n = len(common_assets)
    
    # Ensure Lambda is properly sized
    if Lambda.shape != (n, n):
        if Lambda.shape[0] == 1:  # Diagonal case
            Lambda = np.diag(np.full(n, Lambda[0, 0]))
        else:
            # Truncate or pad Lambda to match
            min_dim = min(n, Lambda.shape[0])
            Lambda = Lambda[:min_dim, :min_dim]
            if Lambda.shape[0] < n:
                # Pad with identity
                Lambda_padded = np.eye(n)
                Lambda_padded[:Lambda.shape[0], :Lambda.shape[1]] = Lambda
                Lambda = Lambda_padded
    
    # Convert to numpy
    target_vec = target_aligned.values
    cur_vec = cur_aligned.values
    
    # Calculate optimal weights: (I + κ*Λ)^{-1} * (target + κ*Λ*cur)
    I = np.eye(n)
    A = I + kappa * Lambda
    
    try:
        A_inv = linalg.inv(A)
        optimal_vec = A_inv @ (target_vec + kappa * Lambda @ cur_vec)
    except linalg.LinAlgError:
        # Fallback to pseudo-inverse
        A_inv = linalg.pinv(A)
        optimal_vec = A_inv @ (target_vec + kappa * Lambda @ cur_vec)
    
    # Ensure weights sum to 1
    optimal_vec = optimal_vec / optimal_vec.sum() if optimal_vec.sum() != 0 else optimal_vec
    
    return pd.Series(optimal_vec, index=common_assets, name='optimal_weights')


def implementable_efficient_frontier(risk_grid: np.ndarray, model: Callable, 
                                   Sigma: np.ndarray, Lambda: np.ndarray,
                                   n_simulations: int = 100) -> pd.DataFrame:
    """
    Build the Implementable Efficient Frontier (σ, k(σ)) by maximizing expected net return
    at each risk level with quadratic costs.
    
    Formula: k(σ) = max_π E[r_{π,t+1}^{net}] s.t. E[π_t^T * Σ * π_t] = σ^2
    
    Args:
        risk_grid: Array of target risk levels
        model: Function that returns target weights given date
        Sigma: Covariance matrix
        Lambda: Cost matrix
        n_simulations: Number of simulation periods
        
    Returns:
        DataFrame with columns ['risk', 'expected_return', 'sharpe']
    """
    results = []
    
    for target_risk in risk_grid:
        try:
            # Optimize for this risk level
            optimal_return = _optimize_risk_level(target_risk, model, Sigma, Lambda, n_simulations)
            sharpe = optimal_return / target_risk if target_risk > 0 else 0
            
            results.append({
                'risk': target_risk,
                'expected_return': optimal_return,
                'sharpe': sharpe
            })
        except Exception as e:
            print(f"Warning: Failed to optimize for risk level {target_risk}: {e}")
            results.append({
                'risk': target_risk,
                'expected_return': 0,
                'sharpe': 0
            })
    
    return pd.DataFrame(results)


def _optimize_risk_level(target_risk: float, model: Callable, Sigma: np.ndarray, 
                        Lambda: np.ndarray, n_simulations: int) -> float:
    """
    Optimize expected return for a given risk level.
    """
    def objective(weights):
        # Ensure weights sum to 1
        weights = weights / weights.sum()
        
        # Calculate risk constraint
        portfolio_variance = weights.T @ Sigma @ weights
        risk_penalty = (portfolio_variance - target_risk**2)**2
        
        # Simulate returns (simplified - in practice use actual return model)
        expected_return = np.random.normal(0.05, 0.15)  # Placeholder
        
        # Add cost penalty
        cost_penalty = 0.5 * weights.T @ Lambda @ weights
        
        return -(expected_return - cost_penalty) + 1000 * risk_penalty
    
    # Initial guess
    n_assets = Sigma.shape[0]
    x0 = np.ones(n_assets) / n_assets
    
    # Constraints: weights sum to 1
    constraints = {'type': 'eq', 'fun': lambda x: x.sum() - 1}
    
    # Bounds: weights between 0 and 1
    bounds = [(0, 1) for _ in range(n_assets)]
    
    try:
        result = optimize.minimize(objective, x0, method='SLSQP', 
                                 constraints=constraints, bounds=bounds)
        return -result.fun
    except:
        return 0.0


def scale_path_to_AUM(frontier: pd.DataFrame, AUMs: List[float], 
                     Lambda_factory: Callable) -> Dict[float, pd.DataFrame]:
    """
    Show how efficient frontier drops with size due to scaling costs.
    
    Args:
        frontier: Base efficient frontier
        AUMs: List of AUM levels to scale to
        Lambda_factory: Function that creates Lambda matrix for given AUM
        
    Returns:
        Dictionary mapping AUM to scaled frontier
    """
    scaled_frontiers = {}
    
    for aum in AUMs:
        # Scale Lambda based on AUM (costs typically scale with trade size)
        Lambda_scaled = Lambda_factory(aum)
        
        # Recalculate frontier with scaled costs
        risk_levels = frontier['risk'].values
        scaled_frontier = implementable_efficient_frontier(
            risk_levels, None, None, Lambda_scaled
        )
        
        scaled_frontiers[aum] = scaled_frontier
    
    return scaled_frontiers



def lambda_from_liquidity(adv: pd.Series, spread_bps: pd.Series, vol: pd.Series, 
                         Y: float = 0.6) -> np.ndarray:
    """
    Calibrate Λ from microstructure data.
    
    Transitory impact: λ_i ∝ Y * σ_i / ADV_i
    
    Args:
        adv: Average Daily Volume series
        spread_bps: Bid-ask spread in basis points
        vol: Volatility series
        Y: Scaling parameter (default 0.6)
        
    Returns:
        Diagonal cost matrix Λ
    """
    if adv.empty or vol.empty:
        return np.array([[1.0]])  # Default cost
    
    # Align indices
    common_assets = adv.index.intersection(vol.index)
    adv_aligned = adv.reindex(common_assets, fill_value=adv.median())
    vol_aligned = vol.reindex(common_assets, fill_value=vol.median())
    
    # Calculate per-asset cost: λ_i ∝ Y * σ_i / ADV_i
    lambda_diag = Y * vol_aligned / adv_aligned
    
    # Add spread component if provided
    if not spread_bps.empty:
        spread_aligned = spread_bps.reindex(common_assets, fill_value=spread_bps.median())
        lambda_diag += spread_aligned / 10000  # Convert bps to decimal
    
    # Create diagonal matrix
    Lambda = np.diag(lambda_diag.values)
    
    return Lambda


def dynamic_participation_rate(alpha_strength: pd.Series, adv: pd.Series, 
                             day_limit: float) -> pd.Series:
    """
    Turn signal strength into ADV-capped participation rate.
    
    Args:
        alpha_strength: Signal strength series
        adv: Average Daily Volume series
        day_limit: Maximum participation as fraction of ADV
        
    Returns:
        Participation rate series
    """
    if alpha_strength.empty or adv.empty:
        return pd.Series(dtype=float)
    
    # Align indices
    common_assets = alpha_strength.index.intersection(adv.index)
    alpha_aligned = alpha_strength.reindex(common_assets, fill_value=0)
    adv_aligned = adv.reindex(common_assets, fill_value=adv.median())
    
    # Calculate participation rate
    # Higher alpha strength → higher participation, but capped by ADV
    participation = alpha_aligned.abs() * day_limit
    
    # Cap at ADV limit
    participation = participation.clip(upper=day_limit)
    
    return participation.rename('participation_rate')


def expected_cost_per_weight_change(w_to_trade: pd.Series, Lambda: np.ndarray) -> float:
    """
    Calculate expected cost per unit weight change for cost diagnostics.
    
    Args:
        w_to_trade: Weight changes to trade
        Lambda: Cost matrix
        
    Returns:
        Expected cost per unit change
    """
    if w_to_trade.empty or Lambda.size == 0:
        return 0.0
    
    # Calculate cost per unit
    cost = quadratic_cost_tau(w_to_trade, Lambda)
    total_change = w_to_trade.abs().sum()
    
    if total_change == 0:
        return 0.0
    
    return cost / total_change

