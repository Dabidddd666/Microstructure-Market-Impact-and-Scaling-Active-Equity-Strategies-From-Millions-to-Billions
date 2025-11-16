# DST Squad Q4 Quantitative Trading System

A comprehensive quantitative trading framework implementing advanced signal generation, backtesting, and portfolio optimization techniques for scalable trading strategies.

## Overview

This system provides a complete pipeline for quantitative trading strategy development, from signal generation to performance evaluation. It features institutional-grade backtesting capabilities with realistic transaction cost modeling, risk neutralization, and capacity-aware portfolio construction.

## Architecture

### Core Components

- **Signal Generation**: Multi-factor alpha signals with investable implementations
- **Risk Neutralization**: Factor-based risk model neutralization 
- **Backtesting Engine**: Realistic simulation with market impact modeling
- **Portfolio Construction**: Capacity-aware position sizing and execution
- **Performance Analytics**: Comprehensive strategy evaluation tools

### Key Features

- **No Look-Ahead Bias**: All signals designed for live implementation
- **Market Impact Modeling**: Realistic transaction cost estimation
- **Factor Attribution**: Performance decomposition and risk analysis
- **Capacity Analysis**: AUM scaling and liquidity constraints
- **Robust Implementation**: Production-ready code with error handling

## Project Structure

```
├── signal_generation_bn.ipynb    # Billion-level signal generation
├── signal_generation_mm.ipynb    # Market microstructure signals  
├── backtesting_helper.py         # Core backtesting framework
├── neutralizing.py               # Risk neutralization engine
├── strategy_scaling.py           # Factor construction & scaling
├── utils.py                      # Shared utility functions
├── sp500_parser.py              # Market data utilities
├── backtesting_result.ipynb     # Performance analysis
└── databento.ipynb              # Data acquisition
```

## Signal Generation

### Alpha Factors

**Technical Signals:**
- Mean reversion z-scores (20/60/120 day windows)
- EMA momentum indicators with volatility normalization
- Donchian channel breakout signals with volume confirmation
- Volatility percentile rankings and regime detection

**Microstructure Signals:**
- Corwin-Schultz bid-ask spread estimation
- Volume surprise and turnover jump indicators
- Amihud illiquidity and price impact measures
- Intraday reversal and gap analysis

**OHLCV Factors:**
- Overnight gap and intraday reversal signals
- Range-to-ATR ratio analysis
- Close-to-low ratio and trend indicators
- Volume-weighted price action metrics

### Signal Processing

- **EWMA Z-Scoring**: Time-series standardization without look-ahead bias
- **Cross-Sectional Winsorization**: Robust outlier handling
- **Factor Combination**: ICIR-weighted composite construction
- **Investability Filtering**: Liquidity and universe constraints

## Backtesting Framework

### Execution Modeling

**Transaction Costs:**
- Commission and spread costs
- Market impact modeling (square-root participation)
- Implementation shortfall estimation
- Capacity-aware position sizing

**Portfolio Management:**
- Dollar-neutral long/short construction
- Dynamic rebalancing with turnover constraints
- Risk factor exposure monitoring
- Performance attribution analysis

### Performance Metrics

- **Risk-Adjusted Returns**: Sharpe ratio, Information ratio
- **Drawdown Analysis**: Maximum drawdown and recovery periods
- **Turnover Analysis**: Single-side and portfolio turnover
- **Capacity Metrics**: AUM scaling and participation rates

## Risk Neutralization

### Risk Factors

**Style Factors:**
- Market beta and correlation measures
- Size, momentum, and volatility exposures
- Quality and profitability factors
- Liquidity and microstructure factors

**Industry Neutralization:**
- GICS sector and sub-industry dummies
- Dynamic sector exposure monitoring
- Industry rotation signal generation

### Neutralization Methods

- **OLS Regression**: Cross-sectional factor neutralization
- **WLS Weighting**: Volatility-adjusted regression weights
- **Rolling Windows**: Dynamic factor estimation
- **Robust Implementation**: Minimum sample size requirements

## Technical Specifications

### Data Requirements

- **Price Data**: OHLCV with daily frequency
- **Volume Data**: Dollar volume and ADV calculations
- **Universe Data**: Stock listings and sector classifications
- **Market Data**: Benchmark returns and risk-free rates

### Performance Benchmarks

- **Capacity**: Designed for institutional AUM ($1B+)
- **Frequency**: Daily rebalancing with weekly optimization
- **Universe**: Liquid large-cap equity universe
- **Constraints**: Long/short with gross exposure limits

## Dependencies

- Python 3.8+
- pandas, numpy, scipy
- scikit-learn, matplotlib
- databento (market data)
- BeautifulSoup (web scraping)

## Implementation Notes

This system implements institutional-grade quantitative trading methodologies with particular attention to:

- **Realistic Implementation**: All signals avoid look-ahead bias
- **Transaction Costs**: Comprehensive cost modeling for live trading
- **Risk Management**: Factor-based risk controls and monitoring
- **Scalability**: Designed for billion-dollar AUM deployment

The framework provides both simple backtesting for research and capacity-aware simulation for production deployment.

---

*Developed for Alphathon only.*
