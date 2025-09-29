# Help text constants for the GUI

WORKFLOW_HELP_TEXT = """
📋 WORKFLOW OVERVIEW

1. FILE SELECTION
   • Trades CSV (Required): Your executed trades with timestamps and PnL
   • OHLCV CSV (Optional): Market data for technical indicators
   • Feature CSVs: External data sources (fear & greed, sentiment, etc.)

2. FEATURE LOADING
   • Click "Load Features" to scan all CSV files
   • System will detect available columns automatically
   • Chart indicators are computed from OHLCV if provided

3. FEATURE SELECTION
   • Select which features to analyze from CSV files
   • Select which technical indicators to include
   • Choose data transformations (z-score, rolling stats, etc.)
   • Set parameters for transformations

4. ANALYSIS SETTINGS
   • Number of bins: How many groups to divide feature values into
   • Binning method:
     - Equal width: Same range size for each bin
     - Equal count: Same number of trades per bin

5. EXECUTION
   • System merges trades with features by timestamp
   • Calculates performance metrics per feature bin
   • Generates comprehensive plots and CSV reports
   • Results saved in organized folders by feature

📊 DATA MERGING PROCESS

The system intelligently merges your trades with feature data:

• Timestamp Alignment: Finds the most recent feature value before each trade
• Forward-Fill Logic: Uses last known feature value if exact match not found
• Multiple Sources: Combines CSV features with computed chart indicators
• No-Leakage: Ensures no future data is used for past trades

📈 OUTPUT STRUCTURE

Results are organized as:
./data/trade_analysis/
├── feature_name_1/
│   ├── Trade_Performance_plots.png
│   ├── Risk_Metrics_plots.png
│   └── OHLCV_Overlay.png (if chart indicator)
├── feature_name_2/
│   └── ...
└── all_features/
    └── trades_features_merged.csv
"""

INDICATORS_HELP_TEXT = """
📊 AVAILABLE INDICATORS

When you provide OHLCV data, the following technical indicators are automatically computed:

🔍 TREND INDICATORS
• EMA (8, 21, 50, 200): Exponential Moving Averages
• SMA (10, 20, 50): Simple Moving Averages
• MACD: Moving Average Convergence Divergence
• MACD Signal & Histogram
• ADX: Average Directional Index (trend strength)
• Parabolic SAR: Stop and Reverse points

📊 MOMENTUM INDICATORS
• RSI: Relative Strength Index (14-period)
• Stochastic %K and %D: Stochastic oscillator
• Williams %R: Williams Percent Range
• CCI: Commodity Channel Index
• MFI: Money Flow Index
• Ultimate Oscillator: Multi-timeframe momentum

📈 VOLATILITY INDICATORS
• Bollinger Bands (Upper, Middle, Lower)
• Bollinger Band Width: Volatility measure
• ATR: Average True Range
• Keltner Channels: Volatility-based channels

💰 VOLUME INDICATORS
• OBV: On-Balance Volume
• VWAP: Volume Weighted Average Price
• Volume SMA: Volume moving average
• Accumulation/Distribution Line
• Chaikin Money Flow

🎯 SUPPORT/RESISTANCE
• Pivot Points: Classical pivot levels
• Fibonacci Retracement levels
• Key price levels based on recent highs/lows

⚡ CUSTOM INDICATORS
• Price Distance from EMAs: How far price is from key moving averages
• Volume Ratio: Current volume vs average
• Volatility Ratio: Current ATR vs historical average

🔄 INDICATOR COMBINATIONS
You can combine multiple indicators in your analysis:
• Use both trend and momentum indicators
• Combine volume with price-based indicators
• Mix different timeframe perspectives
• Apply transformations to normalize different scales

💡 SELECTION TIPS
• Start with 3-5 key indicators to avoid over-complexity
• Choose indicators from different categories (trend, momentum, volume)
• Consider the timeframe of your trades when selecting indicators
• Use transformations to normalize indicators for better comparison
"""

CONVERSIONS_HELP_TEXT = """
🔧 DATA TRANSFORMATIONS

Apply mathematical transformations to normalize and improve feature analysis:

📊 STANDARDIZATION
• zscore: Standard z-score normalization (mean=0, std=1)
• robust_zscore: Uses median and MAD instead of mean/std
• expanding_zscore: Uses only historical data (no future leakage)
• ewm_zscore: Exponentially weighted z-score

🔄 ROLLING STATISTICS
• rolling_zscore: Z-score using rolling window
  Parameters: window=20 (default), leak_safe=true
• rolling_robust_zscore: Robust z-score with rolling window
  Parameters: window=50, leak_safe=true
• rolling_minmax: Rolling min-max normalization
  Parameters: window=100, leak_safe=true

📏 SCALING
• minmax: Scale to 0-1 range using global min/max
• expanding_minmax: Scale using only historical min/max
• normalize: Center and scale by range

📈 CHANGE RATES
• pct_change: Percentage change from previous value
  Parameters: periods=1 (how many periods back)
• diff: Absolute difference from previous value
  Parameters: periods=1
• log_return: Logarithmic return
  Parameters: periods=1

🎯 PERCENTILES
• rank_pct: Convert to percentile ranks (0-1)
• winsorize_rolling: Clip extreme values to rolling percentiles
  Parameters: window=250, lower=0.01, upper=0.99

🔢 PARAMETER EXAMPLES

Rolling Z-Score with custom window:
Parameters: window=30,leak_safe=true

EWM Z-Score with custom span:
Parameters: span=20,leak_safe=true

Percentage change over 5 periods:
Parameters: periods=5

Winsorization removing top/bottom 5%:
Parameters: lower=0.05,upper=0.95,window=200

💡 WHEN TO USE WHICH TRANSFORMATION

• zscore/robust_zscore: When you want to normalize across entire dataset
• rolling_zscore: When you want recent normalization (adaptable to regime changes)
• expanding_zscore: Safest for backtesting (no future data leakage)
• minmax: When you need bounded 0-1 output
• pct_change: For momentum/change analysis
• rank_pct: When relative ranking matters more than absolute values

⚠️ LEAKAGE CONSIDERATIONS

For realistic backtesting, use leak_safe=true options:
• expanding_zscore: Only uses past data
• rolling_zscore with leak_safe=true
• ewm_zscore with leak_safe=true

These ensure no future information contaminates historical analysis.
"""

FILE_FORMATS_HELP_TEXT = """
📁 REQUIRED FILE FORMATS

🔹 TRADES CSV (Required)
Must contain these columns:
• timestamp: Trade execution time (YYYY-MM-DD HH:MM:SS or ISO format)
• realized_pnl: Profit/loss for each trade (numeric)

Optional but recommended:
• closed_timestamp: Trade exit time
• qty/size: Position size (for long/short detection)
• side: 'long' or 'short' (explicit side declaration)

Example:
timestamp,realized_pnl,qty,side
2024-01-15 10:30:00,150.50,1000,long
2024-01-15 11:45:00,-75.25,-500,short

🔹 OHLCV CSV (Optional - for chart indicators)
Must contain these columns:
• timestamp: Candle time
• open: Opening price
• high: Highest price
• low: Lowest price  
• close: Closing price
• volume: Trading volume

Example:
timestamp,open,high,low,close,volume
2024-01-15 10:00:00,45.50,46.20,45.30,46.10,125000
2024-01-15 11:00:00,46.10,46.50,45.80,46.25,98000

🔹 FEATURE CSV FILES (Optional but recommended)
Can contain any numeric features with timestamps:
• timestamp: Feature observation time
• feature1, feature2, ...: Any numeric features

Example - Fear & Greed Index:
timestamp,fear_greed,vix,sentiment_score
2024-01-15 00:00:00,25,18.5,0.65
2024-01-16 00:00:00,31,17.2,0.72

Example - Custom Indicators:
timestamp,rsi_daily,macd_4h,volume_ratio
2024-01-15 08:00:00,65.2,0.15,1.25
2024-01-15 12:00:00,58.7,0.08,0.95

📅 TIMESTAMP FORMATS SUPPORTED

The system automatically detects these formats:
• ISO Format: 2024-01-15T10:30:00
• Standard: 2024-01-15 10:30:00
• Date Only: 2024-01-15 (assumes 00:00:00)
• Unix Timestamp: 1705312200
• With Timezone: 2024-01-15 10:30:00+00:00

🔗 DATA MERGING LOGIC

1. All files are loaded and timestamps converted to datetime
2. Features are aligned to trade timestamps using forward-fill
3. The system finds the most recent feature value before each trade
4. Missing values are handled gracefully (forward-filled or excluded)
5. Chart indicators are computed from OHLCV and merged similarly

📊 OUTPUT FILES

The analysis generates:
• CSV summaries: Performance metrics per feature bin
• PNG plots: Visual analysis for each feature
• trades_features_merged.csv: Complete dataset with all features aligned

💡 TIPS FOR BEST RESULTS

• Ensure timestamps are in chronological order
• Use consistent timezone across all files
• Higher frequency data (1min, 5min) gives better trade alignment
• Include multiple data sources for richer analysis
• Clean your data beforehand (remove outliers, handle missing values)
"""

ADVANCED_HELP_TEXT = """
⚙️ ADVANCED FEATURES & SETTINGS

🎯 BINNING STRATEGIES

Equal Width (Size):
• Divides feature range into equal-sized intervals
• Good for normally distributed features
• May have uneven trade counts per bin
• Example: RSI 0-20, 20-40, 40-60, 60-80, 80-100

Equal Count (Amount):
• Same number of trades in each bin
• Better for skewed distributions
• Adaptive bin boundaries
• Ensures statistical significance in each bin

📊 PERFORMANCE METRICS CALCULATED

For each feature bin, the system computes:

Basic Metrics:
• Mean PnL, Total PnL, Trade Count
• Win Rate, Profit Factor
• Gross Profit, Gross Loss

Risk Metrics:
• Median PnL, PnL Standard Deviation
• Value at Risk (VaR 5%), Conditional VaR
• Sharpe Ratio, Sortino Ratio
• Maximum Drawdown

Advanced Metrics:
• Expectancy, Payoff Ratio
• Average Win, Average Loss
• Max Win/Loss Streaks
• Average Trade Duration

Long/Short Breakdown:
• Separate metrics for long and short trades
• Win rates by direction
• PnL distribution by side

🔄 CORRELATION & LAG ANALYSIS

Cross-Correlation:
• Analyzes lead/lag relationships
• Finds optimal timing offsets
• Identifies predictive patterns

Rolling Correlations:
• Time-varying relationship strength
• Regime change detection
• Correlation stability analysis

🎨 VISUALIZATION OUTPUTS

Per Feature Analysis:
• Performance vs Feature Value (bar charts)
• Risk Metrics Comparison
• Expectancy & Payoff Analysis
• Streak & Drawdown Analysis
• Duration Analysis
• Long/Short Performance Split

Chart Indicator Overlays:
• OHLCV candlestick charts
• Indicator overlay plots
• Visual pattern recognition
• Entry/exit point analysis

📈 MULTI-FEATURE ANALYSIS

Feature Comparison:
• Cross-feature performance matrix
• Feature importance ranking
• Interaction effects
• Portfolio-level insights

Correlation Matrix:
• Feature inter-dependencies
• Redundancy detection
• Diversification opportunities

🔧 TECHNICAL CONSIDERATIONS

Memory Management:
• Large datasets are processed in chunks
• Efficient timestamp alignment algorithms
• Garbage collection for memory optimization

Performance Optimization:
• Vectorized calculations using pandas/numpy
• Parallel processing for independent features
• Caching of computed indicators

Error Handling:
• Graceful handling of missing data
• Timestamp format auto-detection
• Robust numeric conversions
• Detailed error logging

🚀 BEST PRACTICES

Data Quality:
• Clean timestamps and remove duplicates
• Handle missing values appropriately
• Validate data ranges and outliers
• Ensure sufficient data history

Feature Engineering:
• Start with raw features, then apply transformations
• Use leak-safe transformations for backtesting
• Combine multiple timeframes
• Consider regime-dependent features

Analysis Workflow:
• Begin with fewer features to understand patterns
• Gradually add complexity
• Validate results with out-of-sample data
• Document your findings and assumptions

Interpretation:
• Consider statistical significance (trade counts)
• Look for consistent patterns across bins
• Verify results make economic sense
• Be aware of survivorship bias and data mining

💡 TROUBLESHOOTING

Common Issues:
• Timestamp misalignment: Check timezone consistency
• No features loaded: Verify CSV column names
• Empty bins: Reduce number of bins or check data range
• Memory errors: Process smaller date ranges or fewer features
• Slow performance: Reduce data frequency or feature count

Performance Tips:
• Use 1-hour or daily data for initial exploration
• Limit to 10-15 features for comprehensive analysis
• Pre-filter trades to relevant time periods
• Use SSD storage for large datasets
"""

# Conversion options for feature transformations
CONVERSION_OPTIONS = [
    "None", "zscore", "robust_zscore", "rolling_zscore",
    "expanding_zscore", "ewm_zscore", "rolling_robust_zscore",
    "minmax", "rolling_minmax", "expanding_minmax",
    "pct_change", "diff", "log_return"
]
