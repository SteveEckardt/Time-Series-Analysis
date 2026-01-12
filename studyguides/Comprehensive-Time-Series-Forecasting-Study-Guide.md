# COMPREHENSIVE TIME SERIES FORECASTING STUDY GUIDE

---

## TABLE OF CONTENTS

1. [Your Project Summary](#1-your-project-summary)
2. [Core Python/Pandas Keywords](#2-core-pythonpandas-keywords)
3. [Time Series Concepts](#3-time-series-concepts)
4. [SARIMA Model Explained](#4-sarima-model-explained)
5. [Workshop Techniques Reference](#5-workshop-techniques-reference)
6. [Common Questions & Answers](#6-common-questions--answers)
7. [Sample Code Examples](#7-sample-code-examples)
8. [Quick Reference Card](#8-quick-reference-card)
9. [Final Checklist](#9-final-checklist)

---

## 1. YOUR PROJECT SUMMARY

### Dataset:
- **Location:** Boujdour, Morocco (3 zones combined)
- **Period:** Sept 14, 2022 - May 24, 2024 (21 months)
- **Original:** 88,890 measurements (30-min intervals)
- **Analysis:** Monthly totals

### Key Findings:
- **Monthly Seasonality (STRONG):** Oct peak (~775k), July low (~515k) - 50% swing
- **Weekly Pattern (WEAK):** Weekdays 147, Weekends 142 - only 4% difference
- **Daily Pattern (STRONG):** 9 PM peak (185), 8 AM low (114) - 62% swing

### Model: SARIMA(1,0,1)(1,0,1,12)
- Training: 18 months
- Testing: 3 months (MAE ~10% on complete months)
- Forecast: 12 months (June 2024 - May 2025)

### Forecast Results:
- **Peak:** Sept 2024 (690,011 units)
- **Low:** May 2025 (549,368 units)
- **Average:** 615,145 units/month
- **Annual:** ~7.4 million units

### Business Impact:
- Plan for 20% higher demand Sept-Oct vs summer
- Schedule maintenance July-Aug (low demand)
- Ensure infrastructure handles ~690k peak

---

## 2. CORE PYTHON/PANDAS KEYWORDS

### DATA LOADING & INSPECTION

```python
pd.read_csv('file.csv')        # Load CSV into DataFrame
df.head(n)                     # First n rows (default 5)
df.tail(n)                     # Last n rows
df.shape                       # (rows, columns)
df.columns                     # Column names
df.info()                      # Data types, null counts
df.describe()                  # Statistics summary
```

### DATA MANIPULATION

```python
df['new_col'] = df['A'] + df['B']  # Create column
df.set_index('column')             # Make column the index
df.reset_index()                   # Index back to column
df.dropna()                        # Remove missing values
df.fillna(value)                   # Fill missing values
df['col'].shift(n)                 # Move data by n periods
```

### DATETIME OPERATIONS

```python
pd.to_datetime(df['column'])       # Convert to datetime
df['DateTime'].dt.month            # Extract month (1-12)
df['DateTime'].dt.dayofweek        # Day of week (0=Mon)
df['DateTime'].dt.hour             # Hour (0-23)
df['DateTime'].dt.year             # Year
df['DateTime'].dt.to_period('M')   # Period format ('2023-01')
pd.date_range(start, periods, freq) # Create date sequence
```

### GROUPBY & AGGREGATION

```python
df.groupby('column')['value'].mean()  # Group and average
df.groupby('col')['val'].sum()        # Group and sum
df.groupby('col')['val'].agg(['mean','sum','std'])  # Multiple
```

### TIME SERIES RESAMPLING

```python
# Must have datetime index first
df.resample('D').sum()     # Daily frequency, sum values
df.resample('W').mean()    # Weekly, average
df.resample('ME').sum()    # Month End, sum
df.resample('MS').sum()    # Month Start, sum
# Returns Series - use .to_frame() to convert to DataFrame
```

### USEFUL CALCULATIONS

```python
df['col'].mean()           # Average
df['col'].sum()            # Total
df['col'].min() / .max()   # Min/Max
df['col'].std()            # Standard deviation
df['col'].median()         # Middle value (robust to outliers)
df['col'].quantile(0.25)   # 25th percentile (Q1)
df['col'].value_counts()   # Frequency of each value
df['col'].rolling(window=7).mean()  # 7-period moving average
```

### FILTERING DATA

```python
df[df['col'] > 100]                    # Single condition
df[(df['col'] > 100) & (df['col'] < 200)]  # AND condition
df[(df['col'] < 50) | (df['col'] > 150)]   # OR condition
df[df['col'].isin(['A', 'B', 'C'])]   # Match list
df.loc[df['col'] > 100, 'other_col']  # Filter rows, select column
```

---

## 3. TIME SERIES CONCEPTS

### Stationarity
**Definition:** Statistical properties (mean, variance) don't change over time

**Why it matters:** ARIMA/SARIMA models assume stationarity

**How to check:**
- Visual: Plot looks "flat" without trend
- Statistical: ADF test p-value < 0.05

**Example:**
- Stationary: Daily temperature in a stable climate
- Non-stationary: Company revenue growing year-over-year

**Fix:** Differencing (subtract previous value)

### Seasonality
**Definition:** Regular, repeating pattern at fixed intervals

**Examples:**
- Ice cream sales peak every summer
- Electricity use spikes every evening
- Retail sales increase every December

**Detection:**
- Visual: Repeating peaks/valleys in plot
- ACF plot: Spikes at seasonal lags (12, 24, 36 for monthly)

### Trend
**Definition:** Long-term direction (up or down)

**Examples:**
- Population growth
- Technology adoption curve
- Climate warming

**Removal:** Differencing or detrending

### Autocorrelation (ACF)
**Definition:** Correlation between time series and its lagged version

**Interpretation:**
- High ACF at lag 1: Tomorrow looks like today
- High ACF at lag 12: This month looks like same month last year

**Use:** Identify seasonality and MA parameters

### Partial Autocorrelation (PACF)
**Definition:** Correlation between time series and lag, removing intermediate lags

**Use:** Determine AR parameters for ARIMA

### Differencing
**Definition:** Subtract previous value from current

**Formula:** value(t) - value(t-1)

**Purpose:** Remove trend, achieve stationarity

**Types:**
- First order: value(t) - value(t-1)
- Second order: Apply differencing twice
- Seasonal: value(t) - value(t-12) for monthly data

### Residuals
**Definition:** Errors = Actual - Predicted

**Good residuals:**
- Mean close to 0
- Normally distributed
- No patterns (white noise)
- No autocorrelation

**Bad residuals:**
- Systematic patterns remaining
- ACF spikes
- Trending

---

## 4. SARIMA MODEL EXPLAINED

### Full Name
**Seasonal AutoRegressive Integrated Moving Average**

### When to Use
- Data with seasonal patterns (monthly, quarterly, yearly cycles)
- Regular time intervals
- Enough data (1.5-2 full seasonal cycles minimum)

### Parameters: SARIMA(p,d,q)(P,D,Q,s)

#### Non-Seasonal Components (p,d,q)
- **p (AR order):** How many past values to use
  - p=1: Uses yesterday's value
  - p=2: Uses last 2 days
- **d (Differencing):** How many times to difference
  - d=0: Already stationary
  - d=1: One differencing needed
- **q (MA order):** How many past forecast errors to use
  - q=1: Corrects based on last error
  - q=2: Uses last 2 errors

#### Seasonal Components (P,D,Q,s)
- **P (Seasonal AR):** Past seasonal values
  - P=1: Uses value from 1 year ago
- **D (Seasonal Diff):** Seasonal differencing
  - D=0: Seasonal pattern stable
  - D=1: Remove seasonal trend
- **Q (Seasonal MA):** Past seasonal errors
  - Q=1: Corrects based on error from 1 year ago
- **s (Season Length):** Period of seasonality
  - s=12 for monthly data with yearly cycle
  - s=7 for daily data with weekly cycle
  - s=4 for quarterly data

### Your Model: SARIMA(1,0,1)(1,0,1,12)

**Non-seasonal (1,0,1):**
- AR(1): Uses last month's value
- d=0: Data already stationary
- MA(1): Corrects based on last month's error

**Seasonal (1,0,1,12):**
- P=1: Uses value from 12 months ago
- D=0: Seasonal pattern stable
- Q=1: Corrects based on error from 12 months ago
- s=12: Yearly seasonal cycle

**Why these parameters?**
- Data is stationary (ADF p-value = 0.0003) → d=0, D=0
- Limited data (21 months) → Keep simple (1,1) to avoid overfitting
- Clear 12-month cycle → s=12

### Model Selection Process

1. **Check Stationarity**
   - Run ADF test
   - If p-value ≥ 0.05, increase d

2. **Identify Seasonality**
   - Plot data, look for repeating pattern
   - Check ACF for spikes at seasonal lags
   - Set s = seasonal period

3. **Start Simple**
   - Try (1,d,1)(1,D,1,s) first
   - Only increase if residuals show patterns

4. **Evaluate**
   - Check AIC/BIC (lower is better)
   - Examine residuals (should be white noise)
   - Validate on test set

---

## 5. WORKSHOP TECHNIQUES REFERENCE

### Data Cleaning

#### Outlier Detection (IQR Method)
```python
Q1 = df['col'].quantile(0.25)
Q3 = df['col'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = (df['col'] < lower_bound) | (df['col'] > upper_bound)
print(f"Found {outliers.sum()} outliers")

# Replace with rolling median
df.loc[outliers, 'col'] = df['col'].rolling(window=24, center=True).median()
```

### Exploratory Data Analysis

#### Monthly Pattern Analysis
```python
df['month'] = df['DateTime'].dt.month
monthly_pattern = df.groupby('month')['value'].mean()
monthly_pattern.plot(kind='bar')
plt.title('Average Value by Month')
plt.xlabel('Month')
plt.ylabel('Average Value')
plt.show()
```

#### Day of Week Pattern
```python
df['day_of_week'] = df['DateTime'].dt.dayofweek
day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
daily_pattern = df.groupby('day_of_week')['value'].mean()
daily_pattern.index = day_names
daily_pattern.plot(kind='bar')
plt.show()
```

#### Hourly Pattern
```python
df['hour'] = df['DateTime'].dt.hour
hourly_pattern = df.groupby('hour')['value'].mean()
hourly_pattern.plot()
plt.title('Average Value by Hour of Day')
plt.xlabel('Hour (0-23)')
plt.ylabel('Average Value')
plt.show()
```

### Stationarity Testing

```python
from statsmodels.tsa.stattools import adfuller

result = adfuller(df['value'].dropna())
print(f"ADF Statistic: {result[0]:.6f}")
print(f"p-value: {result[1]:.6f}")
print(f"Critical Values:")
for key, value in result[4].items():
    print(f"  {key}: {value:.3f}")

if result[1] < 0.05:
    print("✓ Data is STATIONARY")
else:
    print("✗ Data is NOT STATIONARY - differencing needed")
```

### Model Building

#### Train/Test Split
```python
train_size = len(monthly_df) - 3
train_data = monthly_df[:train_size]
test_data = monthly_df[train_size:]

print(f"Training: {len(train_data)} months")
print(f"Testing: {len(test_data)} months")
```

#### Fit SARIMA Model
```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(train_data['total'],
                order=(1, 0, 1),
                seasonal_order=(1, 0, 1, 12))

fitted_model = model.fit(disp=False, maxiter=200)
print("✓ Model fitted successfully")
print(fitted_model.summary())
```

#### Generate Predictions
```python
predictions = fitted_model.forecast(steps=3)
```

### Model Validation

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

mae = mean_absolute_error(test_data['total'], predictions)
rmse = np.sqrt(mean_squared_error(test_data['total'], predictions))
mape = np.mean(np.abs((test_data['total'] - predictions) / test_data['total'])) * 100

print(f"MAE: {mae:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"MAPE: {mape:.2f}%")
```

### Visualization

```python
plt.figure(figsize=(14, 7))
plt.plot(train_data.index, train_data['total'],
         label='Training Data', color='blue', marker='o', linewidth=2)
plt.plot(test_data.index, test_data['total'],
         label='Actual (Test)', color='green', marker='o', linewidth=2)
plt.plot(test_data.index, predictions,
         label='Predictions', color='red', marker='s',
         linestyle='--', linewidth=2)
plt.axvline(x=train_data.index[-1], color='gray',
            linestyle=':', linewidth=2, label='Train/Test Split')
plt.title('Model Validation: Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Monthly Usage')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 6. COMMON QUESTIONS & ANSWERS

### Q: "Why monthly instead of daily forecasts?"

**Answer:** With only 21 months of data, we need at least 1.5-2 full seasonal cycles to reliably model the 12-month seasonality. Monthly aggregation gives us clean seasonal patterns without overfitting to daily noise.

### Q: "How accurate is your forecast?"

**Answer:** The model achieved ~10% MAE on complete test months. March and April predictions were within 1-3% of actual values. May showed higher error (28%) because the dataset ends May 24th - an incomplete month.

### Q: "Can you forecast 2-3 years out?"

**Answer:** Yes, but forecast uncertainty increases with distance. I'm comfortable with 12-18 month forecasts given our data. Beyond that, the model assumes historical patterns continue, which becomes less reliable.

### Q: "What if there's extreme weather or policy changes?"

**Answer:** Current SARIMA model only knows historical patterns. For external factors, we'd need SARIMAX (adds external variables like temperature, economic indicators). That would require additional data collection.

### Q: "Why is the data stationary with obvious seasonality?"

**Answer:** Stationarity means the seasonal pattern is stable and repeating consistently. The mean and variance of the seasonal cycle don't change over time. The pattern repeats reliably, making it stationary.

### Q: "How would you improve this model?"

**Three ways:**
1. **More data:** Additional years would improve parameter estimation
2. **External variables:** Add temperature, economic data, holidays (SARIMAX)
3. **Model comparison:** Test Prophet, LSTM, or ensemble methods
4. **Higher frequency:** Daily forecasts for operational planning

### Q: "What are the biggest limitations?"

1. **Limited historical data:** Only 1.75 seasonal cycles
2. **No external factors:** Weather, economy, policy not included
3. **Assumes continuity:** Pattern must remain stable
4. **Incomplete months:** Sept 2022 and May 2024 are partial

### Q: "How do you know your model is good?"

**Four checks:**
1. **Validation metrics:** 10% MAE is good (target <15%)
2. **Residuals:** Should look like random noise, no patterns
3. **Business sense:** Predictions align with known patterns (summer low, fall peak)
4. **AIC/BIC:** Compare alternative models, choose lower values

### Q: "Why these specific SARIMA parameters?"

**Data-driven choices:**
- **d=0, D=0:** ADF test showed stationarity (p=0.0003)
- **s=12:** Clear yearly seasonal cycle in data
- **Simple (1,1) terms:** Limited data (21 months) means simple model to avoid overfitting
- **Validated:** Residuals showed no remaining patterns

---

## 7. SAMPLE CODE EXAMPLES

### Complete End-to-End Example

```python
# === IMPORTS ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

# === LOAD DATA ===
df = pd.read_csv('Boujdour.csv')
df['DateTime'] = pd.to_datetime(df['DateTime'])
df['total'] = df['zone1'] + df['zone2'] + df['zone3']

print(f"Loaded {len(df):,} records")
print(f"Date range: {df['DateTime'].min()} to {df['DateTime'].max()}")

# === CLEAN OUTLIERS (IQR Method) ===
Q1 = df['total'].quantile(0.25)
Q3 = df['total'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = (df['total'] < lower) | (df['total'] > upper)
print(f"Found {outliers.sum()} outliers ({outliers.sum()/len(df)*100:.2f}%)")

df.loc[outliers, 'total'] = df['total'].rolling(
    window=48, center=True
).median()[outliers]

# === AGGREGATE TO MONTHLY ===
df_indexed = df.set_index('DateTime')
monthly_df = df_indexed['total'].resample('ME').sum().to_frame()
print(f"Aggregated to {len(monthly_df)} months")

# === STATIONARITY TEST ===
result = adfuller(monthly_df['total'])
print(f"\nADF Test Results:")
print(f"p-value: {result[1]:.6f}")
if result[1] < 0.05:
    print("✓ Data is stationary")
else:
    print("✗ Data needs differencing")

# === TRAIN/TEST SPLIT ===
train_size = len(monthly_df) - 3
train = monthly_df[:train_size]
test = monthly_df[train_size:]

print(f"\nTraining: {len(train)} months")
print(f"Testing: {len(test)} months")

# === BUILD MODEL ===
print("\nFitting SARIMA(1,0,1)(1,0,1,12)...")
model = SARIMAX(train['total'],
                order=(1, 0, 1),
                seasonal_order=(1, 0, 1, 12))
fitted = model.fit(disp=False, maxiter=200)
print("✓ Model fitted")

# === VALIDATE ON TEST SET ===
predictions = fitted.forecast(steps=len(test))

mae = mean_absolute_error(test['total'], predictions)
rmse = np.sqrt(mean_squared_error(test['total'], predictions))
mape = np.mean(np.abs((test['total'] - predictions) / test['total'])) * 100

print(f"\nValidation Metrics:")
print(f"MAE: {mae:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"MAPE: {mape:.2f}%")

# === RETRAIN ON FULL DATA FOR FORECAST ===
print("\nRetraining on complete dataset...")
full_model = SARIMAX(monthly_df['total'],
                     order=(1, 0, 1),
                     seasonal_order=(1, 0, 1, 12))

full_fitted = full_model.fit(disp=False, maxiter=200)
future_forecast = full_fitted.forecast(steps=12)

# Create future dates
future_dates = pd.date_range(
    start=monthly_df.index[-1] + pd.DateOffset(months=1),
    periods=12,
    freq='ME'
)

# Display forecast
forecast_df = pd.DataFrame({
    'Month': future_dates,
    'Forecasted_Usage': future_forecast.values
})

print("\n12-MONTH FORECAST:")
print("=" * 60)
for date, value in zip(future_dates, future_forecast):
    print(f"{date.strftime('%B %Y'):20s}: {value:>12,.2f} units")
print("=" * 60)
print(f"Average: {future_forecast.mean():,.2f} units/month")
print(f"Peak: {future_dates[future_forecast.argmax()].strftime('%B %Y')} "
      f"({future_forecast.max():,.2f})")
print(f"Low: {future_dates[future_forecast.argmin()].strftime('%B %Y')} "
      f"({future_forecast.min():,.2f})")

# === FINAL VISUALIZATION ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

# Full view
ax1.plot(monthly_df.index, monthly_df['total'],
         label='Historical', color='blue', marker='o', linewidth=2)
ax1.plot(future_dates, future_forecast,
         label='12-Month Forecast', color='red', marker='s',
         linestyle='--', linewidth=2)
ax1.axvline(x=monthly_df.index[-1], color='green',
            linestyle=':', linewidth=2, label='Forecast Start')
ax1.set_title('Complete Time Series + 12-Month Forecast', fontsize=14)
ax1.set_ylabel('Monthly Usage')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Zoomed view (last 6 months + forecast)
last_6 = monthly_df.tail(6)
ax2.plot(last_6.index, last_6['total'],
         label='Recent Historical', color='blue', marker='o', linewidth=2)
ax2.plot(future_dates, future_forecast,
         label='12-Month Forecast', color='red', marker='s',
         linestyle='--', linewidth=2)
ax2.axvline(x=monthly_df.index[-1], color='green',
            linestyle=':', linewidth=2)
ax2.set_title('Detailed View: Last 6 Months + Forecast', fontsize=14)
ax2.set_ylabel('Monthly Usage')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nAnalysis complete!")
```

---

## 8. QUICK REFERENCE CARD

### STATIONARY vs NON-STATIONARY
- **Stationary:** Mean/variance constant → Ready for ARIMA
- **Non-stationary:** Trending up/down → Need differencing
- **Test:** ADF p-value < 0.05 = Stationary ✓

### SARIMA PARAMETERS
- **(p,d,q):** Non-seasonal (AR, Diff, MA)
- **(P,D,Q,s):** Seasonal (AR, Diff, MA, Period)
- **Your model:** (1,0,1)(1,0,1,12)

### EVALUATION METRICS
- **MAE:** Average error (lower better)
- **RMSE:** Penalizes big errors (lower better)
- **MAPE:** Error as % (lower better)
- **AIC/BIC:** Model quality (lower better)

### RESIDUALS SHOULD BE:
- Centered at 0
- Normally distributed
- No ACF spikes (white noise)

### WHEN TO USE WHAT
- **SARIMA:** Pure seasonal forecasting
- **SARIMAX:** Add external variables (weather, etc.)
- **ML (XGBoost/RF):** Many features, complex patterns
- **Change Point:** Detect regime shifts

### YOUR DATA INSIGHTS
- Oct peak (775k), Jul low (515k)
- Weekdays > Weekends (weak)
- 9 PM peak, 8 AM low (strong)
- Stationary (p=0.0003)
- 12-month seasonal cycle

---

## 9. FINAL CHECKLIST

### Before Meeting:
- ✓ Understand your data (patterns, seasonality)
- ✓ Can explain SARIMA parameters
- ✓ Know model performance (~10% error)
- ✓ Understand limitations (limited data, no external variables)
- ✓ Have business implications ready
- ✓ Questions prepared about deliverables

### Can You Answer:
- ✓ **Why monthly vs daily?** (Data amount, seasonal cycle)
- ✓ **Why stationary with seasonality?** (Stable repeating pattern)
- ✓ **Why these SARIMA parameters?** (Simple, limited data)
- ✓ **Why May prediction off?** (Incomplete month)
- ✓ **How to improve?** (More data, add variables, retrain)

### Key Message:
"I built a SARIMA forecast capturing Boujdour's seasonal electricity pattern, validated at ~10% error, forecasting 12 months ahead with clear business implications for capacity planning and maintenance scheduling."

---

## Author

Steve Eckardt  
