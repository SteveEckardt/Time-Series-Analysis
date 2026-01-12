# GLOSSARY: Technical Terms & Python Methods

Quick reference guide for time series analysis terminology and Python methods used in electricity forecasting projects.

---

## Statistical & Time Series Concepts

### ARIMA (AutoRegressive Integrated Moving Average)
- Statistical model for time series forecasting
- AR = uses past values, I = differencing, MA = uses past errors

### SARIMA (Seasonal ARIMA)
- ARIMA extended with seasonal components
- Handles data with repeating patterns (monthly, yearly cycles)

### Stationarity
- Property where mean and variance remain constant over time
- Required for ARIMA/SARIMA models
- Non-stationary data has trends or changing variance

### ADF Test (Augmented Dickey-Fuller Test)
- Statistical test to check if data is stationary
- p-value < 0.05 → stationary
- p-value ≥ 0.05 → non-stationary (needs differencing)

### Differencing
- Subtracting previous value from current value
- Removes trend to achieve stationarity
- First differencing: value(t) - value(t-1)

### Seasonality
- Regular, predictable pattern that repeats at fixed intervals
- Example: Higher electricity use every August

### Trend
- Long-term increase or decrease in data over time
- Example: Demand growing year-over-year

### Outlier
- Data point significantly different from others
- Often indicates measurement error or unusual event

### IQR (Interquartile Range)
- Middle 50% of data (between 25th and 75th percentiles)
- Used to detect outliers: values beyond Q1-1.5×IQR or Q3+1.5×IQR

### Residuals
- Difference between actual and predicted values
- Used to evaluate model accuracy

### Rolling Window
- Moving calculation over fixed number of periods
- Example: 24-hour rolling average

---

## Model Parameters

### SARIMA(p,d,q)(P,D,Q,s)

**Non-seasonal parameters:**
- **p** = Autoregressive order (how many past values to use)
- **d** = Differencing order (how many times to difference)
- **q** = Moving average order (how many past errors to use)

**Seasonal parameters:**
- **P** = Seasonal autoregressive order
- **D** = Seasonal differencing order
- **Q** = Seasonal moving average order
- **s** = Season length (12 for monthly data with yearly pattern)

---

## Performance Metrics

### MAE (Mean Absolute Error)
- Average of absolute differences between actual and predicted
- In same units as data
- Formula: mean(|actual - predicted|)

### RMSE (Root Mean Squared Error)
- Square root of average squared errors
- Penalizes large errors more than MAE
- Formula: sqrt(mean((actual - predicted)²))

### MAPE (Mean Absolute Percentage Error)
- Average absolute error as percentage
- Easy to interpret: 5% MAPE = 5% average error
- Formula: mean(|actual - predicted| / actual) × 100

### AIC (Akaike Information Criterion)
- Measure of model quality balancing fit and complexity
- Lower is better
- Used to compare different models

### BIC (Bayesian Information Criterion)
- Similar to AIC but penalizes complexity more heavily
- Lower is better

### R² (R-squared / Coefficient of Determination)
- Proportion of variance explained by model
- Range: 0 to 1 (1 = perfect fit)

---

## Data Concepts

### Time Series
- Data points indexed in time order
- Example: hourly electricity demand

### Aggregation
- Combining data into larger time periods
- Example: hourly → daily → monthly

### Granularity
- Level of detail in data
- Higher granularity = more frequent measurements (hourly vs monthly)

### Lag
- Shifted version of time series
- Lag-1 = previous value, Lag-12 = value from 12 periods ago

### Train/Test Split
- Dividing data into training set (build model) and test set (evaluate)
- Common: 70-80% train, 20-30% test

### Forecast Horizon
- How far into the future predictions extend
- Example: 12-month forecast horizon

### Overfitting
- Model too complex, fits training data perfectly but fails on new data
- Prevented by using simpler models, more data, validation

---

## Python Methods: pandas

### `pd.read_csv('file.csv')`
- Load CSV file into DataFrame

### `df.head(n)` / `df.tail(n)`
- View first/last n rows

### `df.describe()`
- Summary statistics (mean, std, min, max, quartiles)

### `df.info()`
- Data types and non-null counts

### `df.shape`
- Dimensions (rows, columns)

### `df['column']`
- Select single column (returns Series)

### `df[['col1', 'col2']]`
- Select multiple columns (returns DataFrame)

### `df.loc[rows, columns]`
- Select by labels

### `df.iloc[rows, columns]`
- Select by integer position

### `df.dropna()`
- Remove rows with missing values

### `df.fillna(value)`
- Fill missing values with specified value

### `df.groupby('column').agg(function)`
- Group data and apply aggregation
- Example: `df.groupby('month').mean()`

### `df.sort_values('column')`
- Sort by column values

### `df.reset_index(drop=True)`
- Reset row indices

### `df.set_index('column')`
- Set column as index

### `pd.to_datetime(df['column'])`
- Convert to datetime format

### `df['date'].dt.year` / `.dt.month` / `.dt.day`
- Extract date components

### `df['date'].dt.dayofweek`
- Get day of week (0=Monday, 6=Sunday)

### `df.resample('ME').sum()`
- Resample time series to different frequency
- 'ME' = month end, 'D' = daily, 'W' = weekly

### `df.rolling(window=n).mean()`
- Calculate rolling average over n periods

---

## Python Methods: numpy

### `np.array([1, 2, 3])`
- Create numpy array

### `np.mean(data)` / `np.median(data)`
- Calculate mean/median

### `np.std(data)`
- Standard deviation

### `np.min(data)` / `np.max(data)`
- Minimum/maximum values

### `np.sum(data)`
- Sum of all elements

### `np.sqrt(data)`
- Square root

### `np.abs(data)`
- Absolute value

### `np.percentile(data, q)`
- Calculate percentile (q=25 for Q1, q=75 for Q3)

### `np.arange(start, stop, step)`
- Create range of values

### `np.linspace(start, stop, num)`
- Create evenly spaced values

---

## Python Methods: matplotlib

### `plt.figure(figsize=(width, height))`
- Create new figure with dimensions

### `plt.subplots(rows, cols)`
- Create figure with multiple subplots

### `plt.plot(x, y)`
- Line plot

### `plt.scatter(x, y)`
- Scatter plot

### `plt.bar(x, y)` / `plt.boxplot(data)`
- Bar plot / Box plot

### `plt.title('text')`
- Add title

### `plt.xlabel('text')` / `plt.ylabel('text')`
- Add axis labels

### `plt.legend()`
- Show legend

### `plt.grid()`
- Add grid lines

### `plt.axvline(x=value)` / `plt.axhline(y=value)`
- Add vertical/horizontal line

### `plt.tight_layout()`
- Adjust spacing to prevent overlap

### `plt.savefig('filename.png', dpi=150)`
- Save figure to file

### `plt.show()`
- Display figure

### `plt.tick_params(axis='x', rotation=45)`
- Rotate tick labels

---

## Python Methods: statsmodels

### `adfuller(data)`
- Augmented Dickey-Fuller stationarity test
- Returns: (statistic, p-value, lags, nobs, critical_values, icbest)

### `SARIMAX(data, order=(p,d,q), seasonal_order=(P,D,Q,s))`
- Create SARIMA model
- order = non-seasonal parameters
- seasonal_order = seasonal parameters

### `.fit(disp=False, maxiter=200)`
- Train/fit the model
- disp=False suppresses output
- maxiter = maximum iterations

### `.forecast(steps=n)`
- Predict n steps into future

### `.aic` / `.bic`
- Access model quality metrics

---

## Python Methods: sklearn

### `mean_absolute_error(actual, predicted)`
- Calculate MAE

### `mean_squared_error(actual, predicted)`
- Calculate MSE
- Use `np.sqrt()` for RMSE

---

## Python Operators & Syntax

### `|` (pipe)
- Logical OR operator
- Example: `(x < 5) | (x > 10)` = x less than 5 OR greater than 10

### `&` (ampersand)
- Logical AND operator
- Example: `(x > 5) & (x < 10)` = x between 5 and 10

### f-string formatting
- `f"{value:,.0f}"` = format with commas, 0 decimals
- `f"{value:.2f}"` = 2 decimal places
- `f"{date.strftime('%b %Y')}"` = format date as "Jan 2025"

### List comprehension
- `[expression for item in list if condition]`
- Compact way to create lists

### `.strftime('%format')`
- Format datetime as string
- '%Y' = year, '%m' = month number, '%b' = month name, '%d' = day

---

## Common Abbreviations

| Abbreviation | Meaning |
|--------------|---------|
| **MW** | Megawatt (unit of power) |
| **MW-hr** | Megawatt-hour (unit of energy) |
| **YoY** | Year-over-Year |
| **Q1/Q3** | First/Third Quartile (25th/75th percentile) |
| **Std Dev** | Standard Deviation |
| **df** | DataFrame (common variable name) |
| **ax/axes** | Axis/Axes (matplotlib subplot reference) |
| **ADF** | Augmented Dickey-Fuller |
| **AR** | Autoregressive |
| **MA** | Moving Average |
| **ARIMA** | AutoRegressive Integrated Moving Average |
| **SARIMA** | Seasonal ARIMA |

---

## Author

Steve Eckardt  
