# Python/Pandas/Plotting Keywords - Quick Reference

Fast lookup guide for common Python data analysis operations.

---

## PANDAS CORE

### Loading & Inspection

```python
pd.read_csv('file.csv')    # Load CSV file into DataFrame
df.head(n)                 # Show first n rows (default 5)
df.tail(n)                 # Show last n rows (default 5)
df.shape                   # Returns (rows, columns) tuple
df.columns                 # List of column names
df.index                   # Row labels/index
```

### Selection

```python
df['column']               # Select single column (returns Series)
df[['col1', 'col2']]       # Select multiple columns (returns DataFrame)
```

### Index Operations

```python
df.set_index('column')     # Make column the index
df.reset_index()           # Convert index back to column
```

### Handling Missing Data

```python
df.dropna()                # Remove rows with missing values
df.fillna(value)           # Fill missing values
```

---

## DATETIME OPERATIONS

### Conversion

```python
pd.to_datetime()           # Convert string/numbers to datetime
```

### Extraction

```python
.dt.month                  # Extract month number (1-12)
.dt.dayofweek              # Extract day of week (0=Monday, 6=Sunday)
.dt.hour                   # Extract hour (0-23)
.dt.year                   # Extract year
.dt.to_period('M')         # Convert to period (e.g., '2023-01' for monthly)
```

### Date Generation

```python
pd.date_range()            # Create sequence of dates
pd.DateOffset()            # Add time intervals to dates
```

---

## GROUPBY & AGGREGATION

### Basic Operations

```python
df.groupby('column')       # Group rows by unique values in column
.mean()                    # Calculate average
.sum()                     # Calculate total
.count()                   # Count non-null values
.min() / .max()            # Minimum/maximum values
.std()                     # Standard deviation
.agg()                     # Apply multiple aggregation functions
```

### Example Usage

```python
df.groupby('month')['sales'].mean()
df.groupby('category').agg(['mean', 'sum', 'count'])
```

---

## TIME SERIES

### Resampling

```python
.resample('freq')          # Change time frequency (must have datetime index)
```

**Frequency Options:**
- `'D'` = Daily
- `'W'` = Weekly
- `'ME'` = Month End
- `'MS'` = Month Start

### Shifting & Rolling

```python
.shift(n)                  # Move data forward/backward by n periods
.rolling(window=n)         # Create rolling/moving window
```

### Example Usage

```python
df.resample('ME').sum()                    # Aggregate to monthly totals
df['value'].rolling(window=7).mean()       # 7-period moving average
```

---

## MATPLOTLIB (plt)

### Figure Creation

```python
plt.figure(figsize=(w,h))  # Create new figure with width, height
plt.subplots(rows, cols)   # Create multiple plots
```

### Plot Types

```python
plt.plot(x, y)             # Create line plot
plt.bar(x, y)              # Create bar plot
```

### Labels & Titles

```python
plt.title('text')          # Add title
plt.xlabel('text')         # Add x-axis label
plt.ylabel('text')         # Add y-axis label
plt.legend()               # Show legend
```

### Customization

```python
plt.grid()                 # Add grid lines
plt.xticks()               # Customize x-axis tick marks
plt.yticks()               # Customize y-axis tick marks
plt.axvline(x=value)       # Add vertical line
plt.tight_layout()         # Adjust spacing to prevent label cutoff
```

### Display

```python
plt.show()                 # Display the plot
```

---

## PANDAS PLOTTING

### Basic Plotting

```python
df.plot()                  # Quick plot (uses matplotlib underneath)
```

### Plot Types

```python
kind='line'                # Line plot (default)
kind='bar'                 # Bar plot
kind='barh'                # Horizontal bar plot
```

### Customization Options

| Parameter | Description | Example |
|-----------|-------------|---------|
| `figsize=(w,h)` | Set figure size | `figsize=(12,6)` |
| `color='name'` | Set color | `color='blue'` |
| `marker='o'` | Add markers to points | `marker='o'` |
| `linestyle='--'` | Line style | `linestyle='--'` |
| `linewidth=n` | Line thickness | `linewidth=2` |
| `label='text'` | Label for legend | `label='Sales'` |

### Example Usage

```python
df.plot(kind='line', figsize=(12,6), color='blue', marker='o', linewidth=2)
```

---

## NUMPY

### Statistical Functions

```python
np.mean()                  # Calculate average
np.sum()                   # Calculate sum
np.std()                   # Standard deviation
np.min() / np.max()        # Min/max values
```

### Mathematical Functions

```python
np.abs()                   # Absolute value
np.sqrt()                  # Square root
```

---

## STATSMODELS

### Stationarity Testing

```python
adfuller(data)             # Augmented Dickey-Fuller stationarity test
```

**Returns:** tuple: `(statistic, p-value, lags, nobs, critical_values, icbest)`

### SARIMA Modeling

```python
SARIMAX(data, order=(p,d,q), seasonal_order=(P,D,Q,s))  # Create SARIMA model
```

**Parameters:**
- `order`: (autoregressive, differencing, moving average)
- `seasonal_order`: (seasonal AR, seasonal diff, seasonal MA, season length)

### Model Operations

```python
.fit()                     # Train/fit the model
.forecast(steps=n)         # Predict n steps into future
```

---

## SKLEARN METRICS

### Error Metrics

```python
mean_absolute_error(actual, predicted)         # Average absolute difference
mean_squared_error(actual, predicted)          # Average squared difference
np.sqrt(mean_squared_error())                  # RMSE (root mean squared error)
```

---

## STRING FORMATTING

### f-string Formatting

```python
f"{value:,}"               # Format with comma separators (1,234,567)
f"{value:.2f}"             # Format to 2 decimal places (123.45)
f"{value:,.2f}"            # Comma + 2 decimals (1,234.56)
```

### Date Formatting

```python
.strftime('%B %Y')         # Format date ('January 2024')
```

**Common Date Format Codes:**
- `%Y` = Year (2024)
- `%m` = Month number (01-12)
- `%B` = Month name (January)
- `%d` = Day (01-31)

---

## COMMON PARAMETER NAMES

### Plot Styling

| Parameter | Description | Example Values |
|-----------|-------------|----------------|
| `figsize=(width, height)` | Plot dimensions in inches | `(12, 6)` |
| `color=` | Color name or code | `'blue'`, `'red'`, `'#FF5733'` |
| `label=` | Text for legend | `'Sales Data'` |
| `linewidth=` or `lw=` | Line thickness | `2`, `3` |
| `linestyle=` or `ls=` | Line style | `'-'`, `'--'`, `':'`, `'-.'` |
| `marker=` | Point marker | `'o'`, `'s'`, `'^'`, `'x'` |
| `alpha=` | Transparency | `0.5` (0=invisible, 1=opaque) |
| `rotation=` | Rotate labels (degrees) | `45`, `90` |
| `fontsize=` | Text size | `12`, `14` |
| `grid=True/False` | Show/hide grid | `True`, `False` |

---

## COMMON WORKFLOW PATTERNS

### Pattern 1: Load → Clean → Group → Plot

```python
# Load data
df = pd.read_csv('file.csv')

# Convert datetime
df['DateTime'] = pd.to_datetime(df['DateTime'])

# Group and aggregate
monthly = df.groupby('month')['total'].mean()

# Plot
monthly.plot(kind='bar')
plt.show()
```

### Pattern 2: Resample → Aggregate → Forecast

```python
# Resample to monthly frequency
monthly_df = df.set_index('DateTime').resample('ME').sum()

# Build SARIMA model
model = SARIMAX(monthly_df, 
                order=(1,0,1), 
                seasonal_order=(1,0,1,12))

# Fit and forecast
forecast = model.fit().forecast(steps=12)
```

### Pattern 3: Load → Clean Outliers → Analyze

```python
# Load data
df = pd.read_csv('file.csv')

# Detect outliers (IQR method)
Q1 = df['value'].quantile(0.25)
Q3 = df['value'].quantile(0.75)
IQR = Q3 - Q1
outliers = (df['value'] < Q1 - 1.5*IQR) | (df['value'] > Q3 + 1.5*IQR)

# Replace outliers
df.loc[outliers, 'value'] = df['value'].rolling(window=24).median()

# Continue analysis
monthly = df.groupby('month')['value'].mean()
```

### Pattern 4: Train/Test Split → Validate

```python
# Split data
train_size = int(len(df) * 0.8)
train = df[:train_size]
test = df[train_size:]

# Build and fit model
model = SARIMAX(train, order=(1,0,1), seasonal_order=(1,0,1,12))
fitted = model.fit()

# Predict and validate
predictions = fitted.forecast(steps=len(test))
mae = mean_absolute_error(test, predictions)
print(f"MAE: {mae:,.2f}")
```

---

## QUICK TIPS

### Performance
- Use `.loc[]` and `.iloc[]` for selection (faster than chaining)
- Avoid loops when possible (use vectorized operations)
- Use `df.query()` for complex filtering

### Memory
- Use `dtype` parameter in `read_csv()` to specify data types
- Drop unused columns early: `df.drop(columns=['col1', 'col2'])`
- Use `pd.to_datetime()` with `format=` parameter for faster conversion

### Debugging
- Use `df.info()` to check data types and null values
- Use `df.describe()` for quick statistics
- Use `df.sample(10)` to view random rows

---

## Author

Steve Eckardt  
