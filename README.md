# Time Series Forecasting: Bangladesh Electricity Demand

Practical time series analysis and forecasting project using 10 years of Bangladesh electricity grid data. Built SARIMA model achieving 4.59% MAPE to forecast 24-month demand for infrastructure planning.

## Project Overview

**Objective:** Forecast monthly electricity demand for Bangladesh power grid over 24-month horizon to support capacity planning and resource allocation.

**Dataset:**
- Source: Bangladesh Power Development Board
- Period: April 2015 - May 2025 (10 years)
- Granularity: 92,650 hourly measurements
- Growth: 115% increase over period

## Key Findings

**Demand Patterns:**
- Strong upward trend: +11.5% annual growth
- Clear seasonality: 12-month cycle
- Peak demand: August (summer cooling)
- Low demand: December (moderate climate)
- Seasonal swing: 47% difference peak-to-low

**Forecast Results (June 2025 - May 2027):**
- Average monthly demand: 9.5M MW-hours
- Projected peak: August 2026 (11.2M MW-hours)
- Projected low: December 2026 (6.7M MW-hours)
- Growth rate: 8% year-over-year

## Technical Approach

### 1. Data Preparation
- **Outlier Detection:** IQR method identified 96 erroneous values (0.1%)
- **Cleaning:** Replaced outliers with rolling median
- **Aggregation:** Hourly → Monthly totals (92,650 hours → 122 months)

### 2. Model Selection
**SARIMA(1,1,1)(1,1,1,12)**
- Non-seasonal: (p=1, d=1, q=1)
- Seasonal: (P=1, D=1, Q=1, s=12)
- Stationarity: ADF test confirmed differencing needed (original p=0.84, after differencing p=0.000006)

### 3. Model Validation
- Training set: 98 months (April 2015 - May 2023)
- Test set: 24 months (June 2023 - May 2025)

**Performance Metrics:**
- MAPE: 4.59% (Excellent - target <10%)
- MAE: 376,340 MW-hours
- RMSE: 426,875 MW-hours

## Technologies Used

- **Python 3.x**
- **pandas** - Data manipulation
- **numpy** - Numerical operations
- **matplotlib** - Visualization
- **statsmodels** - SARIMA implementation
- **sklearn** - Validation metrics

## Project Structure
```
├── data/
│   └── Boujdour.csv
│   └── Bangladesh.csv
├── notebooks/
│   ├── Eckardt_Morocco_Electricity_TimeSeries_Analysis
│   ├── Eckardt_TimeSeries_Final_WorkBook (Bangladesh Electricity)
└── README.md
```

## Key Insights

1. **Predictable Growth:** Strong upward trend with consistent seasonal patterns makes forecasting reliable
2. **Capacity Planning:** Infrastructure must handle 47% seasonal variation
3. **Peak Preparation:** August consistently shows highest demand (cooling needs)
4. **Model Reliability:** 4.59% MAPE demonstrates excellent forecast accuracy

## Limitations

- Assumes historical patterns continue
- Does not account for:
  - Major policy changes
  - Climate disruptions
  - Economic shocks
  - New industrial development

## Future Improvements

1. Incorporate exogenous variables (temperature, GDP, population)
2. Test alternative models (Prophet, LSTM)
3. Implement confidence intervals
4. Daily/hourly granularity forecasting
5. Real-time model updating

## Results

Successfully delivered 24-month electricity demand forecast with 4.59% error rate, providing Bangladesh Power Development Board with reliable projections for infrastructure planning and resource allocation.

## Author

Steve Eckardt  
Open Avenues The Build Fellowship 

## Certificate

Completed: December 2025  
Program: Practical Time Series Analysis - The Brainiacs Fellowship  
Project Lead: Aiswarya Jonnalagadda
This project is for educational and portfolio purposes.

## License
MIT
