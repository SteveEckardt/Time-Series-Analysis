# SARIMA Forecasting Cheat Sheet: Boujdour Electricity

Quick reference guide for Morocco electricity forecasting project using SARIMA time series analysis.

---

## Dataset Basics

| Attribute | Value |
|-----------|-------|
| **Source** | Morocco Dataset (UC Irvine), Boujdour city |
| **Time Period** | Sept 14, 2022 → May 24, 2024 (21 months) |
| **Original Frequency** | 30-minute intervals (88,890 measurements) |
| **Analysis Frequency** | Monthly totals (21 data points) |
| **Coverage** | 3 municipal zones combined into city total |

---

## Key Patterns Discovered

### Monthly Seasonality (STRONG)
- **Peak:** October (~775k units) - comfortable weather, high economic activity
- **Low:** July (~515k units) - extreme heat, reduced activity
- **Pattern:** 62% swing from low to high

### Weekly Pattern (WEAK)
- **Weekdays:** ~147 units (30-min avg)
- **Weekends:** ~142 units (30-min avg)
- **Difference:** Only 4%

### Daily Pattern (STRONG)
- **Peak:** 9 PM (185 units) - evening residential use
- **Low:** 8 AM (114 units) - early morning
- **Pattern:** 62% swing from low to high

---

## Model Details

### Model Type
**SARIMA(1,0,1)(1,0,1,12)**

### Parameters Explained

**Non-seasonal (1,0,1):**
- p=1: Autoregressive order (uses 1 past value)
- d=0: No differencing needed
- q=1: Moving average order (uses 1 past error)

**Seasonal (1,0,1,12):**
- P=1: Seasonal AR (uses value from 1 season ago)
- D=0: No seasonal differencing needed
- Q=1: Seasonal MA (uses error from 1 season ago)
- s=12: 12-month seasonal cycle

### Why These Parameters?
- **0 differencing:** Data is stationary (ADF p-value = 0.0003)
- **12-month cycle:** Captures yearly seasonal pattern
- **Simple terms (1,1):** Limited data (21 months) - avoid overfitting

---

## Model Performance

### Validation (3-month test set)
- **Training:** 18 months (Sept 2022 - Feb 2024)
- **Testing:** 3 months (Mar - May 2024)
- **MAE:** 57,138 units (~10% average error)

**Error by Month:**
| Month | Error |
|-------|-------|
| March 2024 | 1.2% |
| April 2024 | 3.1% |
| May 2024 | 28.7% (incomplete data - ends May 24) |

---

## 12-Month Forecast (June 2024 - May 2025)

### Summary Statistics
- **Peak Period:** September 2024 (690,011 units)
- **Low Period:** May 2025 (549,368 units)
- **Average:** 615,145 units/month
- **Annual Total:** ~7.4 million units

### Monthly Breakdown

| Month | Forecast (units) | Notes |
|-------|-----------------|-------|
| June 2024 | 612,691 | |
| July 2024 | 582,635 | Summer low |
| August 2024 | 605,060 | |
| September 2024 | 690,011 | **Peak** |
| October 2024 | 646,840 | |
| November 2024 | 618,919 | |
| December 2024 | 628,928 | |
| January 2025 | 613,368 | |
| February 2025 | 592,438 | |
| March 2025 | 618,003 | |
| April 2025 | 623,478 | |
| May 2025 | 549,368 | **Year low** |

---

## Limitations

1. **Limited data:** Only 1.75 seasonal cycles available
2. **No external variables:** Doesn't account for weather, economy, holidays
3. **Forecast uncertainty:** Accuracy degrades beyond 12 months
4. **Incomplete months:** Sept 2022 and May 2024 are partial

---

## Business Implications

### Operational Planning

| Area | Recommendation |
|------|----------------|
| **Peak Capacity** | Prepare for ~690k in Sept-Oct (20% above summer) |
| **Maintenance Window** | Schedule during July-Aug low demand period |
| **Budget Planning** | Plan for ~7.4M units annually |
| **Infrastructure** | Ensure grid can handle fall peak loads |

---

## Common Questions

### "Can it forecast 2-3 years out?"
Yes, but accuracy degrades significantly. Comfortable forecasting 12-18 months maximum with current data.

### "What about extreme weather events?"
Model only captures historical patterns. Need SARIMAX with temperature/weather data for climate-adjusted forecasts.

### "Why monthly instead of daily forecasts?"
Only 21 months of data available. Need 1.5-2 full seasonal cycles minimum for reliable 12-month seasonality detection.

### "Why is May forecast high but actual value low?"
Dataset ends May 24 (incomplete month). Model predicts full month totals.

### "What makes the data stationary?"
No long-term drift present. Demand remains stable around trend except for predictable seasonal cycles.

---

## One-Sentence Summary

SARIMA predicts future electricity cycles based on historical seasonal patterns, forecasting Boujdour will use ~615k units/month with September peak and summer lows.

---

## Author

Steve Eckardt  
