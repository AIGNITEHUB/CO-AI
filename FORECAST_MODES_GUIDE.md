# 🔮 Forecast Modes Guide

## Overview

The CO-AI system supports **two forecast modes** based on target settings:

1. **📈 Target-Driven Mode** (Default) - Extrapolates toward reduction targets
2. **📊 Historical Trend Forecast Mode** - Uses historical data trends to project future (no targets)

---

## Mode 1: 📈 Target-Driven Mode (Default)

### **When to Use:**
- You have reduction goals/commitments (e.g., Net Zero by 2050)
- Want to see trajectory toward specific targets
- Planning policy pathways to achieve goals

### **Configuration:**
```json
{
  "target_reduction_pct": 100,  // Non-zero target
  "policy_actions": [
    {
      "sector": "Energy",
      "reduction_target_pct": 100,  // Non-zero sector target
      "implementation_years": [2026, 2027, 2028, 2029, 2030],
      "yearly_improvement_pct": [3.0, 6.0, 10.0, 15.0, 22.0]
    }
  ]
}
```

### **Behavior:**
1. **2026-2030:** Use actual policy data
2. **2031-2050:** **Extrapolate linearly** from 22% → 100%
3. Each sector progresses toward its own target
4. Reaches target exactly at target_year (2050)

### **Example Output:**
```
Year    Reduction     Emissions
2030    21.38%        0.354 GtCO₂  (last data)
2035    42.71%        0.258 GtCO₂  (extrapolated)
2040    64.03%        0.162 GtCO₂  (extrapolated)
2045    85.35%        0.066 GtCO₂  (extrapolated)
2050    100.00%       0.000 GtCO₂  (TARGET REACHED ✅)
```

### **Use Cases:**
- ✅ Net Zero commitments
- ✅ Paris Agreement compliance tracking
- ✅ Policy pathway planning
- ✅ Target feasibility analysis

---

## Mode 2: 📊 Historical Trend Forecast Mode

### **When to Use:**
- No specific reduction targets set
- Want to predict realistic trajectory based on policy momentum
- "What if current trends continue?" scenario
- More realistic baseline than flat projection for comparing against ambitious targets

### **Configuration:**
```json
{
  "target_reduction_pct": 0,  // Zero target
  "policy_actions": [
    {
      "sector": "Energy",
      "reduction_target_pct": 0,  // Zero sector target
      "implementation_years": [2026, 2027, 2028, 2029, 2030],
      "yearly_improvement_pct": [3.0, 6.0, 10.0, 15.0, 22.0]
    }
  ]
}
```

### **Behavior:**
1. **2026-2030:** Use actual policy data
2. **Calculate historical trend:** Linear regression on 2026-2030 data to find slope
3. **2031-2050:** **Extrapolate with damping** - trend continues but slows over time
4. **Damping formula:** `1.0 / (1.0 + 0.05 × years_elapsed)` prevents unrealistic growth
5. Captures policy momentum while accounting for diminishing returns

### **Example Output:**
```
Year    Reduction     Emissions     Damping
2030    21.38%        0.354 GtCO₂   N/A (last data)
2035    39.76%        0.271 GtCO₂   0.80 (80% of trend)
2040    52.01%        0.216 GtCO₂   0.67 (67% of trend)
2045    60.76%        0.177 GtCO₂   0.57 (57% of trend)
2050    67.32%        0.147 GtCO₂   0.50 (50% of trend)
```

### **Use Cases:**
- ✅ Momentum-based forecasting (policy trends continue)
- ✅ "What if we maintain current effort?" scenario
- ✅ Policy gap analysis (compare with Target-Driven)
- ✅ Realistic baseline projection
- ✅ Understanding natural progression from implemented policies

---

## Comparison Table

| Feature | Target-Driven Mode | Historical Trend Mode |
|---------|-------------------|-------------------|
| **Target Setting** | Non-zero (e.g., 100%) | Zero (0%) |
| **Extrapolation** | ✅ Yes (linear to target) | ✅ Yes (historical trend + damping) |
| **2031-2050 Trajectory** | Decreasing to target | Decreasing with momentum |
| **Reaches Target** | ✅ Yes (at target_year) | N/A (trend-based) |
| **Use Case** | Policy planning | Momentum forecasting |
| **Method** | Linear interpolation | Linear regression + damping |
| **Realism** | Goal-oriented | Trend-based |

---

## Implementation Logic

### **Code (policy_commitments.py:131-160)**

```python
# FORECAST MODE: If target_pct is 0 or None, use historical trend
if self.target_pct is None or self.target_pct == 0:
    # Calculate historical trend from actual policy data
    if len(self.implementation_years) >= 2:
        # Linear regression on all data points
        years_data = self.implementation_years
        reduction_data = self.yearly_improvement_pct

        # Calculate slope: y = mx + b
        n = len(years_data)
        sum_x = sum(years_data)
        sum_y = sum(reduction_data)
        sum_xy = sum(y * x for x, y in zip(years_data, reduction_data))
        sum_x2 = sum(x * x for x in years_data)
        historical_slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)

        # Apply damping factor (diminishing returns)
        years_elapsed = year - last_data_year
        damping = 1.0 / (1.0 + 0.05 * years_elapsed)

        # Extrapolate with damped trend
        extrapolated = last_data_reduction + historical_slope * years_elapsed * damping
        return min(max(extrapolated, 0.0), 150.0)
    else:
        return last_data_reduction  # Fallback
```

### **Decision Flow:**

```
Year > last_data_year?
  ↓ Yes
  ↓
target_pct == 0?
  ↓ Yes → HISTORICAL TREND MODE
  |       1. Calculate slope from 2026-2030 data
  |       2. Apply damping factor
  |       3. Extrapolate: last + slope × years × damping
  |
  ↓ No → TARGET-DRIVEN MODE
        Extrapolate: last + slope × (year - last_year)
        Cap at target_pct
```

---

## Example Scenarios

### **Scenario 1: Ambitious Net Zero**

**Config:**
```json
{
  "target_reduction_pct": 100,
  "Energy": {"reduction_target_pct": 100},
  "Transport": {"reduction_target_pct": 100},
  ...
}
```

**Result:**
- 2030: 21% reduction
- 2050: **100% reduction** (Net Zero achieved)
- Linear progression

---

### **Scenario 2: Moderate Target**

**Config:**
```json
{
  "target_reduction_pct": 60,
  "Energy": {"reduction_target_pct": 70},
  "Transport": {"reduction_target_pct": 50},
  ...
}
```

**Result:**
- 2030: 21% reduction
- 2050: **60% reduction** (weighted average of sectors)
- Partial decarbonization

---

### **Scenario 3: Historical Trend Forecast**

**Config:**
```json
{
  "target_reduction_pct": 0,
  "Energy": {"reduction_target_pct": 0},
  "Transport": {"reduction_target_pct": 0},
  ...
}
```

**Result:**
- 2030: 21.38% reduction (from 2026-2030 policies)
- 2035: 39.76% reduction (trend continues with 80% damping)
- 2040: 52.01% reduction (trend continues with 67% damping)
- 2050: **67.32% reduction** (trend continues with 50% damping)
- Realistic momentum-based projection

---

## Testing

### **Test Files:**

1. **test_extrapolation_2030.py** - Tests Target-Driven mode (targets > 0)
2. **test_historical_trend.py** - Tests Historical Trend Forecast mode (comprehensive)
3. **test_pure_forecast.py** - Tests with ALL targets = 0 (deprecated - now uses trend)
4. **test_forecast_mode.py** - Tests behavior with country target = 0

### **Test Commands:**

```bash
# Test Target-Driven mode (default)
python test_extrapolation_2030.py

# Test Historical Trend Forecast mode
python test_historical_trend.py

# Compare both modes
python test_forecast_mode.py
```

---

## Data Files

### **Target-Driven Mode:**
- **File:** `data/country_commitments.json`
- **Targets:** All > 0
- **Use:** Default production mode

### **Pure Forecast Mode:**
- **File:** `data/vietnam_forecast_mode.json`
- **Targets:** All = 0
- **Use:** BAU baseline analysis

---

## Switching Between Modes

### **Method 1: Modify JSON targets**

**From Target-Driven → Pure Forecast:**
```json
// Change all targets from this:
"reduction_target_pct": 100

// To this:
"reduction_target_pct": 0
```

**From Pure Forecast → Target-Driven:**
```json
// Change all targets from this:
"reduction_target_pct": 0

// To this:
"reduction_target_pct": 100
```

### **Method 2: Use separate JSON files**

```python
# Target-Driven
commitment = load_country_commitment('Vietnam', 'data')

# Pure Forecast
with open('data/vietnam_forecast_mode.json') as f:
    commitment = load_country_commitment_from_json(f.read(), 'Vietnam')
```

---

## Interpretation Guide

### **Target-Driven Results:**

```
2050: 100% reduction → 0.000 GtCO₂
```

**Means:**
- "If we follow this trajectory, we'll reach Net Zero by 2050"
- Assumes continuous policy implementation
- Shows **what we need to achieve**

### **Historical Trend Forecast Results:**

```
2050: 67.32% reduction → 0.147 GtCO₂
```

**Means:**
- "If current policy momentum continues (with diminishing returns), we'll reach 67% reduction"
- Realistic projection based on 2026-2030 trends
- Shows **what will happen if we maintain current effort level**
- More optimistic than "do nothing", less ambitious than Net Zero target

### **Gap Analysis:**

```
Target-Driven (2050):      0.000 GtCO₂ (100% reduction - Net Zero)
Historical Trend (2050):   0.147 GtCO₂ (67.32% reduction)
─────────────────────────────────────
GAP:                       0.147 GtCO₂  ⚠️

Interpretation:
"Current momentum gets us 67% of the way. We need additional
 policies to close the remaining 32.68% gap to reach Net Zero."
```

---

## Validation Checks

### **Target-Driven Mode:**

✅ **Check 1:** Continuous progression
- Reduction increases every year
- No plateau effect

✅ **Check 2:** Target reached
- Exactly 100% at target_year
- Not exceeded

✅ **Check 3:** Linear rate
- Slope = (target - last_data) / years_remaining
- Consistent across sectors

### **Historical Trend Forecast Mode:**

✅ **Check 1:** Continued improvement after 2030
- Reduction increases beyond 2030 based on historical trend
- Emissions decrease but at diminishing rate (damping effect)

✅ **Check 2:** Damping factor applied correctly
- Short-term (2-5 years): ~80-90% of trend applied
- Long-term (20+ years): ~50% of trend applied
- Prevents unrealistic exponential growth

✅ **Check 3:** Sector-specific trends
- Each sector extrapolates based on its own historical slope
- High-momentum sectors (e.g., Forestry +6.75%/yr) grow faster
- Low-momentum sectors (e.g., Agriculture +3.75%/yr) grow slower

---

## Best Practices

### **✅ DO:**

1. **Use Target-Driven for:**
   - Policy planning and commitments
   - NDC (Nationally Determined Contributions) tracking
   - Climate goal feasibility analysis

2. **Use Historical Trend Forecast for:**
   - Momentum-based baseline projections
   - "Current effort continues" scenario
   - Realistic reference case for policy gap analysis
   - Understanding diminishing returns from existing policies

3. **Run Both Modes:**
   - Compare trajectories to identify policy gap
   - Target-Driven shows where you **want** to be
   - Historical Trend shows where you'll **likely** be
   - Gap = additional policies needed

### **❌ DON'T:**

1. **Mix modes unintentionally:**
   - Don't set some sectors to 0, others to 100
   - Be consistent within scenario

2. **Over-interpret Historical Trend:**
   - It assumes trends continue - they may not
   - External shocks (tech breakthroughs, economic crises) can change trajectory
   - Real emissions may differ due to unforeseen factors

3. **Forget uncertainty:**
   - Both modes assume linearity
   - Real-world is non-linear (technology breakthroughs, shocks)

---

## Future Enhancements

### **Planned (AI/ML Integration):**

1. **Prophet Model** (Phase 2)
   - Non-linear trend extrapolation
   - Incorporate GDP, population growth
   - Uncertainty bands

2. **XGBoost** (Phase 3)
   - Feature-rich forecasting
   - Technology adoption curves
   - Policy effectiveness modeling

3. **LSTM** (Phase 4)
   - Complex pattern recognition
   - Cross-sector interactions
   - Historical analogs from other countries

4. **Ensemble** (Phase 5)
   - Combine linear + Prophet + XGBoost + LSTM
   - Best-estimate forecast
   - Confidence intervals

See: `AI_FORECASTING_METHODS.md` for details

---

## FAQ

**Q: Can I have different targets for different sectors?**
A: Yes! Each sector has its own `reduction_target_pct`. National target is weighted average.

**Q: What if I only want Pure Forecast for some sectors?**
A: Set those sectors' `reduction_target_pct = 0`, keep others > 0.

**Q: How accurate is the linear extrapolation?**
A: It's a simplification. Real trajectories may be S-curved or exponential. See AI/ML guide for improvements.

**Q: Can I change target mid-way (e.g., 2040)?**
A: Currently no. Add 2040 to `implementation_years` as a milestone instead.

**Q: What if my data goes to 2035 instead of 2030?**
A: System auto-adjusts. Extrapolation starts from last data year.

---

## Summary

| Mode | Config | Behavior | Use Case |
|------|--------|----------|----------|
| **Target-Driven** | Targets > 0 | Extrapolate to target | Policy planning |
| **Historical Trend** | Targets = 0 | Extrapolate with damped trend | Momentum forecasting |

**Key Takeaway:** Use **both modes** to understand:
- Where you **want to go** (Target-Driven → 100% reduction by 2050)
- Where you'll **likely end up** (Historical Trend → ~67% reduction by 2050)
- The **GAP** you need to close (~33% additional reduction needed)

---

**Last Updated:** 2025-10-17
**Status:** ✅ Production Ready
**Related Docs:**
- `AI_FORECASTING_METHODS.md` - Advanced forecasting
- `EXTRAPOLATION_UPDATE_SUMMARY.md` - Technical details
- `LOGIC_DOCUMENTATION.md` - Full system docs
