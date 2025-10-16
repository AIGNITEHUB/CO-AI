# 🌳 Carbon Sink Calculation - Emitters vs Sinks

## 🎯 Overview

This document explains how carbon sinks (like forestry) are handled differently from emission sources in the policy commitment calculation.

---

## 🔴 **Problem: Why Separate Emitters and Sinks?**

### **The Fundamental Difference**

**EMITTERS** (positive baseline):
- Release CO2 into the atmosphere
- Goal: **REDUCE emissions** (bring them closer to zero)
- Example: Energy sector emits 210 MtCO2/year → reduce to 0

**SINKS** (negative baseline):
- Absorb CO2 from the atmosphere
- Goal: **INCREASE removal capacity** (absorb more CO2)
- Example: Forestry absorbs -15 MtCO2/year → increase to -45

### **Semantic Issue**

The field `reduction_target_pct` has **opposite meanings**:

| Sector Type | `reduction_target_pct` Means | Example |
|-------------|------------------------------|---------|
| **Emitter** | "Reduce emissions by X%" | 100% = eliminate all emissions |
| **Sink** | "Increase removal by X%" | 200% = triple removal capacity |

**This is why they must be calculated separately!**

---

## 📊 **Vietnam Data Example**

### **Baseline Emissions (2020)**

```
EMITTERS:
  Energy:       +210.0 MtCO2  (50.0% of gross)
  Transport:     +85.0 MtCO2  (20.2% of gross)
  Industry:      +75.0 MtCO2  (17.9% of gross)
  Agriculture:   +35.0 MtCO2  (8.3% of gross)
  ─────────────────────────────────────────
  Gross Total:  +405.0 MtCO2

SINKS:
  Forestry:      -15.0 MtCO2  (absorbs CO2)
  ─────────────────────────────────────────
  Net Total:    +390.0 MtCO2
```

### **Target (2050)**

```
EMITTERS:
  Energy:         0.0 MtCO2  (100% reduction)
  Transport:      0.0 MtCO2  (100% reduction)
  Industry:       0.0 MtCO2  (100% reduction)
  Agriculture:    3.5 MtCO2  (90% reduction)
  ─────────────────────────────────────────
  Gross Total:    3.5 MtCO2

SINKS:
  Forestry:     -45.0 MtCO2  (200% increase = 3x removal)
  ─────────────────────────────────────────
  Net Total:    -41.5 MtCO2  (CARBON NEGATIVE! 🎉)
```

**Result:** Not just Net Zero, but actually removing more CO2 than emitting!

---

## 🔧 **Corrected Calculation Method**

### **OLD Method (WRONG) ❌**

```python
# Treated all sectors the same way
for action in policy_actions:
    weighted_reduction = (action.share_pct / 100) * (reduction_pct / 100)
    total += weighted_reduction
```

**Problem:**
- Forestry with share_pct = 3.57% and reduction_target_pct = 200%
- Contributed: 3.57% × 200% = **7.14%** to total reduction
- But this is semantically wrong! Forestry doesn't "reduce emissions" - it removes CO2!

---

### **NEW Method (CORRECT) ✅**

```python
def calculate_annual_reduction_fraction(self, year: int) -> float:
    emitters_reduction = 0.0
    sinks_contribution = 0.0

    for action in self.policy_actions:
        reduction_pct = action.get_reduction_for_year(year, self.target_year)

        if action.baseline_emissions_mtco2 > 0:
            # EMITTER: Use share_pct weighted reduction
            weighted = (action.share_pct / 100) * (reduction_pct / 100)
            emitters_reduction += weighted

        elif action.baseline_emissions_mtco2 < 0:
            # SINK: Calculate additional removal in MtCO2
            baseline_removal = abs(action.baseline_emissions_mtco2)
            additional_removal = baseline_removal * (reduction_pct / 100)

            # Convert to equivalent reduction fraction
            baseline_mtco2 = self.baseline_emissions_gtco2 * 1000
            sinks_contribution += additional_removal / baseline_mtco2

    total = emitters_reduction + sinks_contribution
    return min(total, 1.0)
```

---

## 📈 **Step-by-Step Example: Year 2050**

### **Step 1: Calculate Reduction from EMITTERS**

Use `share_pct` (recalculated to exclude sinks):

```
Energy:      51.85% × 100% = 0.5185  (51.85%)
Transport:   20.99% × 100% = 0.2099  (20.99%)
Industry:    18.52% × 100% = 0.1852  (18.52%)
Agriculture:  8.64% ×  90% = 0.0778  (7.78%)
──────────────────────────────────────────────
Emitters Total:              0.9914  (99.14%)
```

**Interpretation:** Emitters reduce their baseline by 99.14% (almost fully eliminated).

---

### **Step 2: Calculate Contribution from SINKS**

Calculate additional removal in **MtCO2**, then convert:

```
Forestry Baseline Removal: 15 MtCO2 per year
Forestry Improvement: 200%

Additional Removal = 15 × (200 / 100) = 30 MtCO2

Convert to Reduction Fraction:
  Baseline Net Emissions: 420 MtCO2
  Sink Contribution: 30 / 420 = 0.0714 (7.14%)
```

**Interpretation:** The 30 MtCO2 of additional CO2 removal is equivalent to reducing baseline emissions by 7.14%.

---

### **Step 3: Total Reduction**

```
Total Reduction = Emitters + Sinks
                = 99.14% + 7.14%
                = 106.28%

Capped at 100% for target year
```

**Result:** Vietnam achieves **Net Zero and beyond** (carbon negative)!

---

## 🔄 **Share_pct Recalculation**

### **OLD share_pct (WRONG)**

Based on net emissions (including sink):

```
Base: 390 MtCO2 (net)

Energy:      210/390 = 53.85%
Transport:    85/390 = 21.79%
Industry:     75/390 = 19.23%
Agriculture:  35/390 = 8.97%
Forestry:    -15/390 = -3.85% → Set to 3.57% (arbitrary)
───────────────────────────────
Total: ~100%
```

**Problem:** Sink is forced into share_pct artificially.

---

### **NEW share_pct (CORRECT)**

Based on **gross emissions** (emitters only):

```
Base: 405 MtCO2 (gross, emitters only)

Energy:      210/405 = 51.85%
Transport:    85/405 = 20.99%
Industry:     75/405 = 18.52%
Agriculture:  35/405 = 8.64%
Forestry:         0 = 0.00%  (Not included in share calculation)
───────────────────────────────
Total: 100%
```

**Result:** Clean, logical share distribution among emitters.

---

## 📝 **Updated JSON Structure**

### **Before (Old)**

```json
{
  "sector": "Forestry",
  "baseline_emissions_mtco2": -15.0,
  "share_pct": 3.57,
  "reduction_target_pct": 200.0
}
```

**Issue:** `share_pct = 3.57%` was arbitrary and caused wrong calculation.

---

### **After (New)**

```json
{
  "sector": "Forestry",
  "baseline_emissions_mtco2": -15.0,
  "share_pct": 0.0,
  "reduction_target_pct": 200.0
}
```

**Fix:** `share_pct = 0.0` because Forestry is NOT included in weighted emitter calculation.

**Note:** The field is kept for consistency, but value is 0.

---

## 🔍 **How to Interpret reduction_target_pct**

### **For EMITTERS (baseline > 0)**

```
reduction_target_pct = % of baseline to eliminate

Examples:
  100% = Full decarbonization (0 emissions)
   90% = Reduce to 10% of baseline
   50% = Cut emissions in half
```

**Formula:**
```
Target Emissions = Baseline × (1 - reduction_target_pct/100)
```

**Example:** Energy baseline 210 MtCO2, target 100%
```
Target = 210 × (1 - 100/100) = 0 MtCO2 ✓
```

---

### **For SINKS (baseline < 0)**

```
reduction_target_pct = % to increase removal capacity

Examples:
  200% = Triple removal capacity (2x increase)
  100% = Double removal capacity (1x increase)
   50% = Increase removal by 50%
```

**Formula:**
```
Target Removal = |Baseline| × (1 + reduction_target_pct/100)
```

**Example:** Forestry baseline -15 MtCO2, target 200%
```
Target = -15 × (1 + 200/100) = -45 MtCO2 ✓
```

---

## 📊 **Comparison: OLD vs NEW Calculation**

### **Year 2024**

**OLD Method:**
```
Total = (50.0% × 4.0%) + (20.24% × 2.0%) + (17.86% × 3.0%) +
        (8.33% × 2.5%) + (3.57% × 8.0%)
      = 2.00% + 0.40% + 0.54% + 0.21% + 0.29%
      = 3.44%
```

**NEW Method:**
```
Emitters:
  (51.85% × 4.0%) + (20.99% × 2.0%) + (18.52% × 3.0%) + (8.64% × 2.5%)
  = 2.07% + 0.42% + 0.56% + 0.22%
  = 3.27%

Sinks:
  15 × (8/100) / 420 = 1.2 / 420 = 0.29%

Total = 3.27% + 0.29% = 3.56%
```

**Difference:** 3.44% → 3.56% (+0.12%)

---

### **Year 2050**

**OLD Method:**
```
Total = (50.0% × 100%) + (20.24% × 100%) + (17.86% × 100%) +
        (8.33% × 90%) + (3.57% × 200%)
      = 50.0% + 20.24% + 17.86% + 7.50% + 7.14%
      = 102.74% → Capped at 100%
```

**NEW Method:**
```
Emitters:
  (51.85% × 100%) + (20.99% × 100%) + (18.52% × 100%) + (8.64% × 90%)
  = 51.85% + 20.99% + 18.52% + 7.78%
  = 99.14%

Sinks:
  15 × (200/100) / 420 = 30 / 420 = 7.14%

Total = 99.14% + 7.14% = 106.28% → Capped at 100%
```

**Difference:** Both reach 100%, but new method is semantically correct.

---

## ✅ **Validation**

### **Test Results**

Running `test_vietnam_policy.py`:

```
✓ Test 1: Loading Vietnam commitment data
  - Forestry share_pct: 0.0% ✓
  - All emitters sum to ~100% ✓

✓ Test 2: Sector breakdown
  - Energy: 210.0 MtCO₂
  - Forestry: -15.0 MtCO₂ (negative) ✓

✓ Test 3: Annual reduction calculation
  - 2024: 3.55% (increased from 3.43%) ✓
  - 2030: 37.65% (increased from 35.88%) ✓

✓ Test 4: Policy-driven pathway forecast
  - Uses corrected calculation ✓
  - Reaches Net Zero by 2050 ✓

✓ Test 5: Compare pathways
  - Policy-driven distinct from exponential ✓
```

All tests pass with corrected logic! ✅

---

## 🎯 **Key Takeaways**

1. **Emitters and Sinks are fundamentally different**
   - Emitters: Reduce emissions (toward 0)
   - Sinks: Increase removal (away from 0, more negative)

2. **share_pct should only include emitters**
   - Base: Gross emissions (sum of positive baselines)
   - Sinks: Excluded from share_pct (set to 0)

3. **reduction_target_pct has dual meaning**
   - For emitters: % to reduce
   - For sinks: % to increase removal

4. **Calculation must separate the two**
   - Emitters: Weighted by share_pct
   - Sinks: Calculate in MtCO2, then convert to fraction

5. **Result: More accurate and semantically correct**
   - Total reduction properly accounts for both
   - Can exceed 100% if sinks are very strong
   - Achieves true Net Zero or carbon negative status

---

## 📚 **References**

- **Implementation:** `logic/policy_commitments.py:125-169`
- **Data structure:** `data/country_commitments.json`
- **Tests:** `test_vietnam_policy.py`, `test_extrapolation.py`
- **Related docs:** `POLICY_COMMITMENTS_GUIDE.md`, `POLICY_PATHWAY_SHAPE_EXPLAINED.md`

---

## 🚀 **For Other Countries**

When creating commitment data for other countries:

1. **Identify emitters vs sinks**
   - Positive baseline = emitter
   - Negative baseline = sink

2. **Calculate share_pct for emitters only**
   ```
   share_pct = (sector_baseline / gross_emissions) × 100
   ```
   Where `gross_emissions = sum of all positive baselines`

3. **Set share_pct = 0 for all sinks**

4. **Define reduction_target_pct appropriately**
   - Emitters: How much to reduce (typically 90-100%)
   - Sinks: How much to increase removal (can be >100%)

5. **Run validation**
   ```bash
   python test_vietnam_policy.py
   ```
   Adapt the test script for your country's data.

---

**Last Updated:** 2025-10-16
**Status:** ✅ Implemented and Tested
