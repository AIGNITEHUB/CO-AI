# Forecast Mode and Target Mode Separation - Implementation Summary

## Overview
Successfully separated forecast mode and target mode logic in both backend and UI to prevent bugs from overlapping functionality.

## Key Changes

### 1. UI Updates (app.py)

#### Mode Selection (Lines 111-122)
- **Before**: Single checkbox "Enable detailed policy tracking"
- **After**: Radio button with 3 options:
  - None (no policy tracking)
  - Forecast Mode (policy-driven projection)
  - Target Mode (achieve specific target)

#### Mode-Specific Upload Info (Lines 128-131)
- Forecast Mode: Shows info about projecting emissions based on historical trends
- Target Mode: Shows info about calculating pathway to achieve target

#### Mode Validation (Lines 151-169)
- Added validation to ensure uploaded JSON mode matches selected UI mode
- Prevents mismatch errors

#### Display Summary (Lines 192-209)
**Forecast Mode displays:**
- Projection Year (instead of Target Year)
- Baseline Emissions
- Baseline Year

**Target Mode displays:**
- Target Year
- Target Reduction %
- Baseline Year

#### Info Messages (Lines 246-249)
- Different messages for forecast vs target mode
- Forecast: "Projecting emissions based on policy actions and historical trends"
- Target: "Target year, reduction %, and pathway are fixed"

#### Input Controls (Lines 254-293)
**Forecast Mode:**
- Shows "Projection year" instead of "Target year"
- Mode selector: "Policy-driven Projection" (disabled)
- No reduction % slider

**Target Mode:**
- Shows "Target year"
- Commitment type: "Percentage Reduction" (disabled)
- Reduction % slider (disabled, from JSON)

#### Emissions Display (Lines 298-345)
**Forecast Mode (Lines 298-312):**
- Shows "Projected Emissions" calculated from policy actions
- Displays delta from baseline
- No user input required

**Target Mode (Lines 314-345):**
- Shows "Target Emissions" with reduction slider
- User can adjust (or locked from JSON)
- Calculated from target_reduction_pct

#### Pathway Selection (Lines 350-366)
**Forecast Mode:**
- Shows "Projection pathway" with "Policy-driven projection"

**Target Mode:**
- Shows "Reduction pathway shape" with "Policy-driven (based on actual actions)"

#### Button and Success Messages (Lines 486-521)
**Forecast Mode:**
- Button: "Generate emissions projection"
- Success: "Emissions projection generated"

**Target Mode:**
- Button: "Generate commitment scenario"
- Success: "Commitment scenario generated"

### 2. Backend Logic (policy_commitments.py)

#### CountryCommitment Class (Lines 202-217)
- Added `mode` field: "forecast" or "target"
- Added `is_forecast_mode` property
- `target_reduction_pct` is optional in forecast mode
- `target_emissions_gtco2` returns baseline for forecast mode

#### PolicyAction Extrapolation (Lines 131-160)
**Forecast Mode (target_pct is 0 or None):**
- Uses historical trend from policy data
- Calculates slope via linear regression
- Applies damping factor for long-term projections
- Caps at 150% for realistic bounds

**Target Mode (target_pct > 0):**
- Extrapolates linearly toward target
- Uses target_year for slope calculation
- Caps at target_pct

#### Validation (Lines 464-469)
- Checks mode field
- Requires target_reduction_pct only for target mode
- Allows forecast mode without target_reduction_pct

### 3. JSON File Structure

#### Forecast Mode (vietnam_forecast_mode.json)
```json
{
  "Vietnam": {
    "mode": "forecast",
    "target_year": 2050,
    "baseline_year": 2025,
    "baseline_emissions_gtco2": 0.45,
    "policy_actions": [
      {
        "reduction_target_pct": 0,  // 0 triggers historical trend
        ...
      }
    ]
  }
}
```

#### Target Mode (example)
```json
{
  "Vietnam": {
    "mode": "target",
    "target_year": 2050,
    "baseline_year": 2025,
    "baseline_emissions_gtco2": 0.45,
    "target_reduction_pct": 50.0,  // Required for target mode
    "policy_actions": [
      {
        "reduction_target_pct": 60.0,  // Specific target
        ...
      }
    ]
  }
}
```

## Test Results

### Forecast Mode Test (vietnam_forecast_mode.json)
- Validation: PASS
- Mode detection: Correct (is_forecast_mode = True)
- Target reduction: None (as expected)
- Projections:
  - 2030: 21.38% reduction → 0.3538 GtCO2
  - 2050: 67.32% reduction → 0.1471 GtCO2 (auto-extrapolated)
- Policy actions: All loaded correctly with 0% targets

### Validation Tests
- Forecast mode without target_reduction_pct: PASS
- Target mode with target_reduction_pct: PASS
- Target mode without target_reduction_pct: FAIL (correctly rejected)

## Benefits

1. **Clear Separation**: No confusion between forecast and target modes
2. **Correct Logic**: Forecast uses historical trends, target uses specific goals
3. **Better UX**: Clear labels and info for each mode
4. **Validation**: Prevents uploading wrong JSON for selected mode
5. **Flexible**: Supports both pure forecasting and target-driven scenarios

## Usage

### For Forecast Mode:
1. Select "Forecast Mode" in the radio button
2. Upload JSON with `"mode": "forecast"` (no target_reduction_pct needed)
3. Policy actions should have `reduction_target_pct: 0` to use historical trends
4. System projects emissions based on policy momentum

### For Target Mode:
1. Select "Target Mode" in the radio button
2. Upload JSON with `"mode": "target"` and `"target_reduction_pct"`
3. Policy actions should have specific reduction targets
4. System calculates pathway to achieve the target

## Files Modified
- `app.py`: UI separation (lines 111-521)
- `logic/policy_commitments.py`: Backend logic (already correct)
- `test_forecast_target_modes.py`: New test file
- `FORECAST_TARGET_MODE_SEPARATION.md`: This documentation

## Next Steps (Optional)
1. Create a target mode JSON example for Vietnam
2. Add visual indicators (colors/icons) for mode selection
3. Add tooltips explaining the difference between modes
4. Consider adding a "hybrid" mode that combines both approaches
