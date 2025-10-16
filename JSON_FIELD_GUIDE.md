# JSON Field Guide - Policy Commitment Data

## Overview
This guide explains all fields in the policy commitment JSON files for both Forecast Mode and Target Mode.

---

## Country-Level Fields

### Required Fields (Both Modes)

#### `country` (string)
- **Purpose**: Name of the country
- **Example**: `"Vietnam"`, `"USA"`, `"Japan"`
- **Notes**: Must match the top-level key in the JSON

#### `mode` (string)
- **Purpose**: Determines how emissions are calculated
- **Options**:
  - `"forecast"` - Project based on policy actions and historical trends
  - `"target"` - Calculate pathway to achieve specific reduction target
- **Example**: `"mode": "forecast"`
- **Default**: `"target"` (for backward compatibility)

#### `target_year` (integer)
- **Purpose**:
  - **Forecast Mode**: Final year for projection
  - **Target Mode**: Year to achieve reduction target
- **Example**: `2050`, `2060`, `2100`
- **Range**: Must be > baseline_year, typically 2030-2100

#### `baseline_year` (integer)
- **Purpose**: Reference year for baseline emissions
- **Example**: `2020`, `2025`, `2030`
- **Notes**: All reductions are calculated relative to this year

#### `baseline_emissions_gtco2` (float)
- **Purpose**: Total emissions at baseline year in Gigatons CO2
- **Example**: `0.45` (450 million tons)
- **Unit**: GtCO2 (1 GtCO2 = 1000 MtCO2)
- **Notes**: Sum of all sector baselines should approximately equal this

#### `policy_actions` (array)
- **Purpose**: List of sector-specific policy actions
- **Type**: Array of PolicyAction objects
- **Min**: 1 action required
- **Example**: See Policy Action Fields section below

---

### Mode-Specific Fields

#### `target_reduction_pct` (float)
- **When Required**:
  - ✅ **Required** for Target Mode
  - ❌ **Not used** in Forecast Mode (omit or set to null)
- **Purpose**: Overall emission reduction target as percentage of baseline
- **Example**: `50.0` means reduce by 50% (e.g., 100 → 50 GtCO2)
- **Range**: 0.0 to 100.0
- **Notes**:
  - In Target Mode: System calculates pathway to achieve this
  - In Forecast Mode: System projects based on policy momentum instead

---

## Policy Action Fields

Each policy action represents a sector-specific emission reduction initiative.

### Required Fields (Both Modes)

#### `sector` (string)
- **Purpose**: Name of the emissions sector
- **Example**: `"Energy"`, `"Transport"`, `"Industry"`, `"Agriculture"`, `"Forestry"`, `"Buildings"`
- **Notes**: Can be any descriptive name

#### `action_name` (string)
- **Purpose**: Description of the specific policy action
- **Example**:
  - `"Coal phase-out & Renewable expansion"`
  - `"Electric vehicle adoption"`
  - `"Reforestation & Forest conservation"`
- **Notes**: Should be specific and descriptive

#### `baseline_year` (integer)
- **Purpose**: Reference year for this sector's baseline
- **Example**: `2025`
- **Notes**: Usually same as country-level baseline_year

#### `baseline_emissions_mtco2` (float)
- **Purpose**: Sector emissions at baseline year in Million tons CO2
- **Example**: `225.0` (225 million tons)
- **Unit**: MtCO2 (1 GtCO2 = 1000 MtCO2)
- **Special**:
  - **Positive values**: Emission sources (e.g., Energy: 225.0)
  - **Negative values**: Carbon sinks (e.g., Forestry: -17.0)

#### `implementation_years` (array of integers)
- **Purpose**: Years when policy data is available
- **Example**: `[2026, 2027, 2028, 2029, 2030]`
- **Notes**:
  - Must be in ascending order
  - Must be >= baseline_year
  - Can extend beyond or stop before target_year
  - System auto-extrapolates if gap exists

#### `yearly_improvement_pct` (array of floats)
- **Purpose**: Cumulative reduction/increase percentage for each year
- **Example**: `[3.0, 6.0, 10.0, 15.0, 22.0]`
- **Unit**: Percentage (%)
- **Notes**:
  - Must have same length as implementation_years
  - Should be cumulative (increasing)
  - For **emitters** (positive baseline): % reduction in emissions
  - For **sinks** (negative baseline): % increase in removal capacity

#### `status` (string)
- **Purpose**: Current status of policy implementation
- **Options**:
  - `"On track"` - Progressing as planned
  - `"Behind schedule"` - Lagging behind targets
  - `"Ahead"` - Exceeding expectations
  - `"At risk"` - Facing challenges
- **Example**: `"On track"`
- **Notes**: For display/reporting only, doesn't affect calculations

---

### Target Fields (Required - at least one)

#### `reduction_target_pct` (float)
- **When Used**: For emitting sectors (positive baseline)
- **Purpose**: Target reduction percentage by target_year
- **Example**: `60.0` means reduce by 60%
- **Range**: 0.0 to 100.0
- **Notes**:
  - **Forecast Mode**: Set to `0` to use historical trend extrapolation
  - **Target Mode**: Set to actual reduction target

#### `removal_increase_pct` (float)
- **When Used**: For carbon sink sectors (negative baseline)
- **Purpose**: Target increase in removal capacity by target_year
- **Example**: `200.0` means triple the removal (100% baseline + 200% increase)
- **Range**: 0.0 to 1000.0+ (can exceed 100% for sinks)
- **Notes**:
  - **Forecast Mode**: Set to `0` to use historical trend
  - **Target Mode**: Set to target increase

**Important**: Each action must have either `reduction_target_pct` OR `removal_increase_pct` (or both). For sinks, prefer `removal_increase_pct`.

---

### Optional Fields

#### `share_pct` (float)
- **Purpose**: Manually override the sector's share of total emissions
- **Auto-calculated**: If omitted, calculated as `(baseline_emissions_mtco2 / total_gross_emissions) * 100`
- **Example**: `51.96` for Energy sector
- **Unit**: Percentage of total emissions (%)
- **Notes**:
  - **Auto-calculation**:
    - For emitters: Share = (sector_baseline / sum_of_all_positive_baselines) × 100
    - For sinks: Always 0% (not counted in share)
  - **Manual override**: Set this field to override calculation
  - Used for weighted reduction calculations

---

## Complete Examples

### Forecast Mode Example

```json
{
  "Vietnam": {
    "country": "Vietnam",
    "mode": "forecast",
    "target_year": 2050,
    "baseline_year": 2025,
    "baseline_emissions_gtco2": 0.45,
    "policy_actions": [
      {
        "sector": "Energy",
        "action_name": "Coal phase-out & Renewable expansion (solar, wind, offshore wind)",
        "baseline_year": 2025,
        "baseline_emissions_mtco2": 225.0,
        "reduction_target_pct": 0,
        "implementation_years": [2026, 2027, 2028, 2029, 2030],
        "yearly_improvement_pct": [3.0, 6.0, 10.0, 15.0, 22.0],
        "status": "On track"
      },
      {
        "sector": "Forestry",
        "action_name": "Reforestation & Forest conservation (carbon sink expansion)",
        "baseline_year": 2025,
        "baseline_emissions_mtco2": -17.0,
        "removal_increase_pct": 0,
        "implementation_years": [2026, 2027, 2028, 2029, 2030],
        "yearly_improvement_pct": [5.0, 10.0, 16.0, 23.0, 32.0],
        "status": "Ahead"
      }
    ]
  }
}
```

**Forecast Mode Notes:**
- No `target_reduction_pct` at country level
- Policy actions have `reduction_target_pct: 0` to trigger historical trend extrapolation
- System projects emissions based on policy momentum from 2026-2030 data
- Auto-extrapolates from 2030 to 2050 using learned trends

---

### Target Mode Example

```json
{
  "Vietnam": {
    "country": "Vietnam",
    "mode": "target",
    "target_year": 2050,
    "baseline_year": 2025,
    "baseline_emissions_gtco2": 0.45,
    "target_reduction_pct": 50.0,
    "policy_actions": [
      {
        "sector": "Energy",
        "action_name": "Coal phase-out & Renewable expansion",
        "baseline_year": 2025,
        "baseline_emissions_mtco2": 225.0,
        "reduction_target_pct": 65.0,
        "implementation_years": [2026, 2030, 2035, 2040, 2045, 2050],
        "yearly_improvement_pct": [5.0, 15.0, 30.0, 45.0, 55.0, 65.0],
        "status": "On track"
      },
      {
        "sector": "Forestry",
        "action_name": "Reforestation & Forest conservation",
        "baseline_year": 2025,
        "baseline_emissions_mtco2": -17.0,
        "removal_increase_pct": 300.0,
        "implementation_years": [2026, 2030, 2035, 2040, 2045, 2050],
        "yearly_improvement_pct": [20.0, 60.0, 120.0, 180.0, 240.0, 300.0],
        "status": "Ahead"
      }
    ]
  }
}
```

**Target Mode Notes:**
- Has `target_reduction_pct: 50.0` at country level (reduce by 50%)
- Policy actions have specific targets (Energy: 65%, Forestry: 300% increase)
- System calculates pathway to achieve 50% overall reduction
- Policy data extends to 2050 to match target_year (no extrapolation needed)

---

## Calculation Logic

### Share Percentage (Auto-calculated)
```
For emitters (baseline > 0):
  share_pct = (baseline_emissions_mtco2 / gross_emissions) × 100
  where gross_emissions = sum of all positive baselines

For sinks (baseline < 0):
  share_pct = 0 (not included in share calculation)
```

**Example:**
- Energy: 225 MtCO2
- Transport: 90 MtCO2
- Industry: 80 MtCO2
- Agriculture: 38 MtCO2
- Forestry: -17 MtCO2

```
Gross emissions = 225 + 90 + 80 + 38 = 433 MtCO2
Energy share = (225 / 433) × 100 = 51.96%
Transport share = (90 / 433) × 100 = 20.79%
Forestry share = 0% (it's a sink)
```

---

### Annual Reduction Calculation

**For a given year:**

1. **Within data range**: Interpolate between data points
2. **Before data range**: Return 0 (no action yet)
3. **After data range**: Auto-extrapolate
   - **Forecast Mode** (target = 0): Use historical trend with damping
   - **Target Mode** (target > 0): Linear extrapolation toward target

---

### Total Country Reduction

```
For each sector:
  If emitter (baseline > 0):
    weighted_reduction = (share_pct / 100) × (reduction_pct / 100)

  If sink (baseline < 0):
    additional_removal = abs(baseline) × (increase_pct / 100)
    equivalent_reduction = additional_removal / country_baseline_mtco2

Total reduction fraction = sum(all weighted reductions + sink contributions)
```

---

## Validation Rules

### Country Level
- ✅ `country` must be non-empty string
- ✅ `mode` must be "forecast" or "target"
- ✅ `target_year` > `baseline_year`
- ✅ `baseline_emissions_gtco2` must be positive
- ✅ `policy_actions` must have at least 1 item
- ✅ If `mode == "target"`, must have `target_reduction_pct`
- ✅ If `mode == "forecast"`, `target_reduction_pct` is optional

### Policy Action Level
- ✅ All required fields must be present
- ✅ `implementation_years` and `yearly_improvement_pct` must have same length
- ✅ `implementation_years` must be in ascending order
- ✅ Must have either `reduction_target_pct` or `removal_increase_pct`
- ✅ If `baseline_emissions_mtco2 < 0`, should use `removal_increase_pct`
- ✅ `status` should be a valid status string

---

## Tips and Best Practices

### For Forecast Mode
1. Set `reduction_target_pct: 0` at policy action level to use historical trends
2. Provide at least 3-5 years of implementation data for reliable trend calculation
3. System will auto-extrapolate with damping (slowing growth) for long-term projections
4. Works best when policy data shows clear trends

### For Target Mode
1. Set specific `reduction_target_pct` or `removal_increase_pct` for each action
2. Ensure sector targets are realistic and aligned with country target
3. Provide milestone data points across the entire period if possible
4. System will interpolate/extrapolate to fill gaps

### For Both Modes
1. Use consistent baseline_year across country and all actions
2. Ensure sum of sector baselines ≈ country baseline (accounting for sinks)
3. Use negative values for carbon sinks (forestry, ocean, etc.)
4. Keep yearly_improvement_pct cumulative (increasing over time)
5. Provide more data points for better accuracy

### Common Mistakes to Avoid
- ❌ Mixing up MtCO2 and GtCO2 units
- ❌ Using target_reduction_pct in forecast mode
- ❌ Forgetting negative sign for carbon sinks
- ❌ Making yearly_improvement_pct non-cumulative
- ❌ Having implementation_years out of order
- ❌ Mismatched array lengths for years and percentages

---

## Units Reference

| Field | Unit | Example | Conversion |
|-------|------|---------|------------|
| baseline_emissions_gtco2 | GtCO2 | 0.45 | 1 GtCO2 = 1000 MtCO2 |
| baseline_emissions_mtco2 | MtCO2 | 225.0 | 1 MtCO2 = 0.001 GtCO2 |
| reduction_target_pct | % | 60.0 | Percentage (0-100) |
| removal_increase_pct | % | 200.0 | Percentage (0-1000+) |
| share_pct | % | 51.96 | Percentage (0-100) |
| yearly_improvement_pct | % | 22.0 | Percentage (0-100+) |

---

## Need Help?

- Check validation errors in the UI when uploading
- Run `test_forecast_target_modes.py` to test your JSON
- See `data/vietnam_forecast_mode.json` for a working example
- Review `FORECAST_TARGET_MODE_SEPARATION.md` for mode differences
