# 📘 Policy Commitments Configuration Guide (Updated)

## 🎯 Overview

Hệ thống Policy Commitments tracking đã được cập nhật với các tính năng mới:
- ✅ **Upload JSON file** thay vì load file có sẵn
- ✅ **Field `share_pct`** để tính weighted reduction chính xác hơn
- ✅ **Field `target_reduction_pct`** thay vì hardcode target emissions
- ✅ **Rename `net_zero_year` → `target_year`** linh hoạt hơn

---

## 📁 Cấu trúc JSON File

### **Root Level**
```json
{
  "CountryName1": { ... },
  "CountryName2": { ... }
}
```

### **Country Object**

```json
{
  "country": "Vietnam",                    // Tên quốc gia
  "target_year": 2050,                     // Năm đạt mục tiêu (thay vì net_zero_year)
  "target_reduction_pct": 100.0,           // % giảm so với baseline (0-100)
  "baseline_year": 2020,                   // Năm baseline
  "baseline_emissions_gtco2": 0.42,        // Phát thải baseline (GtCO₂)
  "policy_actions": [ ... ]                // Array các policy actions
}
```

#### **Thay đổi so với version cũ:**
| Field cũ | Field mới | Lý do |
|----------|-----------|-------|
| `net_zero_year` | `target_year` | Không phải lúc nào cũng là Net Zero (có thể giảm 80%, 50%...) |
| *(không có)* | `target_reduction_pct` | Cho phép nhập % giảm thay vì hardcode 0 emissions |

---

### **Policy Action Object**

```json
{
  "sector": "Energy",
  "action_name": "Coal phase-out & Renewable expansion",
  "baseline_year": 2020,
  "baseline_emissions_mtco2": 210.0,
  "share_pct": 50.0,                       // [MỚI] Tỷ trọng sector
  "reduction_target_pct": 50.0,
  "implementation_years": [2024, 2025, 2026, 2027, 2028, 2029, 2030],
  "yearly_improvement_pct": [4.0, 8.5, 14.0, 20.0, 26.5, 33.0, 40.0],
  "status": "On track"
}
```

#### **Field Mới: `share_pct`**

**Ý nghĩa:** Tỷ trọng phần trăm của sector trong **tổng phát thải GROSS** (chỉ emitters).

**⚠️ LƯU Ý QUAN TRỌNG: Chỉ tính cho EMITTERS (baseline > 0)**

**Công thức:**
```
gross_emissions = sum(baseline_mtco2 for sectors where baseline > 0)
share_pct = (sector_baseline_mtco2 / gross_emissions) × 100
```

**Ví dụ với Vietnam:**
```
EMITTERS (baseline > 0):
  Energy:      210 MtCO₂
  Transport:    85 MtCO₂
  Industry:     75 MtCO₂
  Agriculture:  35 MtCO₂
  ────────────────────
  Gross Total: 405 MtCO₂

SINKS (baseline < 0):
  Forestry:    -15 MtCO₂  ← KHÔNG tính vào gross!

Calculation:
  Energy share:      210 / 405 × 100 = 51.85%
  Transport share:    85 / 405 × 100 = 20.99%
  Industry share:     75 / 405 × 100 = 18.52%
  Agriculture share:  35 / 405 × 100 = 8.64%
  Forestry share:     0.00%  ← Set to 0 (is a sink)
```

**Tại sao cần `share_pct`?**
- ✅ Tính weighted reduction chính xác hơn
- ✅ Sector có tỷ trọng lớn có impact nhiều hơn lên tổng reduction
- ✅ Giúp forecast pathway realistic hơn
- ✅ Tách riêng emitters và sinks để tính đúng logic

**📚 Chi tiết:** Xem `CARBON_SINK_CALCULATION.md` để hiểu cách xử lý carbon sinks.

---

## 🔢 Tính toán Weighted Reduction

### **⚠️ QUAN TRỌNG: Phân biệt EMITTERS và SINKS**

Hệ thống tính toán **tách riêng** hai loại sector:

1. **EMITTERS** (baseline > 0): Phát thải CO2, mục tiêu GIẢM
2. **SINKS** (baseline < 0): Hấp thụ CO2, mục tiêu TĂNG removal

### **Logic mới (corrected - separates emitters and sinks):**

```python
def calculate_annual_reduction_fraction(year: int) -> float:
    emitters_reduction = 0.0
    sinks_contribution = 0.0

    for action in policy_actions:
        reduction_pct = action.get_reduction_for_year(year)

        if action.baseline_emissions_mtco2 > 0:
            # EMITTER: Use share_pct weighted reduction
            weighted = (action.share_pct / 100) * (reduction_pct / 100)
            emitters_reduction += weighted

        elif action.baseline_emissions_mtco2 < 0:
            # SINK: Calculate additional removal in MtCO2
            baseline_removal = abs(action.baseline_emissions_mtco2)
            additional_removal = baseline_removal * (reduction_pct / 100)

            # Convert to equivalent reduction fraction
            baseline_mtco2 = country_baseline_gtco2 * 1000
            sinks_contribution += additional_removal / baseline_mtco2

    # Total = emitters reduction + sink contribution
    return min(emitters_reduction + sinks_contribution, 1.0)
```

### **Ví dụ tính toán: Năm 2030**

**Scenario:**
- Energy (51.85% share): 40% reduction
- Transport (20.99% share): 32% reduction
- Industry (18.52% share): 30% reduction
- Agriculture (8.64% share): 28% reduction
- Forestry (0% share, -15 MtCO2 baseline): 62% sink increase

**Bước 1: Tính reduction từ EMITTERS**
```
Energy:      51.85% × 40% = 20.74%
Transport:   20.99% × 32% = 6.72%
Industry:    18.52% × 30% = 5.56%
Agriculture:  8.64% × 28% = 2.42%
────────────────────────────────
Emitters Total:        35.44%
```

**Bước 2: Tính contribution từ SINKS**
```
Forestry baseline removal: 15 MtCO2
Forestry improvement: 62%
Additional removal: 15 × (62/100) = 9.3 MtCO2

Convert to fraction:
  Baseline: 420 MtCO2
  Sink contribution: 9.3 / 420 = 2.21%
```

**Bước 3: Tổng cộng**
```
Total Reduction = 35.44% + 2.21% = 37.65%
```

✅ **Logic chính xác và semantically correct!**

**📚 Chi tiết đầy đủ:** Xem `CARBON_SINK_CALCULATION.md` để hiểu sâu về cách tính.

---

## 📤 Workflow Upload JSON trong UI

### **Bước 1: Mở expander "🎯 Add Net Zero / Commitment Scenario"**

### **Bước 2: Upload JSON file**
```
📤 Upload Policy Commitment Data (Optional)
[Drag & Drop JSON file here]
```

### **Bước 3: Validation tự động**
```
✓ JSON structure is valid
📋 Found 1 country/countries: Vietnam
```

### **Bước 4: Select country**
```
Select country from uploaded data: [Vietnam ▼]
```

### **Bước 5: Enable tracking**
```
☑ 📋 Enable detailed policy tracking for Vietnam
✓ Loaded 5 policy actions for Vietnam

Target Year: 2050
Target Reduction: 100.0%
Baseline Year: 2020
```

### **Bước 6: Choose pathway**
```
Reduction pathway shape:
○ Policy-driven (based on actual actions)  ← MỚI
○ Exponential decay (realistic)
○ Linear reduction
○ S-curve (slow-fast-slow)
○ Custom milestones
```

### **Bước 7: Generate scenario**
```
🚀 Generate commitment scenario
```

---

## 🎨 Hiển thị trên UI

### **Sector Tabs (với share_pct):**

```
┌─────────────────────────────────────────────────────────┐
│  Energy (50.0%)  |  Transport (20.2%)  |  Industry...   │
├─────────────────────────────────────────────────────────┤
│  [Active Tab: Energy (50.0%)]                           │
│                                                          │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌───┐ │
│  │ Baseline   │  │ Sector     │  │ Reduction  │  │...│ │
│  │ Emissions  │  │ Share      │  │ Target     │  └───┘ │
│  │ 210.0 Mt   │  │ 50.00%     │  │ 50.0%      │        │
│  └────────────┘  └────────────┘  └────────────┘        │
│                      ↑ NEW METRIC                        │
└─────────────────────────────────────────────────────────┘
```

### **Metrics hiển thị:**
| Metric | Mô tả | Ví dụ |
|--------|-------|-------|
| **Baseline Emissions** | Phát thải baseline của sector | 210.0 Mt CO₂ |
| **Sector Share** | Tỷ trọng % trong tổng (MỚI) | 50.00% |
| **Reduction Target** | Mục tiêu % giảm | 50.0% |
| **Progress (2030)** | Tiến độ hiện tại | 40.0% (+6.5% YoY) |

---

## 🧮 Tính `share_pct` cho quốc gia của bạn

### **Bước 1: Tổng hợp baseline emissions**
```
Total = Σ sector_baseline_mtco2
```

**Lưu ý với carbon sinks (e.g., Forestry):**
- Nếu `baseline_emissions_mtco2 < 0` → Đây là carbon sink
- Tính share dựa trên absolute value hoặc tách riêng

### **Bước 2: Tính share từng sector**
```
share_pct = (sector_baseline / total_baseline) × 100
```

### **Bước 3: Validate tổng**
```
Σ share_pct ≈ 100%  (± 0.5% tolerance)
```

### **Ví dụ: USA**

Giả sử USA baseline = 5.0 GtCO₂ (2020):
- Energy: 2000 MtCO₂ → 2000/5000 = **40.0%**
- Transport: 1800 MtCO₂ → 1800/5000 = **36.0%**
- Industry: 900 MtCO₂ → 900/5000 = **18.0%**
- Agriculture: 600 MtCO₂ → 600/5000 = **12.0%**
- Forestry (sink): -300 MtCO₂ → Tách riêng hoặc -6.0%
- **Net baseline:** 5000 - 300 = 4700 MtCO₂

**Option 1 (exclude sinks from share):**
```json
{
  "sector": "Energy",
  "share_pct": 42.55,  // 2000/4700
  ...
},
{
  "sector": "Forestry",
  "share_pct": -6.38,  // -300/4700 (negative share)
  ...
}
```

**Option 2 (relative to gross):**
```json
{
  "sector": "Energy",
  "share_pct": 40.0,   // 2000/5000
  ...
},
{
  "sector": "Forestry",
  "baseline_emissions_mtco2": -300,
  "share_pct": 6.0,    // Treat as positive contribution
  ...
}
```

---

## ✅ Validation Checklist

Khi tạo JSON file mới:

- [ ] **Country level:**
  - [ ] `target_year` > `baseline_year`
  - [ ] `target_reduction_pct` trong khoảng 0-100
  - [ ] `baseline_emissions_gtco2` match với sum of sectors (± tolerance)
  - [ ] `country` field match với key trong root object

- [ ] **Policy actions:**
  - [ ] Mỗi sector có `share_pct` > 0 (hoặc < 0 nếu là sink)
  - [ ] Sum of `share_pct` ≈ 100%
  - [ ] `implementation_years` và `yearly_improvement_pct` cùng length
  - [ ] `yearly_improvement_pct` tăng dần (cumulative)
  - [ ] `status` là một trong: "On track", "Behind schedule", "Ahead"

- [ ] **Logical consistency:**
  - [ ] `reduction_target_pct` <= max(`yearly_improvement_pct`)
  - [ ] Nếu `status = "Ahead"` → progress > expected
  - [ ] Nếu `status = "Behind schedule"` → progress < expected

---

## 🔧 Troubleshooting

### ❌ Error: "Sum of share_pct is not 100%"
**Cause:** Các share_pct không tổng bằng 100%
**Fix:** Recalculate hoặc normalize:
```python
shares = [50.0, 20.24, 17.86, 8.33, 3.57]
total = sum(shares)  # 100.0
normalized = [s / total * 100 for s in shares]
```

### ❌ Error: "Arrays have different length"
**Cause:** `implementation_years` và `yearly_improvement_pct` khác length
**Fix:** Ensure cùng số phần tử:
```json
"implementation_years": [2024, 2025, 2026],
"yearly_improvement_pct": [5.0, 12.0, 20.0]  // 3 elements
```

### ❌ Error: "yearly_improvement_pct not monotonic"
**Cause:** Giá trị giảm thay vì tăng
**Fix:** Đảm bảo cumulative:
```json
// ❌ Wrong
"yearly_improvement_pct": [5.0, 12.0, 10.0, 20.0]  // 12→10 giảm!

// ✅ Correct
"yearly_improvement_pct": [5.0, 12.0, 15.0, 20.0]  // Luôn tăng
```

### ❌ Warning: "Country not found in JSON"
**Cause:** Key trong JSON không match với selection
**Fix:** Check spelling và case-sensitive:
```json
{
  "vietnam": { ... }  // ❌ lowercase
  "Vietnam": { ... }  // ✅ Correct
}
```

---

## 📊 Examples

### **Example 1: Vietnam (100% Net Zero by 2050)**
✅ File: `data/country_commitments.json`

### **Example 2: Template with comments**
✅ File: `data/country_commitments_template.json`

### **Example 3: Tạo file mới cho USA**

```json
{
  "USA": {
    "country": "USA",
    "target_year": 2050,
    "target_reduction_pct": 80.0,  // Giảm 80%, không phải Net Zero
    "baseline_year": 2005,
    "baseline_emissions_gtco2": 6.13,
    "policy_actions": [
      {
        "sector": "Electricity",
        "action_name": "Clean Energy Standard & Grid modernization",
        "baseline_year": 2005,
        "baseline_emissions_mtco2": 2400.0,
        "share_pct": 39.15,
        "reduction_target_pct": 100.0,
        "implementation_years": [2024, 2026, 2028, 2030],
        "yearly_improvement_pct": [15.0, 35.0, 55.0, 75.0],
        "status": "On track"
      },
      // ... thêm sectors khác
    ]
  }
}
```

---

## 🚀 Quick Start

1. **Copy template:**
   ```bash
   cp data/country_commitments_template.json data/my_country.json
   ```

2. **Edit fields:** Replace placeholders với data thực

3. **Calculate share_pct:**
   ```python
   total = sum(sector['baseline_emissions_mtco2'] for sector in actions)
   for action in actions:
       action['share_pct'] = (action['baseline_emissions_mtco2'] / total) * 100
   ```

4. **Validate:**
   - Upload file vào UI
   - Kiểm tra validation messages

5. **Test:**
   - Generate policy-driven scenario
   - So sánh với exponential pathway

---

## 📖 API Reference

### **Functions:**

```python
load_country_commitment_from_json(json_content: str, country: str) -> CountryCommitment
```
Load commitment từ JSON string.

```python
validate_commitment_json(json_content: str) -> tuple[bool, str]
```
Validate structure, returns (is_valid, error_message).

```python
get_countries_from_json(json_content: str) -> list[str]
```
Extract list of country names.

### **Properties:**

```python
CountryCommitment.target_emissions_gtco2 -> float
```
Tính target emissions từ baseline và reduction %:
```python
= baseline_emissions_gtco2 * (1 - target_reduction_pct / 100)
```

---

## 💡 Best Practices

1. **Naming conventions:**
   - Country names: Title case ("Vietnam", "USA")
   - Sectors: Standardized ("Energy", "Transport", "Industry")

2. **Data sources:**
   - Cite sources trong `action_name`
   - Example: "Coal phase-out (per DPPA 2023)"

3. **Progress tracking:**
   - Update quarterly or annually
   - Keep historical data

4. **Version control:**
   - Date-stamp JSON files: `vietnam_2025Q1.json`
   - Git commit với meaningful messages

---

## 📞 Support

- **Documentation:** `POLICY_COMMITMENTS_GUIDE.md` (this file)
- **Template:** `data/country_commitments_template.json`
- **Example:** `data/country_commitments.json` (Vietnam)
- **Test script:** `test_vietnam_policy.py`
