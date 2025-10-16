# 📊 Policy-Driven Pathway Shape Explained

## 🎯 Tại sao Policy-Driven khác Exponential Decay?

### **Trước đây (Version cũ):**

❌ **Vấn đề:**
```python
# Old logic
policy_acceleration = 1.0 + reduction_fraction * 0.5
adjusted_value = start_emissions * np.exp(-k_base * years_elapsed * policy_acceleration)
```

**Kết quả:** Vẫn dùng exponential base với acceleration → Shape giống exponential!

**So sánh (old):**
```
Year  Policy-Driven  Exponential  Difference
2024     0.439         0.440        -0.001  ← Gần như giống nhau!
2029     0.186         0.212        -0.026
2034     0.077         0.103        -0.026
```

---

### **Bây giờ (Version mới):**

✅ **Giải pháp:**
```python
# New logic - Direct application from baseline
reduction_fraction = country_commitment.calculate_annual_reduction_fraction(year)
policy_emissions = baseline_emissions * (1 - reduction_fraction)
```

**Kết quả:** Tạo **staged/stepped reduction** theo policy milestones!

**So sánh (new):**
```
Year  Policy-Driven  Exponential  Difference
2024     0.406         0.440        -0.034  ← Nhanh hơn ban đầu
2029     0.294         0.212        +0.082  ← CHẬM hơn! (staged approach)
2034     0.195         0.103        +0.093  ← Maintain cao hơn
2039     0.109         0.050        +0.059
2044     0.044         0.024        +0.020
2049     0.000         0.012        -0.012  ← Đều đạt target
```

---

## 📈 Visualization Shape

### **Exponential Decay:**
```
Emissions
  │
  │ ╲                   ← Smooth curve
  │  ╲
  │   ╲╲
  │     ╲╲
  │       ╲╲
  │         ╲╲
  │           ╲╲____
  │                  ╲___
  └────────────────────────── Time
```
**Đặc điểm:** Giảm nhanh ban đầu, chậm dần về cuối.

---

### **Policy-Driven (Staged):**
```
Emissions
  │
  │ ╲                   ← Stage 1 (2024-2030): Early action
  │  ╲╲
  │    ╲___
  │        ╲            ← Stage 2 (2031-2040): Accelerated
  │         ╲╲╲
  │            ╲___
  │                ╲    ← Stage 3 (2041-2050): Final push
  │                 ╲╲╲╲
  └────────────────────────── Time
```
**Đặc điểm:** Stepped reduction theo policy milestones, có plateau giữa các stages.

---

## 🔢 Dữ liệu đã update

### **Data structure (extended to 2050):**

```json
{
  "implementation_years": [2024, 2025, 2026, 2027, 2028, 2029, 2030, 2035, 2040, 2045, 2050],
  "yearly_improvement_pct": [4.0, 8.5, 14.0, 20.0, 26.5, 33.0, 40.0, 60.0, 80.0, 92.0, 100.0]
}
```

**Key milestones:**
- **2030:** 40% reduction (Early commitment phase)
- **2035:** 60% reduction (Jump +20% - acceleration)
- **2040:** 80% reduction (Jump +20% - maintained pace)
- **2045:** 92% reduction (Jump +12% - final push)
- **2050:** 100% reduction (Net Zero achieved)

---

## 🤔 Nếu data chỉ đến 2030 mà target 2050?

### **Scenario:** Data stops at 2030 (40%), target is 2050 (100%)

```json
"implementation_years": [2024, 2025, 2026, 2027, 2028, 2029, 2030],
"yearly_improvement_pct": [4.0, 8.5, 14.0, 20.0, 26.5, 33.0, 40.0]
```

### **Logic xử lý trong code:**

#### **1. PolicyAction.get_reduction_for_year()**
```python
def get_reduction_for_year(self, year: int) -> float:
    # If year is after last implementation year, use last value
    if year >= self.implementation_years[-1]:
        return self.yearly_improvement_pct[-1]  # ← Returns 40.0 for 2031-2050
```

**Kết quả:**
- 2024-2030: Follow data (4% → 40%)
- 2031-2050: **Plateau tại 40%** (không tăng thêm)

---

#### **2. _policy_driven_pathway() calculation**
```python
for year in [2024...2050]:
    reduction_fraction = calculate_annual_reduction_fraction(year)
    # 2024-2030: Tăng dần
    # 2031-2050: Constant tại ~0.36 (weighted average of 40% across sectors)

    policy_emissions = baseline_emissions * (1 - reduction_fraction)
    # 2024: 0.42 * (1 - 0.034) = 0.406 GtCO₂
    # 2030: 0.42 * (1 - 0.364) = 0.267 GtCO₂
    # 2031-2050: 0.42 * (1 - 0.364) = 0.267 GtCO₂ ← STUCK!
```

**Vấn đề:**
```
Emissions
  │
  │ ╲╲╲╲                ← Giảm đến 2030
  │     ╲_______________  ← PLATEAU 2031-2050 (không giảm thêm!)
  │
  │                     Target (0.0) not reached! ❌
  └────────────────────────── Time
      2030            2050
```

---

### **Giải pháp:**

#### **Option 1: Extend data to 2050 (✅ ĐÃ IMPLEMENT)**

Update JSON với milestones đến 2050:
```json
"implementation_years": [..., 2030, 2035, 2040, 2045, 2050],
"yearly_improvement_pct": [..., 40.0, 60.0, 80.0, 92.0, 100.0]
```

**Kết quả:** Shape follows milestones đến cuối!

---

#### **Option 2: Interpolation logic (Chưa implement)**

Nếu không có data sau 2030, code có thể tự extrapolate:

```python
def get_reduction_for_year(self, year: int) -> float:
    # ... existing logic ...

    # If year is past last data point, extrapolate to target
    if year > self.implementation_years[-1]:
        last_year = self.implementation_years[-1]
        last_reduction = self.yearly_improvement_pct[-1]

        years_to_target = target_year - last_year
        years_elapsed = year - last_year

        # Linear interpolation to reduction target
        slope = (self.reduction_target_pct - last_reduction) / years_to_target
        return min(last_reduction + slope * years_elapsed, self.reduction_target_pct)
```

**Ví dụ:**
- 2030: 40% (last data)
- Target 2050: 100% (from JSON)
- 2040: 40% + (100-40)/20 * 10 = **70%** (interpolated)

---

### **Hiện tại:** ✅ **Extend data to 2050**

Cách này realistic hơn vì:
- Governments thường có roadmap dài hạn
- Policy milestones được plan trước
- Transparency cao hơn

---

## 🔒 Disabled inputs khi enable Policy Tracking

### **Lý do disable:**

Khi upload JSON và enable policy tracking, các parameters sau bị **lock** vì đã define trong JSON:

| Input | Locked? | Lý do |
|-------|---------|-------|
| **Target year** | 🔒 Yes | Từ JSON `target_year` |
| **Commitment type** | 🔒 Yes | Force "Percentage Reduction" |
| **Reduction %** | 🔒 Yes | Từ JSON `target_reduction_pct` |
| **Pathway shape** | 🔒 Yes | Force "Policy-driven" |
| **Milestones** | ❌ Hidden | Không cần (có sẵn trong actions) |

---

### **UI hiển thị:**

```
🔒 Policy tracking enabled: Using parameters from uploaded JSON (Vietnam).
    Target year, reduction %, and pathway are fixed based on policy data.

┌────────────────────────────────────────────┐
│ Target year:     2050  🔒 (disabled)       │
│ Commitment type: Percentage Reduction 🔒   │
│ Reduction %:     100.0%  🔒 (disabled)     │
│ Pathway shape:   Policy-driven 🔒          │
└────────────────────────────────────────────┘
```

---

### **User workflow:**

1. **Upload JSON** → System validates
2. **Select country** → Load commitment data
3. **Enable tracking** → All inputs locked 🔒
4. **View sector breakdown** → See detailed policy actions
5. **Generate scenario** → Only "🚀 Generate" button active

**Không thể override** vì:
- JSON là source of truth
- Đảm bảo consistency với policy data
- Tránh user nhập conflict values

---

## 📊 Comparison Summary

### **Key Differences:**

| Aspect | Exponential Decay | Policy-Driven (New) |
|--------|-------------------|---------------------|
| **Base logic** | Mathematical curve | Policy milestone-based |
| **Shape** | Smooth exponential | Staged/stepped |
| **Early phase (2024-2030)** | Slow start | Faster (policy push) |
| **Mid phase (2031-2040)** | Accelerating | Plateau → Jump (staged) |
| **Late phase (2041-2050)** | Slowing down | Accelerate to target |
| **Data dependency** | None (pure math) | Requires policy milestones |
| **Realism** | Generic assumption | Based on actual policies |

---

### **Visual Comparison:**

```
Emissions (GtCO₂)
0.5 ┤
    │  ╲ Exponential (smooth)
0.4 ┤   ╲
    │    ╲╲
0.3 ┤      ╲╲___
    │     Policy╲  (staged)
0.2 ┤            ╲___
    │                 ╲___
0.1 ┤                     ╲╲╲
    │                         ╲___
0.0 ┼─────────────────────────────
    2024  2030  2035  2040  2045  2050

    ← Policy maintains higher emissions mid-term
      (realistic transition time)

    → Both reach target by 2050
```

---

## 🎯 Takeaways

1. **Policy-driven shape khác biệt rõ ràng** với exponential:
   - Staged reduction theo milestones
   - Không smooth như exponential
   - Realistic hơn với policy implementation

2. **Data structure extended đến target year:**
   - Tránh plateau effect
   - Shape rõ ràng đến cuối
   - User có thể update progress theo năm

3. **UI inputs disabled khi policy tracking:**
   - Đảm bảo consistency
   - JSON là single source of truth
   - Tránh conflicts

4. **Calculation dùng weighted reduction:**
   - `share_pct` quan trọng
   - Sectors lớn có impact nhiều hơn
   - Kết quả chính xác hơn

---

## 📝 Next Steps

### **Để customize cho quốc gia khác:**

1. **Extend implementation_years:**
   ```json
   "implementation_years": [2024, ..., 2050]
   ```

2. **Define milestones:**
   ```json
   "yearly_improvement_pct": [5, ..., 95]  // Progressive increase
   ```

3. **Set share_pct correctly:**
   ```
   Total all shares = 100% (approximately)
   ```

4. **Upload và test:**
   - Validate JSON
   - Check shape khác với exponential
   - Compare visually

---

## 🔍 Debugging Tips

### **Shape vẫn giống exponential?**

**Check:**
1. Data có extend đến target year không?
2. Milestones có staged progression không? (không nên linear)
3. share_pct có đúng không?

**Fix:** Add more aggressive milestones:
```json
// Instead of linear
"yearly_improvement_pct": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

// Try staged
"yearly_improvement_pct": [5, 10, 18, 30, 45, 65, 78, 88, 95, 100]
```

---

### **Emissions không reach target?**

**Check:**
1. Last milestone = target_reduction_pct?
2. Last implementation_year = target_year?

**Fix:**
```json
{
  "target_year": 2050,
  "target_reduction_pct": 100.0,
  "implementation_years": [..., 2050],  // ← Must include target_year
  "yearly_improvement_pct": [..., 100.0] // ← Must reach target %
}
```

---

## 📚 References

- **Implementation:** `logic/forecast.py::_policy_driven_pathway()`
- **Data structure:** `logic/policy_commitments.py::PolicyAction`
- **Sample data:** `data/country_commitments.json`
- **UI logic:** `app.py` (lines 176-290)
- **Tests:** `test_vietnam_policy.py`
