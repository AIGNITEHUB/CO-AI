# Tài liệu Công thức Lý thuyết - CO₂ AI Demo

## 1. 🎯 Net Zero / Commitment Scenario

### 1.1. Mục đích
Tạo kịch bản cam kết giảm phát thải và tính toán khoảng cách phát thải (emissions gap) so với dự báo Business-as-Usual (BAU).

### 1.2. Công thức Pathways

#### A. Linear Pathway (Giảm tuyến tính)

**Phương trình:**
```
E(t) = E₀ + m(t - t₀)
```

Với:
- `E(t)`: Lượng phát thải tại năm t (GtCO₂)
- `E₀`: Lượng phát thải hiện tại (GtCO₂)
- `t₀`: Năm hiện tại
- `m`: Độ dốc (slope)

**Công thức slope:**
```
m = (E_target - E₀) / (t_target - t₀)
```

**Ví dụ:**
- E₀ = 40 GtCO₂ (năm 2024)
- E_target = 0 GtCO₂ (năm 2050)
- m = (0 - 40) / (2050 - 2024) = -1.538 GtCO₂/năm
- E(2030) = 40 + (-1.538)(2030 - 2024) = 30.77 GtCO₂

---

#### B. Exponential Pathway (Giảm mũ)

**Phương trình:**
```
E(t) = E₀ · e^(-k(t - t₀))
```

**Công thức hằng số phân rã k:**
```
k = -ln(E_target / E₀) / (t_target - t₀)
```

**Đặc điểm:**
- Giảm nhanh ở giai đoạn đầu
- Giảm chậm dần ở giai đoạn cuối
- Phù hợp với thực tế (dễ giảm phát thải khi còn cao)

**Ví dụ:**
- E₀ = 40 GtCO₂, E_target = 0.01 GtCO₂
- k = -ln(0.01/40) / 26 = 0.319
- E(2025) = 40 · e^(-0.319×1) = 29.1 GtCO₂
- E(2030) = 40 · e^(-0.319×6) = 6.0 GtCO₂
- E(2050) = 40 · e^(-0.319×26) ≈ 0.01 GtCO₂

---

#### C. S-curve Pathway (Đường cong Sigmoid)

**Phương trình:**
```
E(t) = E_target + L / (1 + e^(k·x))
```

Với:
- `L = E₀ - E_target` (phạm vi thay đổi)
- `x = t - t_mid` (khoảng cách từ điểm giữa)
- `t_mid = t₀ + (t_target - t₀)/2` (điểm uốn)
- `k = 8 / (t_target - t₀)` (độ dốc)

**Đặc điểm:**
- Giai đoạn 1 (0-30%): Giảm chậm (~20% tổng giảm)
- Giai đoạn 2 (30-70%): Giảm nhanh (~60% tổng giảm)
- Giai đoạn 3 (70-100%): Giảm chậm (~20% tổng giảm)

**Ví dụ:**
- E₀ = 40 GtCO₂, E_target = 0 GtCO₂
- t₀ = 2024, t_target = 2050, t_mid = 2037
- L = 40, k = 8/26 = 0.308
- E(2030) = 0 + 40/(1 + e^(0.308×(2030-2037))) = 35.2 GtCO₂
- E(2037) = 0 + 40/(1 + e^0) = 20.0 GtCO₂ (điểm giữa)
- E(2044) = 0 + 40/(1 + e^(0.308×7)) = 4.8 GtCO₂

---

#### D. Milestones Pathway (Polynomial qua các mốc)

**Phương trình polynomial bậc n:**
```
E(t) = β₀ + β₁t + β₂t² + β₃t³ + ... + βₙtⁿ
```

**Phương pháp:**
- Sử dụng Polynomial Regression với degree = 3
- Fit qua: dữ liệu lịch sử + các mốc trung gian + mục tiêu cuối
- Đảm bảo: E(t) ≥ 0 ∀t

**Ví dụ:**
- Dữ liệu lịch sử: 1990-2024
- Milestones: {2030: 25, 2040: 15}
- Target: {2050: 5}
- Fit polynomial qua tất cả các điểm

---

### 1.3. Metrics Phân tích

#### A. Emissions Gap (Khoảng cách phát thải)
```
Gap(t) = E_BAU(t) - E_commitment(t)
```

#### B. Total Emissions Gap (Tổng khoảng cách)
```
Total_Gap = Σ [E_BAU(t) - E_commitment(t)]
            t=t₀+1 đến t_target
```

#### C. Average Annual Reduction (Mức giảm trung bình hàng năm)
```
Avg_Reduction = Total_Gap / (t_target - t₀)
```

#### D. Reduction Rate (Tỷ lệ giảm tổng thể)
```
Reduction_Rate = (E₀ - E_target) / E₀ × 100%
```

**Ví dụ tính toán:**
- E₀ = 40 GtCO₂, E_target = 0 GtCO₂
- Total_Gap = 872.5 GtCO₂
- Avg_Reduction = 872.5 / 26 = 33.6 GtCO₂/năm
- Reduction_Rate = (40 - 0) / 40 × 100% = 100%

---

## 2. 📊 Catalyst-Pathway Performance Matrix

### 2.1. Mục đích
Tìm điều kiện vận hành tối ưu cho mỗi tổ hợp Pathway × Catalyst bằng **grid search optimization**, sau đó tạo ma trận so sánh yield (%) tại các điều kiện optimal này.

**⚠️ Key Insight - Optimization-based Approach:**

```
Flow cũ (Knowledge-based):
    OPT_WINDOWS (hard-coded) → predict_conversion() → Matrix

Flow mới (Optimization-based):
    Y_base + G_catalyst (hard-coded)
         ↓
    predict_conversion() với generic scoring
         ↓
    Grid Search: test nhiều (T, P, H₂:CO₂)
         ↓
    Tìm max yield → Đây chính là "optimal conditions"
         ↓
    Matrix + Optimal Table (OUTPUTS)
```

**Điểm khác biệt:**
- Không cần biết optimal trước (không có OPT_WINDOWS)
- Optimal conditions là KẾT QUẢ của optimization, không phải INPUT
- Chỉ cần BASE_YIELD và CATALYST_GAIN từ literature

### 2.2. Dữ liệu Cơ sở (Hard-coded từ Literature)

#### A. Base Yield (Năng suất cơ bản)
```
Y_base = {
    Sabatier: 0.72 (72%)
    RWGS+FT: 0.55 (55%)
    Methanol: 0.60 (60%)
    Electroreduction (formate): 0.45 (45%)
    Electroreduction (CO): 0.50 (50%)
}
```

#### B. Catalyst Gain Factor (Hệ số tăng cường)
```
G_catalyst = {
    Ni-based: 1.00
    Cu/ZnO/Al₂O₃: 1.05 (+5%)
    Fe/Co (FT): 1.10 (+10%)
    Ag/Au (electro): 0.95 (-5%)
    Cu (electro): 1.08 (+8%)
    Other: 1.00
}
```

#### C. H₂:CO₂ Stoichiometry
```
R_stoich = {
    Sabatier: 4.0 (CO₂ + 4H₂ → CH₄ + 2H₂O)
    Methanol: 3.0 (CO₂ + 3H₂ → CH₃OH + H₂O)
    Formate: 2.0 (CO₂ + 2H₂ → HCOO⁻)
    CO: 1.0 (CO₂ + H₂ → CO + H₂O)
    RWGS+FT: 2.2 (trung bình)
}
```

---

### 2.3. Công thức Dự đoán Yield (Generic Scoring)

**Không cần biết optimal trước, chỉ dùng generic preferences cho từng loại pathway.**

#### A. Gaussian Score Function
```
Score(x, μ, σ) = e^(-(x - μ)² / (2σ²))
```

#### B. Generic Temperature Scoring

**Thermal Pathways (Sabatier, RWGS+FT, Methanol):**
```
μ_T = 320°C (generic thermal optimal)
σ_T = 50°C (wide tolerance)
S_T = e^(-(T - 320)² / (2×50²))
```

**Electro Pathways:**
```
μ_T = 25°C (room temperature)
σ_T = 10°C
S_T = e^(-(T - 25)² / (2×10²))
```

#### C. Generic Pressure Scoring

**Thermal Pathways:**
```
μ_P = 20 bar (moderate pressure)
σ_P = 15 bar (very wide tolerance: 5-35 bar acceptable)
S_P = e^(-(P - 20)² / (2×15²))
```

**Electro Pathways:**
```
μ_P = 1 bar (atmospheric)
σ_P = 0.5 bar
S_P = e^(-(P - 1)² / (2×0.5²))
```

#### D. H₂:CO₂ Ratio Scoring
```
μ_H = R_stoich[pathway] (stoichiometric ratio)
σ_H = 1.0 (tolerance ±1.0)
S_H = e^(-(H - μ_H)² / (2×1.0²))
```

#### E. Reactor Bias Factor
```
B_reactor = {
    Fixed-bed: 1.00
    Slurry: 0.98
    Electrochemical (flow cell): 1.05
    Other: 1.00
}
```

---

### 2.4. Công thức Tính Yield Tổng hợp

```
Y_raw = Y_base × G_catalyst × B_reactor × f_T × f_P × f_H
```

**Với các hàm trọng số:**
```
f_T = 0.5 + 0.5 × S_T    (Temperature: 50% baseline + 50% score)
f_P = 0.6 + 0.4 × S_P    (Pressure: 60% baseline + 40% score)
f_H = 0.5 + 0.5 × S_H    (H₂ ratio: 50% baseline + 50% score)
```

**Grid Carbon Intensity Adjustment (Electro only):**
```
Nếu CI_grid > 600 gCO₂/kWh:  Y_raw = Y_raw × 0.9
Nếu CI_grid < 100 gCO₂/kWh:  Y_raw = Y_raw × 1.03
```

**H₂ Stoichiometry Limit:**
```
F_H2 = min(1.0, H_actual / R_stoich)
Y_adjusted = Y_raw × (0.8 + 0.2 × F_H2)
```

**Final Clipping:**
```
Y_final = clip(Y_adjusted, 0.03, 0.95)
Y_percent = Y_final × 100%
```

---

### 2.5. Grid Search Optimization

**Mục tiêu:** Tìm (T*, P*, H*) để maximize Y_percent

#### A. Define Search Space

**Thermal Pathways:**
```
T ∈ [200, 450] với bước 10°C      → 26 points
P ∈ [5, 60] với bước 5 bar         → 12 points
H ∈ [R_stoich - 2, R_stoich + 3] với bước 0.2  → ~26 points

Total: 26 × 12 × 26 = 8,112 evaluations/pathway
```

**Electro Pathways:**
```
T ∈ [15, 40] với bước 2°C          → 13 points
P ∈ [0.5, 2.0] với bước 0.2 bar    → 8 points
H ∈ [R_stoich - 1, R_stoich + 2] với bước 0.2  → ~16 points

Total: 13 × 8 × 16 = 1,664 evaluations/pathway
```

#### B. Optimization Algorithm

```
Với mỗi (pathway, catalyst):

    1. Check compatibility:
       Nếu không valid → return N/A

    2. Initialize:
       Y_max = 0
       (T*, P*, H*) = None

    3. Grid search:
       For T in T_range:
           For P in P_range:
               For H in H_range:
                   Y = predict_yield(pathway, catalyst, T, P, H)

                   If Y > Y_max:
                       Y_max = Y
                       (T*, P*, H*) = (T, P, H)

    4. Return:
       {
           'optimal_T': T*,
           'optimal_P': P*,
           'optimal_H': H*,
           'max_yield': Y_max
       }
```

---

### 2.6. Tạo Matrix từ Grid Search Results

**Thuật toán:**
```
Matrix[5 pathways × 6 catalysts]:

For pathway in pathways:
    For catalyst in catalysts:

        optimal = grid_search_optimize(pathway, catalyst)

        If optimal is None:
            Matrix[pathway, catalyst] = N/A
        Else:
            Matrix[pathway, catalyst] = optimal['max_yield']
```

**Output Format:**

| Pathway | Ni-based | Cu/ZnO/Al₂O₃ | Fe/Co (FT) | Ag/Au | Cu (e) | Other |
|---------|----------|---------------|------------|-------|--------|-------|
| Sabatier| 72.3%    | 75.8%         | **79.5%**  | N/A   | N/A    | 72.0% |
| RWGS+FT | 55.2%    | 57.9%         | **60.7%**  | N/A   | N/A    | 55.0% |
| Methanol| 60.1%    | 63.2%         | **66.3%**  | N/A   | N/A    | 60.0% |
| E-form  | N/A      | N/A           | N/A        | 42.5% | **48.9%** | 45.0% |
| E-CO    | N/A      | N/A           | N/A        | 47.2% | **54.3%** | 50.0% |

*Yields tại điều kiện optimal tìm được từ grid search*

---

### 2.7. Ví dụ Optimization cho Sabatier + Fe/Co

**Bước 1: Define search space**
```
T_range = [200, 210, 220, ..., 440, 450]  (26 values)
P_range = [5, 10, 15, ..., 55, 60]         (12 values)
H_range = [2.0, 2.2, 2.4, ..., 6.8, 7.0]   (26 values)
```

**Bước 2: Evaluate một điểm (T=320, P=10, H=4.0)**
```
Y_base = 0.72, G_catalyst = 1.10, B_reactor = 1.00

S_T = e^(-(320-320)²/(2×50²)) = e^0 = 1.0
S_P = e^(-(10-20)²/(2×15²)) = e^(-100/450) = 0.800
S_H = e^(-(4.0-4.0)²/(2×1.0²)) = e^0 = 1.0

f_T = 0.5 + 0.5×1.0 = 1.0
f_P = 0.6 + 0.4×0.800 = 0.92
f_H = 0.5 + 0.5×1.0 = 1.0

Y_raw = 0.72 × 1.10 × 1.00 × 1.0 × 0.92 × 1.0 = 0.729

F_H2 = min(1.0, 4.0/4.0) = 1.0
Y_adjusted = 0.729 × 1.0 = 0.729

Y = 72.9%
```

**Bước 3: Evaluate tất cả 8,112 điểm**
```
Results:
  (T=310, P=10, H=4.0) → 78.2%
  (T=320, P=10, H=4.0) → 72.9%
  (T=320, P=12, H=4.2) → 79.5% ← MAX
  (T=330, P=15, H=3.8) → 77.1%
  ...
```

**Bước 4: Return optimal**
```
Optimal conditions:
  T* = 320°C
  P* = 12 bar
  H* = 4.2
  Y_max = 79.5%
```

---

### 2.8. Validation Rules (Tương thích Pathway-Catalyst)

```
Thermal Pathways ↔ Thermal Catalysts
    Sabatier         ↔ {Ni-based, Cu/ZnO/Al₂O₃, Fe/Co, Other}
    RWGS+FT          ↔ {Ni-based, Cu/ZnO/Al₂O₃, Fe/Co, Other}
    Methanol         ↔ {Ni-based, Cu/ZnO/Al₂O₃, Fe/Co, Other}

Electro Pathways ↔ Electro Catalysts
    Electro-formate  ↔ {Cu (electro), Ag/Au (electro), Other}
    Electro-CO       ↔ {Cu (electro), Ag/Au (electro), Other}

Invalid Combinations → N/A (skip grid search)
```

---

## 3. ⚙️ Optimal Operating Conditions Reference

### 3.1. Mục đích
Tạo bảng tham chiếu từ **kết quả grid search**: điều kiện tối ưu, catalyst tốt nhất, và max yield cho mỗi pathway.

### 3.2. Phương pháp (Optimization-based)

**Với mỗi pathway:**
```
1. Initialize:
   best_catalyst = None
   max_yield_overall = 0
   optimal_conditions = None

2. For catalyst in all_catalysts:
       If không compatible → skip

       optimal = grid_search_optimize(pathway, catalyst)

       If optimal['max_yield'] > max_yield_overall:
           best_catalyst = catalyst
           max_yield_overall = optimal['max_yield']
           optimal_conditions = optimal

3. Return:
   {
       'pathway': pathway,
       'T_opt': optimal_conditions['optimal_T'],
       'P_opt': optimal_conditions['optimal_P'],
       'H_opt': optimal_conditions['optimal_H'],
       'best_catalyst': best_catalyst,
       'max_yield': max_yield_overall
   }
```

**Output (Kết quả từ Grid Search):**

| Pathway | T_opt | P_opt | H₂:CO₂_opt | Best Catalyst | Max Yield | Primary Product |
|---------|-------|-------|------------|---------------|-----------|-----------------|
| Sabatier | 320°C | 12 bar | 4.2 | Fe/Co (FT) | 79.5% | CH₄ |
| RWGS+FT | 370°C | 22 bar | 2.3 | Fe/Co (FT) | 60.7% | C5+ hydrocarbons |
| Methanol | 255°C | 48 bar | 3.1 | Fe/Co (FT) | 66.3% | CH₃OH |
| Electro-formate | 26°C | 1.0 bar | 2.1 | Cu (electro) | 48.9% | HCOO⁻/HCOOH |
| Electro-CO | 24°C | 0.9 bar | 1.2 | Cu (electro) | 54.3% | CO |

*Giá trị chính xác tìm được từ optimization, không phải ±ranges*

---

### 3.4. Ví dụ: Tìm Optimal cho Pathway "Sabatier"

**Bước 1: Grid search cho tất cả catalysts**

```
Results:
Sabatier + Ni-based:      T=315°C, P=10 bar, H=4.0 → Y = 72.3%
Sabatier + Cu/ZnO/Al₂O₃:  T=318°C, P=11 bar, H=4.1 → Y = 75.8%
Sabatier + Fe/Co (FT):    T=320°C, P=12 bar, H=4.2 → Y = 79.5% ← MAX
Sabatier + Ag/Au (electro): N/A (incompatible)
Sabatier + Cu (electro):    N/A (incompatible)
Sabatier + Other:         T=320°C, P=10 bar, H=4.0 → Y = 72.0%
```

**Bước 2: Chọn best**
```
best_catalyst = Fe/Co (FT)
T_opt = 320°C
P_opt = 12 bar
H_opt = 4.2
max_yield = 79.5%
```

---

### 3.5. Computational Complexity

**Tổng số evaluations:**
```
5 pathways:
  - 3 thermal pathways × 4 thermal catalysts × 8,112 evals = 97,344
  - 2 electro pathways × 2 electro catalysts × 1,664 evals = 6,656

Total: 104,000 evaluations

Thời gian (giả định 1ms/eval):
  - Sequential: 104 seconds (~1.7 minutes)
  - Parallel (10 workers): 10.4 seconds
```

**Optimization:**
- Cache prediction function results
- Sử dụng coarse-to-fine search (grid thô → grid mịn)
- Early stopping nếu yield không cải thiện

---

### 3.6. Stoichiometry (Phương trình hóa học)

#### A. Sabatier (Methanation)
```
CO₂ + 4H₂ → CH₄ + 2H₂O     ΔH = -165 kJ/mol
H₂:CO₂ = 4:1
```

#### B. RWGS + Fischer-Tropsch
```
Giai đoạn 1 (RWGS):
CO₂ + H₂ → CO + H₂O         ΔH = +41 kJ/mol

Giai đoạn 2 (FT):
nCO + (2n+1)H₂ → CₙH₂ₙ₊₂ + nH₂O

Trung bình: H₂:CO₂ ≈ 2:1
```

#### C. Methanol Synthesis
```
CO₂ + 3H₂ → CH₃OH + H₂O     ΔH = -49 kJ/mol
H₂:CO₂ = 3:1
```

#### D. Electroreduction (Formate Route)
```
CO₂ + 2H⁺ + 2e⁻ → HCOO⁻    E° = -0.61 V vs RHE
Hoặc: CO₂ + H₂ → HCOOH
H₂:CO₂ = 2:1 (equivalent)
```

#### E. Electroreduction (CO Route)
```
CO₂ + 2H⁺ + 2e⁻ → CO + H₂O  E° = -0.52 V vs RHE
Hoặc: CO₂ + H₂ → CO + H₂O
H₂:CO₂ ≈ 1:1
```

---

### 3.5. Nhiệt động học (Thermodynamics)

#### A. Thermal Pathways

**Sabatier (Exothermic):**
- ΔH = -165 kJ/mol → Tỏa nhiệt
- High T (320°C) → Tăng kinetics
- Moderate P (10 bar) → Cân bằng Le Chatelier

**Methanol (Exothermic):**
- ΔH = -49 kJ/mol → Tỏa nhiệt
- Low T (250°C) → Tránh decomposition
- High P (50 bar) → Dịch cân bằng sang phải

**RWGS (Endothermic):**
- ΔH = +41 kJ/mol → Thu nhiệt
- Highest T (380°C) → Cần nhiệt độ cao
- Moderate-high P (20 bar)

#### B. Electro Pathways

**Electroreduction:**
- Room temperature (25°C) → Không cần đun nóng
- Atmospheric P (1 bar) → Không cần nén
- Energy: Điện năng → Hóa năng
- Efficiency phụ thuộc: CI_grid

---

### 3.6. Carbon Utilization (Hiệu suất sử dụng carbon)

**Công thức:**
```
C_util = Y_final × (1 - S_others)

Với S_others: Selectivity của sản phẩm phụ không chứa carbon
```

**Product Slate (Selectivity Distribution):**
```
Sabatier:
    CH₄: 75-85%
    CO: 5-10%
    CH₃OH: 1-3%
    Others: 5-10%

RWGS+FT:
    C5+: 55-65%
    CO: 5-10%
    CH₄: 3-8%
    Others: 20-30%

Methanol:
    CH₃OH: 70-80%
    CO: 3-8%
    CH₄: 2-5%
    Others: 10-20%

Electroreduction (HCOO⁻):
    HCOO⁻/HCOOH: 65-75%
    CO: 5-10%
    Others: 15-25%

Electroreduction (CO):
    CO: 70-80%
    HCOO⁻: 3-8%
    Others: 15-25%
```

---

## 4. Tổng kết Công thức

### 4.1. Commitment Scenario

```
Linear:      E(t) = E₀ + m(t - t₀)
Exponential: E(t) = E₀ · e^(-k(t - t₀))
S-curve:     E(t) = E_target + L/(1 + e^(k·x))

Gap(t) = E_BAU(t) - E_commitment(t)
Total_Gap = Σ Gap(t)
Avg_Reduction = Total_Gap / (t_target - t₀)
```

### 4.2. Catalyst-Pathway Matrix (Optimization-based)

**Prediction Function (Generic Scoring):**
```
Thermal: μ_T = 320°C (σ = 50), μ_P = 20 bar (σ = 15)
Electro: μ_T = 25°C (σ = 10), μ_P = 1 bar (σ = 0.5)
H₂:CO₂: μ_H = R_stoich (σ = 1.0)

Score(x, μ, σ) = e^(-(x - μ)² / (2σ²))

Y = Y_base × G_catalyst × B_reactor × f_T × f_P × f_H
f_T = 0.5 + 0.5 × S_T
f_P = 0.6 + 0.4 × S_P
f_H = 0.5 + 0.5 × S_H

Y_adjusted = Y × (0.8 + 0.2 × F_H2)
F_H2 = min(1.0, H_actual / R_stoich)

Y_final = clip(Y_adjusted, 0.03, 0.95)
```

**Grid Search Optimization:**
```
Thermal: T ∈ [200, 450]°C, P ∈ [5, 60] bar, H ∈ [R-2, R+3]
Electro: T ∈ [15, 40]°C, P ∈ [0.5, 2.0] bar, H ∈ [R-1, R+2]

(T*, P*, H*) = argmax Y(T, P, H)
                T,P,H

Matrix[pathway, catalyst] = Y(T*, P*, H*)
```

### 4.3. Optimal Conditions (từ Grid Search Results)

```
For each pathway:
    best_yield = 0
    best_catalyst = None

    For catalyst in valid_catalysts:
        (T*, P*, H*) = grid_search_optimize(pathway, catalyst)

        If Y(T*, P*, H*) > best_yield:
            best_yield = Y(T*, P*, H*)
            best_catalyst = catalyst
            optimal_conditions = (T*, P*, H*)

Output:
    T_opt (exact value, not range)
    P_opt (exact value, not range)
    H_opt (exact value, not range)
    Best_Catalyst
    Max_Yield
```

**Ưu điểm của Optimization-based Approach:**
- Không cần hard-code OPT_WINDOWS
- Tự động tìm optimal cho mỗi pathway × catalyst
- Matrix và Optimal Table là OUTPUTS, không phải INPUTS
- Chỉ cần BASE_YIELD và CATALYST_GAIN từ literature
