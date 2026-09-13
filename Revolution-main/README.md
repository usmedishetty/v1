# Indian Infrastructure Projects Synthetic Dataset & Land Acquisition CRS Pipeline

A realistic, high-fidelity synthetic dataset generator, feature engineering pipeline, and **Composite Risk Score (CRS)** benchmark for Indian infrastructure projects across **Highways**, **Railways**, **Airports**, and **Renewable Energy**.

The dataset and risk modeling pipeline incorporate empirical patterns from **Land Conflict Watch (LCW)**, **PMGSY GIS standards**, the **RFCTLARR Act, 2013**, and central infrastructure awarding data.

---

## 1. Quick Start & Execution

### Prerequisites
- Python 3.8+ or [`uv`](https://github.com/astral-sh/uv)
- Standard library only (zero external dependencies)

### Run Dataset Generation & CRS Pipeline
```bash
# 1. Generate realistic dataset (e.g. 12,500 records)
python generate_dataset.py --records 12500 --output indian_infrastructure_projects_dataset.csv

# 2. Apply CRS algorithm corrections, feature engineering, and validation
python apply_crs_corrections.py --input indian_infrastructure_projects_dataset.csv --output indian_infrastructure_projects_dataset.csv

# Or execute with uv
uv run python generate_dataset.py --records 12500
uv run python apply_crs_corrections.py
```

---

## 2. Dataset Schema & Column Dictionary

The dataset contains 29 comprehensive columns categorized below:

### A. Base Physical, Socio-Legal & Workflow Metrics
| Column Name | Type | Description |
| :--- | :--- | :--- |
| `project_id` | String | Unique standard identifier (`NHAI-MH-2023-0014`, `MoR-UP-2024-0102`, `AAI-AP-2019-0001`, `MNRE-RJ-2024-0001`). |
| `project_type` | Categorical | Sector: `Highway`, `Railway`, `Airport`, `Renewable_Energy`. |
| `state` | Categorical | 28 Indian States + 8 Union Territories (36 total). |
| `district` | Categorical | Real district mapped hierarchically under PMGSY GIS standards. |
| `terrain_type` | Categorical | `Urban`, `Rural_Agri`, `Forest_Eco_Sensitive`, `Tribal_Schedule_V`. |
| `land_area_hectares` | Float | Right-of-way / project land footprint in hectares (5.0 to 1000.0 ha). |
| `estimated_cost_inr_crore`| Float | Total capital expenditure estimate in ₹ Crores (₹14.0 to ₹15,000+ Cr). |
| `project_start_year` | Integer | Initiation year (2015 to 2026). |
| `affected_families_count` | Integer | Total families displaced or affected by land acquisition (10 to 7500+). |
| `title_dispute_rate_percent`| Float | Percentage of land parcels under title disputes / customary claims (0% to 100%). |
| `local_protest_flag` | Boolean | Whether active community / landowner opposition or protest occurred (`True`/`False`). |
| `compensation_multiplier_demand` | Float | Multiplier demanded over circle/market rate (1.0x to 4.0x). |
| `sia_approval_status` | Categorical | Social Impact Assessment status: `Approved`, `Pending`, `Rejected`, `Exempted`. |
| `section_11_notification_days` | Integer | Preliminary acquisition notification timeline under RFCTLARR (30 to 730 days). |
| `forest_clearance_status` | Categorical | MoEFCC clearance: `Not_Required`, `Stage_1_Pending`, `Approved`. |
| `fund_disbursement_percent` | Float | Proportion of allocated land compensation funds disbursed to date (0% to 100%). |

### B. Engineered Risk Sub-Indices (All bounded [0, 1])
| Column Name | Type | Formula / Mapping Logic |
| :--- | :--- | :--- |
| `sia_approval_status_risk_score` | Float | `Approved`/`Exempted` $\to 0.0$; `Pending` ($\le 180\text{d}$) $\to 0.5$; `Pending` ($>180\text{d}$) or `Rejected` $\to 1.0$. |
| `forest_clearance_status_risk_score` | Float | `Approved`/`Not_Required` $\to 0.0$; `Stage_1_Pending` ($\le 180\text{d}$) $\to 0.5$; `Stage_1_Pending` ($>180\text{d}$) or `Rejected` $\to 1.0$. |
| `C_r` | Float | Compensation Risk: If `fund_disbursement == 0` $\to 1.0$. Else $\min(1.0, \max(0.0, \text{multiplier} - 1.0))$. Fallback $0.5$. |
| `F_r` | Float | Affected Families Scaling: $\frac{\log_{10}(\text{Affected\_Families} + 1)}{\log_{10}(F_{\max} + 1)}$ where $F_{\max} = \text{Percentile}_{95}$. Capped at $1.0$. |
| `H_r` | Float | Historical Inertia: $\frac{\text{GroupMeanDays}(\text{state}, \text{project\_type})}{\max(\text{AllGroupMeans})}$, capped at $1.0$. |
| `W_r` | Float | Weather Vulnerability Index: Empirical state-level climate/disaster index ($0.40$ to $0.85$). |
| `P_r` | Float | Protest Score: Binary flag mapping (`True` $\to 0.7$, `False` $\to 0.0$). |

### C. Feature Engineering & Composite Risk Score
| Column Name | Type | Description |
| :--- | :--- | :--- |
| `land_area_log` | Float | $\log_{10}(\text{land\_area\_hectares} + 1)$ |
| `project_age_years` | Integer | $\text{current\_year} - \text{project\_start\_year}$ |
| `delay_binary_label` | Integer | $1$ if $\text{section\_11\_notification\_days} > 365$, else $0$ |
| `delay_risk_tier` | Categorical | `Low` ($\le 90\text{d}$), `Medium` ($91-180\text{d}$), `High` ($181-365\text{d}$), `Very_High` ($>365\text{d}$) |
| `CRS` | Float | **Composite Risk Score** $[0, 100]$: $$100 \times \left( 0.35 \times \frac{F_r + C_r}{2} + 0.30 \times \frac{\text{SIA}_r + \text{Forest}_r}{2} + 0.20 \times W_r + 0.15 \times P_r \right)$$ |
| `CRS_tier` | Categorical | `Low` ($<25$), `Moderate` ($25-50$), `High` ($50-75$), `Critical` ($\ge 75$) |

---

## 3. Statistical Summary & Verification (12,500 Records)

```
================================================================================
VALIDATION & INTEGRITY CHECKS
================================================================================
[SUCCESS] All validation checks passed!
  - Total records processed : 12,500
  - NaN / Missing values     : 0 (0.00%)
  - Sub-indices [0, 1] bound : 100% verified
  - CRS [0, 100] bound       : 100% verified

================================================================================
COMPOSITE RISK SCORE (CRS) & FEATURE SUMMARY DISTRIBUTIONS
================================================================================
Metric                                  Min     Mean   Median      Max   StdDev
--------------------------------------------------------------------------------
Affected Families (F_r)                0.30     0.75     0.76     1.00     0.15
Compensation Risk (C_r)                0.01     0.94     1.00     1.00     0.15
Historical Inertia (H_r)               0.15     0.52     0.52     1.00     0.08
Weather Vulnerability (W_r)            0.40     0.60     0.60     0.85     0.12
Protest Score (P_r)                    0.00     0.19     0.00     0.70     0.31
SIA Status Risk                        0.00     0.45     0.00     1.00     0.48
Forest Clearance Risk                  0.00     0.20     0.00     1.00     0.40
Log Land Area (ha)                     0.78     2.24     2.31     3.00     0.39
Project Age (Years)                    0.00     4.58     4.00    11.00     2.85
Composite Risk Score (CRS)            22.77    54.04    53.50    90.70    12.81

--- Delay Risk Tier Distribution ---
  * Low         :    296 (  2.4%)
  * Medium      :  1,965 ( 15.7%)
  * High        :  5,027 ( 40.2%)
  * Very_High   :  5,212 ( 41.7%)

--- Delay Binary Label (> 365 days) ---
  * Severe Delay (>365d) :  5,212 (41.7%)
  * Standard (<=365d)    :  7,288 (58.3%)

--- CRS Risk Tier Distribution ---
  * Low         :     11 (  0.1%)
  * Moderate    :  5,163 ( 41.3%)
  * High        :  6,488 ( 51.9%)
  * Critical    :    838 (  6.7%)
```

---

## 4. References

1. **Land Conflict Watch (LCW)**: [https://www.landconflictwatch.org/](https://www.landconflictwatch.org/) — Socio-environmental land conflict database and community protest triggers.
2. **Ministry of Rural Development (PMGSY)**: [https://pmgsy.dord.gov.in/](https://pmgsy.dord.gov.in/) — Spatial GIS administrative boundary standards and district naming conventions.
3. **The RFCTLARR Act, 2013**: Ministry of Law and Justice, Government of India — Statutory Social Impact Assessment (SIA) processes, Section 11 preliminary notifications, and Solatium / Multiplier schedules.
4. **MoEFCC PARIVESH Portal**: Environmental, Forest and Wildlife Clearance guidelines under the Forest (Conservation) Act, 1980.
