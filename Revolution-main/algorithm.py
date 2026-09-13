import numpy as np
import pandas as pd
import math
from typing import Dict, Any, Tuple

class RiskEngine:
    """
    Production-Grade Risk Engine for SIH Problem Statement 26017
    Features:
      - Bounded [0, 100] guarantee with numerical safety
      - Bayesian shrinkage for historical delay priors
      - Feature attribution & statutory LARR Act lapse detection
      - Vectorized SIMD pipeline for high-throughput batch scoring
    """

    TERRAIN_WEIGHTS = {
        "plain": 0.1, "agricultural": 0.2, "plain/agricultural": 0.2,
        "urban": 0.4, "semi-arid": 0.3, "coastal": 0.5, "wetland": 0.6,
        "coastal/wetland": 0.5, "hilly": 0.7, "mountainous": 0.8,
        "hilly/mountainous": 0.8, "dense forest": 1.0, "forest": 0.9,
        "riverbed": 0.7, "default": 0.5
    }

    def __init__(self, f_max: float = 50000.0, d_max: float = 1000.0, 
                 n_base: float = 60.0, n_limit: float = 365.0):
        self.f_max = max(1.0, f_max)
        self.d_max = max(1.0, d_max)
        self.n_base = n_base
        self.n_limit = n_limit
        self.log_fmax_plus1 = math.log10(self.f_max + 1)
        self.historical_stats = {}
        self.global_mean_delay = 0.0

    def fit_historical_priors(self, df: pd.DataFrame, delay_col: str = "Actual_Delay_Days",
                             state_col: str = "State", proj_col: str = "Project_Type",
                             m_smoothing: float = 5.0):
        """Bayesian empirical mean smoothing to prevent cold-start anomalies."""
        if delay_col not in df.columns:
            return
        self.global_mean_delay = float(df[delay_col].mean())
        if df[delay_col].max() > 0:
            self.d_max = max(self.d_max, float(df[delay_col].max()))
            
        grp = df.groupby([state_col, proj_col])[delay_col].agg(['count', 'mean']).reset_index()
        for _, row in grp.iterrows():
            st, pr = str(row[state_col]), str(row[proj_col])
            n, mean_val = row['count'], row['mean']
            smoothed = (n * mean_val + m_smoothing * self.global_mean_delay) / (n + m_smoothing)
            self.historical_stats[(st, pr)] = smoothed

    def score_single(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Single-record evaluation with explainability breakdown and statutory warning."""
        eps = 1e-6

        # 1. Socio-Political Risk (R_SP)
        p_raw = record.get("Local_Protest_Flag", False)
        p_r = 1.0 if str(p_raw).strip().lower() in ["true", "1", "yes", "high"] else 0.0

        c_offer = max(0.0, float(record.get("C_offer", 0.0)))
        c_demand = max(0.0, float(record.get("C_demand", 0.0)))
        if c_offer <= eps:
            c_r = 1.0 if c_demand > 0 else 0.0
        else:
            c_r = max(0.0, min(1.0, (c_demand - c_offer) / c_offer))

        aff_fam = max(0.0, float(record.get("Affected_Families_Count", 0.0)))
        f_r = min(1.0, math.log10(aff_fam + 1) / self.log_fmax_plus1)
        r_sp = (0.50 * p_r) + (0.30 * c_r) + (0.20 * f_r)

        # 2. Legal & Regulatory Risk (R_LR)
        t_rate = max(0.0, min(100.0, float(record.get("Title_Dispute_Rate_Percent", 0.0))))
        t_r = t_rate / 100.0

        def parse_clearance(val: Any) -> float:
            if val is None or pd.isna(val): return 0.5
            s = str(val).strip().lower()
            if s in ["approved", "granted", "cleared", "exempted", "not applicable", "n/a"]:
                return 0.0
            elif "pending <= 6" in s or s in ["pending", "in review", "applied"]:
                return 0.5
            elif "pending > 6" in s or s in ["rejected", "denied"]:
                return 1.0
            return 0.5

        sia_r = parse_clearance(record.get("SIA_Approval_Status", "Approved"))
        fc_r = parse_clearance(record.get("Forest_Clearance_Status", "Approved"))
        days_elapsed = max(0.0, float(record.get("Section_11_Notification_Days", 0.0)))
        n_r = max(0.0, min(1.0, (days_elapsed - self.n_base) / max(eps, self.n_limit - self.n_base)))
        r_lr = (0.40 * t_r) + (0.20 * sia_r) + (0.20 * fc_r) + (0.20 * n_r)

        # 3. Environmental & Geo Risk (R_EG)
        terrain_str = str(record.get("Terrain_Type", "default")).strip().lower()
        t_m = self.TERRAIN_WEIGHTS.get(terrain_str, self.TERRAIN_WEIGHTS["default"])
        w_idx = max(0.0, min(10.0, float(record.get("Weather_Index", 1.0))))
        w_r = w_idx / 10.0
        r_eg = (0.50 * t_m) + (0.50 * w_r)

        # 4. Financial & Admin Risk (R_FA)
        f_disb = max(0.0, min(100.0, float(record.get("Fund_Disbursement_Percent", 100.0))))
        f_dr = 1.0 - (f_disb / 100.0)
        st, pr = str(record.get("State", "")), str(record.get("Project_Type", ""))
        h_delay = self.historical_stats.get((st, pr), self.global_mean_delay)
        h_r = max(0.0, min(1.0, h_delay / max(eps, self.d_max)))
        r_fa = (0.60 * f_dr) + (0.40 * h_r)

        # 5. Composite Score & Drivers
        crs = round(max(0.0, min(100.0, ((0.35 * r_sp) + (0.30 * r_lr) + (0.20 * r_eg) + (0.15 * r_fa)) * 100.0)), 2)
        
        risk_band = ("Low Risk" if crs < 35 else "Medium Risk" if crs < 65 else 
                     "High Risk" if crs < 85 else "Critical / Severe Risk")

        contributions = {
            "Socio-Political (R_SP)": round(0.35 * r_sp * 100, 2),
            "Legal & Regulatory (R_LR)": round(0.30 * r_lr * 100, 2),
            "Environmental & Geo (R_EG)": round(0.20 * r_eg * 100, 2),
            "Financial & Admin (R_FA)": round(0.15 * r_fa * 100, 2),
        }

        return {
            "CRS": crs,
            "Risk_Band": risk_band,
            "Dominant_Risk_Driver": max(contributions.items(), key=lambda x: x[1])[0],
            "LARR_Section11_Lapse_Warning": days_elapsed >= self.n_limit,
            "Sub_Indices": {"R_SP": round(r_sp, 4), "R_LR": round(r_lr, 4), 
                            "R_EG": round(r_eg, 4), "R_FA": round(r_fa, 4)},
            "Contributions": contributions
        }

    def score_dataframe_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """High-throughput vectorized execution for batch processing."""
        n, eps = len(df), 1e-6

        # Vectorized R_SP
        p_series = df.get("Local_Protest_Flag", pd.Series([False]*n))
        p_r = (p_series.astype(float).values if p_series.dtype in [bool, np.bool_] 
               else np.where(p_series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"]), 1.0, 0.0))

        c_offer = pd.to_numeric(df.get("C_offer", 1.0), errors="coerce").fillna(1.0).to_numpy(dtype=float)
        c_demand = pd.to_numeric(df.get("C_demand", 1.0), errors="coerce").fillna(1.0).to_numpy(dtype=float)
        c_r = np.clip((c_demand - c_offer) / np.where(c_offer <= eps, eps, c_offer), 0.0, 1.0)
        c_r[(c_offer <= eps) & (c_demand <= eps)] = 0.0

        aff_fam = np.maximum(0.0, pd.to_numeric(df.get("Affected_Families_Count", 0), errors="coerce").fillna(0).to_numpy(dtype=float))
        f_r = np.clip(np.log10(aff_fam + 1.0) / self.log_fmax_plus1, 0.0, 1.0)
        r_sp = (0.50 * p_r) + (0.30 * c_r) + (0.20 * f_r)

        # Vectorized R_LR
        t_r = np.clip(pd.to_numeric(df.get("Title_Dispute_Rate_Percent", 0), errors="coerce").fillna(0).to_numpy(dtype=float) / 100.0, 0.0, 1.0)
        
        def parse_vec_clearance(col_name: str) -> np.ndarray:
            s = df.get(col_name, pd.Series(["Approved"]*n)).astype(str).str.strip().str.lower()
            arr = np.full(len(s), 0.5, dtype=float)
            arr[s.isin(["approved", "granted", "cleared", "exempted", "not applicable", "n/a"])] = 0.0
            arr[s.str.contains("pending > 6|pending_gt_6|rejected|denied", regex=True)] = 1.0
            return arr

        sia_r, fc_r = parse_vec_clearance("SIA_Approval_Status"), parse_vec_clearance("Forest_Clearance_Status")
        days = np.maximum(0.0, pd.to_numeric(df.get("Section_11_Notification_Days", 0), errors="coerce").fillna(0).to_numpy(dtype=float))
        n_r = np.clip((days - self.n_base) / max(eps, self.n_limit - self.n_base), 0.0, 1.0)
        r_lr = (0.40 * t_r) + (0.20 * sia_r) + (0.20 * fc_r) + (0.20 * n_r)

        # Vectorized R_EG
        terrain_series = df.get("Terrain_Type", pd.Series(["default"]*n)).astype(str).str.strip().str.lower()
        t_m = terrain_series.map(self.TERRAIN_WEIGHTS).fillna(self.TERRAIN_WEIGHTS["default"]).to_numpy(dtype=float)
        w_r = np.clip(pd.to_numeric(df.get("Weather_Index", 1.0), errors="coerce").fillna(1.0).to_numpy(dtype=float) / 10.0, 0.0, 1.0)
        r_eg = (0.50 * t_m) + (0.50 * w_r)

        # Vectorized R_FA
        f_dr = np.clip(1.0 - (pd.to_numeric(df.get("Fund_Disbursement_Percent", 100.0), errors="coerce").fillna(100.0).to_numpy(dtype=float) / 100.0), 0.0, 1.0)
        states, projs = df.get("State", pd.Series([""]*n)).astype(str).values, df.get("Project_Type", pd.Series([""]*n)).astype(str).values
        h_delays = np.array([self.historical_stats.get((s, p), self.global_mean_delay) for s, p in zip(states, projs)], dtype=float)
        h_r = np.clip(h_delays / max(eps, self.d_max), 0.0, 1.0)
        r_fa = (0.60 * f_dr) + (0.40 * h_r)

        # Final CRS Calculation
        crs = np.clip(((0.35 * r_sp) + (0.30 * r_lr) + (0.20 * r_eg) + (0.15 * r_fa)) * 100.0, 0.0, 100.0)

        res_df = df.copy()
        res_df["R_SP"], res_df["R_LR"] = np.round(r_sp, 4), np.round(r_lr, 4)
        res_df["R_EG"], res_df["R_FA"] = np.round(r_eg, 4), np.round(r_fa, 4)
        res_df["CRS"] = np.round(crs, 2)
        res_df["Risk_Band"] = np.select([crs < 35.0, crs < 65.0, crs < 85.0], 
                                        ["Low Risk", "Medium Risk", "High Risk"], default="Critical / Severe Risk")
        res_df["LARR_Sec11_Lapse_Warning"] = days >= self.n_limit
        return res_df
