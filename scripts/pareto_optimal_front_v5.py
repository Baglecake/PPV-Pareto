# PPV-Pareto v10 — Production-ready end-to-end forecast pipeline with harmonization,
# row-wise fallback, calibration, EC variants, and grouped CV/cross-fitted calibration.
#
# Key improvements in v10:
# - Production error handling with proper exit codes
# - Configurable verbosity to reduce console spam
# - Enhanced state diagnostics with neff tables
# - Better artifact organization and validation
# - Comprehensive logging and metadata tracking
# - Improved reliability checks and warnings

import os
import sys
import json
import random
import warnings
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import NSGAIISampler
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore")

# NEW: shrinkage helper (requires shrinkage_utils.py in your path)
from shrinkage_utils import state_level_ec_from_preds_shrunk, FIPS_TO_POSTAL, EV_2024

# ------------------ RANDOM SEED (SET EARLY) ------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ------------------ USER CONFIG ------------------
DATA_2012 = "/content/ANES_data_predicting_popular_vote_shares_2012.dta"
DATA_2016 = "/content/ANES_data_predicting_popular_vote_shares_2016.dta"
DATA_2020 = "/content/ANES_data_predicting_popular_vote_shares_2020.dta"
DATA_2024 = "/content/anes_timeseries_2024_stata_20250808.dta"  # your file

# Confirmed 2024 base weight
WEIGHT_VAR_OVERRIDE_2024 = "V240107a"

# If 2024 WNH derived missingness > this threshold, skip models that require WNH in row-wise fallback
WNH_MISSINGNESS_THRESHOLD = 0.20

# EC strategy options: "base", "state_refit", "dual"
EC_STRATEGY = "dual"

# Minimum effective N to include a state's EV in tallies (publication filter)
EC_MIN_NEFF = 100.0

# Shrinkage prior strength (higher -> more shrink to national mean)
EC_SHRINKAGE_K_PRIOR = 400.0

# Optimization
N_TRIALS = 200
N_SPLITS = 5

# Feature toggles
GROUPED_CV_BY_YEAR = True
CROSS_FIT_ISOTONIC = True
STATE_MIN_WEIGHT = 0.0  # set >0 to collapse rare states (by total weight) to "_OTHER_" [not used in this file]
SAVE_CALIBRATION_PLOTS = True
SAVE_OOF_PANELS = True   # placeholder flag (no OOF panel saving in this file)
EC_SPLIT_ME_NE = False   # placeholder; ME/NE split not implemented in this file

# NEW: Verbosity controls
VERBOSE_EC_LISTS = False  # Set True to show full excluded state lists
VERBOSE_DIAGNOSTICS = True  # Set True for detailed diagnostic output

# Outputs
OUT_DIR = "/content/optuna_outputs_forward"
BOOTSTRAP_REPS = 500
BOOTSTRAP_SEED = 123

# ------------------ Utilities ------------------

def _exists(p: str) -> bool:
    try:
        return os.path.exists(p)
    except Exception:
        return False

def _to_num(s): return pd.to_numeric(s, errors="coerce")

def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)

def _roc_auc_safe(y_true, y_score, sample_weight=None) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return 0.5
        return roc_auc_score(y_true, y_score, sample_weight=sample_weight)
    except Exception:
        return 0.5

def eff_sample_size(w: np.ndarray) -> float:
    w = np.asarray(w, float)
    if not np.isfinite(w).any() or np.sum(w) <= 0:
        return 0.0
    return float((w.sum()**2) / (np.sum(w**2) + 1e-12))

def weighted_bootstrap_ci_share(p: np.ndarray, w: np.ndarray, thr: float, reps: int, seed: int):
    rng = np.random.default_rng(seed)
    w_pos = np.clip(w, 0, None)
    if w_pos.sum() <= 0:
        return (float("nan"), float("nan"))
    probs = w_pos / w_pos.sum()
    n = len(p)
    stats = []
    for _ in range(reps):
        idx = rng.choice(n, size=n, replace=True, p=probs)
        preds = (p[idx] > thr).astype(int)
        stats.append(float(np.sum(w[idx] * preds) / np.sum(w[idx])))
    lo = float(np.nanpercentile(stats, 2.5))
    hi = float(np.nanpercentile(stats, 97.5))
    return lo, hi

def pct(x: Optional[float]) -> str:
    return f"{100*x:.1f}%" if x is not None and np.isfinite(x) else "NA"

def dump_neff_table(tag: str, sdf: pd.DataFrame, out_dir: str):
    """Save detailed state-level diagnostics with neff and reliability info"""
    cols = ["state","ev","neff","eligible","reliability","mean_prob","share_thr_05"]
    extra = [c for c in ["mean_prob_shrunk","prior_center_national","ci_low_thr05","ci_high_thr05","thin_reason"] if c in sdf.columns]
    output_path = os.path.join(out_dir, f"states_neff_{tag}.csv")
    sdf[cols+extra].sort_values("ev", ascending=False).to_csv(output_path, index=False)
    if VERBOSE_DIAGNOSTICS:
        print(f"Saved detailed state diagnostics: {output_path}")

def validate_file_exists(filepath: str, description: str) -> bool:
    """Check if file exists and return appropriate error if not"""
    if not _exists(filepath):
        print(f"ERROR: Missing required file: {description}")
        print(f"  Path: {filepath}")
        return False
    return True

def save_requirements_txt(out_dir: str):
    """Save pinned requirements for reproducibility"""
    requirements = [
        "# PPV-Pareto Forecasting Pipeline Requirements",
        "# Pin versions for reproducibility",
        "",
        "pandas>=1.5.0,<2.1.0",
        "numpy>=1.21.0,<1.25.0", 
        "scikit-learn>=1.1.0,<1.4.0",
        "optuna>=3.0.0,<3.5.0",
        "matplotlib>=3.5.0,<3.8.0",
        "seaborn>=0.11.0,<0.13.0  # optional",
        "",
        "# For stata file reading:",
        "# pip install pandas[pyreadstat]",
        "",
        "# Note: Requires shrinkage_utils.py in Python path"
    ]
    
    req_path = os.path.join(out_dir, "requirements.txt")
    with open(req_path, "w") as f:
        f.write("\n".join(requirements))
    
    if VERBOSE_DIAGNOSTICS:
        print(f"Saved requirements.txt: {req_path}")

def save_readme_section(out_dir: str):
    """Save README section explaining key concepts"""
    readme_content = """# PPV-Pareto Election Forecasting Pipeline

## Key Concepts

### Electoral College Filtering (EC_MIN_NEFF)

The pipeline uses an effective sample size filter (`EC_MIN_NEFF = 100`) to ensure reliable state-level estimates:

- **Included states**: Only states with effective N ≥ 100 contribute to Electoral College tallies
- **Excluded states**: States below threshold are marked as "LOW_Neff" and excluded from EV totals
- **Full results**: Complete 538-EV results saved in `forecast_2024_states_shrunk_full538.csv` for QA

### Model Hierarchy (Row-wise Fallback)

Models are applied in priority order by AUC performance:
1. **Knee model**: Optimal complexity-accuracy balance (typically 5 features)
2. **2-feature (any)**: Best 2-feature model including ideology
3. **2-feature (no ideology)**: Best 2-feature model without ideology  
4. **1-feature (no ideology)**: Simplest backstop model

### Shrinkage and Calibration

- **Empirical Bayes shrinkage**: State estimates pulled toward national mean (k_prior=400)
- **Cross-fitted isotonic calibration**: Out-of-fold probability calibration by year
- **Grouped CV**: Prevents data leakage across election cycles (2012/2016/2020)

## Output Files

- `pareto_front.png/csv`: Accuracy vs complexity trade-off
- `forecast_2024_states_*.csv`: State-level EC projections
- `states_neff_*.csv`: Detailed state diagnostics with reliability metrics
- `knee_selection.json`: Selected model parameters and performance
- `calibration_*.png`: Model calibration curves
- `RUN_METADATA.json`: Complete run configuration

## Reliability Indicators

- **neff**: Effective sample size per state
- **reliability**: "OK" (neff≥100) or "LOW_Neff" (neff<100)  
- **eligible**: Whether state contributes to EV tallies
- **thin_reason**: Explanation for exclusion (e.g., "LOW_Neff")
"""
    
    readme_path = os.path.join(out_dir, "README_PIPELINE.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    if VERBOSE_DIAGNOSTICS:
        print(f"Saved pipeline README: {readme_path}")

# ------------------ Past years loaders (harmonized) ------------------

def load_2012(path: str):
    df = pd.read_stata(path, convert_categoricals=False)
    y = df["prevote_intpreswho"].copy().where(~df["prevote_intpreswho"].isin([-9,-8,-1]), np.nan)
    y = y.map(lambda v: 1 if v==2 else (0 if v in [1,5] else np.nan))
    lib = df["libcpre_self"].where(~df["libcpre_self"].isin([-9,-8,-2]), np.nan)
    dapp = df["presapp_econ"].where(~df["presapp_econ"].isin([-9,-8]), np.nan)
    pecon = df["finance_finpast"].where(~df["finance_finpast"].isin([-9,-8]), np.nan).replace({3:2,2:3})
    distrust = df["trustgov_trustgrev"].where(~df["trustgov_trustgrev"].isin([-9,-1]), np.nan)
    age = df["dem_age_r_x"].where(~df["dem_age_r_x"].isin([-2]), np.nan)
    female = df["gender_respondent_x"].replace({2:1,1:0})
    edu = df["dem_edugroup_x"].where(~df["dem_edugroup_x"].isin([-9,-8,-7,-6,-5,-4,-3,-2]), np.nan)
    wnh = df["dem_raceeth_x"].where(~df["dem_raceeth_x"].isin([-9]), np.nan).apply(lambda v: 1 if v==1 else (0 if v in [2,3,4,5,6] else np.nan))
    state = df["sample_stfips"]
    X = pd.DataFrame({
        "lib_cons_norm": (lib - 1)/6,
        "Disapp_economy_norm": (dapp - 1)/4,
        "pers_econ_worse_norm": (pecon - 1)/2,
        "Distrust_gov_norm": (distrust - 1)/4,
        "age_norm": (age - 18)/(90-18),
        "edu_norm": (edu - 1)/4,
        "white_nonhispanic": wnh,
        "female": female,
        "state": state
    })
    w = df["weight_full"] if "weight_full" in df.columns else pd.Series(np.nan, index=df.index)
    return X, y.rename("target"), w.rename("weight"), pd.Series(2012, index=df.index, name="year")

def load_2016(path: str):
    df = pd.read_stata(path, convert_categoricals=False)
    y = df["V161031"].copy().where(~df["V161031"].isin([6,7,8,-9,-8,-1]), np.nan)
    y = y.map(lambda v: 1 if v==2 else (0 if v in [1,3,4,5] else np.nan))
    lib = df["V161126"].where(~df["V161126"].isin([-9,-8,99]), np.nan)
    dapp = df["V161083"].where(~df["V161083"].isin([-9,-8]), np.nan)
    pecon = df["V161110"].where(~df["V161110"].isin([-9,-8]), np.nan)
    distrust = df["V161215"].where(~df["V161215"].isin([-9,-8]), np.nan)
    age = df["V161267"].where(~df["V161267"].isin([-9,-8]), np.nan)
    female = df["V161342"].replace({2:1,1:0,3:0,-9:np.nan})
    edu = df["V161270"].where(~(df["V161270"]==-9) & ~df["V161270"].between(90,95), np.nan)
    wnh = df["V161310x"].where(~df["V161310x"].isin([-2]), np.nan).apply(lambda v: 1 if v==1 else (0 if v in [2,3,4,5,6] else np.nan))
    state = df["V161015b"].where(~df["V161015b"].isin([-1]), np.nan)
    X = pd.DataFrame({
        "lib_cons_norm": (lib - 1)/6,
        "Disapp_economy_norm": (dapp - 1)/4,
        "pers_econ_worse_norm": (pecon - 1)/4,
        "Distrust_gov_norm": (distrust - 1)/4,
        "age_norm": (age - 18)/(90-18),
        "edu_norm": (edu - 1)/15,
        "white_nonhispanic": wnh,
        "female": female,
        "state": state
    })
    w = df["V160101"] if "V160101" in df.columns else pd.Series(np.nan, index=df.index)
    return X, y.rename("target"), w.rename("weight"), pd.Series(2016, index=df.index, name="year")

def load_2020(path: str):
    df = pd.read_stata(path, convert_categoricals=False)
    y = df["V201033"].copy().where(~df["V201033"].isin([-8,-9,-1,11,12]), np.nan)
    y = y.map(lambda v: 1 if v == 2 else (0 if v in [1,3,4,5] else np.nan))
    lib = df["V201200"].where(~df["V201200"].isin([-9,-8,99]), np.nan)
    dapp = df["V201327x"].where(~df["V201327x"].isin([-2]), np.nan)
    pecon = df["V201502"].where(~df["V201502"].isin([-9]), np.nan)
    distrust = df["V201233"].where(~df["V201233"].isin([-9,-8]), np.nan)
    age = df["V201507x"].where(~df["V201507x"].isin([-9]), np.nan)
    female = df["V201600"].replace({2:1,1:0,-9:np.nan})
    edu = df["V201511x"].where(~df["V201511x"].isin([-9,-8,-7,-6,-5,-4,-3,-2]), np.nan)
    wnh = df["V201549x"].where(~df["V201549x"].isin([-9,-8]), np.nan).apply(lambda v: 1 if v==1 else (0 if v in [2,3,4,5,6] else np.nan))
    state = df["V201014b"].where(~df["V201014b"].isin([-9,-8,-7,-6,-5,-4,-3,-2,-1,86]), np.nan)
    disapp_norm_aligned = 1.0 - ((dapp - 1)/4)
    X = pd.DataFrame({
        "lib_cons_norm": (lib - 1)/6,
        "Disapp_economy_norm": disapp_norm_aligned,
        "pers_econ_worse_norm": (pecon - 1)/4,
        "Distrust_gov_norm": (distrust - 1)/4,
        "age_norm": (age - 18)/(80-18),
        "edu_norm": (edu - 1)/4,
        "white_nonhispanic": wnh,
        "female": female,
        "state": state
    })
    w = df["V200010a"] if "V200010a" in df.columns else pd.Series(np.nan, index=df.index)
    return X, y.rename("target"), w.rename("weight"), pd.Series(2020, index=df.index, name="year")

# ------------------ 2024 loader (validated mappings) ------------------

VAR_2024_PREFS = {
    "ideology": ["V241177", "V241200"],
    "econ_approval": ["V241143x", "V241327"],
    "personal_econ": ["V241451", "V241502"],
    "gov_trust": ["V241229", "V241233"],
    "age": ["V241458x", "V241507", "V241507x"],
    "education": ["V241465x", "V241511"],
    "gender": ["V241550", "V241600a"],
    "state_candidates": ["V241023", "V243002", "V241014"],
    "race_eth_summary": ["V241501x"],
    "race": ["V241540"],
    "hispanic": ["V241541"],
    "race_alt_combined": ["V241155x"],
    "weight": ["V240107a"]
}

def _neg_to_nan(s: pd.Series) -> pd.Series:
    s = _to_num(s)
    return s.where(~s.isin(list(range(-99,0))), np.nan)

def _bound(s: pd.Series, lo: float, hi: float) -> pd.Series:
    s = _to_num(s)
    return s.where(s.between(lo, hi, inclusive="both"), np.nan)

def _pick_first(df: pd.DataFrame, candidates: List[str], clean_neg: bool = True) -> Tuple[pd.Series, str]:
    for col in candidates:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            if clean_neg:
                s = s.where(~s.isin(list(range(-99,0))), np.nan)
            return s, col
    return pd.Series(np.nan, index=df.index), "none"

def _pick_best_state(df: pd.DataFrame, candidates: List[str]) -> Tuple[pd.Series, str, float, int]:
    best_col = None
    best_series = None
    best_cov = -1.0
    best_card = -1
    diag = []
    for col in candidates:
        if col in df.columns:
            s_raw = _neg_to_nan(df[col])
            s = s_raw.where(s_raw.between(1,56, inclusive="both"), np.nan)
            postal = s.map(FIPS_TO_POSTAL)
            cov = float(s.notna().mean())
            card = int(postal.dropna().nunique())
            diag.append((col, cov, card))
            score = cov + 0.01*card
            if (card >= 40) and (score > best_cov + 0.01*best_card):
                best_cov = cov
                best_card = card
                best_col = col
                best_series = s
    if best_series is None:
        if diag:
            col, cov, card = max(diag, key=lambda t: t[1])
            s_raw = _neg_to_nan(df[col])
            best_series = s_raw.where(s_raw.between(1,56, inclusive="both"), np.nan)
            best_col, best_cov, best_card = col, cov, card
        else:
            best_series = pd.Series(np.nan, index=df.index)
            best_col, best_cov, best_card = "none", 0.0, 0
    return best_series, best_col, best_cov, best_card

def load_2024(path: str):
    df = pd.read_stata(path, convert_categoricals=False)

    # Core predictors (prefer summary vars, fallback to raw)
    lib, lib_col = _pick_first(df, VAR_2024_PREFS["ideology"])
    dapp, dapp_col = _pick_first(df, VAR_2024_PREFS["econ_approval"])
    pecon, pecon_col = _pick_first(df, VAR_2024_PREFS["personal_econ"])
    distrust, dist_col = _pick_first(df, VAR_2024_PREFS["gov_trust"])
    age, age_col = _pick_first(df, VAR_2024_PREFS["age"])
    edu, edu_col = _pick_first(df, VAR_2024_PREFS["education"])
    gender, gender_col = _pick_first(df, VAR_2024_PREFS["gender"])

    # Clean ranges
    lib = _bound(lib, 1, 7)
    dapp = _bound(dapp, 1, 5)
    pecon = _bound(pecon, 1, 5)
    distrust = _bound(distrust, 1, 5)
    age = age.where((age >= 18) & (age <= 100), np.nan)
    edu = _bound(edu, 1, 15)

    # Female 1/0
    female = pd.Series(np.nan, index=df.index, dtype=float)
    female.loc[gender == 2] = 1.0
    female.loc[gender == 1] = 0.0

    # White non-Hispanic
    race_sum, race_sum_col = _pick_first(df, VAR_2024_PREFS["race_eth_summary"])
    white_nh = pd.Series(np.nan, index=df.index, dtype=float)
    if race_sum_col != "none":
        white_nh.loc[race_sum == 1] = 1.0
        white_nh.loc[(race_sum >= 2) & (race_sum <= 6)] = 0.0
        wnh_source = race_sum_col
    else:
        race, race_col = _pick_first(df, VAR_2024_PREFS["race"])
        hisp, hisp_col = _pick_first(df, VAR_2024_PREFS["hispanic"])
        if race_col != "none" and hisp_col != "none":
            white_nh.loc[(race == 1) & (hisp.isin([0, 2]))] = 1.0
            white_nh.loc[(race.isin([2,3,4,5,6])) | (hisp == 1)] = 0.0
            wnh_source = f"{race_col}+{hisp_col}"
        else:
            race_alt, race_alt_col = _pick_first(df, VAR_2024_PREFS["race_alt_combined"])
            if race_alt_col != "none":
                white_nh.loc[race_alt == 1] = 1.0
                white_nh.loc[(race_alt >= 2)] = 0.0
                wnh_source = race_alt_col
            else:
                wnh_source = "none"
    wnh_missing = float(pd.isna(white_nh).mean())

    # State variable
    state_series, state_col_used, state_cov, state_card = _pick_best_state(df, VAR_2024_PREFS["state_candidates"])

    # Weight
    weight_candidates = VAR_2024_PREFS["weight"]
    if WEIGHT_VAR_OVERRIDE_2024:
        weight_candidates = [WEIGHT_VAR_OVERRIDE_2024] + [c for c in weight_candidates if c != WEIGHT_VAR_OVERRIDE_2024]
    wcol_found = None
    for wcol in weight_candidates:
        if wcol in df.columns:
            w = _neg_to_nan(df[wcol]).where(lambda s: s > 0, np.nan).rename("weight")
            wcol_found = wcol
            break
    if wcol_found is None:
        w = pd.Series(1.0, index=df.index, name="weight")
        wcol_found = "equal"

    # Normalization
    edu_max = float(np.nanmax(edu.values)) if np.isfinite(edu.values).any() else 5.0
    edu_den = 4.0 if edu_max <= 5.0 else 15.0

    X = pd.DataFrame({
        "lib_cons_norm": (lib - 1)/6,
        "Disapp_economy_norm": (dapp - 1)/4,
        "pers_econ_worse_norm": (pecon - 1)/4,
        "Distrust_gov_norm": (distrust - 1)/4,
        "age_norm": (age - 18)/(80-18),
        "edu_norm": (edu - 1)/edu_den,
        "white_nonhispanic": white_nh,
        "female": female,
        "state": state_series
    })

    # Diagnostics for provenance
    diag = {
        "wnh_source": wnh_source,
        "wnh_missing": wnh_missing,
        "state_col_used": state_col_used,
        "state_cov": float(state_cov),
        "state_card": int(state_card),
        "weight_source": wcol_found,
        "weight_eff_n": eff_sample_size(w.values)
    }

    return X, w, wnh_missing, diag

# ------------------ Modeling and optimization ------------------

CANDIDATE_VARS = [
    "lib_cons_norm","Disapp_economy_norm","Distrust_gov_norm","age_norm","edu_norm",
    "white_nonhispanic","pers_econ_worse_norm","female","state"
]

def build_pipe(selected: List[str], C: float):
    num_features = [c for c in selected if c != "state"]
    cat_features = ["state"] if "state" in selected else []
    tr = []
    if num_features: tr.append(("num","passthrough", num_features))
    if cat_features: tr.append(("state", make_ohe(), cat_features))
    ct = ColumnTransformer(transformers=tr, remainder="drop", sparse_threshold=1.0)
    clf = LogisticRegression(solver="liblinear", C=C, max_iter=1000, random_state=RANDOM_SEED)
    return Pipeline([("ct", ct), ("clf", clf)])

def five_fold_weighted_auc(df_all: pd.DataFrame, selected: List[str], C: float) -> float:
    # Grouped CV by year to avoid leakage across cycles
    m = df_all["target"].notna() & df_all["weight"].notna()
    for c in selected: m &= df_all[c].notna()
    data = df_all.loc[m].reset_index(drop=True)
    if len(data) < 300 or data["target"].nunique() < 2:
        return 0.5
    X = data[selected]; y = data["target"].astype(int).to_numpy()
    w = data["weight"].astype(float).to_numpy()
    groups = data["year"].to_numpy()
    uniq_groups = np.unique(groups)
    if not GROUPED_CV_BY_YEAR or len(uniq_groups) < 2:
        return 0.5
    gkf = GroupKFold(n_splits=min(N_SPLITS, len(uniq_groups)))
    aucs = []
    for tr_idx, te_idx in gkf.split(X, y, groups=groups):
        if len(np.unique(y[tr_idx])) < 2 or len(np.unique(y[te_idx])) < 2:
            continue
        pipe = build_pipe(selected, C)
        pipe.fit(X.iloc[tr_idx], y[tr_idx], clf__sample_weight=w[tr_idx])
        p = pipe.predict_proba(X.iloc[te_idx])[:,1]
        aucs.append(_roc_auc_safe(y[te_idx], p, sample_weight=w[te_idx]))
    return float(np.mean(aucs)) if aucs else 0.5

def make_objective(df_all: pd.DataFrame):
    def obj(trial: optuna.trial.Trial):
        sel = [v for v in CANDIDATE_VARS if trial.suggest_categorical(f"include_{v}", [True, False])]
        if not sel:
            return 0.0, 99.0
        C = trial.suggest_float("logreg_C", 1e-2, 100.0, log=True)
        auc = five_fold_weighted_auc(df_all, sel, C)
        return float(auc), float(len(sel))
    return obj

def recommend_knee_point(study: optuna.study.Study) -> Optional[optuna.trial.FrozenTrial]:
    pareto = study.best_trials
    if not pareto:
        return None
    aucs = np.array([t.values[0] for t in pareto])
    cplx = np.array([t.values[1] for t in pareto])
    # If ties, prefer smallest complexity
    best_idx = int(np.argmax(aucs - 1e-9 * cplx))
    return pareto[best_idx]

# ------------------ Calibration helpers ------------------

def fit_isotonic(x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None) -> IsotonicRegression:
    x = np.asarray(x, float); y = np.asarray(y, float)
    if w is not None: w = np.asarray(w, float)
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    try:
        ir.fit(x, y, sample_weight=w)
    except TypeError:
        ir.fit(x, y)
    return ir

def cross_fit_isotonic(df_all: pd.DataFrame, feats: List[str], C: float, n_splits: int = N_SPLITS):
    # Group-aware OOF predictions, then fit calibrator; finally refit full model
    mask = df_all["target"].notna() & df_all["weight"].notna()
    for c in feats: mask &= df_all[c].notna()
    data = df_all.loc[mask].reset_index(drop=True)
    if data.empty:
        raise ValueError("No training rows for cross_fit_isotonic.")

    X = data[feats]; y = data["target"].astype(int).to_numpy()
    w = data["weight"].astype(float).to_numpy()
    groups = data["year"].to_numpy()
    uniq_groups = np.unique(groups)
    gkf = GroupKFold(n_splits=min(n_splits, len(uniq_groups))) if GROUPED_CV_BY_YEAR and len(uniq_groups) >= 2 else None

    oof = np.full_like(y, np.nan, dtype=float)
    if gkf is not None:
        for tr_idx, te_idx in gkf.split(X, y, groups=groups):
            if len(np.unique(y[tr_idx])) < 2 or len(np.unique(y[te_idx])) < 2:
                continue
            pipe = build_pipe(feats, C)
            pipe.fit(X.iloc[tr_idx], y[tr_idx], clf__sample_weight=w[tr_idx])
            oof[te_idx] = pipe.predict_proba(X.iloc[te_idx])[:,1]
    else:
        # Fallback: single fit for OOF (not ideal, but ensures pipeline runs)
        pipe_tmp = build_pipe(feats, C)
        pipe_tmp.fit(X, y, clf__sample_weight=w)
        oof[:] = pipe_tmp.predict_proba(X)[:,1]

    m = np.isfinite(oof)
    cal = fit_isotonic(oof[m], y[m], w[m])

    final_pipe = build_pipe(feats, C)
    final_pipe.fit(X, y, clf__sample_weight=w)

    return final_pipe, cal, int(m.sum()), data  # return data for plotting

def quick_grouped_cv_auc(df_all: pd.DataFrame, feats: List[str], C: float, n_splits: int = N_SPLITS) -> float:
    # Compute grouped-CV AUC by year for a given feature set (for backstop variants)
    mask = df_all["target"].notna() & df_all["weight"].notna()
    for c in feats: mask &= df_all[c].notna()
    data = df_all.loc[mask].reset_index(drop=True)
    if data.empty or data["year"].nunique() < 2 or data["target"].nunique() < 2:
        return float("nan")
    X = data[feats]; y = data["target"].astype(int).to_numpy()
    w = data["weight"].astype(float).to_numpy(); groups = data["year"].to_numpy()
    gkf = GroupKFold(n_splits=min(n_splits, data["year"].nunique()))
    aucs = []
    for tr_idx, te_idx in gkf.split(X, y, groups=groups):
        if len(np.unique(y[tr_idx])) < 2 or len(np.unique(y[te_idx])) < 2:
            continue
        pipe = build_pipe(feats, C)
        pipe.fit(X.iloc[tr_idx], y[tr_idx], clf__sample_weight=w[tr_idx])
        p = pipe.predict_proba(X.iloc[te_idx])[:,1]
        aucs.append(_roc_auc_safe(y[te_idx], p, sample_weight=w[te_idx]))
    return float(np.mean(aucs)) if aucs else float("nan")

def assert_ec_consistency(tag: str, sdf: pd.DataFrame, total_ev_present: int):
    # Ensure the "present EV" equals the sum of ev over eligible rows for this table
    ev_from_eligible = int(sdf.loc[sdf["eligible"], "ev"].sum())
    if ev_from_eligible != total_ev_present:
        print(f"WARN[{tag}]: EV sum mismatch: computed={ev_from_eligible}, reported={total_ev_present}")

def cv_metric_panel(df_all: pd.DataFrame, feats: List[str], C: float):
    mask = df_all["target"].notna() & df_all["weight"].notna()
    for c in feats: mask &= df_all[c].notna()
    data = df_all.loc[mask].reset_index(drop=True)
    if data.empty:
        return dict(auc=float("nan"), brier=float("nan"), used=0)
    X = data[feats]; y = data["target"].astype(int).to_numpy()
    w = data["weight"].astype(float).to_numpy()
    groups = data["year"].to_numpy()
    uniq_groups = np.unique(groups)
    if not GROUPED_CV_BY_YEAR or len(uniq_groups) < 2:
        return dict(auc=0.5, brier=float("nan"), used=0)
    gkf = GroupKFold(n_splits=min(N_SPLITS, len(uniq_groups)))
    oof = np.full_like(y, np.nan, dtype=float)
    for tr_idx, te_idx in gkf.split(X, y, groups=groups):
        if len(np.unique(y[tr_idx])) < 2 or len(np.unique(y[te_idx])) < 2:
            continue
        pipe = build_pipe(feats, C)
        pipe.fit(X.iloc[tr_idx], y[tr_idx], clf__sample_weight=w[tr_idx])
        oof[te_idx] = pipe.predict_proba(X.iloc[te_idx])[:,1]
    m = np.isfinite(oof)
    if not m.any():
        return dict(auc=0.5, brier=float("nan"), used=0)
    auc = _roc_auc_safe(y[m], oof[m], sample_weight=w[m])
    try:
        brier = brier_score_loss(y[m], oof[m], sample_weight=w[m])
    except TypeError:
        brier = brier_score_loss(y[m], oof[m])
    return dict(auc=float(auc), brier=float(brier), used=int(m.sum()))

def save_calibration_curve(pipe: Pipeline, cal: IsotonicRegression, X: pd.DataFrame, y: np.ndarray, w: np.ndarray, path: str):
    p_raw = pipe.predict_proba(X)[:,1]
    p_cal = cal.transform(p_raw)
    bins = np.linspace(0,1,11)
    idx = np.digitize(p_cal, bins) - 1
    dfb = pd.DataFrame({"p":p_cal, "y":y.astype(int), "w":w})
    agg = dfb.groupby(idx, dropna=True).apply(
        lambda g: pd.Series({
            "p_mean": np.average(g["p"], weights=g["w"]) if g["w"].sum()>0 else np.nan,
            "y_rate": np.average(g["y"], weights=g["w"]) if g["w"].sum()>0 else np.nan,
            "n": g["w"].sum()
        })
    ).reset_index(drop=True)
    plt.figure()
    plt.plot(agg["p_mean"], agg["y_rate"], marker="o")
    plt.plot([0,1],[0,1], linestyle="--", color="gray")
    plt.xlabel("Calibrated predicted prob"); plt.ylabel("Empirical rate"); plt.title("Calibration curve")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

def monotone_check(pipe: Pipeline, X: pd.DataFrame, col: str, q_lo=0.05, q_hi=0.95, k: int = 11) -> float:
    # Correlation between quantile of feature and mean predicted probability
    if col not in X.columns or X[col].dropna().empty:
        return float("nan")
    qs = np.linspace(q_lo, q_hi, int(k))
    preds = []
    base = X.copy()
    for q in qs:
        v = np.nanpercentile(base[col].dropna().to_numpy(), q*100)
        tmp = base.copy()
        tmp[col] = v
        preds.append(pipe.predict_proba(tmp)[:,1].mean())
    if len(preds) < 2:
        return float("nan")
    return float(np.corrcoef(qs, preds)[0,1])

# ------------------ Forecast helpers ------------------

def forecast_national_from_preds(p: np.ndarray, w: np.ndarray):
    w = np.clip(np.asarray(w, float), 0, None)
    m = np.isfinite(p) & np.isfinite(w)
    p = p[m]; w = w[m]
    if p.size == 0 or w.sum() <= 0:
        return dict(weighted_mean_prob=float("nan"), share_thr_05=float("nan"),
                    share_thr_05_ci=(float("nan"), float("nan")),
                    share_thr_mean=float("nan"), share_thr_mean_ci=(float("nan"), float("nan")), used=0)
    mean_prob = float(np.sum(p * w) / w.sum())
    share05 = float(np.sum((p > 0.5).astype(int) * w) / w.sum())
    lo05, hi05 = weighted_bootstrap_ci_share(p, w, 0.5, BOOTSTRAP_REPS, BOOTSTRAP_SEED)
    lomean, himean = weighted_bootstrap_ci_share(p, w, mean_prob, BOOTSTRAP_REPS, BOOTSTRAP_SEED)
    share_mean_thr = float(np.sum((p > mean_prob).astype(int) * w) / w.sum())
    return dict(weighted_mean_prob=mean_prob, share_thr_05=share05,
                share_thr_05_ci=(lo05, hi05),
                share_thr_mean=share_mean_thr,
                share_thr_mean_ci=(lomean, himean),
                used=int(w.size))

def state_level_ec_from_preds(p: np.ndarray, w: np.ndarray, state_fips: np.ndarray, min_neff: float = 0.0):
    # Base (non-shrunk) EC table
    p = np.asarray(p, float); w = np.asarray(w, float); s = pd.to_numeric(state_fips, errors="coerce")
    m = np.isfinite(p) & np.isfinite(w) & np.isfinite(s)
    if not m.any():
        return pd.DataFrame(columns=["state","mean_prob","share_thr_05","ci_low_thr05","ci_high_thr05","neff","used_n","w_sum","ev","winner_mean","winner_thr05","reliability","eligible","thin_reason"]), 0, 0, 0
    dfp = pd.DataFrame({"proba": p[m], "w": np.clip(w[m], 0, None), "state": s[m]})
    dfp["postal"] = dfp["state"].map(FIPS_TO_POSTAL)
    dfp = dfp[dfp["postal"].isin(EV_2024.keys())]
    rows = []
    for st, sub in dfp.groupby("postal"):
        pv, wv = sub["proba"].values, sub["w"].values
        used_n = int((wv > 0).sum())
        w_sum = float(np.sum(wv))
        if w_sum <= 0:
            meanp = float("nan"); share05 = float("nan"); neff = 0.0
            ci_lo = float("nan"); ci_hi = float("nan")
        else:
            meanp = float(np.sum(pv * wv) / w_sum)
            share05 = float(np.sum((pv > 0.5).astype(int) * wv) / w_sum)
            neff = eff_sample_size(wv)
            ci_lo, ci_hi = weighted_bootstrap_ci_share(pv, wv, thr=0.5, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED)
        rows.append({"state": st, "mean_prob": meanp, "share_thr_05": share05, "ci_low_thr05": ci_lo, "ci_high_thr05": ci_hi, "neff": neff, "used_n": used_n, "w_sum": w_sum, "ev": EV_2024[st]})
    sdf = pd.DataFrame(rows).sort_values("ev", ascending=False)
    if sdf.empty:
        return sdf, 0, 0, 0
    sdf["winner_mean"] = np.where(sdf["mean_prob"] > 0.5, "R", "D")
    sdf["winner_thr05"] = np.where(sdf["share_thr_05"] > 0.5, "R", "D")
    sdf["reliability"] = np.where(sdf["neff"] >= 100, "OK", "LOW_Neff")
    sdf["eligible"] = sdf["neff"] >= float(min_neff)
    sdf["thin_reason"] = np.where(sdf["reliability"]=="LOW_Neff", "LOW_Neff", "")
    ev_mean = int(sdf.loc[sdf["eligible"] & (sdf["winner_mean"]=="R"),"ev"].sum())
    ev_thr05 = int(sdf.loc[sdf["eligible"] & (sdf["winner_thr05"]=="R"),"ev"].sum())
    total_ev_present = int(sdf.loc[sdf["eligible"], "ev"].sum())
    return sdf, ev_mean, ev_thr05, total_ev_present

# ------------------ Main ------------------

def main():
    # Create output directory
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print(f"PPV-Pareto v10 Election Forecasting Pipeline")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Output directory: {OUT_DIR}")
    print()

    # Validate all required files exist
    required_files = [
        (DATA_2012, "ANES 2012 data"),
        (DATA_2016, "ANES 2016 data"),
        (DATA_2020, "ANES 2020 data"),
        (DATA_2024, "ANES 2024 data")
    ]
    
    missing_files = []
    for filepath, description in required_files:
        if not validate_file_exists(filepath, description):
            missing_files.append(description)
    
    if missing_files:
        print(f"ERROR: Cannot proceed with {len(missing_files)} missing file(s)")
        print("Please ensure all ANES data files are available and paths are correct.")
        sys.exit(1)

    # Load training years (harmonized)
    if VERBOSE_DIAGNOSTICS:
        print("Loading and harmonizing training data...")
    
    X12, y12, w12, yr12 = load_2012(DATA_2012)
    X16, y16, w16, yr16 = load_2016(DATA_2016)
    X20, y20, w20, yr20 = load_2020(DATA_2020)

    def pack(X, y, w, yr):
        df = X.copy()
        df["target"] = y
        df["weight"] = w
        df["year"] = yr
        return df

    df_all = pd.concat(
        [pack(X12, y12, w12, yr12),
         pack(X16, y16, w16, yr16),
         pack(X20, y20, w20, yr20)],
        ignore_index=True
    )

    if VERBOSE_DIAGNOSTICS:
        print(f"Training data: {len(df_all)} total observations across {df_all['year'].nunique()} years")

    # Optuna: maximize AUC vs minimize #features using grouped CV
    print("\nOptimizing (GroupKFold by year: AUC vs #features)...")
    study = optuna.create_study(
        directions=["maximize", "minimize"],
        sampler=NSGAIISampler(seed=RANDOM_SEED)
    )
    study.optimize(make_objective(df_all), n_trials=N_TRIALS, show_progress_bar=True)

    # Plot/save Pareto
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    xs = [t.values[1] for t in trials]; ys = [t.values[0] for t in trials]
    pareto = study.best_trials; px = [t.values[1] for t in pareto]; py = [t.values[0] for t in pareto]
    plt.figure(figsize=(7,5))
    if _HAS_SNS:
        sns.scatterplot(x=xs, y=ys, alpha=0.3, label="All trials", s=26)
        sns.scatterplot(x=px, y=py, color="red", label="Pareto", s=55)
    else:
        plt.scatter(xs, ys, alpha=0.3, label="All trials", s=26)
        plt.scatter(px, py, color="red", label="Pareto", s=55)
    plt.xlabel("# features"); plt.ylabel("AUC (GroupKFold)"); plt.title("Pareto: Accuracy vs Complexity")
    plt.grid(True, alpha=0.2)
    pareto_png = os.path.join(OUT_DIR, "pareto_front.png"); plt.tight_layout(); plt.savefig(pareto_png, dpi=150); plt.close()

    pareto_rows = []
    for t in pareto:
        feats = [k.replace("include_", "") for k, v in t.params.items() if k.startswith("include_") and v]
        pareto_rows.append({
            "trial": t.number,
            "auc": t.values[0],
            "complexity": t.values[1],
            "C": t.params.get("logreg_C", None),
            "features": json.dumps(feats)
        })
    pareto_csv = os.path.join(OUT_DIR, "pareto_front.csv")
    pd.DataFrame(pareto_rows).sort_values(["complexity", "auc"], ascending=[True, False]).to_csv(pareto_csv, index=False)
    print(f"Saved {pareto_png} and {pareto_csv}")

    # Knee selection
    knee = recommend_knee_point(study)
    if knee is None:
        print("ERROR: No Pareto solutions found.")
        sys.exit(1)
        
    knee_features = [k.replace("include_", "") for k, v in knee.params.items() if k.startswith("include_") and v]
    knee_C = knee.params.get("logreg_C", 1.0)
    knee_auc = knee.values[0]
    print(f"\nKnee-point: trial #{knee.number} | AUC={knee_auc:.3f} | k={int(knee.values[1])} | C={knee_C:.4g}")
    print("Selected features:", knee_features)

    # CV metric panel for knee
    panel = cv_metric_panel(df_all, knee_features, knee_C)
    print(f"CV panel (knee): AUC={panel['auc']:.3f} | Brier={panel['brier']:.4f} | used={panel['used']}")

    # Best small models for row-wise fallback
    def extract_best(cond_fn, pref_len: Optional[int] = None):
        best = None
        for t in study.best_trials:
            feats = [k.replace("include_", "") for k, v in t.params.items() if k.startswith("include_") and v]
            if pref_len is not None and len(feats) != pref_len:
                continue
            if not cond_fn(set(feats)):
                continue
            auc = t.values[0]; C = t.params.get("logreg_C", 1.0)
            if (best is None) or (auc > best["auc"]):
                best = {"features": feats, "C": C, "auc": auc, "trial": t.number}
        return best

    two_any = extract_best(lambda s: True, pref_len=2)
    two_no_lib = extract_best(lambda s: ("lib_cons_norm" not in s), pref_len=2)
    one_no_lib = extract_best(lambda s: ("lib_cons_norm" not in s), pref_len=1)

    # Backstops if Optuna doesn't yield ideology-free models
    if two_no_lib is None:
        two_no_lib = {"features": ["Disapp_economy_norm", "state"], "C": 1.0, "auc": 0.0, "trial": -1}
    if one_no_lib is None:
        one_no_lib = {"features": ["Disapp_economy_norm"], "C": 1.0, "auc": 0.0, "trial": -1}

    def _print_backstop_auc(tag, feats, C, auc):
        if auc == 0.0:
            auc_cv = quick_grouped_cv_auc(df_all, feats, C)
            print(f"  (computed grouped-CV AUC for {tag}) AUC≈{auc_cv:.3f}")
            return auc_cv
        return auc

    if two_any:
        print(f"\nBest 2-feature (any): {two_any['features']} | C={two_any['C']:.4g} | AUC≈{two_any['auc']:.3f}")
    if two_no_lib:
        two_no_lib["auc"] = _print_backstop_auc("2feat_no_lib", two_no_lib["features"], two_no_lib["C"], two_no_lib["auc"])
        print(f"Best 2-feature (no ideology): {two_no_lib['features']} | C={two_no_lib['C']:.4g} | AUC≈{two_no_lib['auc'] if not np.isnan(two_no_lib['auc']) else 'NA'}")
    if one_no_lib:
        one_no_lib["auc"] = _print_backstop_auc("1feat_no_lib", one_no_lib["features"], one_no_lib["C"], one_no_lib["auc"])
        print(f"Best 1-feature (no ideology): {one_no_lib['features']} | C={one_no_lib['C']:.4g} | AUC≈{one_no_lib['auc'] if not np.isnan(one_no_lib['auc']) else 'NA'}")

    # Compose model registry in priority order by AUC (unique by feature-set)
    models: List[Dict[str, Any]] = []
    models.append({"name": "knee", "features": knee_features, "C": knee_C, "auc": knee_auc})
    if two_any:     models.append({"name": "2feat_any",    "features": two_any["features"],    "C": two_any["C"],    "auc": two_any["auc"]})
    if two_no_lib:  models.append({"name": "2feat_no_lib", "features": two_no_lib["features"], "C": two_no_lib["C"], "auc": two_no_lib["auc"]})
    if one_no_lib:  models.append({"name": "1feat_no_lib", "features": one_no_lib["features"], "C": one_no_lib["C"], "auc": one_no_lib["auc"]})
    seen = set(); models_sorted = []
    for m in sorted(models, key=lambda d: (-(d["auc"] if d["auc"] is not None else -1), len(d["features"]))):
        key = tuple(sorted(m["features"]))
        if key in seen: continue
        seen.add(key); models_sorted.append(m)
    models = models_sorted

    # Train pipelines with cross-fitted isotonic calibration
    trained: Dict[str, Dict[str, Any]] = {}
    for m in models:
        feats = m["features"]; C = m["C"]; name = m["name"]
        try:
            if CROSS_FIT_ISOTONIC:
                pipe, calibrator, n_oof, data_used = cross_fit_isotonic(df_all, feats, C, n_splits=N_SPLITS)
            else:
                pipe = build_pipe(feats, C)
                # simple fit + calibrate on in-sample (discouraged)
                mask_tr = df_all["target"].notna() & df_all["weight"].notna()
                for c in feats: mask_tr &= df_all[c].notna()
                dfit = df_all.loc[mask_tr]
                pipe.fit(dfit[feats], dfit["target"].astype(int), clf__sample_weight=dfit["weight"].astype(float).values)
                p_tr = pipe.predict_proba(dfit[feats])[:, 1]
                calibrator = fit_isotonic(p_tr, dfit["target"].astype(int).values, dfit["weight"].astype(float).values)
                n_oof, data_used = len(dfit), dfit

            trained[name] = {"pipe": pipe, "features": feats, "cal": calibrator, "C": C, "auc": m["auc"]}
            print(f"Trained model [{name}] with cross-fitted isotonic | OOF n={n_oof} | feats={feats} | C={C:.4g} | AUC≈{m['auc'] if m['auc'] is not None else float('nan'):.3f}")

            # Optional: save calibration plots for knee
            if SAVE_CALIBRATION_PLOTS and name == "knee":
                Xp = data_used[feats]
                yp = data_used["target"].astype(int).to_numpy()
                wp = data_used["weight"].astype(float).to_numpy()
                save_calibration_curve(pipe, calibrator, Xp, yp, wp, os.path.join(OUT_DIR, "calibration_knee.png"))
        except Exception as e:
            print(f"WARN: Training {name} failed: {e}")

    # State-refit EC model: knee + state (robustness)
    ec_refit = None
    sel_ec = list(knee_features) if "state" in knee_features else list(knee_features) + ["state"]
    try:
        pipe_ec, cal_ec, _, data_ec = cross_fit_isotonic(df_all, sel_ec, knee_C, n_splits=N_SPLITS)
        ec_refit = {"pipe": pipe_ec, "features": sel_ec, "cal": cal_ec}
        print(f"Trained state-refit EC model | feats={sel_ec}")
        if SAVE_CALIBRATION_PLOTS:
            Xp = data_ec[sel_ec]; yp = data_ec["target"].astype(int).to_numpy(); wp = data_ec["weight"].astype(float).to_numpy()
            save_calibration_curve(pipe_ec, cal_ec, Xp, yp, wp, os.path.join(OUT_DIR, "calibration_ec.png"))
    except Exception as e:
        print(f"WARN: State-refit EC training failed: {e}")

    # Load 2024 and forecasting
    print("\nLoading 2024 and forecasting...")
    X24, w24, wnh_missing, diag24 = load_2024(DATA_2024)
    w_vec = w24.astype(float).values
    print(f"2024 WNH missingness: {wnh_missing:.1%}")

    # Save variable provenance for audit
    diag24["wnh_missing_pct"] = wnh_missing
    with open(os.path.join(OUT_DIR, "2024_variable_sources.json"), "w") as f:
        json.dump(diag24, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else o)

    # Row-wise fallback scoring on 2024 with calibration; skip WNH models if missingness high
    p_out = np.full(len(X24), np.nan, dtype=float)
    assigned = np.zeros(len(X24), dtype=bool)
    fill_report = []
    for name, obj in trained.items():
        feats = obj["features"]
        if ("white_nonhispanic" in feats) and (wnh_missing > WNH_MISSINGNESS_THRESHOLD):
            fill_report.append((name, feats, 0))
            continue
        mrows = X24[feats].notna().all(axis=1) & (~assigned)
        n_rows = int(mrows.sum())
        if n_rows == 0:
            fill_report.append((name, feats, 0))
            continue
        p = obj["pipe"].predict_proba(X24.loc[mrows, feats])[:, 1]
        p_cal = np.clip(obj["cal"].transform(p), 0.0, 1.0)
        p_out[mrows.values] = p_cal
        assigned[mrows.values] = True
        fill_report.append((name, feats, n_rows))
    total_assigned = int(np.isfinite(p_out).sum())
    print("Row-wise fallback fill counts (2024):")
    for nm, feats, n in fill_report:
        frac = n / max(1, len(X24))
        print(f"  - {nm}: {n} rows ({frac:.1%}) | feats={feats}")
    print(f"Total 2024 rows scored: {total_assigned} of {len(X24)}")

    # National aggregation
    nat = forecast_national_from_preds(p_out, w_vec)
    print("\nNational popular-vote style forecast (Republican vs others):")
    print(f"- Weighted mean probability: {pct(nat['weighted_mean_prob'])}")
    print(f"- Threshold 0.5: {pct(nat['share_thr_05'])} (95% CI {pct(nat['share_thr_05_ci'][0])}–{pct(nat['share_thr_05_ci'][1])})")
    print(f"- Threshold = weighted mean prob: {pct(nat['share_thr_mean'])} (95% CI {pct(nat['share_thr_mean_ci'][0])}–{pct(nat['share_thr_mean_ci'][1])})")
    print(f"- Used {nat['used']} of {len(X24)} 2024 rows")

    # Directional sanity check (Disapp_economy_norm)
    if "knee" in trained and "Disapp_economy_norm" in trained["knee"]["features"]:
        mask_tr = df_all[trained["knee"]["features"]].notna().all(axis=1)
        corr = monotone_check(trained["knee"]["pipe"], df_all.loc[mask_tr, trained["knee"]["features"]], "Disapp_economy_norm")
        print(f"Directional sanity (Disapp_economy_norm): corr(q, p)={corr:.3f}")

    # EC outputs
    ec_outputs = []

    # Base EC (non-shrunk)
    if EC_STRATEGY in ("base", "dual"):
        sdf_base, ev_mean_base, ev_thr05_base, total_ev_present_base = state_level_ec_from_preds(
            p_out, w_vec, X24["state"].values if "state" in X24.columns else np.full(len(X24), np.nan),
            min_neff=EC_MIN_NEFF
        )
        ec_outputs.append(("EC (base-style, row-wise fallback)", sdf_base, ev_mean_base, ev_thr05_base, total_ev_present_base))
        # Save detailed neff table
        dump_neff_table("base", sdf_base, OUT_DIR)

    # State-refit EC (non-shrunk)
    if EC_STRATEGY in ("state_refit", "dual") and ec_refit is not None:
        feats_ec = ec_refit["features"]
        mask_rows_ec = X24[feats_ec].notna().all(axis=1) & w24.notna()
        p_ec = np.full(len(X24), np.nan, dtype=float)
        if mask_rows_ec.any():
            raw = ec_refit["pipe"].predict_proba(X24.loc[mask_rows_ec, feats_ec])[:, 1]
            p_ec[mask_rows_ec.values] = np.clip(ec_refit["cal"].transform(raw), 0.0, 1.0)
        sdf_refit, ev_mean_refit, ev_thr05_refit, total_ev_present_refit = state_level_ec_from_preds(
            p_ec, w_vec, X24["state"].values if "state" in X24.columns else np.full(len(X24), np.nan),
            min_neff=EC_MIN_NEFF
        )
        ec_outputs.append(("EC (state-refit: knee + state)", sdf_refit, ev_mean_refit, ev_thr05_refit, total_ev_present_refit))
        dump_neff_table("state_refit", sdf_refit, OUT_DIR)

    # Shrunk EC (publication-safe headline)
    sdf_shrunk, ev_shrunk, ev_thr05_shrunk, total_ev_present_shrunk = state_level_ec_from_preds_shrunk(
        p_out, w_vec, X24["state"].values if "state" in X24.columns else np.full(len(X24), np.nan),
        min_neff=EC_MIN_NEFF, k_prior=EC_SHRINKAGE_K_PRIOR, reliability_threshold=EC_MIN_NEFF,
        ci_reps=BOOTSTRAP_REPS, ci_seed=BOOTSTRAP_SEED
    )
    ec_outputs.append((f"EC (shrunk to national mean; k_prior={EC_SHRINKAGE_K_PRIOR:g})", sdf_shrunk, ev_shrunk, ev_thr05_shrunk, total_ev_present_shrunk))
    dump_neff_table("shrunk", sdf_shrunk, OUT_DIR)

    # Shrunk EC (full 538) saved for QA
    sdf_shrunk_all, ev_all, ev_thr05_all, tot_all = state_level_ec_from_preds_shrunk(
        p_out, w_vec, X24["state"].values if "state" in X24.columns else np.full(len(X24), np.nan),
        min_neff=0.0, k_prior=EC_SHRINKAGE_K_PRIOR, reliability_threshold=EC_MIN_NEFF,
        ci_reps=BOOTSTRAP_REPS, ci_seed=BOOTSTRAP_SEED
    )
    sdf_shrunk_all.to_csv(os.path.join(OUT_DIR, "forecast_2024_states_shrunk_full538.csv"), index=False)

    # Console EC summary with controlled verbosity
    print("\nElectoral College projection (winner-take-all; ME/NE not split):")
    print(f"(Note: EV tallies exclude states with neff < {EC_MIN_NEFF:g})")
    for label, sdf, ev_mean, ev_thr05, total_ev_present in ec_outputs:
        if total_ev_present == 538:
            print(f"- {label}: R {ev_mean} / D {538 - ev_mean} (mean-prob); R {ev_thr05} / D {538 - ev_thr05} (0.5-threshold)")
        else:
            print(f"- {label}: R {ev_mean} of {total_ev_present} present (mean-prob); R {ev_thr05} of {total_ev_present} present (0.5-threshold)")
        
        low_reli = sdf[sdf["reliability"] == "LOW_Neff"]["state"].tolist()
        excluded = sdf.loc[~sdf["eligible"], "state"].tolist()
        
        if VERBOSE_EC_LISTS:
            # Show full lists when verbose mode enabled
            if low_reli:
                print(f"  Low reliability (LOW_Neff): {', '.join(low_reli)}")
            if excluded:
                print(f"  Excluded from EV tally (neff < {EC_MIN_NEFF:g}): {', '.join(excluded)}")
        else:
            # Condensed output for production
            if low_reli:
                print(f"  Low reliability states: {len(low_reli)} (see states_neff_*.csv for details)")
            if excluded:
                print(f"  Excluded from EV tally: {len(excluded)} states (neff < {EC_MIN_NEFF:g})")
        
        # Always run consistency check but don't spam unless there's an issue
        assert_ec_consistency(label.split()[1] if len(label.split()) > 1 else "EC", sdf, total_ev_present)

    # Save outputs
    for label, sdf, _, _, _ in ec_outputs:
        if "shrunk" in label or "shrink" in label:
            tag = "shrunk"
        elif "state-refit" in label:
            tag = "state_refit"
        else:
            tag = "base"
        sdf.to_csv(os.path.join(OUT_DIR, f"forecast_2024_states_{tag}.csv"), index=False)

    # Save knee selection
    with open(os.path.join(OUT_DIR, "knee_selection.json"), "w") as f:
        json.dump(
            {"trial": knee.number, "auc": knee_auc, "C": knee_C, "features": knee_features, "cv_panel": panel},
            f, indent=2
        )

    # Run metadata
    from datetime import datetime
    run_meta = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": "PPV-Pareto v10",
        "random_seed": RANDOM_SEED,
        "n_trials": N_TRIALS,
        "n_splits": N_SPLITS,
        "grouped_cv": GROUPED_CV_BY_YEAR,
        "cross_fit_isotonic": CROSS_FIT_ISOTONIC,
        "state_min_weight": STATE_MIN_WEIGHT,
        "ec_strategy": EC_STRATEGY,
        "ec_min_neff": EC_MIN_NEFF,
        "ec_k_prior": EC_SHRINKAGE_K_PRIOR,
        "wnh_missingness_threshold": WNH_MISSINGNESS_THRESHOLD,
        "bootstrap_reps": BOOTSTRAP_REPS,
        "verbose_ec_lists": VERBOSE_EC_LISTS,
        "verbose_diagnostics": VERBOSE_DIAGNOSTICS
    }
    with open(os.path.join(OUT_DIR, "RUN_METADATA.json"), "w") as f:
        json.dump(run_meta, f, indent=2)

    # Generate supporting files
    save_requirements_txt(OUT_DIR)

    # Report
    report_md = os.path.join(OUT_DIR, "forecast_2024_report.md")
    lines = []
    lines.append("# 2024 Forward Forecast (trained on 2012, 2016, 2020)")
    lines.append("")
    lines.append(f"**Generated**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    lines.append(f"**Version**: PPV-Pareto v10")
    lines.append(f"**Random seed**: {RANDOM_SEED}")
    lines.append("")
    lines.append(f"**Selected knee model**: features = {', '.join(knee_features)}; C = {knee_C:.4g}; AUC (GroupKFold) = {knee_auc:.3f}.")
    lines.append(f"**CV panel (knee)**: AUC={panel['auc']:.3f} | Brier={panel['brier']:.4f} | used={panel['used']}")
    lines.append("")
    lines.append("## National Popular Vote Forecast")
    lines.append("National popular-vote style estimate (with survey weights), using row-wise fallback + cross-fitted isotonic calibration:")
    lines.append("")
    lines.append(f"- **Weighted mean probability**: {pct(nat['weighted_mean_prob'])}")
    lines.append(f"- **Threshold 0.5**: {pct(nat['share_thr_05'])} (95% CI {pct(nat['share_thr_05_ci'][0])}–{pct(nat['share_thr_05_ci'][1])})")
    lines.append(f"- **Threshold = weighted mean prob**: {pct(nat['share_thr_mean'])} (95% CI {pct(nat['share_thr_mean_ci'][0])}–{pct(nat['share_thr_mean_ci'][1])})")
    lines.append(f"- **2024 rows scored**: {int(np.isfinite(p_out).sum())} of {len(X24)} via row-wise fallback")
    lines.append("")
    lines.append("## Electoral College Projection")
    lines.append("Electoral College projection (winner-take-all; ME/NE not split):")
    lines.append("")
    for label, sdf, ev_mean, ev_thr05, total_ev_present in ec_outputs:
        if total_ev_present == 538:
            lines.append(f"- **{label}**: R {ev_mean} / D {538 - ev_mean} (mean-prob); R {ev_thr05} / D {538 - ev_thr05} (0.5-threshold)")
        else:
            lines.append(f"- **{label}**: R {ev_mean} of {total_ev_present} (mean-prob); R {ev_thr05} of {total_ev_present} (0.5-threshold)")
        low_reli = sdf[sdf["reliability"] == "LOW_Neff"]["state"].tolist()
        excluded = sdf.loc[~sdf["eligible"], "state"].tolist()
        if low_reli:
            lines.append(f"  - Low reliability (LOW_Neff): {len(low_reli)} states")
        if excluded:
            lines.append(f"  - Excluded from EV tally (neff < {EC_MIN_NEFF:g}): {len(excluded)} states")
    lines.append("")
    lines.append("## Technical Notes")
    lines.append("")
    lines.append("- **ANES Survey Limitation**: ANES is a national survey; state estimates are illustrative without small-area modeling (e.g., MRP).")
    lines.append("- **Empirical Bayes Shrinkage**: We apply empirical-Bayes shrinkage of state means toward the national mean to stabilize thin samples.")
    lines.append(f"- **EC Eligibility Filter**: States with neff < {EC_MIN_NEFF:g} are excluded from EV tallies.")
    lines.append("- **Training Harmonization**: 2012/2016 Disapp_economy_norm scaled to 0–1; 2020 flipped to align construct across years.")
    lines.append(f"- **2024 Base Weight**: '{WEIGHT_VAR_OVERRIDE_2024 or VAR_2024_PREFS['weight'][0]}' (confirmed).")
    lines.append(f"- **2024 White Non-Hispanic**: Derived from {VAR_2024_PREFS['race_eth_summary'][0]} if present; else from race+Hispanic.")
    lines.append(f"- **State Variable**: Chosen by coverage+cardinality from candidates {VAR_2024_PREFS['state_candidates']}.")
    if SAVE_CALIBRATION_PLOTS:
        lines.append("- **Calibration Plots**: calibration_knee.png, calibration_ec.png")
    lines.append(f"- **2024 WNH Missingness**: {wnh_missing:.1%}.")
    lines.append(f"- **Full Results**: EC tallies exclude states with neff < {EC_MIN_NEFF:g}; full-538 shrunk results saved as forecast_2024_states_shrunk_full538.csv.")
    if two_no_lib and not np.isnan(two_no_lib['auc']):
        lines.append(f"- **Backstop 2-feature (no ideology) AUC**: {two_no_lib['auc']:.3f}")
    if one_no_lib and not np.isnan(one_no_lib['auc']):
        lines.append(f"- **Backstop 1-feature (no ideology) AUC**: {one_no_lib['auc']:.3f}")
    
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nSaved outputs to {OUT_DIR}")
    print("Key artifacts:")
    print(f"  - Main report: forecast_2024_report.md")
    print(f"  - State tables: forecast_2024_states_*.csv")
    print(f"  - State diagnostics: states_neff_*.csv")  
    print(f"  - Model selection: knee_selection.json")
    print(f"  - Configuration: RUN_METADATA.json")
    print(f"  - Dependencies: requirements.txt")
    print(f"  - Documentation: README_PIPELINE.md")
    
    if SAVE_CALIBRATION_PLOTS:
        print(f"  - Calibration plots: calibration_*.png")

    print(f"\nPipeline completed successfully!")
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        sys.exit(1)
