%%writefile pareto_optimal_front_ca_v3.py

# PPV-Pareto CA v3 — Train on 2011 & 2015; forecast 2021 OOS (no leakage)
# - GroupKFold by year (LOYO-style) for Optuna selection
# - Cross-fitted isotonic calibration on training years only
# - Row-wise fallback like U.S. pipeline (knee + compact backstops)
# - Province-level diagnostics (no EC logic)
#
# Files expected (your filenames):
#   - DATA_2004_2011 = "2004_to_2011ces.dta"            [optional fallback to source some 2011 vars]
#   - DATA_2011      = "CPS&PES&MBS&WEB_2011_final.dta"
#   - DATA_2015      = "CES2015_CPS-PES-MBS_complete-v2.dta"
#   - DATA_2021      = "2021 Canadian Election Study v2.0.dta"
#
# This script auto-detects key variables via value labels and metadata (using pyreadstat),
# so you don't need to splice anything manually. It prints provenance for audit.
#
# Output dir:
#   OUT_DIR = "optuna_outputs_ca_2021_oos"
#
# Requirements:
#   pip install pandas numpy scikit-learn optuna matplotlib seaborn pyreadstat

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

import pyreadstat

warnings.filterwarnings("ignore")

# ------------------ RANDOM SEED ------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ------------------ USER CONFIG ------------------
DATA_2004_2011 = "2004_to_2011ces.dta"  # optional (used only if needed)
DATA_2011 = "CPS&PES&MBS&WEB_2011_final.dta"
DATA_2015 = "CES2015_CPS-PES-MBS_complete-v2.dta"
DATA_2021 = "2021 Canadian Election Study v2.0.dta"

# Preferred 2021 keys (will be auto-detected if missing)
KEY_VARS_2021 = {
    "vote_choice": "cps21_votechoice",
    "ideology": "cps21_lr_self",
    "econ_retro": "cps21_econ_retro",
    "gender": "cps21_gender",
    "age": "cps21_age",
    "education": "cps21_education",
    "province": "cps21_province",
    "weight": "cps21_weight_general_all",
}

# Outputs
OUT_DIR = "optuna_outputs_ca_2021_oos"
BOOTSTRAP_REPS = 500
BOOTSTRAP_SEED = 123

# Optimization
N_TRIALS = 200
N_SPLITS = 5

# Feature toggles
GROUPED_CV_BY_YEAR = True
CROSS_FIT_ISOTONIC = True
SAVE_CALIBRATION_PLOTS = True

# Incumbent by year; used to align econ sign so higher => more CPC-favorable
INCUMBENT_BY_YEAR = {2011: "CPC", 2015: "CPC", 2021: "LPC"}

VERBOSE = True

# ------------------ Utils ------------------

def _exists(p: str) -> bool:
    try:
        return os.path.exists(p)
    except Exception:
        return False

def validate_file_exists(filepath: str, description: str) -> bool:
    if not filepath or not _exists(filepath):
        print(f"ERROR: Missing required file for {description}: {filepath}")
        return False
    return True

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

def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)

def _to_num(s): return pd.to_numeric(s, errors="coerce")

def _normalize_01_series(s: Optional[pd.Series], index, lo: Optional[float]=None, hi: Optional[float]=None) -> pd.Series:
    if s is None:
        return pd.Series(np.nan, index=index)
    s = _to_num(s)
    if lo is None:
        lo = float(np.nanmin(s.values)) if np.isfinite(s.values).any() else 0.0
    if hi is None:
        hi = float(np.nanmax(s.values)) if np.isfinite(s.values).any() else 1.0
    rng = hi - lo
    if not np.isfinite(rng) or rng <= 0:
        rng = 1.0
    return (s - lo) / rng

def _clean_pos(s: Optional[pd.Series], index=None) -> pd.Series:
    if s is None:
        return pd.Series(np.nan, index=index, name="weight")
    s = _to_num(s)
    return s.where(s > 0, np.nan)

def _infer_female(series: Optional[pd.Series], index, numeric_female_val: Optional[int]=2) -> pd.Series:
    if series is None:
        return pd.Series(np.nan, index=index)
    s = series.copy()
    female = pd.Series(np.nan, index=s.index, dtype=float)
    if s.dtype == object:
        sc = s.astype(str).str.lower().str.strip()
        female.loc[sc.str.contains("female", na=False)] = 1.0
        female.loc[sc.str.contains("male", na=False)] = 0.0
    else:
        if numeric_female_val is not None:
            female.loc[s == numeric_female_val] = 1.0
            female.loc[(s != numeric_female_val) & s.notna()] = 0.0
    return female

def _sgn_for_year(year: int) -> float:
    return +1.0 if INCUMBENT_BY_YEAR.get(year, "LPC") != "CPC" else -1.0

def _to_str_with_nan(s: Optional[pd.Series], index) -> pd.Series:
    """
    Cast to string categories consistently while preserving NaNs.
    """
    if s is None:
        return pd.Series(np.nan, index=index)
    out = s.astype(object)
    mask_na = pd.isna(out)
    out = out.astype(str)
    out[mask_na] = np.nan
    return out

# ------------------ Vote detection via value labels ------------------

PARTY_TOKENS = {
    "conservative": 1, "cpc": 1, "tory": 1,
    "liberal": 0, "lpc": 0,
    "ndp": 0, "new democratic": 0,
    "bloc": 0, "bq": 0,
    "green": 0, "gpc": 0,
    "people": 0, "ppc": 0,
    "independent": 0, "other": 0, "none": 0
}

def detect_vote_var(df: pd.DataFrame, meta, prefer_pes: bool = True) -> Optional[str]:
    var2labelset = meta.variable_to_label or {}
    labelsets = meta.value_labels or {}
    name2label = meta.column_names_to_labels or {}

    candidates = []
    for var, lset in var2labelset.items():
        if lset not in labelsets:
            continue
        labels_map = labelsets[lset]
        vals = [str(v).lower() for v in labels_map.values()]
        party_hits = sum(any(tk in v for tk in PARTY_TOKENS.keys()) for v in vals)
        if party_hits >= 2:
            var_low = var.lower()
            lab_low = (name2label.get(var, "") or "").lower()
            score = party_hits
            if "vote" in var_low or "vote" in lab_low: score += 1
            if "federal" in lab_low or "federal" in var_low: score += 1
            if prefer_pes and ("pes" in var_low or "post" in lab_low): score += 2
            if not prefer_pes and ("cps" in var_low or "campaign" in lab_low): score += 1
            candidates.append((score, var))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    chosen = candidates[0][1]
    if VERBOSE:
        print(f"Detected vote variable: {chosen} (score-ranked)")
    return chosen

def map_vote_to_cpc(df: pd.DataFrame, meta, var: str) -> pd.Series:
    s = df[var]
    y = pd.Series(np.nan, index=s.index, dtype=float)

    var2labelset = meta.variable_to_label or {}
    labelsets = meta.value_labels or {}
    lset = var2labelset.get(var, None)
    if lset and (lset in labelsets):
        labels_map = labelsets[lset]  # code -> text
        inv_map = {code: str(txt).lower() for code, txt in labels_map.items()}
        for code, txt in inv_map.items():
            if any(tok in txt for tok in PARTY_TOKENS.keys()):
                target = 1.0 if ("conservative" in txt or "cpc" in txt or "tory" in txt) else 0.0
                y.loc[s == code] = target
        bad_tokens = ["don", "refus", "not vote", "none of the above", "dk", "na", "invalid"]
        for code, txt in inv_map.items():
            if any(bt in txt for bt in bad_tokens):
                y.loc[s == code] = np.nan
    else:
        low = s.astype(str).str.lower().str.strip()
        y.loc[low.str.contains("conservative", na=False)] = 1.0
        non_c = ["liberal","ndp","new democratic","bloc","green","people","ppc","independent","other"]
        for tk in non_c:
            y.loc[low.str.contains(tk, na=False)] = 0.0

    return y

# ------------------ Search helpers ------------------

def find_first(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    return None

def search_any_by_label(meta, any_keywords: List[str]) -> Optional[str]:
    name2label = meta.column_names_to_labels or {}
    for var, lab in name2label.items():
        hay = f"{var} | {lab or ''}".lower()
        if any(kw.lower() in hay for kw in any_keywords):
            return var
    return None

def detect_ideology_var(df: pd.DataFrame, meta) -> Optional[str]:
    cand = find_first(df, ["cps21_lr_self", "lr_self", "left_right", "left_right_self"])
    return cand or search_any_by_label(meta, ["left-right", "left right", "lr self", "ideology", "left/right"])

def detect_econ_var(df: pd.DataFrame, meta) -> Optional[str]:
    cand = find_first(df, ["cps21_econ_retro", "econ_retro", "economy_retro"])
    return cand or search_any_by_label(meta, ["econom", "better", "worse", "past year", "retrospective"])

def detect_education_var(df: pd.DataFrame, meta) -> Optional[str]:
    cand = find_first(df, ["cps21_education", "education", "educ"])
    return cand or search_any_by_label(meta, ["educ", "education", "school", "degree"])

def detect_age_var(df: pd.DataFrame, meta) -> Optional[str]:
    cand = find_first(df, ["cps21_age", "age_years", "age"])
    return cand or search_any_by_label(meta, ["age"])

def detect_province_var(df: pd.DataFrame, meta) -> Optional[str]:
    cand = find_first(df, ["cps21_province", "province", "prov"])
    return cand or search_any_by_label(meta, ["province", "prov", "region"])

def detect_gender_var(df: pd.DataFrame, meta) -> Optional[str]:
    cand = find_first(df, ["cps21_gender", "gender", "sex", "cps21_genderid"])
    return cand or search_any_by_label(meta, ["gender","sex"])

def pick_weight_2021(df: pd.DataFrame, meta) -> Optional[str]:
    prefs = [
        "cps21_weight_general_all",
        "cps21_weight_general",
        "cps21_weight",
        "weight_general_all",
        "weight"
    ]
    cand = find_first(df, prefs)
    if cand:
        return cand
    candidates = [c for c in df.columns if "weight" in c.lower() or c.lower().endswith("wt") or "wgt" in c.lower()]
    return candidates[0] if candidates else None

# ------------------ Canadian loaders ------------------

def load_2015(path: str):
    df, meta = pyreadstat.read_dta(path, apply_value_formats=False, formats_as_category=False)
    if VERBOSE:
        print(f"Loaded 2015: df shape={df.shape}")

    vote_var = detect_vote_var(df, meta, prefer_pes=True)
    if not vote_var:
        raise ValueError("Could not auto-detect 2015 vote variable with party value-labels.")
    y = map_vote_to_cpc(df, meta, vote_var)

    ideol_var = detect_ideology_var(df, meta)
    econ_var = detect_econ_var(df, meta)
    edu_var = detect_education_var(df, meta)
    age_var = detect_age_var(df, meta)
    prov_var = find_first(df, ["PES15_PROVINCE", "CPS15_PROVINCE", "CS_Province"])
    gender_var = "rgender" if "rgender" in df.columns else detect_gender_var(df, meta)
    weight_var = find_first(df, [
        "WeightBYPopul_count_Prov_Age_Gender",
        "WeightBYPopul_count_Prov_HHsize",
        "WeightTOsampBYPopul_count_Prov_Age",
        "WeightTOsampBYPopul_count_Prov_H",
        "weightEQ1"
    ]) or "weightEQ1"

    if VERBOSE:
        print("2015 selected vars:", dict(vote=vote_var, ideology=ideol_var, econ=econ_var,
                                          education=edu_var, age=age_var, province=prov_var,
                                          gender=gender_var, weight=weight_var))

    lib_cons_norm = _normalize_01_series(df[ideol_var] if ideol_var else None, df.index, lo=0.0, hi=10.0)
    econ_norm = _normalize_01_series(df[econ_var] if econ_var else None, df.index)
    econ_cpc_aligned = _sgn_for_year(2015) * econ_norm

    age_norm = _normalize_01_series(df[age_var] if age_var else None, df.index, lo=18.0, hi=90.0)
    edu_norm = _normalize_01_series(df[edu_var] if edu_var else None, df.index)
    female = _infer_female(df[gender_var] if gender_var else None, df.index)
    province = _to_str_with_nan(df[prov_var] if prov_var else None, df.index)

    X = pd.DataFrame({
        "lib_cons_norm": lib_cons_norm,
        "econ_cpc_aligned": econ_cpc_aligned,
        "age_norm": age_norm,
        "edu_norm": edu_norm,
        "female": female,
        "province": province
    })

    w = _clean_pos(df[weight_var] if weight_var else None, index=df.index).rename("weight")
    return X, y.rename("target"), w

def load_2011(path: str, fallback_2004_2011: Optional[str] = None):
    df, meta = pyreadstat.read_dta(path, apply_value_formats=False, formats_as_category=False)
    if VERBOSE:
        print(f"Loaded 2011: df shape={df.shape}")

    vote_var = detect_vote_var(df, meta, prefer_pes=True)
    if not vote_var and fallback_2004_2011 and _exists(fallback_2004_2011):
        dfcum, metacum = pyreadstat.read_dta(fallback_2004_2011, apply_value_formats=False, formats_as_category=False)
        vote_var = detect_vote_var(dfcum, metacum, prefer_pes=True)
    if not vote_var:
        raise ValueError("Could not auto-detect 2011 vote variable with party value-labels.")

    y = map_vote_to_cpc(df, meta, vote_var)

    ideol_var = search_any_by_label(meta, ["left-right", "left right", "lr self", "ideology", "left/right"])
    econ_var = search_any_by_label(meta, ["econom", "better", "worse", "past year", "retrospective"])
    edu_var = search_any_by_label(meta, ["educ", "education", "school", "degree"])
    age_var = find_first(df, ["age_years", "age"]) or search_any_by_label(meta, ["age"])
    prov_var = find_first(df, ["PROVINCE11", "CS_PROVINCE", "PROVINCE"])
    weight_var = find_first(df, ["WeightBYNadults_and_TotPopn", "PROVINCIAL_WEIGHT", "PROVINCIAL_WEIGHT11"])
    gender_var = "RGENDER11" if "RGENDER11" in df.columns else search_any_by_label(meta, ["gender","sex"])

    if VERBOSE:
        print("2011 selected vars:", dict(vote=vote_var, ideology=ideol_var, econ=econ_var,
                                          education=edu_var, age=age_var, province=prov_var,
                                          gender=gender_var, weight=weight_var))

    lib_cons_norm = _normalize_01_series(df[ideol_var] if ideol_var else None, df.index, lo=0.0, hi=10.0)
    econ_norm = _normalize_01_series(df[econ_var] if econ_var else None, df.index)
    econ_cpc_aligned = _sgn_for_year(2011) * econ_norm

    age_norm = _normalize_01_series(df[age_var] if age_var else None, df.index, lo=18.0, hi=90.0)
    edu_norm = _normalize_01_series(df[edu_var] if edu_var else None, df.index)
    female = _infer_female(df[gender_var] if gender_var else None, df.index, numeric_female_val=2)
    province = _to_str_with_nan(df[prov_var] if prov_var else None, df.index)

    X = pd.DataFrame({
        "lib_cons_norm": lib_cons_norm,
        "econ_cpc_aligned": econ_cpc_aligned,
        "age_norm": age_norm,
        "edu_norm": edu_norm,
        "female": female,
        "province": province
    })

    w = _clean_pos(df[weight_var] if weight_var else None, index=df.index).rename("weight")
    return X, y.rename("target"), w

def load_2021(path: str):
    df, meta = pyreadstat.read_dta(path, apply_value_formats=False, formats_as_category=False)
    if VERBOSE:
        print(f"Loaded 2021: df shape={df.shape}")

    vote_col = KEY_VARS_2021["vote_choice"] if KEY_VARS_2021["vote_choice"] in df.columns else detect_vote_var(df, meta, prefer_pes=True)
    if not vote_col:
        raise ValueError("2021: Could not find vote choice variable (cps21_votechoice missing; auto-detect failed).")

    ideol_col = KEY_VARS_2021["ideology"] if KEY_VARS_2021["ideology"] in df.columns else detect_ideology_var(df, meta)
    econ_col = KEY_VARS_2021["econ_retro"] if KEY_VARS_2021["econ_retro"] in df.columns else detect_econ_var(df, meta)
    gender_col = detect_gender_var(df, meta)
    age_col = KEY_VARS_2021["age"] if KEY_VARS_2021["age"] in df.columns else detect_age_var(df, meta)
    edu_col = KEY_VARS_2021["education"] if KEY_VARS_2021["education"] in df.columns else detect_education_var(df, meta)
    prov_col = KEY_VARS_2021["province"] if KEY_VARS_2021["province"] in df.columns else detect_province_var(df, meta)
    weight_col = KEY_VARS_2021["weight"] if KEY_VARS_2021["weight"] in df.columns else pick_weight_2021(df, meta)

    y = map_vote_to_cpc(df, meta, vote_col)

    lib_cons_norm = _normalize_01_series(df[ideol_col] if ideol_col else None, df.index, lo=0.0, hi=10.0)
    econ_norm = _normalize_01_series(df[econ_col] if econ_col else None, df.index)
    econ_cpc_aligned = _sgn_for_year(2021) * econ_norm
    age_norm = _normalize_01_series(df[age_col] if age_col else None, df.index, lo=18.0, hi=90.0)
    edu_norm = _normalize_01_series(df[edu_col] if edu_col else None, df.index)
    female = _infer_female(df[gender_col] if gender_col else None, df.index, numeric_female_val=2)
    province = _to_str_with_nan(df[prov_col] if prov_col else None, df.index)

    X = pd.DataFrame({
        "lib_cons_norm": lib_cons_norm,
        "econ_cpc_aligned": econ_cpc_aligned,
        "age_norm": age_norm,
        "edu_norm": edu_norm,
        "female": female,
        "province": province
    })
    w = _clean_pos(df[weight_col] if weight_col else None, index=df.index).rename("weight")

    diag = {
        "vote_col": vote_col,
        "ideology_col": ideol_col,
        "econ_col": econ_col,
        "gender_col": gender_col,
        "age_col": age_col,
        "education_col": edu_col,
        "province_col": prov_col,
        "weight_col": weight_col,
        "weight_eff_n": eff_sample_size(w.fillna(0).values)
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "2021_variable_sources.json"), "w") as f:
        json.dump(diag, f, indent=2)

    if VERBOSE:
        print("2021 selected vars:", diag)

    return X, y.rename("target"), w

# ------------------ Modeling ------------------

CANDIDATE_VARS = ["lib_cons_norm","econ_cpc_aligned","age_norm","edu_norm","female","province"]

def build_pipe(selected: List[str], C: float):
    num_features = [c for c in selected if c != "province"]
    cat_features = ["province"] if "province" in selected else []
    tr = []
    if num_features: tr.append(("num","passthrough", num_features))
    if cat_features: tr.append(("province", make_ohe(), cat_features))
    ct = ColumnTransformer(transformers=tr, remainder="drop", sparse_threshold=1.0)
    clf = LogisticRegression(solver="liblinear", C=C, max_iter=1000, random_state=RANDOM_SEED)
    return Pipeline([("ct", ct), ("clf", clf)])

def five_fold_weighted_auc(df_all: pd.DataFrame, selected: List[str], C: float) -> float:
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
        aucs.append(roc_auc_score(y[te_idx], p, sample_weight=w[te_idx]) if len(np.unique(y[te_idx]))==2 else 0.5)
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
    best_idx = int(np.argmax(aucs - 1e-9 * cplx))
    return pareto[best_idx]

# ------------------ Calibration ------------------

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
    mask = df_all["target"].notna() & df_all["weight"].notna()
    for c in feats: mask &= df_all[c].notna()
    data = df_all.loc[mask].reset_index(drop=True)
    if data.empty or data["year"].nunique() < 2:
        raise ValueError("Need ≥2 training years for cross-fit isotonic.")
    X = data[feats]; y = data["target"].astype(int).to_numpy()
    w = data["weight"].astype(float).to_numpy()
    groups = data["year"].to_numpy()
    gkf = GroupKFold(n_splits=min(n_splits, data["year"].nunique()))
    oof = np.full_like(y, np.nan, dtype=float)
    for tr_idx, te_idx in gkf.split(X, y, groups=groups):
        if len(np.unique(y[tr_idx])) < 2 or len(np.unique(y[te_idx])) < 2:
            continue
        pipe = build_pipe(feats, C)
        pipe.fit(X.iloc[tr_idx], y[tr_idx], clf__sample_weight=w[tr_idx])
        oof[te_idx] = pipe.predict_proba(X.iloc[te_idx])[:,1]
    m = np.isfinite(oof)
    cal = fit_isotonic(oof[m], y[m], w[m])
    final_pipe = build_pipe(feats, C)
    final_pipe.fit(X, y, clf__sample_weight=w)
    return final_pipe, cal, int(m.sum()), data

def cv_metric_panel(df_all: pd.DataFrame, feats: List[str], C: float):
    mask = df_all["target"].notna() & df_all["weight"].notna()
    for c in feats: mask &= df_all[c].notna()
    data = df_all.loc[mask].reset_index(drop=True)
    if data.empty or data["year"].nunique() < 2:
        return dict(auc=float("nan"), brier=float("nan"), used=0)
    X = data[feats]; y = data["target"].astype(int).to_numpy()
    w = data["weight"].astype(float).to_numpy()
    groups = data["year"].to_numpy()
    gkf = GroupKFold(n_splits=min(N_SPLITS, data["year"].nunique()))
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
    auc = roc_auc_score(y[m], oof[m], sample_weight=w[m])
    try:
        brier = brier_score_loss(y[m], oof[m], sample_weight=w[m])
    except TypeError:
        brier = brier_score_loss(y[m], oof[m])
    return dict(auc=float(auc), brier=float(brier), used=int(m.sum()))

# ------------------ Forecast aggregation ------------------

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

def province_level_diagnostics(p: np.ndarray, w: np.ndarray, prov: np.ndarray) -> pd.DataFrame:
    p = np.asarray(p, float); w = np.asarray(w, float); s = pd.Series(prov)
    m = np.isfinite(p) & np.isfinite(w) & s.notna().to_numpy()
    if not m.any():
        return pd.DataFrame(columns=["province","mean_prob","share_thr_05","ci_low_thr05","ci_high_thr05","neff","used_n","w_sum"])
    dfp = pd.DataFrame({"proba": p[m], "w": np.clip(w[m], 0, None), "province": s[m].values})
    rows = []
    for pv, sub in dfp.groupby("province"):
        pv_probs = sub["proba"].values; wv = sub["w"].values
        used_n = int((wv > 0).sum()); w_sum = float(np.sum(wv))
        if w_sum <= 0:
            meanp = float("nan"); share05 = float("nan"); neff = 0.0; ci_lo = float("nan"); ci_hi = float("nan")
        else:
            meanp = float(np.sum(pv_probs * wv) / w_sum)
            share05 = float(np.sum((pv_probs > 0.5).astype(int) * wv) / w_sum)
            neff = eff_sample_size(wv)
            ci_lo, ci_hi = weighted_bootstrap_ci_share(pv_probs, wv, 0.5, BOOTSTRAP_REPS, BOOTSTRAP_SEED)
        rows.append({"province": str(pv), "mean_prob": meanp, "share_thr_05": share05,
                     "ci_low_thr05": ci_lo, "ci_high_thr05": ci_hi, "neff": neff,
                     "used_n": used_n, "w_sum": w_sum})
    return pd.DataFrame(rows).sort_values("w_sum", ascending=False)

# ------------------ Main ------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Validate files
    reqs = [
        (DATA_2011, "CES 2011 data"),
        (DATA_2015, "CES 2015 data"),
        (DATA_2021, "CES 2021 data")
    ]
    missing = False
    for fp, desc in reqs:
        if not validate_file_exists(fp, desc):
            missing = True
    if missing:
        print("ERROR: Missing required file(s); aborting.")
        sys.exit(1)

    # Load training years (2011, 2015)
    print("Loading training data (2011, 2015) with auto-detected variables...")
    X11, y11, w11 = load_2011(DATA_2011, fallback_2004_2011=DATA_2004_2011 if _exists(DATA_2004_2011) else None)
    X15, y15, w15 = load_2015(DATA_2015)

    def pack(X, y, w, yr):
        df = X.copy()
        df["target"] = y
        df["weight"] = w
        df["year"] = yr
        return df

    df_all = pd.concat([pack(X11,y11,w11,2011), pack(X15,y15,w15,2015)], ignore_index=True)
    print(f"Training rows with target present: {(df_all['target'].notna()).sum()} across {df_all['year'].nunique()} years")

    # Optuna: Pareto (AUC vs #features) with GroupKFold by year
    print("\nOptimizing (GroupKFold by year: AUC vs #features)...")
    study = optuna.create_study(directions=["maximize", "minimize"], sampler=NSGAIISampler(seed=RANDOM_SEED))
    study.optimize(make_objective(df_all), n_trials=N_TRIALS, show_progress_bar=True)

    # Pareto plot + CSV
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
    plt.xlabel("# features"); plt.ylabel("AUC (GroupKFold)"); plt.title("Pareto: Accuracy vs Complexity (Canada)")
    plt.grid(True, alpha=0.2)
    pareto_png = os.path.join(OUT_DIR, "pareto_front.png"); plt.tight_layout(); plt.savefig(pareto_png, dpi=150); plt.close()
    pareto_rows = []
    for t in pareto:
        feats = [k.replace("include_", "") for k, v in t.params.items() if k.startswith("include_") and v]
        pareto_rows.append({"trial": t.number, "auc": t.values[0], "complexity": t.values[1],
                            "C": t.params.get("logreg_C", None), "features": json.dumps(feats)})
    pareto_csv = os.path.join(OUT_DIR, "pareto_front.csv")
    pd.DataFrame(pareto_rows).sort_values(["complexity","auc"], ascending=[True, False]).to_csv(pareto_csv, index=False)
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

    # CV panel
    panel = cv_metric_panel(df_all, knee_features, knee_C)
    print(f"CV panel (knee): AUC={panel['auc']:.3f} | Brier={panel['brier']:.4f} | used={panel['used']}")

    # Backstops (like U.S. pipeline)
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

    if two_no_lib is None:
        two_no_lib = {"features": ["econ_cpc_aligned", "province"], "C": 1.0, "auc": 0.0, "trial": -1}
    if one_no_lib is None:
        one_no_lib = {"features": ["econ_cpc_aligned"], "C": 1.0, "auc": 0.0, "trial": -1}

    def quick_auc(df_all, feats, C):
        m = df_all["target"].notna() & df_all["weight"].notna()
        for c in feats: m &= df_all[c].notna()
        data = df_all.loc[m].reset_index(drop=True)
        if data.empty or data["year"].nunique() < 2: return float("nan")
        X = data[feats]; y = data["target"].astype(int).to_numpy()
        w = data["weight"].astype(float).to_numpy(); groups = data["year"].to_numpy()
        gkf = GroupKFold(n_splits=min(N_SPLITS, data["year"].nunique()))
        aucs = []
        for tr_idx, te_idx in gkf.split(X, y, groups=groups):
            pipe = build_pipe(feats, C)
            pipe.fit(X.iloc[tr_idx], y[tr_idx], clf__sample_weight=w[tr_idx])
            p = pipe.predict_proba(X.iloc[te_idx])[:,1]
            aucs.append(roc_auc_score(y[te_idx], p, sample_weight=w[te_idx]))
        return float(np.mean(aucs)) if aucs else float("nan")

    if two_no_lib and two_no_lib["auc"] == 0.0:
        two_no_lib["auc"] = quick_auc(df_all, two_no_lib["features"], two_no_lib["C"])
    if one_no_lib and one_no_lib["auc"] == 0.0:
        one_no_lib["auc"] = quick_auc(df_all, one_no_lib["features"], one_no_lib["C"])

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

    # Train with cross-fitted isotonic
    trained: Dict[str, Dict[str, Any]] = {}
    for m in models:
        feats = m["features"]; C = m["C"]; name = m["name"]
        try:
            pipe, calibrator, n_oof, data_used = cross_fit_isotonic(df_all, feats, C, n_splits=N_SPLITS)
            trained[name] = {"pipe": pipe, "features": feats, "cal": calibrator, "C": C, "auc": m["auc"]}
            print(f"Trained [{name}] | OOF n={n_oof} | feats={feats} | C={C:.4g} | AUC≈{m['auc'] if m['auc'] is not None else float('nan'):.3f}")
            if SAVE_CALIBRATION_PLOTS and name == "knee":
                Xp = data_used[feats]
                yp = data_used["target"].astype(int).to_numpy()
                wp = data_used["weight"].astype(float).to_numpy()
                p_raw = pipe.predict_proba(Xp)[:,1]
                p_cal = calibrator.transform(p_raw)
                bins = np.linspace(0,1,11)
                idx = np.digitize(p_cal, bins) - 1
                dfb = pd.DataFrame({"p":p_cal, "y":yp.astype(int), "w":wp})
                agg = dfb.groupby(idx, dropna=True).apply(
                    lambda g: pd.Series({
                        "p_mean": np.average(g["p"], weights=g["w"]) if g["w"].sum()>0 else np.nan,
                        "y_rate": np.average(g["y"], weights=g["w"]) if g["w"].sum()>0 else np.nan
                    })
                ).reset_index(drop=True)
                plt.figure()
                plt.plot(agg["p_mean"], agg["y_rate"], marker="o")
                plt.plot([0,1],[0,1], linestyle="--", color="gray")
                plt.xlabel("Calibrated predicted prob"); plt.ylabel("Empirical rate"); plt.title("Calibration curve (train years)")
                plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "calibration_knee.png"), dpi=150); plt.close()
        except Exception as e:
            print(f"WARN: Training {name} failed: {e}")

    # Forecast 2021 OOS
    print("\nScoring 2021 out-of-sample...")
    X21, y21_opt, w21 = load_2021(DATA_2021)  # y21_opt not used for training
    p21 = np.full(len(X21), np.nan, dtype=float)
    assigned = np.zeros(len(X21), dtype=bool)
    fill_report = []
    for name, obj in trained.items():
        feats = obj["features"]
        mrows = X21[feats].notna().all(axis=1) & w21.notna() & (~assigned)
        n_rows = int(mrows.sum())
        if n_rows == 0:
            fill_report.append((name, feats, 0))
            continue
        p = obj["pipe"].predict_proba(X21.loc[mrows, feats])[:, 1]
        p_cal = np.clip(obj["cal"].transform(p), 0.0, 1.0)
        p21[mrows.values] = p_cal
        assigned[mrows.values] = True
        fill_report.append((name, feats, n_rows))
    total_assigned = int(np.isfinite(p21).sum())
    print("Row-wise fallback fill counts (2021):")
    for nm, feats, n in fill_report:
        print(f"  - {nm}: {n} rows ({n/len(X21):.1%}) | feats={feats}")
    print(f"Total 2021 rows scored: {total_assigned} of {len(X21)}")

    # National aggregation
    nat = forecast_national_from_preds(p21, w21.astype(float).values)
    print("\n2021 Forecast-style summary (CPC vs Others):")
    print(f"- Weighted mean probability (CPC): {pct(nat['weighted_mean_prob'])}")
    print(f"- Threshold 0.5 (CPC share): {pct(nat['share_thr_05'])} (95% CI {pct(nat['share_thr_05_ci'][0])}–{pct(nat['share_thr_05_ci'][1])})")
    print(f"- Threshold = weighted mean prob: {pct(nat['share_thr_mean'])} (95% CI {pct(nat['share_thr_mean_ci'][0])}–{pct(nat['share_thr_mean_ci'][1])})")
    print(f"- Used {nat['used']} of {len(X21)} 2021 rows")

    # Province diagnostics
    if "province" in X21.columns:
        prov_diag = province_level_diagnostics(p21, w21.astype(float).values, X21["province"].values)
        prov_csv = os.path.join(OUT_DIR, "provinces_diagnostics_2021.csv")
        prov_diag.to_csv(prov_csv, index=False)
        print(f"Saved province-level diagnostics: {prov_csv}")

    # Save knee selection + metadata
    with open(os.path.join(OUT_DIR, "knee_selection.json"), "w") as f:
        json.dump({"trial": knee.number, "auc": knee_auc, "C": knee_C, "features": knee_features, "cv_panel": panel}, f, indent=2)

    from datetime import datetime
    run_meta = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": "PPV-Pareto CA v3",
        "random_seed": RANDOM_SEED,
        "n_trials": N_TRIALS,
        "n_splits": N_SPLITS,
        "grouped_cv": True,
        "cross_fit_isotonic": True,
        "training_years": [2011, 2015],
        "forecast_year": 2021,
        "feature_space": CANDIDATE_VARS,
        "dataset_paths": {"2004_2011": DATA_2004_2011, "2011": DATA_2011, "2015": DATA_2015, "2021": DATA_2021},
        "incumbent_by_year": INCUMBENT_BY_YEAR,
        "notes": [
            "Targets auto-detected via value labels; CPC=1, others=0.",
            "econ_cpc_aligned flips by incumbent so higher => more CPC-favorable across years.",
            "No training leakage: 2021 scored out-of-sample.",
            "Province coerced to string across years to avoid dtype mismatch during OHE."
        ]
    }
    with open(os.path.join(OUT_DIR, "RUN_METADATA.json"), "w") as f:
        json.dump(run_meta, f, indent=2)

    # Report
    report_md = os.path.join(OUT_DIR, "forecast_2021_report.md")
    lines = []
    lines.append("# 2021 Forecast (Canada; CPC vs Others; trained on 2011+2015)")
    lines.append("")
    lines.append(f"Generated (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Version: PPV-Pareto CA v3 | Seed: {RANDOM_SEED}")
    lines.append(f"Selected knee: feats = {', '.join(knee_features)}; C = {knee_C:.4g}; AUC (GroupKFold) = {knee_auc:.3f}")
    lines.append(f"CV panel (knee): AUC={panel['auc']:.3f} | Brier={panel['brier']:.4f} | used={panel['used']}")
    lines.append("")
    lines.append("## National Forecast-Style Summary (CPC vs Others)")
    lines.append(f"- Weighted mean probability (CPC): {pct(nat['weighted_mean_prob'])}")
    lines.append(f"- Threshold 0.5 (CPC share): {pct(nat['share_thr_05'])} (95% CI {pct(nat['share_thr_05_ci'][0])}–{pct(nat['share_thr_05_ci'][1])})")
    lines.append(f"- Threshold = weighted mean prob: {pct(nat['share_thr_mean'])} (95% CI {pct(nat['share_thr_mean_ci'][0])}–{pct(nat['share_thr_mean_ci'][1])})")
    lines.append(f"- 2021 rows scored: {int(np.isfinite(p21).sum())} of {len(X21)} via row-wise fallback")
    if "province" in X21.columns:
        lines.append("")
        lines.append("## Province Diagnostics")
        lines.append("- See provinces_diagnostics_2021.csv for mean prob, share>0.5, and effective N.")
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Save requirements for reproducibility
    reqs = [
        "# PPV-Pareto CA Pipeline Requirements",
        "pandas>=1.5.0,<2.1.0",
        "numpy>=1.21.0,<1.25.0",
        "scikit-learn>=1.1.0,<1.4.0",
        "optuna>=3.0.0,<3.5.0",
        "matplotlib>=3.5.0,<3.8.0",
        "seaborn>=0.11.0,<0.13.0  # optional",
        "pyreadstat>=1.2.0"
    ]
    with open(os.path.join(OUT_DIR, "requirements.txt"), "w") as f:
        f.write("\n".join(reqs))

    print(f"\nSaved outputs to {OUT_DIR}")
    print("  - Pareto: pareto_front.png, pareto_front.csv")
    print("  - Calibration: calibration_knee.png")
    print("  - Province diagnostics: provinces_diagnostics_2021.csv")
    print("  - Model selection: knee_selection.json")
    print("  - Config: RUN_METADATA.json")
    print("  - Report: forecast_2021_report.md")
    print("  - requirements.txt")
    print("\nPipeline completed successfully!")
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
