# PPV-Pareto v9 — End-to-end forecast pipeline with harmonization, row-wise fallback, calibration, and EC variants.
# - Optimizes on 2012/2016/2020 (5-fold CV, AUC vs #features), selects knee
# - Training harmonization:
#     * 2012/2016 Disapp_economy_norm scaled to 0–1 ((dapp-1)/4)
#     * 2020 Disapp_economy_norm flipped to align construct “higher = disapproval of a Democratic incumbent”
# - Refit knee on all past years (weighted)
# - 2024 loader (validated mappings + range checks + best state selection)
#     * Weight: V240107a (confirmed)
#     * white_nonhispanic: prefer V241501x summary; fallback to V241540+V241541; fallback2: V241155x if present
#     * State: choose by valid coverage + cardinality (prefer V241023; fallback to V243002, then V241014)
# - Row-wise fallback scoring on 2024 (no imputation):
#     * Score each case with the best Pareto model it qualifies for (knee → best 2-feature → best 2-feature no-ideology → best 1-feature no-ideology)
# - Probability calibration:
#     * Isotonic calibrator(s) on pooled 2012/2016/2020 predictions (weighted) applied to 2024 probabilities
# - EC reporting:
#     * EC (base-style): aggregate row-wise fallback probabilities to states
#     * EC (state-refit): refit knee+state for EC scoring only (robustness)
#     * Optional min_neff: omit ultra-thin states from EV tallies for transparency
# - Outputs: Pareto plot/CSV, national summary with CIs, EC tables, report.

import os
import json
import random
import warnings
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import NSGAIISampler
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore")

# ------------------ USER CONFIG ------------------
DATA_2012 = "/content/ANES_data_predicting_popular_vote_shares_2012.dta"
DATA_2016 = "/content/ANES_data_predicting_popular_vote_shares_2016.dta"
DATA_2020 = "/content/ANES_data_predicting_popular_vote_shares_2020.dta"
DATA_2024 = "/content/anes_timeseries_2024_stata_20250808.dta"  # your file

# Confirmed 2024 base weight
WEIGHT_VAR_OVERRIDE_2024 = "V240107a"

# If 2024 WNH derived missingness > this threshold, skip models that require WNH in row-wise fallback
WNH_MISSINGNESS_THRESHOLD = 0.20

# EC strategy options: "base" (only base-style), "state_refit" (only refit with state), "dual" (report both)
EC_STRATEGY = "dual"

# Minimum effective N to include a state's EV in tallies (0 to include all)
EC_MIN_NEFF = 0.0  # set to 100.0 for a publication filter

# Optimization
N_TRIALS = 200
N_SPLITS = 5
RANDOM_SEED = 42

# Outputs
OUT_DIR = "/content/optuna_outputs_forward"
BOOTSTRAP_REPS = 500
BOOTSTRAP_SEED = 123

# EC map (winner-take-all; ME/NE not split)
EV_2024 = {
    "AL": 9,"AK": 3,"AZ": 11,"AR": 6,"CA": 54,"CO": 10,"CT": 7,"DE": 3,"DC": 3,"FL": 30,"GA": 16,
    "HI": 4,"ID": 4,"IL": 19,"IN": 11,"IA": 6,"KS": 6,"KY": 8,"LA": 8,"ME": 4,"MD": 10,"MA": 11,"MI": 15,
    "MN": 10,"MS": 6,"MO": 10,"MT": 4,"NE": 5,"NV": 6,"NH": 4,"NJ": 14,"NM": 5,"NY": 28,"NC": 16,"ND": 3,
    "OH": 17,"OK": 7,"OR": 8,"PA": 19,"RI": 4,"SC": 9,"SD": 3,"TN": 11,"TX": 40,"UT": 6,"VT": 3,"VA": 13,
    "WA": 12,"WV": 4,"WI": 10,"WY": 3
}
FIPS_TO_POSTAL = {
    1:"AL",2:"AK",4:"AZ",5:"AR",6:"CA",8:"CO",9:"CT",10:"DE",11:"DC",12:"FL",13:"GA",15:"HI",16:"ID",17:"IL",
    18:"IN",19:"IA",20:"KS",21:"KY",22:"LA",23:"ME",24:"MD",25:"MA",26:"MI",27:"MN",28:"MS",29:"MO",30:"MT",
    31:"NE",32:"NV",33:"NH",34:"NJ",35:"NM",36:"NY",37:"NC",38:"ND",39:"OH",40:"OK",41:"OR",42:"PA",44:"RI",
    45:"SC",46:"SD",47:"TN",48:"TX",49:"UT",50:"VT",51:"VA",53:"WA",54:"WV",55:"WI",56:"WY"
}

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ------------------ Utilities ------------------

def _exists(p: str) -> bool:
    try: return os.path.exists(p)
    except: return False

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

# ------------------ Past years loaders (mirror Stata, harmonized) ------------------

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
        "Disapp_economy_norm": (dapp - 1)/4,             # harmonized 0..1
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
        "Disapp_economy_norm": (dapp - 1)/4,             # harmonized 0..1
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
    # Flip direction for GOP incumbent year so the construct is aligned across years
    disapp_norm_aligned = 1.0 - ((dapp - 1)/4)
    X = pd.DataFrame({
        "lib_cons_norm": (lib - 1)/6,
        "Disapp_economy_norm": disapp_norm_aligned,      # aligned construct
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

# ------------------ 2024 loader (corrected codebook mappings + validation) ------------------

VAR_2024_PREFS = {
    # Preferred then fallback candidates for each concept
    "ideology": ["V241177", "V241200"],          # summary 7-pt, fallback raw 7-pt
    "econ_approval": ["V241143x", "V241327"],    # summary 1..5, fallback raw item
    "personal_econ": ["V241451", "V241502"],     # summary vs last year, fallback raw
    "gov_trust": ["V241229", "V241233"],
    "age": ["V241458x", "V241507", "V241507x"],
    "education": ["V241465x", "V241511"],
    "gender": ["V241550", "V241600a"],           # 1=male,2=female or similar
    "state_candidates": ["V241023", "V243002", "V241014"],  # prefer registration state
    "race_eth_summary": ["V241501x"],            # 1 = White, non-Hispanic (preferred)
    "race": ["V241540"],
    "hispanic": ["V241541"],
    "race_alt_combined": ["V241155x"],           # if present/usable
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
    msg = " | ".join([f"{c}:{cov*100:.1f}% ({card} states)" for c, cov, card in diag]) if diag else "no candidates found"
    if best_series is None:
        print("WARNING: No suitable state variable met cardinality threshold. Falling back to best coverage.")
        if diag:
            col, cov, card = max(diag, key=lambda t: t[1])
            s_raw = _neg_to_nan(df[col])
            best_series = s_raw.where(s_raw.between(1,56, inclusive="both"), np.nan)
            best_col, best_cov, best_card = col, cov, card
        else:
            best_series = pd.Series(np.nan, index=df.index)
            best_col, best_cov, best_card = "none", 0.0, 0
    print(f"2024 state candidates coverage -> {msg}")
    print(f"Using state variable: {best_col} (valid coverage {best_cov*100:.1f}%, distinct states {best_card})")
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
    # edu summary expected 1..5 (or similar), bound generously and normalize later
    edu = _bound(edu, 1, 15)  # supports 2016-like coding too

    # Female 1/0
    female = pd.Series(np.nan, index=df.index, dtype=float)
    female.loc[gender == 2] = 1.0
    female.loc[gender == 1] = 0.0

    # White non-Hispanic
    # 1) try summary V241501x (1 = White non-Hispanic)
    race_sum, race_sum_col = _pick_first(df, VAR_2024_PREFS["race_eth_summary"])
    white_nh = pd.Series(np.nan, index=df.index, dtype=float)
    if race_sum_col != "none":
        white_nh.loc[race_sum == 1] = 1.0
        white_nh.loc[(race_sum >= 2) & (race_sum <= 6)] = 0.0
        wnh_source = race_sum_col
    else:
        # 2) fallback: derive from race + hisp
        race, race_col = _pick_first(df, VAR_2024_PREFS["race"])
        hisp, hisp_col = _pick_first(df, VAR_2024_PREFS["hispanic"])
        if race_col != "none" and hisp_col != "none":
            white_nh.loc[(race == 1) & (hisp.isin([0, 2]))] = 1.0
            white_nh.loc[(race.isin([2,3,4,5,6])) | (hisp == 1)] = 0.0
            wnh_source = f"{race_col}+{hisp_col}"
        else:
            # 3) fallback2: a combined alt if available
            race_alt, race_alt_col = _pick_first(df, VAR_2024_PREFS["race_alt_combined"])
            if race_alt_col != "none":
                # Heuristic: treat lowest code as missing; map 1->WNH, others 0 if plausible
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
        print("WARNING: 2024 base weight not found; using equal weights (1.0).")
        w = pd.Series(1.0, index=df.index, name="weight")
        wcol_found = "equal"

    # Normalization
    # For education, normalize adaptively: if max <=5 assume 1..5; else Stata 2016 style 1..15
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

    print("2024 variable coverage after cleaning (non-missing):")
    for col in ["lib_cons_norm","Disapp_economy_norm","white_nonhispanic","age_norm","edu_norm","female"]:
        cov = float(X[col].notna().mean()); print(f"  - {col}: {cov*100:.1f}%")
    print(f"2024 WNH source: {wnh_source}; mean={float(np.nanmean(white_nh.values)):.3f}, missing={wnh_missing*100:.1f}%")
    print(f"2024 weight source: {wcol_found}; coverage={w.notna().mean()*100:.1f}% | eff_N={eff_sample_size(w.values):.1f}")

    return X, w, wnh_missing

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
    m = df_all["target"].notna() & df_all["weight"].notna()
    for c in selected: m &= df_all[c].notna()
    data = df_all.loc[m].reset_index(drop=True)
    if len(data) < 300 or data["target"].nunique() < 2: return 0.5
    X = data[selected]; y = data["target"].astype(int).values; w = data["weight"].astype(float).values
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    aucs = []
    for tr_idx, te_idx in skf.split(X, y):
        pipe = build_pipe(selected, C)
        pipe.fit(X.iloc[tr_idx], y[tr_idx], clf__sample_weight=w[tr_idx])
        p = pipe.predict_proba(X.iloc[te_idx])[:,1]
        aucs.append(_roc_auc_safe(y[te_idx], p, sample_weight=w[te_idx]))
    return float(np.mean(aucs)) if aucs else 0.5

def make_objective(df_all: pd.DataFrame):
    def obj(trial: optuna.trial.Trial):
        sel = [v for v in CANDIDATE_VARS if trial.suggest_categorical(f"include_{v}", [True, False])]
        if not sel: return 0.0, 99.0
        C = trial.suggest_float("logreg_C", 1e-2, 100.0, log=True)
        auc = five_fold_weighted_auc(df_all, sel, C)
        return float(auc), float(len(sel))
    return obj

def recommend_knee_point(study: optuna.study.Study) -> Optional[optuna.trial.FrozenTrial]:
    pareto = study.best_trials
    if not pareto: return None
    aucs = np.array([t.values[0] for t in pareto])
    cplx = np.array([t.values[1] for t in pareto])
    max_auc, min_c = aucs.max(), cplx.min()
    auc_range = max(1e-12, aucs.max()-aucs.min())
    c_range = max(1e-12, cplx.max()-cplx.min())
    d = np.sqrt(((max_auc - aucs)/auc_range)**2 + ((cplx - min_c)/c_range)**2)
    return pareto[int(np.argmin(d))]

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
    p = np.asarray(p, float); w = np.asarray(w, float); s = pd.to_numeric(state_fips, errors="coerce")
    m = np.isfinite(p) & np.isfinite(w) & np.isfinite(s)
    if not m.any():
        return pd.DataFrame(columns=["state","mean_prob","share_thr_05","neff","used_n","w_sum","ev","winner_mean","winner_thr05","reliability","eligible"]), 0, 0, 0
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
        else:
            meanp = float(np.sum(pv * wv) / w_sum)
            share05 = float(np.sum((pv > 0.5).astype(int) * wv) / w_sum)
            neff = eff_sample_size(wv)
        rows.append({"state": st, "mean_prob": meanp, "share_thr_05": share05, "neff": neff, "used_n": used_n, "w_sum": w_sum, "ev": EV_2024[st]})
    sdf = pd.DataFrame(rows).sort_values("ev", ascending=False)
    if sdf.empty:
        return sdf, 0, 0, 0
    sdf["winner_mean"] = np.where(sdf["mean_prob"] > 0.5, "R", "D")
    sdf["winner_thr05"] = np.where(sdf["share_thr_05"] > 0.5, "R", "D")
    sdf["reliability"] = np.where(sdf["neff"] >= 100, "OK", "LOW_Neff")
    sdf["eligible"] = sdf["neff"] >= float(min_neff)
    ev_mean = int(sdf.loc[sdf["eligible"] & (sdf["winner_mean"]=="R"),"ev"].sum())
    ev_thr05 = int(sdf.loc[sdf["eligible"] & (sdf["winner_thr05"]=="R"),"ev"].sum())
    total_ev_present = int(sdf.loc[sdf["eligible"], "ev"].sum())
    return sdf, ev_mean, ev_thr05, total_ev_present

# ------------------ Main ------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Ensure files exist
    for p in [DATA_2012, DATA_2016, DATA_2020, DATA_2024]:
        if not _exists(p):
            print(f"Missing file: {p}")
            return

    # Load training years (harmonized)
    X12, y12, w12, yr12 = load_2012(DATA_2012)
    X16, y16, w16, yr16 = load_2016(DATA_2016)
    X20, y20, w20, yr20 = load_2020(DATA_2020)

    def pack(X,y,w,yr):
        df = X.copy(); df["target"]=y; df["weight"]=w; df["year"]=yr; return df

    df_all = pd.concat([pack(X12,y12,w12,yr12),
                        pack(X16,y16,w16,yr16),
                        pack(X20,y20,w20,yr20)], ignore_index=True)

    # Optuna: maximize AUC (5-fold CV, weighted) vs minimize #features
    print("\nOptimizing (5-fold CV AUC vs #features)...")
    study = optuna.create_study(directions=["maximize","minimize"], sampler=NSGAIISampler(seed=RANDOM_SEED))
    study.optimize(make_objective(df_all), n_trials=N_TRIALS, show_progress_bar=True)

    # Plot/save Pareto
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    xs = [t.values[1] for t in trials]; ys = [t.values[0] for t in trials]
    pareto = study.best_trials; px = [t.values[1] for t in pareto]; py = [t.values[0] for t in pareto]
    plt.figure(figsize=(7,5))
    sns.scatterplot(x=xs,y=ys,alpha=0.3,label="All trials", s=26)
    sns.scatterplot(x=px,y=py,color="red",label="Pareto", s=55)
    plt.xlabel("# features"); plt.ylabel("AUC (5-fold, weighted)"); plt.title("Pareto: Accuracy vs Complexity")
    plt.grid(True, alpha=0.2)
    pareto_png = os.path.join(OUT_DIR, "pareto_front.png"); plt.tight_layout(); plt.savefig(pareto_png, dpi=150); plt.show()
    pareto_rows = []
    for t in pareto:
        feats = [k.replace("include_","") for k,v in t.params.items() if k.startswith("include_") and v]
        pareto_rows.append({"trial": t.number, "auc": t.values[0], "complexity": t.values[1], "C": t.params.get("logreg_C", None), "features": json.dumps(feats)})
    pareto_csv = os.path.join(OUT_DIR, "pareto_front.csv")
    pd.DataFrame(pareto_rows).sort_values(["complexity","auc"], ascending=[True,False]).to_csv(pareto_csv, index=False)
    print(f"Saved {pareto_png} and {pareto_csv}")

    # Knee selection
    knee = recommend_knee_point(study)
    if knee is None:
        print("No Pareto solutions found."); return
    knee_features = [k.replace("include_","") for k,v in knee.params.items() if k.startswith("include_") and v]
    knee_C = knee.params.get("logreg_C", 1.0)
    knee_auc = knee.values[0]
    print(f"\nKnee-point: trial #{knee.number} | AUC={knee_auc:.3f} | k={int(knee.values[1])} | C={knee_C:.4g}")
    print("Selected features:", knee_features)

    # Build candidate models for row-wise fallback (sorted by AUC desc)
    def extract_best(cond_fn, pref_len: Optional[int] = None):
        best = None
        for t in study.best_trials:
            feats = [k.replace("include_","") for k,v in t.params.items() if k.startswith("include_") and v]
            if pref_len is not None and len(feats) != pref_len: continue
            if not cond_fn(set(feats)): continue
            auc = t.values[0]; C = t.params.get("logreg_C", 1.0)
            if (best is None) or (auc > best["auc"]):
                best = {"features": feats, "C": C, "auc": auc, "trial": t.number}
        return best

    two_any = extract_best(lambda s: True, pref_len=2)
    two_no_lib = extract_best(lambda s: ("lib_cons_norm" not in s), pref_len=2)
    one_no_lib = extract_best(lambda s: ("lib_cons_norm" not in s), pref_len=1)

    if two_any:
        print(f"\nBest 2-feature (any): {two_any['features']} | C={two_any['C']:.4g} | AUC≈{two_any['auc']:.3f}")
    if two_no_lib:
        print(f"Best 2-feature (no ideology): {two_no_lib['features']} | C={two_no_lib['C']:.4g} | AUC≈{two_no_lib['auc']:.3f}")
    if one_no_lib:
        print(f"Best 1-feature (no ideology): {one_no_lib['features']} | C={one_no_lib['C']:.4g} | AUC≈{one_no_lib['auc']:.3f}")

    # Compose model registry in priority order by AUC
    models: List[Dict[str, Any]] = []
    models.append({"name": "knee", "features": knee_features, "C": knee_C, "auc": knee_auc})
    if two_any: models.append({"name": "2feat_any", "features": two_any["features"], "C": two_any["C"], "auc": two_any["auc"]})
    if two_no_lib: models.append({"name": "2feat_no_lib", "features": two_no_lib["features"], "C": two_no_lib["C"], "auc": two_no_lib["auc"]})
    if one_no_lib: models.append({"name": "1feat_no_lib", "features": one_no_lib["features"], "C": one_no_lib["C"], "auc": one_no_lib["auc"]})
    # sort by auc desc, keep unique by tuple(features) to avoid duplicates
    seen = set()
    models_sorted = []
    for m in sorted(models, key=lambda d: d["auc"], reverse=True):
        key = tuple(sorted(m["features"]))
        if key in seen: continue
        seen.add(key); models_sorted.append(m)
    models = models_sorted

    # Train pipelines and calibrators for each model on pooled training
    trained: Dict[str, Dict[str, Any]] = {}
    for m in models:
        feats = m["features"]; C = m["C"]; name = m["name"]
        mask_tr = df_all["target"].notna() & df_all["weight"].notna()
        for c in feats: mask_tr &= df_all[c].notna()
        dfit = df_all.loc[mask_tr]
        if len(dfit) == 0:
            print(f"WARN: No training rows for model {name} ({feats}); skipping.")
            continue
        pipe = build_pipe(feats, C)
        pipe.fit(dfit[feats], dfit["target"].astype(int), clf__sample_weight=dfit["weight"].astype(float).values)
        # Training predictions for calibration
        p_tr = pipe.predict_proba(dfit[feats])[:,1]
        y_tr = dfit["target"].astype(int).values
        w_tr = dfit["weight"].astype(float).values
        calibrator = fit_isotonic(p_tr, y_tr, w_tr)
        trained[name] = {"pipe": pipe, "features": feats, "cal": calibrator, "C": C, "auc": m["auc"]}
        print(f"Trained model [{name}] on {len(dfit)} rows | feats={feats} | C={C:.4g} | AUC≈{m['auc']:.3f}")

    # State-refit EC model: knee + state (robustness)
    ec_refit = None
    sel_ec = list(knee_features) if "state" in knee_features else list(knee_features) + ["state"]
    mask_ec = df_all["target"].notna() & df_all["weight"].notna()
    for c in sel_ec: mask_ec &= df_all[c].notna()
    dfit_ec = df_all.loc[mask_ec]
    if len(dfit_ec) > 0:
        pipe_ec = build_pipe(sel_ec, knee_C)
        pipe_ec.fit(dfit_ec[sel_ec], dfit_ec["target"].astype(int), clf__sample_weight=dfit_ec["weight"].astype(float).values)
        p_ec_tr = pipe_ec.predict_proba(dfit_ec[sel_ec])[:,1]
        cal_ec = fit_isotonic(p_ec_tr, dfit_ec["target"].astype(int).values, dfit_ec["weight"].astype(float).values)
        ec_refit = {"pipe": pipe_ec, "features": sel_ec, "cal": cal_ec}
        print(f"Trained state-refit EC model on {len(dfit_ec)} rows | feats={sel_ec}")
    else:
        print("WARN: No training rows for state-refit EC; skipping that variant.")

    # Load 2024
    print("\nLoading 2024 and forecasting...")
    X24, w24, wnh_missing = load_2024(DATA_2024)
    w_vec = w24.astype(float).values

    # Row-wise fallback scoring on 2024 with calibration, skipping models using WNH if missingness is high
    p_out = np.full(len(X24), np.nan, dtype=float)
    assigned = np.zeros(len(X24), dtype=bool)
    fill_report = []
    for name, obj in trained.items():
        feats = obj["features"]
        if ("white_nonhispanic" in feats) and (wnh_missing > WNH_MISSINGNESS_THRESHOLD):
            continue
        mrows = X24[feats].notna().all(axis=1) & (~assigned)
        n_rows = int(mrows.sum())
        if n_rows == 0:
            fill_report.append((name, feats, 0))
            continue
        p = obj["pipe"].predict_proba(X24.loc[mrows, feats])[:,1]
        p_cal = np.clip(obj["cal"].transform(p), 0.0, 1.0)
        p_out[mrows.values] = p_cal
        assigned[mrows.values] = True
        fill_report.append((name, feats, n_rows))
    total_assigned = int(np.isfinite(p_out).sum())
    print("Row-wise fallback fill counts (2024):")
    for nm, feats, n in fill_report:
        print(f"  - {nm}: {n} rows | feats={feats}")
    print(f"Total 2024 rows scored: {total_assigned} of {len(X24)}")

    # National aggregation (base-style, row-wise fallback calibrated probabilities)
    nat = forecast_national_from_preds(p_out, w_vec)
    print("\nNational popular-vote style forecast (Republican vs others):")
    print(f"- Weighted mean probability: {pct(nat['weighted_mean_prob'])}")
    print(f"- Threshold 0.5: {pct(nat['share_thr_05'])} (95% CI {pct(nat['share_thr_05_ci'][0])}–{pct(nat['share_thr_05_ci'][1])})")
    print(f"- Threshold = weighted mean prob: {pct(nat['share_thr_mean'])} (95% CI {pct(nat['share_thr_mean_ci'][0])}–{pct(nat['share_thr_mean_ci'][1])})")
    print(f"- Used {nat['used']} of {len(X24)} 2024 rows")

    # EC outputs
    ec_outputs = []

    # EC (base-style): use the same p_out
    if EC_STRATEGY in ("base", "dual"):
        sdf_base, ev_mean_base, ev_thr05_base, total_ev_present_base = state_level_ec_from_preds(
            p_out, w_vec, X24["state"].values if "state" in X24.columns else np.full(len(X24), np.nan),
            min_neff=EC_MIN_NEFF
        )
        ec_outputs.append(("EC (base-style, row-wise fallback)", sdf_base, ev_mean_base, ev_thr05_base, total_ev_present_base))

    # EC (state-refit robustness): predict with knee+state refit and calibration
    if EC_STRATEGY in ("state_refit", "dual") and ec_refit is not None:
        feats_ec = ec_refit["features"]
        mask_rows_ec = X24[feats_ec].notna().all(axis=1) & w24.notna()
        p_ec = np.full(len(X24), np.nan, dtype=float)
        if mask_rows_ec.any():
            raw = ec_refit["pipe"].predict_proba(X24.loc[mask_rows_ec, feats_ec])[:,1]
            p_ec[mask_rows_ec.values] = np.clip(ec_refit["cal"].transform(raw), 0.0, 1.0)
        sdf_refit, ev_mean_refit, ev_thr05_refit, total_ev_present_refit = state_level_ec_from_preds(
            p_ec, w_vec, X24["state"].values if "state" in X24.columns else np.full(len(X24), np.nan),
            min_neff=EC_MIN_NEFF
        )
        ec_outputs.append(("EC (state-refit: knee + state)", sdf_refit, ev_mean_refit, ev_thr05_refit, total_ev_present_refit))

    # Console EC summary
    print("\nElectoral College projection (winner-take-all; ME/NE not split):")
    for label, sdf, ev_mean, ev_thr05, total_ev_present in ec_outputs:
        if total_ev_present == 538:
            print(f"- {label}: R {ev_mean} / D {538 - ev_mean} (mean-prob winners); R {ev_thr05} / D {538 - ev_thr05} (0.5-threshold)")
        else:
            print(f"- {label}: R {ev_mean} of {total_ev_present} present (mean-prob); R {ev_thr05} of {total_ev_present} present (0.5-threshold)")
        low_reli = sdf[sdf["reliability"]=="LOW_Neff"]["state"].tolist()
        if low_reli:
            print(f"  Low reliability states: {', '.join(low_reli)}")

    # Save outputs
    os.makedirs(OUT_DIR, exist_ok=True)
    pareto_png = os.path.join(OUT_DIR, "pareto_front.png")
    pareto_csv = os.path.join(OUT_DIR, "pareto_front.csv")
    for label, sdf, _, _, _ in ec_outputs:
        tag = "base" if "base-style" in label else "state_refit"
        sdf.to_csv(os.path.join(OUT_DIR, f"forecast_2024_states_{tag}.csv"), index=False)

    report_md = os.path.join(OUT_DIR, "forecast_2024_report.md")
    lines = []
    lines.append("# 2024 Forward Forecast (trained on 2012, 2016, 2020)")
    lines.append(f"Selected knee model: features = {', '.join(knee_features)}; C = {knee_C:.4g}; AUC (5-fold) = {knee_auc:.3f}.")
    lines.append("National popular-vote style estimate (with survey weights), using row-wise fallback + isotonic calibration:")
    lines.append(f"- Weighted mean probability: {pct(nat['weighted_mean_prob'])}")
    lines.append(f"- Threshold 0.5: {pct(nat['share_thr_05'])} (95% CI {pct(nat['share_thr_05_ci'][0])}–{pct(nat['share_thr_05_ci'][1])})")
    lines.append(f"- Threshold = weighted mean prob: {pct(nat['share_thr_mean'])} (95% CI {pct(nat['share_thr_mean_ci'][0])}–{pct(nat['share_thr_mean_ci'][1])})")
    lines.append(f"- 2024 rows scored: {int(np.isfinite(p_out).sum())} of {len(X24)} via row-wise fallback")
    lines.append("")
    lines.append("Electoral College projection (winner-take-all; ME/NE not split):")
    for label, sdf, ev_mean, ev_thr05, total_ev_present in ec_outputs:
        if total_ev_present == 538:
            lines.append(f"- {label}: R {ev_mean} / D {538 - ev_mean} (mean-prob); R {ev_thr05} / D {538 - ev_thr05} (0.5-threshold)")
        else:
            lines.append(f"- {label}: R {ev_mean} of {total_ev_present} (mean-prob); R {ev_thr05} of {total_ev_present} (0.5-threshold)")
    lines.append("")
    lines.append("Notes:")
    lines.append("- Training harmonization: 2012/2016 Disapp_economy_norm scaled to 0–1; 2020 flipped to align construct across years.")
    lines.append(f"- 2024 base weight used: '{WEIGHT_VAR_OVERRIDE_2024 or VAR_2024_PREFS['weight'][0]}' (confirmed).")
    lines.append(f"- 2024 white_nonhispanic from {VAR_2024_PREFS['race_eth_summary'][0]} if present; else derived from race+Hispanic.")
    lines.append(f"- State variable chosen by valid coverage+cardinality from candidates {VAR_2024_PREFS['state_candidates']}.")
    lines.append(f"- EC strategy: {EC_STRATEGY}; min_neff for EV tallies = {EC_MIN_NEFF}.")
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nSaved outputs to {OUT_DIR}")

if __name__ == "__main__":
    main()
