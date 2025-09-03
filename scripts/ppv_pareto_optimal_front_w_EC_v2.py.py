import os
import sys
import json
import random
import warnings
from typing import Dict, Tuple, List, Any

def _ensure_packages():
    pkgs = ["optuna", "pandas", "numpy", "scikit-learn", "matplotlib", "seaborn", "pyreadstat"]
    import importlib
    to_install = []
    for p in pkgs:
        try:
            importlib.import_module(p if p != "scikit-learn" else "sklearn")
        except Exception:
            to_install.append(p)
    if to_install:
        print("Installing missing packages:", to_install)
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", *to_install])
_ensure_packages()

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import NSGAIISampler
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# ------------------ USER CONFIG ------------------
DATA_PATHS = {
    2012: "/content/ANES_data_predicting_popular_vote_shares_2012.dta",
    2016: "/content/ANES_data_predicting_popular_vote_shares_2016.dta",
    2020: "/content/ANES_data_predicting_popular_vote_shares_2020.dta",
}
WEIGHT_VARS = {
    2012: "weight_full",
    2016: "V160101",
    2020: "V200010a",
}
CSV_2024_PATH = "/content/anes_timeseries_2024_csv_20250808.csv"

# EDIT THIS MAPPING to match your 2024 CSV column names (case-sensitive).
# If you're unsure, run once; the script will print suggestions from your CSV header.
MAP_2024 = {
    # target
    "vote_var": None,              # e.g., "V241033" (if present); 1=Dem, 2=Rep, 3/4/5 other; -8/-9 etc missing
    # predictors
    "lib_cons": None,              # 7-pt ideology self-placement
    "Disapp_economy": None,        # presidential economic approval/disapproval
    "pers_econ_worse": None,       # personal finances vs last year
    "Distrust_gov": None,          # trust in federal government
    "age": None,                   # age
    "female": None,                # gender
    "edu": None,                   # education
    "white_nonhispanic": None,     # race/ethnicity collapsed: white NH=1 else 0 (we’ll recode)
    "state": None,                 # FIPS or state code comparable to prior 'state'
    # weights
    "weight": None,                # pre-election full-sample weight (e.g., "V240010a" or similar)
}

# Optimization config
N_TRIALS = 200
RANDOM_SEED = 42
USE_WEIGHTED_AUC = True
COMPLEXITY_MODE = "features"  # "features" or "params"
OUT_DIR = "/content/optuna_outputs_forward"
BOOTSTRAP_REPS = 500
BOOTSTRAP_SEED = 123

# Electoral votes (2024 apportionment). Maine/Nebraska treated winner-take-all here (limitation).
EV_2024 = {
    "AL": 9,"AK": 3,"AZ": 11,"AR": 6,"CA": 54,"CO": 10,"CT": 7,"DE": 3,"DC": 3,"FL": 30,"GA": 16,
    "HI": 4,"ID": 4,"IL": 19,"IN": 11,"IA": 6,"KS": 6,"KY": 8,"LA": 8,"ME": 4,"MD": 10,"MA": 11,"MI": 15,
    "MN": 10,"MS": 6,"MO": 10,"MT": 4,"NE": 5,"NV": 6,"NH": 4,"NJ": 14,"NM": 5,"NY": 28,"NC": 16,"ND": 3,
    "OH": 17,"OK": 7,"OR": 8,"PA": 19,"RI": 4,"SC": 9,"SD": 3,"TN": 11,"TX": 40,"UT": 6,"VT": 3,"VA": 13,
    "WA": 12,"WV": 4,"WI": 10,"WY": 3
}

# FIPS to postal for aggregation (ANES 'state' often uses FIPS)
FIPS_TO_POSTAL = {
    1:"AL",2:"AK",4:"AZ",5:"AR",6:"CA",8:"CO",9:"CT",10:"DE",11:"DC",12:"FL",13:"GA",15:"HI",16:"ID",17:"IL",
    18:"IN",19:"IA",20:"KS",21:"KY",22:"LA",23:"ME",24:"MD",25:"MA",26:"MI",27:"MN",28:"MS",29:"MO",30:"MT",
    31:"NE",32:"NV",33:"NH",34:"NJ",35:"NM",36:"NY",37:"NC",38:"ND",39:"OH",40:"OK",41:"OR",42:"PA",44:"RI",
    45:"SC",46:"SD",47:"TN",48:"TX",49:"UT",50:"VT",51:"VA",53:"WA",54:"WV",55:"WI",56:"WY"
}
# ---------------------------------------------------

rng = np.random.RandomState(RANDOM_SEED)
random.seed(RANDOM_SEED)

def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)

def _exists(path: str) -> bool:
    try:
        return os.path.exists(path)
    except Exception:
        return False

def _to_num(s):
    return pd.to_numeric(s, errors="coerce")

def _roc_auc_safe(y_true: np.ndarray, y_score: np.ndarray, sample_weight=None) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return 0.5
        return roc_auc_score(y_true, y_score, sample_weight=sample_weight)
    except Exception:
        return 0.5

def _weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(w)
    if m.sum() == 0:
        return float("nan")
    wp = np.clip(w[m], 0, None)
    if wp.sum() <= 0:
        return float("nan")
    return float(np.sum(wp * x[m]) / np.sum(wp))

def _weighted_share_from_threshold(p: np.ndarray, w: np.ndarray, thr: float) -> float:
    preds = (p > thr).astype(int)
    if np.sum(w) <= 0:
        return float("nan")
    return float(np.sum(w * preds) / np.sum(w))

def weighted_bootstrap_ci_share(p: np.ndarray, w: np.ndarray, thr: float, reps: int, seed: int, alpha: float = 0.05) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    w_pos = np.clip(w, 0, None)
    if w_pos.sum() <= 0:
        return (float("nan"), float("nan"))
    probs = w_pos / w_pos.sum()
    n = len(p)
    stats = []
    for _ in range(reps):
        idx = rng.choice(n, size=n, replace=True, p=probs)
        stats.append(_weighted_share_from_threshold(p[idx], w[idx], thr))
    lo = float(np.nanpercentile(stats, 100*(alpha/2)))
    hi = float(np.nanpercentile(stats, 100*(1 - alpha/2)))
    return lo, hi

def eff_sample_size(w: np.ndarray) -> float:
    w = np.asarray(w, float)
    if not np.isfinite(w).any() or np.sum(w) <= 0:
        return 0.0
    return float((w.sum()**2) / (np.sum(w**2) + 1e-12))

# ------------------ Load & process (match Stata) ------------------

def load_and_process_2012(path: str):
    df = pd.read_stata(path, convert_categoricals=False)

    y = df["prevote_intpreswho"].copy()
    y = y.where(~y.isin([-9, -8, -1]), np.nan)
    y = y.map(lambda v: 1 if v == 2 else (0 if v in [1, 5] else np.nan))

    lib_cons = df["libcpre_self"].copy().where(~df["libcpre_self"].isin([-9, -8, -2]), np.nan)
    Disapp_economy = df["presapp_econ"].copy().where(~df["presapp_econ"].isin([-9, -8]), np.nan)
    pers_econ_worse = df["finance_finpast"].copy().where(~df["finance_finpast"].isin([-9, -8]), np.nan).replace({3:2, 2:3})
    distrust_gov = df["trustgov_trustgrev"].copy().where(~df["trustgov_trustgrev"].isin([-9, -8, -7, -6, -5, -4, -3, -2, -1]), np.nan)
    age = df["dem_age_r_x"].copy().where(~df["dem_age_r_x"].isin([-2]), np.nan)
    female = df["gender_respondent_x"].copy().replace({2:1, 1:0})
    edu = df["dem_edugroup_x"].copy().where(~df["dem_edugroup_x"].isin([-9, -8, -7, -6, -5, -4, -3, -2]), np.nan)
    white_nonhispanic = df["dem_raceeth_x"].copy().where(~df["dem_raceeth_x"].isin([-9]), np.nan).apply(lambda v: 1 if v == 1 else (0 if v in [2,3,4,5,6] else np.nan))
    state = df["sample_stfips"].copy()

    X = pd.DataFrame({
        "lib_cons_norm": (lib_cons - 1.0)/6.0,
        "Disapp_economy_norm": (Disapp_economy - 1.0),
        "Distrust_gov_norm": (distrust_gov - 1.0)/4.0,
        "age_norm": (age - 18.0)/(90.0-18.0),
        "edu_norm": (edu - 1.0)/4.0,
        "white_nonhispanic": white_nonhispanic,
        "pers_econ_worse_norm": (pers_econ_worse - 1.0)/2.0,
        "female": female,
        "state": state,
    })
    w = df["weight_full"] if "weight_full" in df.columns else pd.Series(np.nan, index=df.index)
    return X, y.rename("target"), w.rename("weight"), pd.Series(2012, index=df.index, name="year")

def load_and_process_2016(path: str):
    df = pd.read_stata(path, convert_categoricals=False)

    y = df["V161031"].copy()
    y = y.where(~y.isin([6, 7, 8, -9, -8, -1]), np.nan)
    y = y.map(lambda v: 1 if v == 2 else (0 if v in [1, 3, 4, 5] else np.nan))

    lib_cons = df["V161126"].copy().where(~df["V161126"].isin([-9, -8, 99]), np.nan)
    Disapp_economy = df["V161083"].copy().where(~df["V161083"].isin([-9, -8]), np.nan)
    pers_econ_worse = df["V161110"].copy().where(~df["V161110"].isin([-9, -8]), np.nan)
    distrust_gov = df["V161215"].copy().where(~df["V161215"].isin([-9, -8]), np.nan)
    age = df["V161267"].copy().where(~df["V161267"].isin([-9, -8]), np.nan)
    female = df["V161342"].copy().replace({2:1, 1:0, 3:0, -9:np.nan})
    edu = df["V161270"].copy().where(~df["V161270"].isin([-9]) & ~df["V161270"].between(90,95, inclusive="both"), np.nan)
    white_nonhispanic = df["V161310x"].copy().where(~df["V161310x"].isin([-2]), np.nan).apply(lambda v: 1 if v == 1 else (0 if v in [2,3,4,5,6] else np.nan))
    state = df["V161015b"].copy().where(~df["V161015b"].isin([-1]), np.nan)

    X = pd.DataFrame({
        "lib_cons_norm": (lib_cons - 1.0)/6.0,
        "Disapp_economy_norm": (Disapp_economy - 1.0),
        "Distrust_gov_norm": (distrust_gov - 1.0)/4.0,
        "age_norm": (age - 18.0)/(90.0-18.0),
        "edu_norm": (edu - 1.0)/15.0,
        "white_nonhispanic": white_nonhispanic,
        "pers_econ_worse_norm": (pers_econ_worse - 1.0)/4.0,
        "female": female,
        "state": state,
    })
    w = df["V160101"] if "V160101" in df.columns else pd.Series(np.nan, index=df.index)
    return X, y.rename("target"), w.rename("weight"), pd.Series(2016, index=df.index, name="year")

def load_and_process_2020(path: str):
    df = pd.read_stata(path, convert_categoricals=False)

    y = df["V201033"].copy()
    y = y.where(~y.isin([-8, -9, -1, 11, 12]), np.nan)
    y = y.map(lambda v: 1 if v == 2 else (0 if v in [1, 3, 4, 5] else np.nan))

    lib_cons = df["V201200"].copy().where(~df["V201200"].isin([-9, -8, 99]), np.nan)
    Disapp_economy = df["V201327x"].copy().where(~df["V201327x"].isin([-2]), np.nan)
    pers_econ_worse = df["V201502"].copy().where(~df["V201502"].isin([-9]), np.nan)
    Distrust_gov = df["V201233"].copy().where(~df["V201233"].isin([-9, -8]), np.nan)
    age = df["V201507x"].copy().where(~df["V201507x"].isin([-9]), np.nan)
    female = df["V201600"].copy().replace({2:1, 1:0, -9:np.nan})
    edu = df["V201511x"].copy().where(~df["V201511x"].isin([-9, -8, -7, -6, -5, -4, -3, -2]), np.nan)
    white_nonhispanic = df["V201549x"].copy().where(~df["V201549x"].isin([-9, -8]), np.nan).apply(lambda v: 1 if v == 1 else (0 if v in [2,3,4,5,6] else np.nan))
    state = df["V201014b"].copy().where(~df["V201014b"].isin([-9,-8,-7,-6,-5,-4,-3,-2,-1,86]), np.nan)

    X = pd.DataFrame({
        "lib_cons_norm": (lib_cons - 1.0)/6.0,
        "Disapp_economy_norm": (Disapp_economy - 1.0)/4.0,
        "Distrust_gov_norm": (Distrust_gov - 1.0)/4.0,
        "age_norm": (age - 18.0)/(80.0-18.0),
        "edu_norm": (edu - 1.0)/4.0,
        "white_nonhispanic": white_nonhispanic,
        "pers_econ_worse_norm": (pers_econ_worse - 1.0)/4.0,
        "female": female,
        "state": state,
    })
    w = df["V200010a"] if "V200010a" in df.columns else pd.Series(np.nan, index=df.index)
    return X, y.rename("target"), w.rename("weight"), pd.Series(2020, index=df.index, name="year")

def suggest_2024_mapping(df: pd.DataFrame):
    print("\n2024 CSV columns preview:")
    print(sorted(df.columns)[:50])
    hints = {
        "vote_var": ["V241033","vote","pres","intention","choice"],
        "lib_cons": ["ideology","V241200","lib","conserv"],
        "Disapp_economy": ["econ","approval","presapp","V241327"],
        "pers_econ_worse": ["finance","finpast","worse","V241502"],
        "Distrust_gov": ["trust","gov","V241233"],
        "age": ["age","V241507"],
        "female": ["gender","female","sex","V241600"],
        "edu": ["educ","V241511"],
        "white_nonhispanic": ["race","eth","nonhisp","V241549"],
        "state": ["state","fips","stfips","V241014"],
        "weight": ["weight","pweight","V240010"],
    }
    print("\nMapping hints (search tokens to look for in your columns):")
    for k,v in hints.items():
        print(f"  {k}: tokens -> {v}")

def load_and_process_2024(csv_path: str, mapping: Dict[str,str]):
    if not _exists(csv_path):
        raise FileNotFoundError(f"2024 CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)

    # If mapping incomplete, print suggestions and raise
    missing = [k for k,v in mapping.items() if v is None]
    if missing:
        suggest_2024_mapping(df)
        raise ValueError(f"Please fill MAP_2024 for keys: {missing}")

    def col(name):
        if name not in df.columns:
            raise KeyError(f"Column '{name}' not found in 2024 CSV.")
        return _to_num(df[name])

    y_raw = col(mapping["vote_var"])
    # Map like 2016/2020: 1=Dem, 2=Rep, 3/4/5 other; negative/missing codes -> NaN
    y = y_raw.copy()
    y = y.where(~y.isin([-9,-8,-7,-6,-5,-4,-3,-2,-1,11,12]), np.nan)
    y = y.map(lambda v: 1 if v == 2 else (0 if v in [1,3,4,5] else np.nan))

    lib_cons = col(mapping["lib_cons"])
    # As 2020: recode (-9/-8=.) (99=.)
    lib_cons = lib_cons.where(~lib_cons.isin([-9,-8,99]), np.nan)

    Disapp_economy = col(mapping["Disapp_economy"])
    # Use 2020 scaling if 0–5 like V201327x; treat -2 as missing
    Disapp_economy = Disapp_economy.where(~Disapp_economy.isin([-9,-8,-2]), np.nan)

    pers_econ_worse = col(mapping["pers_econ_worse"])
    pers_econ_worse = pers_econ_worse.where(~pers_econ_worse.isin([-9,-8]), np.nan)

    Distrust_gov = col(mapping["Distrust_gov"])
    Distrust_gov = Distrust_gov.where(~Distrust_gov.isin([-9,-8]), np.nan)

    age = col(mapping["age"]).where(lambda s: ~s.isin([-9,-8,-7,-6,-5,-4,-3,-2,-1]), np.nan)
    female = col(mapping["female"]).replace({2:1, 1:0, -9:np.nan})
    edu = col(mapping["edu"]).where(lambda s: ~s.isin([-9,-8,-7,-6,-5,-4,-3,-2]), np.nan)
    white_nonhispanic = col(mapping["white_nonhispanic"]).where(~col(mapping["white_nonhispanic"]).isin([-9,-8]), np.nan)
    white_nonhispanic = white_nonhispanic.apply(lambda v: 1 if v == 1 else (0 if v in [2,3,4,5,6] else np.nan))
    state = col(mapping["state"]).where(lambda s: ~s.isin([-9,-8,-7,-6,-5,-4,-3,-2,-1,86]), np.nan)
    w = col(mapping["weight"])

    # Normalize per 2020 formulas (closest to recent instrument)
    X = pd.DataFrame({
        "lib_cons_norm": (lib_cons - 1.0)/6.0,
        "Disapp_economy_norm": (Disapp_economy - 1.0)/4.0,
        "pers_econ_worse_norm": (pers_econ_worse - 1.0)/4.0,
        "Distrust_gov_norm": (Distrust_gov - 1.0)/4.0,
        "age_norm": (age - 18.0)/(80.0-18.0),
        "edu_norm": (edu - 1.0)/4.0,
        "white_nonhispanic": white_nonhispanic,
        "female": female,
        "state": state,
    })
    return X, y.rename("target"), w.rename("weight")

# ------------------ Modeling ------------------

CANDIDATE_VARS = [
    "lib_cons_norm","Disapp_economy_norm","Distrust_gov_norm","age_norm","edu_norm",
    "white_nonhispanic","pers_econ_worse_norm","female","state",
]

def build_pipe(selected: List[str], C: float):
    num_features = [c for c in selected if c != "state"]
    cat_features = ["state"] if "state" in selected else []
    transformers = []
    if num_features:
        transformers.append(("num","passthrough", num_features))
    if cat_features:
        transformers.append(("state", make_ohe(), cat_features))
    ct = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=1.0)
    clf = LogisticRegression(solver="liblinear", C=C, max_iter=1000, random_state=RANDOM_SEED)
    return Pipeline([("ct", ct), ("clf", clf)])

def train_eval_leave_one_year_out(df_all: pd.DataFrame, selected: List[str], C: float) -> float:
    years = sorted(df_all["year"].unique())
    aucs = []
    for hold in years:
        train_mask = df_all["year"] != hold
        test_mask = df_all["year"] == hold
        dtr = df_all.loc[train_mask]
        dte = df_all.loc[test_mask]

        # Filter to non-missing in selected, target, weight
        tr_mask = dtr["target"].notna() & dtr["weight"].notna()
        te_mask = dte["target"].notna() & dte["weight"].notna()
        for c in selected:
            tr_mask &= dtr[c].notna()
            te_mask &= dte[c].notna()
        dtr = dtr.loc[tr_mask]
        dte = dte.loc[te_mask]
        if len(dtr) < 200 or dtr["target"].nunique() < 2 or len(dte) < 100 or dte["target"].nunique() < 2:
            aucs.append(0.5)
            continue

        pipe = build_pipe(selected, C)
        pipe.fit(dtr[selected], dtr["target"].astype(int), clf__sample_weight=dtr["weight"].astype(float).values)
        proba = pipe.predict_proba(dte[selected])[:,1]
        auc = _roc_auc_safe(dte["target"].astype(int).values, proba, sample_weight=dte["weight"].astype(float).values) if USE_WEIGHTED_AUC else _roc_auc_safe(dte["target"].astype(int).values, proba)
        aucs.append(auc)
    return float(np.mean(aucs)) if aucs else 0.5

def make_objective(df_all: pd.DataFrame):
    # Precompute number of state levels for "params" complexity if needed
    n_states = int(pd.Series(df_all["state"].dropna().unique()).nunique()) if "state" in df_all.columns else 0
    def objective(trial: optuna.trial.Trial):
        selected = [v for v in CANDIDATE_VARS if trial.suggest_categorical(f"include_{v}", [True, False])]
        if not selected:
            return 0.0, 99.0
        C = trial.suggest_float("logreg_C", 1e-2, 100.0, log=True)

        auc_loyo = train_eval_leave_one_year_out(df_all, selected, C)

        if COMPLEXITY_MODE == "params":
            cplx = len([v for v in selected if v!="state"]) + (n_states if "state" in selected else 0)
        else:
            cplx = len(selected)
        return float(auc_loyo), float(cplx)
    return objective

def recommend_knee_point(study: optuna.study.Study) -> optuna.trial.FrozenTrial:
    pareto = study.best_trials
    if not pareto:
        return None
    aucs = np.array([t.values[0] for t in pareto])
    cplx = np.array([t.values[1] for t in pareto])
    max_auc, min_c = aucs.max(), cplx.min()
    auc_range = max(1e-9, aucs.max()-aucs.min())
    c_range = max(1e-9, cplx.max()-cplx.min())
    d = np.sqrt(((max_auc - aucs)/auc_range)**2 + ((cplx - min_c)/c_range)**2)
    return pareto[int(np.argmin(d))]

# ------------------ Forecast & EC ------------------

def forecast_2024(pipe: Pipeline, X24: pd.DataFrame, w24: pd.Series):
    mask = X24.notna().all(axis=1) & w24.notna()
    X = X24.loc[mask]
    w = w24.loc[mask].astype(float)
    proba = pipe.predict_proba(X)[:,1]
    mean_prob = _weighted_mean(proba, w.values)
    share05 = _weighted_share_from_threshold(proba, w.values, 0.5)
    lo05, hi05 = weighted_bootstrap_ci_share(proba, w.values, 0.5, BOOTSTRAP_REPS, BOOTSTRAP_SEED)
    lomean, himean = weighted_bootstrap_ci_share(proba, w.values, mean_prob if np.isfinite(mean_prob) else 0.5, BOOTSTRAP_REPS, BOOTSTRAP_SEED)
    return dict(
        proba=proba, idx=X.index, weight=w, 
        national=dict(
            weighted_mean_prob=mean_prob,
            share_thr_05=share05, share_thr_05_ci=(lo05, hi05),
            share_thr_mean= _weighted_share_from_threshold(proba, w.values, mean_prob) if np.isfinite(mean_prob) else float("nan"),
            share_thr_mean_ci=(lomean, himean)
        )
    )

def state_level_ec(proba: np.ndarray, idx: pd.Index, w: pd.Series, state_series: pd.Series):
    # Aggregate weighted mean prob by state (FIPS->postal), compute EV winner
    dfp = pd.DataFrame({"proba": proba, "w": w, "state": state_series.loc[idx]})
    # Coerce state to numeric FIPS if possible
    st_num = pd.to_numeric(dfp["state"], errors="coerce")
    # Map to postal; if mapping fails, leave as string and try direct postal
    postal = st_num.map(FIPS_TO_POSTAL)
    # If some missing and original might already be postal strings
    postal = postal.fillna(dfp["state"].astype(str).str.upper().str.strip())
    # Keep only states we have EV for
    dfp["postal"] = postal
    dfp = dfp[dfp["postal"].isin(EV_2024.keys())]

    # Compute weighted mean prob and effective n per state
    results = []
    for st, sub in dfp.groupby("postal"):
        p = sub["proba"].values
        ww = sub["w"].astype(float).values
        meanp = _weighted_mean(p, ww)
        neff = eff_sample_size(ww)
        share05 = _weighted_share_from_threshold(p, ww, 0.5)
        results.append({"state": st, "mean_prob": meanp, "share_thr_05": share05, "neff": neff, "ev": EV_2024[st]})
    sdf = pd.DataFrame(results)
    # Winner by mean probability > 0.5 (Republican)
    sdf["winner_mean"] = np.where(sdf["mean_prob"] > 0.5, "R", "D")
    sdf["winner_thr05"] = np.where(sdf["share_thr_05"] > 0.5, "R", "D")
    ev_mean = int(sdf.loc[sdf["winner_mean"]=="R", "ev"].sum())
    ev_thr05 = int(sdf.loc[sdf["winner_thr05"]=="R", "ev"].sum())
    # Reliability flag
    sdf["reliability"] = np.where(sdf["neff"] >= 100, "OK", "LOW_Neff")
    return sdf.sort_values("ev", ascending=False).reset_index(drop=True), ev_mean, ev_thr05

# ------------------ Main ------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading past ANES datasets...")
    datasets = []
    for yr in [2012, 2016, 2020]:
        p = DATA_PATHS.get(yr)
        if not _exists(p):
            print(f"Missing file for {yr}: {p}")
            return
        if yr == 2012:
            X,y,w,year = load_and_process_2012(p)
        elif yr == 2016:
            X,y,w,year = load_and_process_2016(p)
        else:
            X,y,w,year = load_and_process_2020(p)
        df = X.copy()
        df["target"] = y
        df["weight"] = w
        df["year"] = year
        datasets.append(df)
        print(f"{yr}: labeled={int(y.notna().sum())}, weight_sum={float(np.nansum(w)):.2f}")

    df_all = pd.concat(datasets, axis=0, ignore_index=True)

    print("\nOptimizing with Optuna (leave-one-year-out weighted AUC vs complexity)...")
    sampler = NSGAIISampler(seed=RANDOM_SEED)
    study = optuna.create_study(directions=["maximize","minimize"], sampler=sampler, study_name="loyo_past_years")
    study.optimize(make_objective(df_all), n_trials=N_TRIALS, show_progress_bar=True)

    # Pareto plot
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    xs = [t.values[1] for t in trials]
    ys = [t.values[0] for t in trials]
    pareto = study.best_trials
    px = [t.values[1] for t in pareto]
    py = [t.values[0] for t in pareto]

    plt.figure(figsize=(7,5))
    sns.scatterplot(x=xs,y=ys,alpha=0.3,label="All trials", s=28)
    sns.scatterplot(x=px,y=py,color="red",label="Pareto", s=55)
    plt.xlabel(f"Model complexity ({COMPLEXITY_MODE})")
    plt.ylabel("Weighted AUC (leave-one-year-out)")
    plt.title("Pareto front (2012/2016/2020 LOYO)")
    plt.grid(True, alpha=0.2)
    pareto_png = os.path.join(OUT_DIR, "pareto_loyo.png")
    plt.tight_layout(); plt.savefig(pareto_png, dpi=150); plt.show()
    print(f"Saved {pareto_png}")

    knee = recommend_knee_point(study)
    if knee is None:
        print("No Pareto solutions found.")
        return

    selected = [k.replace("include_","") for k,v in knee.params.items() if k.startswith("include_") and v]
    C = knee.params.get("logreg_C", 1.0)
    print(f"\nKnee-point trial: #{knee.number}, AUC={knee.values[0]:.3f}, complexity={knee.values[1]}, C={C:.3g}")
    print("Selected features:", selected)

    # Refit on all past years (full data, weighted)
    mask = df_all["target"].notna() & df_all["weight"].notna()
    for c in selected:
        mask &= df_all[c].notna()
    dfit = df_all.loc[mask]
    pipe = build_pipe(selected, C)
    pipe.fit(dfit[selected], dfit["target"].astype(int), clf__sample_weight=dfit["weight"].astype(float).values)

    # Load 2024 and forecast
    try:
        X24, y24, w24 = load_and_process_2024(CSV_2024_PATH, MAP_2024)
    except Exception as e:
        print("\nERROR loading 2024 CSV:", e)
        print("Edit MAP_2024 at the top of this script to match your CSV columns.")
        return

    fc = forecast_2024(pipe, X24[selected], w24)
    proba24, idx24, w_idx24 = fc["proba"], fc["idx"], fc["weight"]
    nat = fc["national"]

    # EC by state
    sdf, ev_mean, ev_thr05 = state_level_ec(proba24, idx24, w_idx24, X24["state"])

    # Save CSVs
    pareto_rows = []
    for t in pareto:
        params = t.params
        feats = [k.replace("include_","") for k,v in params.items() if k.startswith("include_") and v]
        pareto_rows.append({"trial": t.number, "auc_loyo": t.values[0], "complexity": t.values[1], "C": params.get("logreg_C", None), "features": json.dumps(feats)})
    pareto_df = pd.DataFrame(pareto_rows).sort_values(["complexity","auc_loyo"], ascending=[True,False])
    pareto_csv = os.path.join(OUT_DIR, "pareto_loyo.csv"); pareto_df.to_csv(pareto_csv, index=False)

    state_csv = os.path.join(OUT_DIR, "forecast_2024_states.csv"); sdf.to_csv(state_csv, index=False)

    # Plain-language report
    def pct(x): return f"{100*x:.1f}%" if x is not None and np.isfinite(x) else "NA"
    report_lines = []
    report_lines.append(f"# 2024 Forward Forecast (trained on 2012, 2016, 2020)")
    report_lines.append("")
    report_lines.append(f"Optimization used leave-one-year-out weighted AUC across past ANES waves, trading off accuracy and simplicity. We selected a knee-point model on the Pareto front and refit it on all past years before forecasting 2024.")
    report_lines.append("")
    report_lines.append(f"Selected model: features = {', '.join(selected)}; C = {C:.3g}.")
    report_lines.append(f"Past-years validation: knee AUC (LOYO, weighted) = {knee.values[0]:.3f}.")
    report_lines.append("")
    report_lines.append("National popular-vote style forecast (Republican vs. others), using survey weights:")
    report_lines.append(f"- Weighted mean probability (expected support): {pct(nat['weighted_mean_prob'])}")
    report_lines.append(f"- Threshold 0.5: {pct(nat['share_thr_05'])} (95% CI {pct(nat['share_thr_05_ci'][0])}–{pct(nat['share_thr_05_ci'][1])})")
    report_lines.append(f"- Threshold = weighted mean probability: {pct(nat['share_thr_mean'])} (95% CI {pct(nat['share_thr_mean_ci'][0])}–{pct(nat['share_thr_mean_ci'][1])})")
    report_lines.append("")
    report_lines.append("Electoral College projection (winner-take-all; ME/NE not split here; caution with small state samples):")
    report_lines.append(f"- EV by mean-probability winners: Republican {ev_mean}, Democrat {538 - ev_mean}")
    report_lines.append(f"- EV by 0.5-threshold winners: Republican {ev_thr05}, Democrat {538 - ev_thr05}")
    low_reli = sdf[sdf["reliability"]=="LOW_Neff"]["state"].tolist()
    if low_reli:
        report_lines.append(f"- Low reliability (small effective N) states: {', '.join(low_reli)}")
    report_lines.append("")
    report_lines.append("Validity and caveats:")
    report_lines.append("- Forecast is strictly forward: model selection uses only past years, then refit on all past data before scoring 2024.")
    report_lines.append("- ANES is optimized for national inference; some states have low effective sample sizes. Treat state calls as indicative.")
    report_lines.append("- Maine/Nebraska are treated as winner-take-all due to lack of congressional-district granularity.")
    report_lines.append("- For higher-fidelity EC forecasts in future work, consider MRP or external state-level poststratification.")
    report_md = os.path.join(OUT_DIR, "forecast_2024_report.md")
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\nSaved outputs:")
    print(f"- Pareto (LOYO) plot: {pareto_png}")
    print(f"- Pareto (LOYO) CSV: {pareto_csv}")
    print(f"- 2024 state forecast CSV: {state_csv}")
    print(f"- 2024 plain-language report: {report_md}")

if __name__ == "__main__":
    main()
