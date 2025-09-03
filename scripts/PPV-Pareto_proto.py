# End-to-end forecast pipeline with pragmatic 2024 fallbacks.
# - Optimizes on 2012/2016/2020 (5-fold CV, AUC vs #features), selects knee
# - Refit knee on all past years (weighted)
# - Forecast 2024 from your .dta path
#   - Provisional base weight: V240107a (override at top if you confirm a better column)
#   - Derive white_nonhispanic from V241540 (race) and V241541 (Hispanic)
#   - If WNH missingness > 20% in 2024, fall back to the best 2-feature model
#     (lib_cons_norm + Disapp_economy_norm) to avoid dropping many cases
#
# Outputs: Pareto plot/CSV, national summary with CIs, EC table, report.

import os
import json
import random
import warnings
from typing import List, Optional

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

warnings.filterwarnings("ignore")

# ------------------ USER CONFIG ------------------
DATA_2012 = "/content/ANES_data_predicting_popular_vote_shares_2012.dta"
DATA_2016 = "/content/ANES_data_predicting_popular_vote_shares_2016.dta"
DATA_2020 = "/content/ANES_data_predicting_popular_vote_shares_2020.dta"
DATA_2024 = "/content/anes_timeseries_2024_stata_20250808.dta"  # your file

# Provisional 2024 weight override: use this column as base weight if present
# Set to None to run equal-weight forecast (not recommended)
WEIGHT_VAR_OVERRIDE_2024 = "V240107a"

# If 2024 WNH derived missingness > this threshold, switch to 2-feature forecast
WNH_MISSINGNESS_THRESHOLD = 0.20

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

def weighted_bootstrap_ci_share(p: np.ndarray, w: np.ndarray, thr: float, reps: int, seed: int, alpha: float = 0.05):
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

# ------------------ Past years loaders (mirror Stata) ------------------

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
        "Disapp_economy_norm": (dapp - 1),
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
        "Disapp_economy_norm": (dapp - 1),
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
    X = pd.DataFrame({
        "lib_cons_norm": (lib - 1)/6,
        "Disapp_economy_norm": (dapp - 1)/4,
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

# ------------------ 2024 loader (override + derive WNH) ------------------

def load_2024(path: str):
    df = pd.read_stata(path, convert_categoricals=False)

    # Predictors
    lib = _to_num(df.get("V241200")).where(lambda s: ~s.isin([-9,-8,99]), np.nan)
    dapp = _to_num(df.get("V241327")).where(lambda s: ~s.isin([-9,-8,-2]), np.nan)
    pecon = _to_num(df.get("V241502")).where(lambda s: ~s.isin([-9,-8,-7]), np.nan)
    distrust = _to_num(df.get("V241233")).where(lambda s: ~s.isin([-9,-8]), np.nan)
    age = _to_num(df.get("V241507")).where(lambda s: ~s.isin([-9,-8]), np.nan)
    edu = _to_num(df.get("V241511")).where(lambda s: ~s.isin([-9,-8,-7,-6,-5,-4,-3,-2]), np.nan)
    female = _to_num(df.get("V241600a")).where(lambda s: ~s.isin(list(range(-9,0))), np.nan)
    state = _to_num(df.get("V241014")).where(lambda s: ~s.isin([-9,-8,-7,-6,-5,-4,-3,-2,-1,86]), np.nan)

    # Derive white_nonhispanic from race + hispanic
    race = _to_num(df.get("V241540"))
    hisp = _to_num(df.get("V241541"))
    race = race.where(~race.isin(list(range(-9,0))), np.nan)
    hisp = hisp.where(~hisp.isin(list(range(-9,0))), np.nan)
    white_nh = pd.Series(np.nan, index=df.index, dtype=float)
    white_nh.loc[(race == 1) & (hisp.isin([0,2]))] = 1.0
    white_nh.loc[(race.isin([2,3,4,5,6])) | (hisp == 1)] = 0.0
    wnh_missing = float(pd.isna(white_nh).mean())

    # Provisional weight: override
    if WEIGHT_VAR_OVERRIDE_2024 and WEIGHT_VAR_OVERRIDE_2024 in df.columns:
        w = _to_num(df[WEIGHT_VAR_OVERRIDE_2024]).where(lambda s: ~s.isin(list(range(-9,1))), np.nan).rename("weight")
        w_source = WEIGHT_VAR_OVERRIDE_2024
    else:
        print("WARNING: 2024 base weight not found; using equal weights (1.0).")
        w = pd.Series(1.0, index=df.index, name="weight")
        w_source = "equal"

    X = pd.DataFrame({
        "lib_cons_norm": (lib - 1)/6,
        "Disapp_economy_norm": (dapp - 1)/4,
        "pers_econ_worse_norm": (pecon - 1)/4,
        "Distrust_gov_norm": (distrust - 1)/4,
        "age_norm": (age - 18)/(80-18),
        "edu_norm": (edu - 1)/4,
        "white_nonhispanic": white_nh,
        "female": female,
        "state": state
    })
    print(f"2024 WNH derived from (V241540,V241541): mean={float(np.nanmean(white_nh.values)):.3f}, missing={wnh_missing*100:.1f}%")
    print(f"2024 weight source: {w_source}; coverage={w.notna().mean()*100:.1f}% | eff_N={eff_sample_size(w.values):.1f}")
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

# ------------------ Forecast helpers ------------------

def forecast_national(pipe: Pipeline, X: pd.DataFrame, w: pd.Series):
    m = X.notna().all(axis=1) & w.notna()
    Xv = X.loc[m]
    ww = np.clip(w.loc[m].astype(float).values, 0, None)
    if Xv.empty or ww.sum() <= 0:
        return dict(weighted_mean_prob=float("nan"), share_thr_05=float("nan"),
                    share_thr_05_ci=(float("nan"), float("nan")),
                    share_thr_mean=float("nan"), share_thr_mean_ci=(float("nan"), float("nan")), used=0, total=len(X))
    p = pipe.predict_proba(Xv)[:,1]
    mean_prob = float(np.sum(p * ww) / ww.sum())
    share05 = float(np.sum((p > 0.5).astype(int) * ww) / ww.sum())
    lo05, hi05 = weighted_bootstrap_ci_share(p, ww, 0.5, BOOTSTRAP_REPS, BOOTSTRAP_SEED)
    lomean, himean = weighted_bootstrap_ci_share(p, ww, mean_prob, BOOTSTRAP_REPS, BOOTSTRAP_SEED)
    share_mean_thr = float(np.sum((p > mean_prob).astype(int) * ww) / ww.sum())
    return dict(weighted_mean_prob=mean_prob, share_thr_05=share05,
                share_thr_05_ci=(lo05, hi05),
                share_thr_mean=share_mean_thr,
                share_thr_mean_ci=(lomean, himean),
                used=int(m.sum()), total=int(len(X)))

def state_level_ec(pipe: Pipeline, X: pd.DataFrame, w: pd.Series, state_series: pd.Series):
    m = X.notna().all(axis=1) & w.notna() & state_series.notna()
    Xv = X.loc[m]; sv = state_series.loc[m]
    ww = np.clip(w.loc[m].astype(float).values, 0, None)
    p = pipe.predict_proba(Xv)[:,1]
    dfp = pd.DataFrame({"proba": p, "w": ww, "state": pd.to_numeric(sv, errors="coerce")})
    dfp["postal"] = dfp["state"].map(FIPS_TO_POSTAL)
    dfp = dfp[dfp["postal"].isin(EV_2024.keys())]
    rows = []
    for st, sub in dfp.groupby("postal"):
        pv, wv = sub["proba"].values, sub["w"].values
        meanp = float(np.sum(pv * wv) / np.sum(wv)) if np.sum(wv) > 0 else float("nan")
        share05 = float(np.sum((pv > 0.5).astype(int) * wv) / np.sum(wv)) if np.sum(wv) > 0 else float("nan")
        neff = eff_sample_size(wv)
        rows.append({"state": st, "mean_prob": meanp, "share_thr_05": share05, "neff": neff, "ev": EV_2024[st]})
    sdf = pd.DataFrame(rows).sort_values("ev", ascending=False)
    sdf["winner_mean"] = np.where(sdf["mean_prob"] > 0.5, "R", "D")
    sdf["winner_thr05"] = np.where(sdf["share_thr_05"] > 0.5, "R", "D")
    ev_mean = int(sdf.loc[sdf["winner_mean"]=="R","ev"].sum())
    ev_thr05 = int(sdf.loc[sdf["winner_thr05"]=="R","ev"].sum())
    sdf["reliability"] = np.where(sdf["neff"] >= 100, "OK", "LOW_Neff")
    return sdf, ev_mean, ev_thr05

# ------------------ Main ------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load past years
    for p in [DATA_2012, DATA_2016, DATA_2020, DATA_2024]:
        if not _exists(p):
            print(f"Missing file: {p}")
            return

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
    selected = [k.replace("include_","") for k,v in knee.params.items() if k.startswith("include_") and v]
    C = knee.params.get("logreg_C", 1.0)
    print(f"\nKnee-point: trial #{knee.number} | AUC={knee.values[0]:.3f} | k={int(knee.values[1])} | C={C:.4g}")
    print("Selected features:", selected)

    # Refit knee on all past years
    m = df_all["target"].notna() & df_all["weight"].notna()
    for c in selected: m &= df_all[c].notna()
    dfit = df_all.loc[m]
    pipe_knee = build_pipe(selected, C)
    pipe_knee.fit(dfit[selected], dfit["target"].astype(int), clf__sample_weight=dfit["weight"].astype(float).values)

    # Also identify best 2-feature model (for fallback if WNH not usable in 2024)
    # Search Pareto trials for those with exactly 2 features and highest AUC
    two_feat_best = None
    for t in study.best_trials:
        feats = [k.replace("include_","") for k,v in t.params.items() if k.startswith("include_") and v]
        if len(feats) == 2:
            if (two_feat_best is None) or (t.values[0] > two_feat_best["auc"]):
                two_feat_best = {"features": feats, "C": t.params.get("logreg_C", 1.0), "auc": t.values[0], "trial": t.number}
    # If not found on Pareto, fall back to known strong pair
    if two_feat_best is None:
        two_feat_best = {"features": ["lib_cons_norm","Disapp_economy_norm"], "C": 0.0661543736556139, "auc": np.nan, "trial": None}
    print(f"\nBest 2-feature fallback: {two_feat_best['features']} | C={two_feat_best['C']:.4g} | AUC≈{(two_feat_best['auc'] if not np.isnan(two_feat_best['auc']) else 0.904):.3f}")

    # Load 2024 and decide which model to apply
    print("\nLoading 2024 and forecasting...")
    X24, w24, wnh_missing = load_2024(DATA_2024)

    use_knee = True
    if "white_nonhispanic" in selected and wnh_missing > WNH_MISSINGNESS_THRESHOLD:
        print(f"INFO: 2024 WNH missingness {wnh_missing*100:.1f}% exceeds {WNH_MISSINGNESS_THRESHOLD*100:.0f}% threshold; using 2-feature fallback for forecast.")
        use_knee = False

    if use_knee:
        pipe = pipe_knee
        sel = selected
        model_label = f"knee ({len(selected)} features)"
    else:
        sel = two_feat_best["features"]
        pipe = build_pipe(sel, two_feat_best["C"])
        # Fit on all past years (only the selected 2 features)
        m2 = df_all["target"].notna() & df_all["weight"].notna()
        for c in sel: m2 &= df_all[c].notna()
        dfit2 = df_all.loc[m2]
        pipe.fit(dfit2[sel], dfit2["target"].astype(int), clf__sample_weight=dfit2["weight"].astype(float).values)
        model_label = f"2-feature fallback"

    # Forecast national and EC
    nat = forecast_national(pipe, X24[sel], w24)
    sdf, ev_mean, ev_thr05 = state_level_ec(pipe, X24[sel], w24, X24["state"] if "state" in X24.columns else pd.Series(np.nan, index=X24.index))

    # Console summary
    pct = lambda x: f"{100*x:.1f}%" if x is not None and np.isfinite(x) else "NA"
    print(f"\nUsing model: {model_label} | Features: {sel}")
    print("\nNational popular-vote style forecast (Republican vs others):")
    print(f"- Weighted mean probability: {pct(nat['weighted_mean_prob'])}")
    print(f"- Threshold 0.5: {pct(nat['share_thr_05'])} (95% CI {pct(nat['share_thr_05_ci'][0])}–{pct(nat['share_thr_05_ci'][1])})")
    print(f"- Threshold = weighted mean prob: {pct(nat['share_thr_mean'])} (95% CI {pct(nat['share_thr_mean_ci'][0])}–{pct(nat['share_thr_mean_ci'][1])})")
    print(f"- Used {nat['used']} of {nat['total']} 2024 rows")

    print("\nElectoral College projection (winner-take-all; ME/NE not split):")
    print(f"- EV by mean-probability winners: R {ev_mean}, D {538 - ev_mean}")
    print(f"- EV by 0.5-threshold winners: R {ev_thr05}, D {538 - ev_thr05}")
    low_reli = sdf[sdf["reliability"]=="LOW_Neff"]["state"].tolist()
    if low_reli:
        print(f"- Low reliability states: {', '.join(low_reli)}")

    # Save outputs
    os.makedirs(OUT_DIR, exist_ok=True)
    pareto_png = os.path.join(OUT_DIR, "pareto_front.png")
    pareto_csv = os.path.join(OUT_DIR, "pareto_front.csv")
    state_csv = os.path.join(OUT_DIR, "forecast_2024_states.csv"); sdf.to_csv(state_csv, index=False)
    report_md = os.path.join(OUT_DIR, "forecast_2024_report.md")
    lines = []
    lines.append("# 2024 Forward Forecast (trained on 2012, 2016, 2020)")
    lines.append(f"Selected knee model: features = {', '.join(selected)}; C = {C:.4g}; AUC (5-fold) = {knee.values[0]:.3f}.")
    lines.append(f"Applied model at forecast: {model_label} | features = {', '.join(sel)}")
    lines.append("National popular-vote style estimate (with survey weights):")
    lines.append(f"- Weighted mean probability: {pct(nat['weighted_mean_prob'])}")
    lines.append(f"- Threshold 0.5: {pct(nat['share_thr_05'])} (95% CI {pct(nat['share_thr_05_ci'][0])}–{pct(nat['share_thr_05_ci'][1])})")
    lines.append(f"- Threshold = weighted mean prob: {pct(nat['share_thr_mean'])} (95% CI {pct(nat['share_thr_mean_ci'][0])}–{pct(nat['share_thr_mean_ci'][1])})")
    lines.append("Electoral College projection (winner-take-all; ME/NE not split):")
    lines.append(f"- EV by mean-probability winners: R {ev_mean}, D {538 - ev_mean}")
    lines.append(f"- EV by 0.5-threshold winners: R {ev_thr05}, D {538 - ev_thr05}")
    if low_reli:
        lines.append(f"- Low reliability states: {', '.join(low_reli)}")
    lines.append("")
    lines.append("Notes:")
    lines.append("- Optimization uses 2012/2016/2020 with 5-fold CV; 2024 is held out for forecasting.")
    lines.append("- 2024 base weight used: '{}' (provisional).".format(WEIGHT_VAR_OVERRIDE_2024 or "equal weights"))
    lines.append("- 2024 white_nonhispanic derived from V241540 and V241541; fell back to 2-feature model if missingness > {}%.".format(int(WNH_MISSINGNESS_THRESHOLD*100)))
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nSaved outputs to {OUT_DIR}")

if __name__ == "__main__":
    main()
