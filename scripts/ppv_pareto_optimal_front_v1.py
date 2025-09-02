import os
import sys
import math
import json
import random
import warnings
from typing import Dict, Tuple, List, Any

# Auto-install dependencies if missing (helpful in Colab)
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
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ------------- USER CONFIG -------------
# Update these paths to point to your .dta files in Colab (e.g., /content/your_file.dta)
DATA_PATHS = {
    2012: "/content/ANES_data_predicting_popular_vote_shares_2012.dta",
    2016: "/content/ANES_data_predicting_popular_vote_shares_2016.dta",
    2020: "/content/ANES_data_predicting_popular_vote_shares_2020.dta",
}

N_TRIALS_PER_YEAR = 200     # Adjust as desired (e.g., 150+ as requested)
RANDOM_SEED = 42
N_SPLITS = 5
# --------------------------------------

rng = np.random.RandomState(RANDOM_SEED)
random.seed(RANDOM_SEED)

def _exists(path: str) -> bool:
    try:
        return os.path.exists(path)
    except Exception:
        return False

def _roc_auc_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # Safe ROC AUC that returns 0.5 if undefined
    try:
        if len(np.unique(y_true)) < 2:
            return 0.5
        return roc_auc_score(y_true, y_score)
    except Exception:
        return 0.5

# --------- Preprocessing per year (replicates .do files) ---------

def load_and_process_2012(path: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_stata(path, convert_categoricals=False)

    # vote_Romney
    y = df["prevote_intpreswho"].copy()
    # 1=Obama, 2=Romney, 5=Other? Set per .do
    y = y.where(~y.isin([-9, -8, -1]), np.nan)
    y = y.map(lambda v: 1 if v == 2 else (0 if v in [1, 5] else np.nan))

    # Ideology
    lib_cons = df["libcpre_self"].copy()
    lib_cons = lib_cons.where(~lib_cons.isin([-9, -8, -2]), np.nan)

    # Disapproval of economy (presapp_econ)
    Disapp_economy = df["presapp_econ"].copy()
    Disapp_economy = Disapp_economy.where(~Disapp_economy.isin([-9, -8]), np.nan)

    # Economic situation
    pers_econ_worse = df["finance_finpast"].copy()
    pers_econ_worse = pers_econ_worse.where(~pers_econ_worse.isin([-9, -8]), np.nan)
    # (3=2) (2=3)
    pers_econ_worse = pers_econ_worse.replace({3: 2, 2: 3})

    # Distrust in government
    distrust_gov = df["trustgov_trustgrev"].copy()
    distrust_gov = distrust_gov.where(~distrust_gov.isin([-9, -8, -7, -6, -5, -4, -3, -2, -1]), np.nan)

    # Age
    age = df["dem_age_r_x"].copy()
    age = age.where(~age.isin([-2]), np.nan)

    # Gender
    female = df["gender_respondent_x"].copy()
    female = female.replace({2: 1, 1: 0})

    # Education
    edu = df["dem_edugroup_x"].copy()
    edu = edu.where(~edu.isin([-9, -8, -7, -6, -5, -4, -3, -2]), np.nan)

    # Ethnicity
    white_nonhispanic = df["dem_raceeth_x"].copy()
    white_nonhispanic = white_nonhispanic.where(~white_nonhispanic.isin([-9]), np.nan)
    white_nonhispanic = white_nonhispanic.apply(lambda v: 1 if v == 1 else (0 if v in [2, 3, 4, 5, 6] else np.nan))

    # State
    state = df["sample_stfips"].copy()

    # Normalizations
    lib_cons_norm = (lib_cons - 1.0) / 6.0
    Disapp_economy_norm = (Disapp_economy - 1.0)  # as in .do
    pers_econ_worse_norm = (pers_econ_worse - 1.0) / 2.0
    Distrust_gov_norm = (distrust_gov - 1.0) / 4.0
    age_norm = (age - 18.0) / (90.0 - 18.0)
    edu_norm = (edu - 1.0) / 4.0

    X = pd.DataFrame({
        "lib_cons_norm": lib_cons_norm,
        "Disapp_economy_norm": Disapp_economy_norm,
        "Distrust_gov_norm": Distrust_gov_norm,
        "age_norm": age_norm,
        "edu_norm": edu_norm,
        "white_nonhispanic": white_nonhispanic,
        "pers_econ_worse_norm": pers_econ_worse_norm,
        "female": female,
        "state": state,
    })
    return X, y.rename("target")

def load_and_process_2016(path: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_stata(path, convert_categoricals=False)

    # vote_Trump
    y = df["V161031"].copy()
    y = y.where(~y.isin([6, 7, 8, -9, -8, -1]), np.nan)
    y = y.map(lambda v: 1 if v == 2 else (0 if v in [1, 3, 4, 5] else np.nan))

    # Ideology
    lib_cons = df["V161126"].copy()
    lib_cons = lib_cons.where(~lib_cons.isin([-9, -8, 99]), np.nan)

    # Disapproval of economy last year
    Disapp_economy = df["V161083"].copy()
    Disapp_economy = Disapp_economy.where(~Disapp_economy.isin([-9, -8]), np.nan)

    # Economic situation
    pers_econ_worse = df["V161110"].copy()
    pers_econ_worse = pers_econ_worse.where(~pers_econ_worse.isin([-9, -8]), np.nan)

    # Distrust in government
    distrust_gov = df["V161215"].copy()
    distrust_gov = distrust_gov.where(~distrust_gov.isin([-9, -8]), np.nan)

    # Age
    age = df["V161267"].copy()
    age = age.where(~age.isin([-9, -8]), np.nan)

    # Gender
    female = df["V161342"].copy()
    female = female.replace({2: 1, 1: 0, 3: 0, -9: np.nan})

    # Education
    edu = df["V161270"].copy()
    edu = edu.where(~edu.isin([-9]) & ~edu.between(90, 95, inclusive="both"), np.nan)

    # Ethnicity
    white_nonhispanic = df["V161310x"].copy()
    white_nonhispanic = white_nonhispanic.where(~white_nonhispanic.isin([-2]), np.nan)
    white_nonhispanic = white_nonhispanic.apply(lambda v: 1 if v == 1 else (0 if v in [2, 3, 4, 5, 6] else np.nan))

    # State
    state = df["V161015b"].copy()
    state = state.where(~state.isin([-1]), np.nan)

    # Normalizations
    lib_cons_norm = (lib_cons - 1.0) / 6.0
    Disapp_economy_norm = (Disapp_economy - 1.0)  # as in .do
    pers_econ_worse_norm = (pers_econ_worse - 1.0) / 4.0
    Distrust_gov_norm = (distrust_gov - 1.0) / 4.0
    age_norm = (age - 18.0) / (90.0 - 18.0)
    edu_norm = (edu - 1.0) / 15.0

    X = pd.DataFrame({
        "lib_cons_norm": lib_cons_norm,
        "Disapp_economy_norm": Disapp_economy_norm,
        "Distrust_gov_norm": Distrust_gov_norm,
        "age_norm": age_norm,
        "edu_norm": edu_norm,
        "white_nonhispanic": white_nonhispanic,
        "pers_econ_worse_norm": pers_econ_worse_norm,
        "female": female,
        "state": state,
    })
    return X, y.rename("target")

def load_and_process_2020(path: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_stata(path, convert_categoricals=False)

    # vote_Trump
    y = df["V201033"].copy()
    y = y.where(~y.isin([-8, -9, -1, 11, 12]), np.nan)
    y = y.map(lambda v: 1 if v == 2 else (0 if v in [1, 3, 4, 5] else np.nan))

    # Ideology
    lib_cons = df["V201200"].copy()
    lib_cons = lib_cons.where(~lib_cons.isin([-9, -8, 99]), np.nan)

    # Disapproval of economy
    Disapp_economy = df["V201327x"].copy()
    Disapp_economy = Disapp_economy.where(~Disapp_economy.isin([-2]), np.nan)

    # Economic situation
    pers_econ_worse = df["V201502"].copy()
    pers_econ_worse = pers_econ_worse.where(~pers_econ_worse.isin([-9]), np.nan)

    # Distrust in gov
    Distrust_gov = df["V201233"].copy()
    Distrust_gov = Distrust_gov.where(~Distrust_gov.isin([-9, -8]), np.nan)

    # Age
    age = df["V201507x"].copy()
    age = age.where(~age.isin([-9]), np.nan)

    # Gender
    female = df["V201600"].copy()
    female = female.replace({2: 1, 1: 0, -9: np.nan})

    # Education
    edu = df["V201511x"].copy()
    edu = edu.where(~edu.isin([-9, -8, -7, -6, -5, -4, -3, -2]), np.nan)

    # Ethnicity
    white_nonhispanic = df["V201549x"].copy()
    white_nonhispanic = white_nonhispanic.where(~white_nonhispanic.isin([-9, -8]), np.nan)
    white_nonhispanic = white_nonhispanic.apply(lambda v: 1 if v == 1 else (0 if v in [2, 3, 4, 5, 6] else np.nan))

    # State
    state = df["V201014b"].copy()
    state = state.where(~state.isin([-9, -8, -7, -6, -5, -4, -3, -2, -1, 86]), np.nan)

    # Normalizations
    lib_cons_norm = (lib_cons - 1.0) / 6.0
    Disapp_economy_norm = (Disapp_economy - 1.0) / 4.0
    pers_econ_worse_norm = (pers_econ_worse - 1.0) / 4.0
    age_norm = (age - 18.0) / (80.0 - 18.0)
    edu_norm = (edu - 1.0) / 4.0
    Distrust_gov_norm = (Distrust_gov - 1.0) / 4.0

    X = pd.DataFrame({
        "lib_cons_norm": lib_cons_norm,
        "Disapp_economy_norm": Disapp_economy_norm,
        "Distrust_gov_norm": Distrust_gov_norm,
        "age_norm": age_norm,
        "edu_norm": edu_norm,
        "white_nonhispanic": white_nonhispanic,
        "pers_econ_worse_norm": pers_econ_worse_norm,
        "female": female,
        "state": state,
    })
    return X, y.rename("target")

def load_dataset(year: int, path: str) -> Tuple[pd.DataFrame, pd.Series]:
    if year == 2012:
        return load_and_process_2012(path)
    elif year == 2016:
        return load_and_process_2016(path)
    elif year == 2020:
        return load_and_process_2020(path)
    else:
        raise ValueError("Year must be one of {2012, 2016, 2020}")

# ------------- Optuna multi-objective setup -------------

def make_objective(X_full: pd.DataFrame, y: pd.Series, year: int):
    # Candidate predictor names (state counts as 1 feature choice, though one-hot expands columns)
    candidate_vars = [
        "lib_cons_norm",
        "Disapp_economy_norm",
        "Distrust_gov_norm",
        "age_norm",
        "edu_norm",
        "white_nonhispanic",
        "pers_econ_worse_norm",
        "female",
        "state",
    ]

    def objective(trial: optuna.trial.Trial):
        # Feature subset selection
        selected = []
        for v in candidate_vars:
            use_v = trial.suggest_categorical(f"include_{v}", [True, False])
            if use_v:
                selected.append(v)

        # Ensure at least one feature is selected; heavily penalize otherwise
        if len(selected) == 0:
            return 0.0, 99.0  # poor AUC, huge complexity => dominated

        # Regularization strength
        C = trial.suggest_float("logreg_C", 1e-2, 100.0, log=True)

        # Build modeling DataFrame subset with selected columns
        cols_needed = selected.copy()
        # Filter rows with no missing in selected features and target
        mask = y.notna()
        for c in cols_needed:
            mask &= X_full[c].notna()

        X = X_full.loc[mask, cols_needed].copy()
        y_sub = y.loc[mask].astype(int).copy()

        # If too few samples after filtering, penalize
        if len(y_sub) < 200 or y_sub.nunique() < 2:
            return 0.5, float(len(selected))

        # Split features into numeric passthrough and categorical (only 'state' is categorical here)
        num_features = [c for c in selected if c != "state"]
        cat_features = ["state"] if "state" in selected else []

        transformers = []
        if num_features:
            transformers.append(("num", "passthrough", num_features))
        if cat_features:
            transformers.append(("state", OneHotEncoder(handle_unknown="ignore", sparse=True), cat_features))

        ct = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            sparse_threshold=1.0,  # keep sparse when state is present
        )

        clf = LogisticRegression(
            solver="liblinear",  # works well with sparse and binary
            C=C,
            max_iter=1000,
            random_state=RANDOM_SEED,
        )

        pipe = Pipeline([
            ("ct", ct),
            ("clf", clf),
        ])

        # 5-fold CV predicted probabilities
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
        try:
            y_proba = cross_val_predict(
                pipe, X, y_sub, cv=cv, method="predict_proba", n_jobs=-1, verbose=0
            )[:, 1]
            auc = _roc_auc_safe(y_sub.values, y_proba)
        except Exception:
            # In case of any failure in fitting
            auc = 0.5

        # Complexity: number of variables chosen (state counts as 1 total, not dummy-expanded count)
        complexity = float(len(selected))
        return auc, complexity

    return objective, candidate_vars

def run_study_for_year(year: int, X: pd.DataFrame, y: pd.Series, n_trials: int) -> optuna.study.Study:
    sampler = NSGAIISampler(seed=RANDOM_SEED)
    study = optuna.create_study(
        directions=["maximize", "minimize"],  # AUC up, complexity down
        sampler=sampler,
        study_name=f"election_{year}_pareto",
    )
    obj, cand = make_objective(X, y, year)
    study.optimize(obj, n_trials=n_trials, show_progress_bar=True)
    return study

# ------------- Visualization and export -------------

def extract_pareto(study: optuna.study.Study) -> List[optuna.trial.FrozenTrial]:
    # For multi-objective studies, best_trials returns Pareto-optimal set
    return study.best_trials

def plot_pareto(study: optuna.study.Study, year: int, out_dir: str):
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    xs = [t.values[1] for t in trials]  # complexity
    ys = [t.values[0] for t in trials]  # AUC

    pareto = extract_pareto(study)
    px = [t.values[1] for t in pareto]
    py = [t.values[0] for t in pareto]

    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=xs, y=ys, alpha=0.4, s=30, label="All trials")
    sns.scatterplot(x=px, y=py, color="red", s=50, label="Pareto-optimal")
    plt.xlabel("Model complexity (number of predictors selected)")
    plt.ylabel("ROC AUC (5-fold CV)")
    plt.title(f"Pareto Front - {year}")
    plt.grid(True, alpha=0.2)
    plt.legend()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"pareto_front_{year}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"Saved {out_path}")

def save_pareto_csv(study: optuna.study.Study, year: int, out_dir: str):
    pareto = extract_pareto(study)
    rows = []
    for t in pareto:
        params = t.params.copy()
        selected = [k.replace("include_", "") for k, v in params.items() if k.startswith("include_") and v]
        rows.append({
            "trial_number": t.number,
            "roc_auc": t.values[0],
            "complexity": t.values[1],
            "logreg_C": params.get("logreg_C", None),
            "selected_features": json.dumps(selected),
        })
    df = pd.DataFrame(rows).sort_values(["complexity", "roc_auc"], ascending=[True, False])
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"pareto_front_{year}.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    return df

def plot_combined_pareto(studies: Dict[int, optuna.study.Study], out_dir: str):
    plt.figure(figsize=(8, 6))
    palette = {2012: "tab:blue", 2016: "tab:green", 2020: "tab:orange"}
    for year, study in studies.items():
        pareto = extract_pareto(study)
        x = [t.values[1] for t in pareto]
        y = [t.values[0] for t in pareto]
        plt.scatter(x, y, s=55, label=f"{year} Pareto", alpha=0.9, c=palette.get(year, None))
    plt.xlabel("Model complexity (number of predictors selected)")
    plt.ylabel("ROC AUC (5-fold CV)")
    plt.title("Pareto Fronts by Year (Overlay)")
    plt.grid(True, alpha=0.2)
    plt.legend()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "pareto_front_combined.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"Saved {out_path}")

def recommend_knee_point(study: optuna.study.Study) -> Dict[str, Any]:
    # Heuristic: choose Pareto trial closest to the "ideal" (max AUC, min complexity)
    pareto = extract_pareto(study)
    if not pareto:
        return {}
    max_auc = max(t.values[0] for t in pareto)
    min_cplx = min(t.values[1] for t in pareto)
    # Normalize distances to [0,1] ranges
    aucs = np.array([t.values[0] for t in pareto])
    cplx = np.array([t.values[1] for t in pareto])
    auc_range = max(1e-9, aucs.max() - aucs.min())
    cplx_range = max(1e-9, cplx.max() - cplx.min())

    # distance to ideal (max_auc, min_cplx)
    dists = np.sqrt(((max_auc - aucs) / auc_range) ** 2 + ((cplx - min_cplx) / cplx_range) ** 2)
    idx = int(np.argmin(dists))
    t = pareto[idx]
    selected = [k.replace("include_", "") for k, v in t.params.items() if k.startswith("include_") and v]
    return {
        "trial_number": t.number,
        "roc_auc": t.values[0],
        "complexity": t.values[1],
        "logreg_C": t.params.get("logreg_C", None),
        "selected_features": selected,
    }

# ----------------- Main runner -----------------

def main():
    print("Starting ANES multi-objective optimization with Optuna (AUC vs. complexity)")
    print("If the .dta paths below are incorrect, edit DATA_PATHS at the top of this script.")
    available_years = []
    for y, p in DATA_PATHS.items():
        if _exists(p):
            available_years.append(y)
        else:
            print(f"Warning: File not found for {y}: {p}. Skipping this year.")
    if not available_years:
        print("No data files found. Please upload your .dta files to Colab and update DATA_PATHS.")
        return

    datasets = {}
    for year in sorted(available_years):
        print(f"\nLoading and preprocessing {year} from {DATA_PATHS[year]} ...")
        X, y = load_dataset(year, DATA_PATHS[year])
        datasets[year] = (X, y)
        print(f"{year}: {len(y.dropna())} labeled rows before feature filtering.")

    studies = {}
    for year in sorted(datasets.keys()):
        X, y = datasets[year]
        print(f"\nRunning Optuna multi-objective study for {year} with {N_TRIALS_PER_YEAR} trials...")
        study = run_study_for_year(year, X, y, N_TRIALS_PER_YEAR)
        studies[year] = study

        print(f"Completed study for {year}.")
        plot_pareto(study, year, out_dir="/content/optuna_outputs")
        pareto_df = save_pareto_csv(study, year, out_dir="/content/optuna_outputs")

        print("Top Pareto candidates (by lowest complexity then highest AUC):")
        display_cols = ["trial_number", "complexity", "roc_auc", "logreg_C", "selected_features"]
        print(pareto_df[display_cols].head(10).to_string(index=False))

        rec = recommend_knee_point(study)
        if rec:
            print("\nRecommended knee-point solution:")
            print(json.dumps(rec, indent=2))
        else:
            print("No Pareto solutions found for recommendation.")

    if len(studies) >= 2:
        print("\nPlotting combined Pareto fronts across years...")
        plot_combined_pareto(studies, out_dir="/content/optuna_outputs")

    print("\nDone. Pareto plots and CSVs saved to /content/optuna_outputs")

if __name__ == "__main__":
    main()
