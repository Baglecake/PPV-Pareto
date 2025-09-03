# PPV-Pareto
Using optuna to generate optimal pareto fronts, this project aims to build on the analysis performed by Camatarri (2024) to maximize predictive accuracy and minimize model complexity.

PPV-Pareto Election Forecasting Pipeline
Key Concepts
Electoral College Filtering (EC_MIN_NEFF)
The pipeline uses an effective sample size filter (EC_MIN_NEFF = 100) to ensure reliable state-level estimates:

Included states: Only states with effective N ≥ 100 contribute to Electoral College tallies
Excluded states: States below threshold are marked as "LOW_Neff" and excluded from EV totals
Full results: Complete 538-EV results saved in forecast_2024_states_shrunk_full538.csv for QA
Model Hierarchy (Row-wise Fallback)
Models are applied in priority order by AUC performance:

Knee model: Optimal complexity-accuracy balance (typically 5 features)
2-feature (any): Best 2-feature model including ideology
2-feature (no ideology): Best 2-feature model without ideology
1-feature (no ideology): Simplest backstop model
Shrinkage and Calibration
Empirical Bayes shrinkage: State estimates pulled toward national mean (k_prior=400)
Cross-fitted isotonic calibration: Out-of-fold probability calibration by year
Grouped CV: Prevents data leakage across election cycles (2012/2016/2020)
Output Files
pareto_front.png/csv: Accuracy vs complexity trade-off
forecast_2024_states_*.csv: State-level EC projections
states_neff_*.csv: Detailed state diagnostics with reliability metrics
knee_selection.json: Selected model parameters and performance
calibration_*.png: Model calibration curves
RUN_METADATA.json: Complete run configuration
Reliability Indicators
neff: Effective sample size per state
reliability: "OK" (neff≥100) or "LOW_Neff" (neff<100)
eligible: Whether state contributes to EV tallies
thin_reason: Explanation for exclusion (e.g., "LOW_Neff")
