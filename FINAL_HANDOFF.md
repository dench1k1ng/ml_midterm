# Final evaluation handoff

Selected model: Gradient Boosting (log1p target).
Selection: lowest mean training 5-fold CV RMSE, fixed KFold seed 42, no extra tuning.

## Results
- CV MAE: $16,559 (fold SD $922).
- CV RMSE: $30,017 (fold SD $8,745).
- CV R²: 0.851.
- CV RMSE reduction vs the median baseline: 63.3%.
- Final test: 146 houses, all rows preserved.
- Test MAE: $14,622.
- Test RMSE: $20,118.
- Test R²: 0.929.
- Mean residual (actual minus predicted): $-573.
- 90th percentile absolute error: $31,662.

## Interpretation and limitations
The test metrics describe a single supplied holdout. CV and test contain different houses,
so their difference is not a paired comparison. Fold SD is not a confidence interval.
The small CV gap between boosting variants does not prove statistical superiority.
R² is not accuracy. Errors can be much larger for individual houses than the mean error.
Random folds do not establish performance in a different city or future market.
No test result was used to choose or retune the model.

## Presentation sources
Use EDA_HANDOFF.md and eda_artifacts for data and EDA, model_comparison.csv for CV,
test_metrics.csv for final results, and test_predictions.csv for editable diagnostic charts.
The PPTX is delivered separately and is not a repository artifact.
