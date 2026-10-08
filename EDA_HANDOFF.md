# House Price Project — EDA handoff

This handoff covers the first team member's work. The full executable analysis and plots are in `main.ipynb`; generated tables and figures are in `eda_artifacts/`.

## Verified dataset facts

- `train.csv`: 1,314 rows and 81 columns.
- `test.csv`: 146 rows and 81 columns.
- Both files contain `Id` and the target `SalePrice`.
- There are no duplicate rows, duplicate IDs, or shared IDs between train and test.
- There are 79 candidate predictors after excluding `Id` and `SalePrice`.
- Raw predictor types: 36 numeric and 43 categorical. `MSSubClass` is numerically stored but semantically categorical.
- Training `SalePrice`: mean about $181,045, median $162,950, range $34,900–$755,000.
- Target skewness is about 1.88, so the distribution has a long right tail.

## Main EDA findings

- The strongest absolute numeric correlations with `SalePrice` are `OverallQual` (0.793), `GrLivArea` (0.697), `GarageCars` (0.642), `GarageArea` (0.633), `TotalBsmtSF` (0.614), and `1stFlrSF` (0.605).
- Price generally increases with overall quality and above-ground living area.
- Neighborhood has a clear association with median price and should be retained as a categorical feature.
- Correlation describes association, not causation, and does not evaluate categorical features by itself.

## Missing-value decisions

Use an explicit `"None"` category when a missing value means that a structure is absent:

- `Alley`, `MasVnrType`;
- `BsmtQual`, `BsmtCond`, `BsmtExposure`, `BsmtFinType1`, `BsmtFinType2`;
- `FireplaceQu`;
- `GarageType`, `GarageFinish`, `GarageQual`, `GarageCond`;
- `PoolQC`, `Fence`, `MiscFeature`.

Use zero for `MasVnrArea` when there is no masonry veneer. Treat `LotFrontage` and `GarageYrBlt` as numeric missing values and impute them inside the pipeline, preferably with a missing indicator. Impute the single missing training value in `Electrical` with the most frequent training category.

## Contract for preprocessing and modeling

1. Keep all 146 test rows.
2. Preserve `Id` only for traceability; do not use it as a model feature.
3. Remove `SalePrice` from both feature matrices.
4. Do not inspect or use test `SalePrice` until the model and all hyperparameters are finalized.
5. Cast `MSSubClass` to a categorical type.
6. Fit imputers and encoders only inside a `Pipeline` and within each training/cross-validation fold.
7. One-hot encode nominal categories with unknown-category handling enabled.
8. Compare all models using the same folds and at least MAE, RMSE, and R².
9. A `log1p(SalePrice)` experiment is reasonable, but final errors must also be reported in dollars after inverse transformation.

No cleaned CSV was created intentionally. A globally preprocessed CSV would make leakage easier and would disconnect preprocessing from cross-validation.
