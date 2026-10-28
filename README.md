# Predictive Analysis of California Census-District House Values (CA2 Project)

End-to-end predictive-analysis project (regression) for the LPU **Predictive Analysis CA2**: data audit, EDA, six measurable research objectives, seven tuned models, error analysis and explainability. **Best model: XGBoost** - test RMSE **$40,980**, MAE $26,522, R^2 0.871.

## 1. Project structure
```
CA2_Project/
|-- data/raw/housing.csv            original dataset (unchanged) + SHA256.txt
|-- data/processed/                 train.csv / test.csv (80/20, income-stratified, seed 42)
|-- data/DATA_DOCUMENTATION.md      provenance, dictionary, known issues
|-- notebooks/California_Housing_Predictive_Analysis.ipynb   executed notebook (all objectives)
|-- src/                            config.py, features.py (imputer + feature engineering + pipeline), evaluate.py, predict.py
|-- figures/                        20 publication-quality figures (PNG, 200 dpi)
|-- results/                        metrics.json (every number in the report), model_comparison.csv, cv_fold_rmse.csv, test_predictions.csv, data_dictionary.csv
|-- models/                         best_model.joblib (full pipeline) + model_card.json
|-- report/                         CA2_Predictive_Analysis_Report.docx (editable) + PDF preview
|-- docs/                           SCREENSHOT_CHECKLIST.md (the viva guide is kept local, see .gitignore)
|-- tools/                          scripts that build the notebook, run it, and generate the report/docs
|-- web/                            static web app (interactive map + price estimator) ready for Vercel
`-- requirements.txt
```

## 2. Setup
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name python3
```
Exact versions used for the reported results are in `requirements-exact.txt` (`pip install -r requirements-exact.txt`). Tested with Python 3.12.3, pandas 3.0.2, NumPy 2.4.4, scikit-learn 1.8.0, XGBoost 3.4.1.

## 3. Reproduce the results
```bash
python tools/build_notebook.py                     # (optional) regenerate the notebook from its builder
jupyter nbconvert --to notebook --execute --inplace notebooks/California_Housing_Predictive_Analysis.ipynb   # ~15-30 min on 1 CPU
python tools/build_docs.py                         # README / data docs from results/metrics.json
python tools/build_report.py                       # DOCX report (needs LibreOffice for page numbers in the contents)
```
Set `QUICK=1` for a fast smoke test (sub-sampled; **never** use its numbers). All randomness uses seed 42.

Use the saved model (run from the project root so that `src` is importable):
```bash
python -m src.predict data/processed/test.csv
```

## 4. Methodology (summary)
1. **Audit** the raw file (missing values, duplicates, consistency, censoring, outliers) - nothing is deleted silently.
2. **Split first** (80/20, stratified on income bands), then explore the training set only.
3. **Leakage-safe pipelines**: imputer -> feature engineering -> log1p/scale/one-hot, fitted on training folds only; test set used once.
4. **Six objectives** (data quality & imputation; income relationship; geography & spatial features; ratio features; model comparison; error analysis & explainability).
5. **Models**: Dummy mean, Ridge, Ridge (log-target), k-NN, Random Forest, HistGradientBoosting, XGBoost; tuned with 3-fold CV (grid/randomised search), selected by 5-fold CV RMSE, compared with the Nadeau-Bengio corrected t-test and a paired bootstrap on the test set.

## 5. Key results
| Model | CV RMSE | Test RMSE | Test MAE | Test R^2 |
|---|---|---|---|---|
| XGBoost | $42,926 | $40,980 | $26,522 | 0.871 |
| HistGB | $44,996 | $43,024 | $28,095 | 0.858 |
| Random Forest | $45,488 | $43,613 | $28,205 | 0.854 |
| k-NN | $56,184 | $53,768 | $36,199 | 0.778 |
| Ridge | $63,389 | $62,098 | $45,173 | 0.704 |
| Ridge (log-target) | $74,289 | $71,014 | $43,775 | 0.613 |
| Dummy (mean) | $115,699 | $114,165 | $90,807 | -0.000 |

* XGBoost vs runner-up HistGB: test RMSE difference -2,044 USD (95 % bootstrap CI -2,687 to -1,411).
* Capped (500,001) districts are 4.3 % of the test set but cause 16 % of the squared error.
* Strong spatial autocorrelation of prices (Moran's I = 0.79) means random CV is optimistic for *new regions* - see report Section 5.3.

## 6. Upload to GitHub (requires your own account)
```bash
cd CA2_Project
git init
git add .
git commit -m "CA2 Predictive Analysis: California housing price prediction"
git branch -M main
# create an EMPTY repository on github.com first (no README), then:
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```
After pushing, check that the notebook renders on GitHub and that figures appear. Then follow `docs/SCREENSHOT_CHECKLIST.md`.

## 7. Deploy the interactive web app on Vercel
The `web/` folder is a static site: the trained model runs in the browser, so no server is needed. Push the repository to GitHub, then on vercel.com choose *Add New, Project*, import the repository, set **Root Directory** to `web`, Framework Preset **Other**, and deploy. Full steps (including the Vercel CLI and local testing) are in `web/README.md`. Regenerate the web model files after retraining with `python tools/export_web.py`.

## 8. Notes
* The dataset reflects 1990 conditions; do not use the model to value current property.
* The report's cover, declaration, certificate and acknowledgement are filled in with typed names only; print and sign by hand wherever LPU requires an actual signature.
* Review the whole project so that you can explain every step; follow your university's academic-integrity and AI-use policy.
