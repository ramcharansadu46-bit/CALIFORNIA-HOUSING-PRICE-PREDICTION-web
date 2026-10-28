# California house-price predictor (static web app)

An interactive map and price estimator for the CA2 Predictive Analysis project. It is a **fully static site**: the trained XGBoost model (800 trees) runs in the visitor's browser in JavaScript, so there is no server, no Python runtime and nothing to pay for beyond Vercel's free tier.

## Files
| File | Purpose |
|---|---|
| `index.html`, `styles.css`, `app.js` | The page, styling and interface logic |
| `predictor.js` | JavaScript port of the pipeline (feature engineering, scaling, one-hot, tree walk) |
| `model.bin`, `model.json` | Exported XGBoost trees (3.6 MB) and preprocessing constants |
| `districts.json` | The 4,128 held-out test districts shown on the map, with actual and predicted values |
| `results.json` | Model comparison table and the six findings shown on the page |
| `assets/` | Figures from the notebook |
| `vercel.json` | Caching and security headers |

`predictor.js` was checked against the Python model on 4,079 test districts: largest difference $0.87, mean $0.11.

## Deploy to Vercel (from the GitHub repository)
1. Push the whole project to GitHub (see the main README).
2. On vercel.com choose **Add New, Project**, then import the repository.
3. Set **Root Directory** to `web`. Framework Preset: **Other**. Leave Build Command and Output Directory empty.
4. Click **Deploy**. You get a URL like `https://<project>.vercel.app`.

## Deploy with the Vercel CLI (no GitHub needed)
```bash
npm i -g vercel
cd web
vercel          # first run: log in and answer the prompts (framework: Other)
vercel --prod   # publish to the production URL
```

## Run locally
```bash
cd web
python -m http.server 8000     # then open http://localhost:8000
```
(Opening `index.html` directly by double-click will not work, because browsers block loading the model files from `file://`.)

## Update the model files
If you retrain the model, regenerate everything with `python tools/export_web.py` from the project root.

## Optional: show your GitHub link
Set `REPO_URL` at the top of `app.js` to your repository URL and a "Source code" link appears in the footer.

## Notes
* Prices are 1990 dollars; this is a learning project and must not be used to value property.
* Fonts load from Google Fonts; if blocked, the page falls back to system fonts.
* The data are the 20 % of census districts held back from training. Check the dataset licence on Kaggle before publishing a public site.
