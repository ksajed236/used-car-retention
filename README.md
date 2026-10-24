# Predicting Used Car Value Retention Ratio

CMPSC 445 final project. Used-car listings scraped from **Craigslist** and **CarMax** are merged with
original MSRPs from **U.S. News & World Report**, and four regression models predict each car's

```
value retention ratio = current listing price / original base MSRP (same make, model, model year)
```

The project also ranks the factors behind depreciation (age, mileage, seller type, model, trim, ...).

| Item | Value |
|---|---|
| Scope | 3 makes x 3 models: Toyota (Camry, Corolla, RAV4), Honda (Civic, Accord, CR-V), Ford (F-150, Escape, Explorer) |
| Raw listings collected | Craigslist 1,312 · CarMax 1,560 (both include Nissan, see below) |
| In-scope before integration | Craigslist 990 · CarMax 1,170 |
| Final integrated dataset | 1,908 listings (Craigslist 744 · CarMax 1,164), 26 unified attributes + target |
| Best model | Gradient boosting: test R² 0.935, test MAE 0.055 (about $1,341 per car) |
| Data scraped | October 23, 2026, within 150 miles of Philadelphia (ZIP 19103) for Craigslist; CarMax ranked by nearest store |

## Repository layout

```
scrapers/                    browser collectors (JavaScript, run in the site's own page)
  craigslist_collector.js    source 1: search pages + every listing's detail page
  carmax_collector.js        source 2: CarMax's inventory search endpoint (JSON)
  usnews_msrp_collector.js   MSRP range for each make / model / model year
src/
  config.py                  paths, study scope, scrape date
  parse_raw.py               step 1: collector JSON -> one CSV per source (source formats kept)
  integrate.py               step 2: unified schema, cleaning, de-duplication, MSRP merge, target
  features.py                step 3: derived features, description keyword features, column roles
  train.py                   step 4: encoding experiment, 4 models, CV + held-out test
  analysis.py                step 5: feature rankings, depreciation rates, partial dependence
  visualize.py               step 6: figures
run_pipeline.py              runs steps 1-6 in order
data/raw/                    collector output (JSON) + first-parsed CSV for each source
data/processed/              integrated_listings.csv, model_ready.csv
results/                     every table the report quotes (CSV / JSON)
figures/                     fig01 ... fig11 (PNG)
```

## How to reproduce

```bash
pip install -r requirements.txt
python run_pipeline.py          # about 3-4 minutes on a laptop; rebuilds everything from data/raw/
```

Single steps can be run from `src/`, e.g. `python train.py "Gradient Boosting"`, `python train.py collect`,
`python analysis.py`, `python visualize.py`.

### Re-collecting the data

The collectors run inside a normal browser session on each site, so they use the same requests the site's
own pages make (Craigslist's server-rendered pages, CarMax's inventory JSON, U.S. News' model-year pages).

1. Open the site (`craigslist.org`, `carmax.com` or `cars.usnews.com`), press F12 and open the Console.
2. Paste the matching file from `scrapers/` and press Enter.
3. Run `await clRun()` / `await cmxRun()` / `await msrpRun()`, then `clDownload()` / `cmxDownload()` /
   `msrpDownload()` to save the JSON file.
4. Put the JSON files in `data/raw/` and run `python run_pipeline.py`.

Requests are spaced out on purpose. U.S. News started refusing requests after about 210 pages at a
400 ms delay, so use `delayMs >= 2000` there.

## Why Nissan is collected but not modeled

Nissan (Altima, Rogue, Sentra) was scraped from both listing sites, but U.S. News stopped serving pages
before the MSRPs for the Rogue (2015-2026) and the Sentra were read. Without an MSRP the target cannot be
computed, and keeping Nissan with only one complete model would break the "3 models per make" rule, so
`config.py` limits the study to Toyota, Honda and Ford. The Nissan rows stay in `data/raw/` and the
cleaning log (`results/cleaning_steps.csv`) records them. To add Nissan back, re-run the MSRP collector for
the missing years and add Nissan to `MAKES_MODELS` in `src/config.py`.

## Main outputs

| File | What it holds |
|---|---|
| `results/cleaning_steps.csv` | listings removed by each cleaning rule, per source |
| `results/eda_*.csv` | attribute statistics, missing values, make/model counts |
| `results/encoding_experiment.csv` | drop vs one-hot vs frequency vs target encoding of the high-cardinality columns |
| `results/model_comparison.csv` | CV and test MAE / RMSE / R² for every model, plus a mean-ratio baseline |
| `results/permutation_importance.csv` | feature ranking for every model (held-out test set) |
| `results/depreciation_rates.csv` | % of value lost per year and per 10,000 miles, by make and model |
| `results/partial_dependence*.csv` | learned depreciation curves for age and mileage |
| `figures/` | data overview, counts, retention vs age/mileage, correlations, model comparison, predicted vs actual, feature rankings, Ridge coefficients, depreciation curves and rates |

The final report (with the AI usage statement) is submitted separately.
