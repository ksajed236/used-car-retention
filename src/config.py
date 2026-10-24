"""Shared paths and constants for the used-car value retention pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"            # exactly what the browser collectors saved (+ first parse to CSV)
PROCESSED = DATA / "processed"  # integrated / model-ready tables
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"

for _p in (RAW, PROCESSED, FIGURES, RESULTS):
    _p.mkdir(parents=True, exist_ok=True)

# collector outputs (JSON) -> first parsed CSV per source
CRAIGSLIST_JSON = RAW / "craigslist_raw.json"
CARMAX_JSON = RAW / "carmax_raw.json"
MSRP_JSON = RAW / "usnews_msrp_raw.json"
CRAIGSLIST_CSV = RAW / "craigslist_listings_parsed.csv"
CARMAX_CSV = RAW / "carmax_listings_parsed.csv"
MSRP_CSV = RAW / "usnews_msrp_parsed.csv"

INTEGRATED_CSV = PROCESSED / "integrated_listings.csv"
MODEL_READY_CSV = PROCESSED / "model_ready.csv"

# Scope of the study: 3 makes x 3 models each.
# Nissan (Altima, Rogue, Sentra) was collected too, but U.S. News stopped serving
# pages before the Rogue 2015-2026 and Sentra MSRPs were read, so Nissan is left
# out of the modeling. To bring it back, re-run the MSRP collector for the missing
# years and add the entry below.
MAKES_MODELS = {
    "Toyota": ["Camry", "Corolla", "RAV4"],
    "Honda": ["Civic", "Accord", "CR-V"],
    "Ford": ["F-150", "Escape", "Explorer"],
}
COLLECTED_NOT_MODELED = {"Nissan": ["Altima", "Rogue", "Sentra"]}

# Data were scraped on 2026-10-23; vehicle age is measured against this date.
SCRAPE_YEAR = 2026
SCRAPE_DATE = "2026-10-23"
PHILLY_LAT, PHILLY_LON = 39.9526, -75.1652  # center of the search radius (ZIP 19103)

RANDOM_STATE = 42
