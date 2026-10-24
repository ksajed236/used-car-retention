"""Step 1 - first parse of each source.

Turns the JSON files written by the three browser collectors (scrapers/*.js)
into one flat CSV per source. Nothing is renamed or cleaned here: each CSV
keeps the source's own column names and string formats ("$13,425",
"116,700", "4 cylinders"), so the raw files show exactly what each site
returned. Cleaning and integration happen in integrate.py.
"""
import json

import pandas as pd

from config import (CARMAX_CSV, CARMAX_JSON, CRAIGSLIST_CSV, CRAIGSLIST_JSON,
                    MSRP_CSV, MSRP_JSON)

CRAIGSLIST_COLUMNS = [
    "source", "post_id", "url", "search_make", "search_model", "search_query",
    "title", "price", "year", "make_model",
    "attr_odometer", "attr_condition", "attr_cylinders", "attr_drive", "attr_fuel",
    "attr_paint_color", "attr_title_status", "attr_transmission", "attr_type",
    "attr_vin",
    "seller_type", "area", "search_location", "latitude", "longitude",
    "posted_at", "updated_at", "n_images", "description", "scraped_at",
]


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_craigslist():
    df = pd.DataFrame(load_json(CRAIGSLIST_JSON))
    for col in CRAIGSLIST_COLUMNS:  # an attribute no seller filled in is absent from every page
        if col not in df.columns:
            df[col] = pd.NA
    extra = [c for c in df.columns if c not in CRAIGSLIST_COLUMNS and df[c].notna().any()]
    df = df[CRAIGSLIST_COLUMNS + extra]
    df.to_csv(CRAIGSLIST_CSV, index=False)
    return df


def parse_carmax():
    df = pd.DataFrame(load_json(CARMAX_JSON))
    df.to_csv(CARMAX_CSV, index=False)
    return df


def parse_msrp():
    df = pd.DataFrame(load_json(MSRP_JSON))
    df.to_csv(MSRP_CSV, index=False)
    return df


def main():
    cl, cm, ms = parse_craigslist(), parse_carmax(), parse_msrp()
    print(f"craigslist: {len(cl):>5} rows x {cl.shape[1]} cols -> {CRAIGSLIST_CSV.name}")
    print(f"carmax:     {len(cm):>5} rows x {cm.shape[1]} cols -> {CARMAX_CSV.name}")
    print(f"msrp:       {len(ms):>5} rows x {ms.shape[1]} cols -> {MSRP_CSV.name}")


if __name__ == "__main__":
    main()
