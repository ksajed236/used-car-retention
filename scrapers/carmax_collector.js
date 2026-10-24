/*
 * Source 2: CarMax used-car inventory
 * ------------------------------------------------------------
 * How to run: open https://www.carmax.com in a browser, open DevTools
 * (F12) -> Console, paste this whole file and press Enter, then run
 *     await cmxRun();       // about 1-2 min
 *     cmxDownload();        // saves carmax_raw.json
 *
 * CarMax's search page is filled by a JSON endpoint
 * (/cars/api/search/run?uri=/cars/<make>/<model>...). We page through that
 * endpoint (100 cars per request) for each make/model, ranked by CarMax's
 * "best match" for ZIP 19103 (nearest stores first), and keep the listing
 * fields we need. List-type fields (features, highlights, packages) are
 * joined into strings so every record is flat.
 */
const CMX_CONFIG = {
  zip: '19103',
  perModel: 130,
  pageSize: 100,
  delayMs: 800,
  models: [
    ['Toyota', 'Camry', 'toyota/camry'],
    ['Toyota', 'Corolla', 'toyota/corolla'],
    ['Toyota', 'RAV4', 'toyota/rav4'],
    ['Honda', 'Civic', 'honda/civic'],
    ['Honda', 'Accord', 'honda/accord'],
    ['Honda', 'CR-V', 'honda/cr-v'],
    ['Ford', 'F-150', 'ford/f150'],
    ['Ford', 'Escape', 'ford/escape'],
    ['Ford', 'Explorer', 'ford/explorer'],
    ['Nissan', 'Altima', 'nissan/altima'],
    ['Nissan', 'Rogue', 'nissan/rogue'],
    ['Nissan', 'Sentra', 'nissan/sentra'],
  ],
};

const CMX_KEEP = [
  'stockNumber', 'vin', 'year', 'make', 'model', 'trim', 'body', 'basePrice', 'originalPrice',
  'hasPriceDrop', 'mileage', 'exteriorColor', 'interiorColor', 'transmission', 'driveTrain',
  'engineSize', 'engineType', 'cylinders', 'horsepower', 'engineTorque', 'mpgCity', 'mpgHighway',
  'fuelType', 'vehicleSize', 'storeName', 'storeCity', 'stateAbbreviation', 'distance',
  'averageRating', 'numberOfReviews', 'numberOfFavorites', 'isNewArrival', 'lastMadeSaleableDate',
  'highlightedFeatures', 'featureScore',
];
const CMX_LISTS = ['features', 'highlights', 'packages', 'priorUseDescriptions', 'vehicleTypes'];

window.__cmx = window.__cmx || { records: [], errors: [], counts: {}, done: false };
const cmxSleep = (ms) => new Promise((r) => setTimeout(r, ms));

// list entries are strings, except priorUseDescriptions which are {name, description} objects
const cmxStr = (v) => (v && typeof v === 'object' ? (v.name ?? JSON.stringify(v)) : String(v));

function cmxFlatten(item, make, model) {
  const rec = { source: 'carmax', search_make: make, search_model: model };
  CMX_KEEP.forEach((k) => { rec[k] = item[k] ?? null; });
  CMX_LISTS.forEach((k) => { rec[k] = Array.isArray(item[k]) ? item[k].map(cmxStr).join('; ') : null; });
  rec.priorUseText = Array.isArray(item.priorUseDescriptions)
    ? item.priorUseDescriptions.map((o) => o && o.description).filter(Boolean).join(' ')
    : null;
  rec.url = `https://www.carmax.com/car/${item.stockNumber}`;
  rec.scraped_at = new Date().toISOString();
  return rec;
}

async function cmxRun() {
  const S = window.__cmx;
  const C = CMX_CONFIG;
  const seen = new Set(S.records.map((r) => r.stockNumber));
  for (const [make, model, slug] of C.models) {
    if (S.counts[slug]) continue;
    let skip = 0;
    let got = 0;
    let total = null;
    while (got < C.perModel && (total === null || skip < total)) {
      const take = Math.min(C.pageSize, C.perModel - got);
      const url = `/cars/api/search/run?uri=/cars/${slug}&skip=${skip}&take=${take}` +
        `&zipCode=${C.zip}&radius=radius-nationwide&shipping=-1&sort=best-match`;
      try {
        const r = await fetch(url, { headers: { accept: 'application/json' } });
        const j = await r.json();
        total = j.totalCount;
        for (const it of j.items || []) {
          if (seen.has(it.stockNumber)) continue;
          seen.add(it.stockNumber);
          S.records.push(cmxFlatten(it, make, model));
          got++;
        }
        if (!j.items || !j.items.length) break;
        skip += take;
      } catch (e) {
        S.errors.push({ url, error: String(e) });
        break;
      }
      await cmxSleep(C.delayMs);
    }
    S.counts[slug] = { collected: got, available: total };
  }
  S.done = true;
  return S.counts;
}

function cmxDownload(name = 'carmax_raw.json') {
  const blob = new Blob([JSON.stringify(window.__cmx.records)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
}
