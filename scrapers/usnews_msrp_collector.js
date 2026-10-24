/*
 * MSRP reference data: U.S. News & World Report Cars model-year pages
 * ------------------------------------------------------------
 * How to run: open https://cars.usnews.com in a browser, open DevTools
 * (F12) -> Console, paste this whole file and press Enter, then run
 *     await msrpRun();      // about 3 min
 *     msrpDownload();       // saves usnews_msrp_raw.json
 *
 * Every model-year review page (cars.usnews.com/cars-trucks/<make>/<model>/<year>)
 * shows the original MSRP range for that model year, from the cheapest trim
 * to the most expensive one, e.g. "$22,970 - 31,370 Original MSRP".
 * We read that range for each make/model/year. Years that do not have a page
 * return 404 and are skipped. The newest model year redirects to the model's
 * main page, so we only keep it when the page heading names the requested year.
 *
 * Note: with delayMs = 400 the site started refusing requests after about
 * 210 pages (the last Nissan pages failed and were recorded as missing).
 * Use delayMs >= 2000 when re-running; msrpRun() skips years already collected
 * (clear window.__msrp.missing first to retry the failed ones).
 */
const MSRP_CONFIG = {
  firstYear: 2000,
  lastYear: 2027,
  delayMs: 400,
  models: [
    ['Toyota', 'Camry', 'toyota/camry'], ['Toyota', 'Corolla', 'toyota/corolla'], ['Toyota', 'RAV4', 'toyota/rav4'],
    ['Honda', 'Civic', 'honda/civic'], ['Honda', 'Accord', 'honda/accord'], ['Honda', 'CR-V', 'honda/cr-v'],
    ['Ford', 'F-150', 'ford/f-150'], ['Ford', 'Escape', 'ford/escape'], ['Ford', 'Explorer', 'ford/explorer'],
    ['Nissan', 'Altima', 'nissan/altima'], ['Nissan', 'Rogue', 'nissan/rogue'], ['Nissan', 'Sentra', 'nissan/sentra'],
  ],
};

window.__msrp = window.__msrp || { records: [], missing: [], done: false };
const msrpSleep = (ms) => new Promise((r) => setTimeout(r, ms));
const MSRP_RE = /\$([\d,]+)\s*-\s*\$?([\d,]+)\s*(Original\s+)?MSRP/;

function msrpParse(html, make, model, year, url, finalUrl) {
  const d = new DOMParser().parseFromString(html, 'text/html');
  const heading = (d.querySelector('h1')?.textContent || d.title || '').replace(/\s+/g, ' ').trim();
  const text = d.body.textContent.replace(/\s+/g, ' ');
  const m = text.match(MSRP_RE);
  if (!m) return null;
  const headingYear = (heading.match(/\b(19|20)\d{2}\b/) || [])[0];
  const redirected = finalUrl && !finalUrl.replace(/\/$/, '').endsWith('/' + year);
  // a redirected page must name the requested year in its heading; otherwise it is a different model year
  if (redirected ? Number(headingYear) !== year : headingYear && Number(headingYear) !== year) return null;
  return {
    make, model, year,
    msrp_low: m[1], msrp_high: m[2],
    label: m[3] ? 'Original MSRP' : 'MSRP',
    page_heading: heading,
    url: finalUrl || url,
    scraped_at: new Date().toISOString(),
  };
}

async function msrpRun() {
  const S = window.__msrp;
  const C = MSRP_CONFIG;
  const done = new Set(S.records.map((r) => `${r.make}|${r.model}|${r.year}`).concat(S.missing));
  for (const [make, model, slug] of C.models) {
    for (let year = C.firstYear; year <= C.lastYear; year++) {
      const key = `${make}|${model}|${year}`;
      if (done.has(key)) continue;
      const url = `https://cars.usnews.com/cars-trucks/${slug}/${year}`;
      try {
        const r = await fetch(url);
        const rec = r.ok ? msrpParse(await r.text(), make, model, year, url, r.url) : null;
        if (rec) S.records.push(rec); else S.missing.push(key);
      } catch (e) {
        S.missing.push(key);
      }
      await msrpSleep(C.delayMs);
    }
  }
  S.done = true;
  return { found: S.records.length, missing: S.missing.length };
}

function msrpDownload(name = 'usnews_msrp_raw.json') {
  const blob = new Blob([JSON.stringify(window.__msrp.records)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
}
