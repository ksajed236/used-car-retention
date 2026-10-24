/*
 * Source 1: Craigslist "cars & trucks" listings
 * ------------------------------------------------------------
 * How to run: open https://www.craigslist.org in a browser, open DevTools
 * (F12) -> Console, paste this whole file and press Enter, then run
 *     await clRun();        // collects everything (about 10-15 min)
 *     clDownload();         // saves craigslist_raw.json
 *
 * Stage 1 - search pages: for every make/model we request the server-rendered
 *           search page (cat=cta, within RADIUS miles of POSTAL, duplicate
 *           posts bundled) and read each result's link, title, price, place.
 * Stage 2 - detail pages: every result page is downloaded and parsed with
 *           DOMParser: year + make/model header, the attribute table
 *           (odometer, condition, fuel, title status, ...), the posting body
 *           (free-text description), map coordinates and post dates.
 *
 * Requests are spaced out (CONCURRENCY workers, DELAY_MS between calls)
 * to stay polite, and progress is checkpointed to localStorage.
 */
const CL_CONFIG = {
  postal: '19103',          // Philadelphia, PA
  radius: 150,              // miles
  maxPerModel: 110,         // cap on detail pages per make/model
  concurrency: 2,
  delayMs: 450,
  models: [
    ['Toyota', 'Camry', 'toyota camry'],
    ['Toyota', 'Corolla', 'toyota corolla'],
    ['Toyota', 'RAV4', 'toyota rav4'],
    ['Honda', 'Civic', 'honda civic'],
    ['Honda', 'Accord', 'honda accord'],
    ['Honda', 'CR-V', 'honda cr-v'],
    ['Ford', 'F-150', 'ford f-150'],
    ['Ford', 'Escape', 'ford escape'],
    ['Ford', 'Explorer', 'ford explorer'],
    ['Nissan', 'Altima', 'nissan altima'],
    ['Nissan', 'Rogue', 'nissan rogue'],
    ['Nissan', 'Sentra', 'nissan sentra'],
  ],
};

window.__cl = window.__cl || { search: [], records: [], errors: [], done: false, stage: 'idle' };

const clSleep = (ms) => new Promise((r) => setTimeout(r, ms));
const clText = (el) => (el ? el.textContent.replace(/\s+/g, ' ').trim() : null);

async function clFetch(url, tries = 3) {
  for (let i = 0; i < tries; i++) {
    try {
      const r = await fetch(url, { credentials: 'include' });
      const html = await r.text();
      if (/This IP has been automatically blocked/i.test(html)) throw new Error('BLOCKED');
      if (r.ok) return html;
    } catch (e) {
      if (String(e).includes('BLOCKED')) throw e;
    }
    await clSleep(1500 * (i + 1));
  }
  throw new Error('fetch failed: ' + url);
}

/* ---------- stage 1: search result pages ---------- */
function clParseSearch(html, make, model, query) {
  const d = new DOMParser().parseFromString(html, 'text/html');
  return [...d.querySelectorAll('li.cl-static-search-result')].map((li) => ({
    search_make: make,
    search_model: model,
    search_query: query,
    url: li.querySelector('a')?.href,
    search_title: li.getAttribute('title') || clText(li.querySelector('.title')),
    search_price: clText(li.querySelector('.price')),
    search_location: clText(li.querySelector('.location')),
  }));
}

/* ---------- stage 2: listing detail page ---------- */
function clParseDetail(html, url) {
  const d = new DOMParser().parseFromString(html, 'text/html');
  const rec = { url };
  rec.title = clText(d.querySelector('#titletextonly'));
  rec.price = clText(d.querySelector('.postingtitletext .price'));
  rec.neighborhood = clText(d.querySelector('.postingtitletext small'));
  rec.year = clText(d.querySelector('.attr.important .year'));
  rec.make_model = clText(d.querySelector('.attr.important .makemodel'));
  d.querySelectorAll('.attrgroup .attr').forEach((a) => {
    const lab = a.querySelector('.labl');
    const val = a.querySelector('.valu');
    if (lab && val) {
      const key = 'attr_' + lab.textContent.replace(':', '').trim().toLowerCase().replace(/\s+/g, '_');
      rec[key] = clText(val);
    }
  });
  const body = d.querySelector('#postingbody');
  if (body) {
    body.querySelectorAll('.print-information, .print-qrcode-container').forEach((n) => n.remove());
    rec.description = body.textContent.replace(/[ \t]+/g, ' ').replace(/\n\s*\n+/g, '\n').trim();
  }
  const map = d.querySelector('#map');
  rec.latitude = map?.dataset.latitude || null;
  rec.longitude = map?.dataset.longitude || null;
  const times = [...d.querySelectorAll('.postinginfos time, .postinginfo time')].map((t) => t.getAttribute('datetime'));
  rec.posted_at = times[0] || null;
  rec.updated_at = times.length > 1 ? times[times.length - 1] : null;
  rec.post_id = (d.body.textContent.match(/post id:\s*(\d+)/) || [])[1] || null;
  const crumbs = [...d.querySelectorAll('.breadcrumbs .crumb')].map(clText).filter(Boolean);
  rec.area = crumbs[0] || null;
  rec.seller_type = crumbs.some((c) => /by owner/i.test(c)) ? 'owner' : crumbs.some((c) => /by dealer/i.test(c)) ? 'dealer' : null;
  rec.n_images = d.querySelectorAll('#thumbs a, .gallery .thumb').length;
  return rec;
}

async function clRun() {
  const S = window.__cl;
  const C = CL_CONFIG;
  S.stage = 'search';
  const seen = new Set(S.search.map((s) => s.url));
  for (const [make, model, query] of C.models) {
    if (S.search.some((s) => s.search_query === query)) continue;
    const url = `/search/area/philadelphia?cat=cta&bundleDuplicates=1&postal=${C.postal}&search_distance=${C.radius}` +
      `&auto_make_model=${encodeURIComponent(query)}`;
    const rows = clParseSearch(await clFetch(url), make, model, query).filter((r) => r.url && !seen.has(r.url));
    rows.forEach((r) => seen.add(r.url));
    S.search.push(...rows.slice(0, C.maxPerModel));
    S.searchCounts = { ...(S.searchCounts || {}), [query]: rows.length };
    await clSleep(C.delayMs);
  }
  S.stage = 'detail';
  const done = new Set(S.records.map((r) => r.url));
  const queue = S.search.filter((s) => !done.has(s.url));
  const worker = async () => {
    while (queue.length) {
      const s = queue.shift();
      try {
        const rec = clParseDetail(await clFetch(s.url), s.url);
        S.records.push({ ...s, ...rec, source: 'craigslist', scraped_at: new Date().toISOString() });
      } catch (e) {
        S.errors.push({ url: s.url, error: String(e) });
        if (String(e).includes('BLOCKED')) { queue.length = 0; }
      }
      if (S.records.length % 25 === 0) {
        try { localStorage.setItem('__cl_records', JSON.stringify(S.records)); } catch (e) { /* quota */ }
      }
      await clSleep(C.delayMs);
    }
  };
  await Promise.all(Array.from({ length: C.concurrency }, worker));
  S.stage = 'finished';
  S.done = true;
  return { search: S.search.length, records: S.records.length, errors: S.errors.length };
}

function clDownload(name = 'craigslist_raw.json') {
  const blob = new Blob([JSON.stringify(window.__cl.records)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
}
