(function () {
  "use strict";
  const REPO_URL = ""; // paste your GitHub repository URL here to show a "Source code" link in the footer

  const $ = (id) => document.getElementById(id);
  const usd = (x) => "$" + Math.round(x).toLocaleString("en-US");
  const usd1k = (x) => usd(Math.round(x / 1000) * 1000);
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const OCEAN_LABEL = { "<1H OCEAN": "Within 1 hour of the ocean", INLAND: "Inland", "NEAR OCEAN": "Near the ocean", "NEAR BAY": "Near San Francisco Bay", ISLAND: "Island" };
  const MODEL_LABEL = { "Dummy (mean)": "Always guess the average", Ridge: "Ridge regression", "Ridge (log-target)": "Ridge, fitted to log price", "k-NN": "k-nearest neighbours", "Random Forest": "Random forest", HistGB: "Gradient boosting (scikit-learn)", XGBoost: "XGBoost" };
  const HUB_LABEL = { SanFrancisco: "San Francisco", LosAngeles: "Los Angeles", SanDiego: "San Diego", SanJose: "San Jose", Sacramento: "Sacramento" };
  const PRESETS = {
    bay: { lon: -122.25, lat: 37.85, income: 4.2, age: 40, hh: 600, rpp: 4.8, bed: 0.2, pph: 2.6 },
    la: { lon: -118.45, lat: 34.02, income: 5.2, age: 35, hh: 700, rpp: 4.8, bed: 0.21, pph: 2.5 },
    valley: { lon: -119.77, lat: 36.75, income: 2.6, age: 28, hh: 450, rpp: 5.0, bed: 0.22, pph: 3.4 },
    north: { lon: -122.4, lat: 40.58, income: 2.9, age: 32, hh: 350, rpp: 5.6, bed: 0.22, pph: 2.5 },
  };
  const FMT = { income: (v) => usd(v * 10000), age: (v) => v + " years", hh: (v) => (+v).toLocaleString("en-US"), rpp: (v) => (+v).toFixed(1), bed: (v) => Math.round(v * 100) + " %", pph: (v) => (+v).toFixed(1) };
  const SLIDERS = ["income", "age", "hh", "rpp", "bed", "pph"];

  const VALUE_STOPS = [[0, [39, 96, 127]], [0.25, [127, 175, 192]], [0.5, [233, 228, 206]], [0.72, [217, 154, 99]], [0.88, [168, 50, 111]], [1, [94, 22, 80]]];
  const ERR_STOPS = [[0, [30, 123, 128]], [0.5, [238, 240, 232]], [1, [201, 130, 31]]];
  function ramp(stops, t) {
    t = clamp(t, 0, 1);
    for (let i = 1; i < stops.length; i++) {
      if (t <= stops[i][0]) {
        const [t0, c0] = stops[i - 1], [t1, c1] = stops[i], u = (t - t0) / (t1 - t0 || 1);
        return `rgb(${Math.round(c0[0] + (c1[0] - c0[0]) * u)},${Math.round(c0[1] + (c1[1] - c0[1]) * u)},${Math.round(c0[2] + (c1[2] - c0[2]) * u)})`;
      }
    }
    const c = stops[stops.length - 1][1];
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  }
  const cssGradient = (stops) => "linear-gradient(90deg," + stops.map(([t, c]) => `rgb(${c[0]},${c[1]},${c[2]}) ${Math.round(t * 100)}%`).join(",") + ")";

  const B = { lonMin: -124.6, lonMax: -114.1, latMin: 32.4, latMax: 42.1 };
  const COS = Math.cos((37.2 * Math.PI) / 180);
  const S = { lon: -122.25, lat: 37.85, income: 3.5, age: 29, hh: 410, rpp: 5.4, bed: 0.21, pph: 2.8, ocean: "NEAR BAY" };
  let model, D, results, ctx, cv, geo = {}, mode = "actual", reveal = 0, dragging = false, colors = {}, lastNear = null;

  /* ───────── data-independent sections ───────── */
  function renderResults() {
    const r = results;
    $("acc_text").textContent = `Guessing the average value for every district would give a typical error (RMSE) of ${usd(r.baseline_rmse)}. ${r.best} has a typical error of ${usd(r.rmse)} on districts it never saw, an average miss of ${usd(r.mae)}, explains ${(r.r2 * 100).toFixed(0)} % of the variation in house value, and lands within 20 % of the true value for ${r.within20.toFixed(0)} % of districts.`;
    const cap = document.querySelector("#models caption"); cap.textContent = cap.textContent.replace("{N}", r.n_test.toLocaleString("en-US"));
    const maxR = Math.max(...r.models.map((m) => m.rmse)), tb = document.querySelector("#models tbody");
    r.models.forEach((m) => {
      const tr = document.createElement("tr");
      if (m.name === r.best) tr.className = "best"; else if (m.name.startsWith("Dummy")) tr.className = "base";
      tr.innerHTML = `<td>${MODEL_LABEL[m.name] || m.name}</td><td><span class="rmse"><span class="bar" style="width:${Math.round((m.rmse / maxR) * 170)}px"></span>${usd(m.rmse)}</span></td><td>${usd(m.mae)}</td><td>${Math.max(0, m.r2).toFixed(2)}</td>`;
      tb.appendChild(tr);
    });
    const qa = $("qa");
    r.findings.forEach((f) => { const dt = document.createElement("dt"); dt.textContent = f.q; const dd = document.createElement("dd"); dd.textContent = f.a; qa.append(dt, dd); });
    if (REPO_URL) { $("repolink").href = REPO_URL; $("repo").hidden = false; }
  }

  /* ───────── map ───────── */
  function layout() {
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
    cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const pad = 28;
    geo = { w, h, s: Math.min((w - 2 * pad) / ((B.lonMax - B.lonMin) * COS), (h - 2 * pad) / (B.latMax - B.latMin)), cx: w / 2, cy: h / 2, lonC: (B.lonMin + B.lonMax) / 2, latC: (B.latMin + B.latMax) / 2 };
  }
  const px = (lon) => geo.cx + (lon - geo.lonC) * COS * geo.s;
  const py = (lat) => geo.cy - (lat - geo.latC) * geo.s;
  const invLon = (x) => geo.lonC + (x - geo.cx) / (COS * geo.s);
  const invLat = (y) => geo.latC - (y - geo.cy) / geo.s;

  function buildColors() {
    colors.actual = D.actual.map((v) => ramp(VALUE_STOPS, (v - 50000) / 450000));
    colors.pred = D.pred.map((v) => ramp(VALUE_STOPS, (v - 50000) / 450000));
    colors.err = D.actual.map((v, i) => ramp(ERR_STOPS, (v - D.pred[i] + 150000) / 300000));
  }

  function draw() {
    if (!ctx || !geo.s) return;
    const { w, h } = geo;
    ctx.clearRect(0, 0, w, h);
    ctx.font = "11px 'Libre Franklin', system-ui, sans-serif"; ctx.textBaseline = "middle";
    ctx.strokeStyle = "#D6DEDC"; ctx.lineWidth = 1; ctx.fillStyle = "#6B7C83";
    for (let lat = 34; lat <= 42; lat += 2) { const y = Math.round(py(lat)) + 0.5; ctx.beginPath(); ctx.moveTo(6, y); ctx.lineTo(w - 6, y); ctx.stroke(); ctx.fillText(lat + "°N", 8, y - 8); }
    for (let lon = -124; lon <= -114; lon += 2) { const x = Math.round(px(lon)) + 0.5; ctx.beginPath(); ctx.moveTo(x, 6); ctx.lineTo(x, h - 6); ctx.stroke(); ctx.fillText(-lon + "°W", x + 4, h - 12); }
    const limit = B.lonMin + reveal * (B.lonMax - B.lonMin), col = colors[mode], r = Math.max(2.1, geo.s / 38);
    for (let i = 0; i < D.lon.length; i++) {
      if (D.lon[i] > limit) continue;
      ctx.fillStyle = col[i]; ctx.globalAlpha = 0.88; ctx.beginPath(); ctx.arc(px(D.lon[i]), py(D.lat[i]), r, 0, 6.2832); ctx.fill();
    }
    ctx.globalAlpha = 1;
    // the five cities used for the distance features
    ctx.fillStyle = "#17303B"; ctx.strokeStyle = "#F8FAF9"; ctx.lineWidth = 1.5;
    Object.entries(model.meta.hubs).forEach(([k, [la, lo]]) => {
      const x = px(lo), y = py(la); ctx.fillRect(x - 4, y - 4, 8, 8); ctx.strokeRect(x - 4, y - 4, 8, 8);
      ctx.fillStyle = "#17303B"; ctx.font = "600 11px 'Libre Franklin', system-ui, sans-serif";
      let leftSide = lo < -120; if (leftSide && x - 9 - ctx.measureText(HUB_LABEL[k]).width < 4) leftSide = false;   // keep the label on screen
      ctx.textAlign = leftSide ? "right" : "left"; ctx.fillText(HUB_LABEL[k], x + (leftSide ? -9 : 9), y + 1); ctx.textAlign = "left";
    });
    // pin
    const x = px(S.lon), y = py(S.lat);
    ctx.lineWidth = 4; ctx.strokeStyle = "#F8FAF9"; ctx.beginPath(); ctx.arc(x, y, 11, 0, 6.2832); ctx.stroke();
    ctx.lineWidth = 2.5; ctx.strokeStyle = "#6F1D5C"; ctx.beginPath(); ctx.arc(x, y, 11, 0, 6.2832); ctx.stroke();
    ctx.fillStyle = "#6F1D5C"; ctx.beginPath(); ctx.arc(x, y, 3, 0, 6.2832); ctx.fill();
  }

  function renderLegend() {
    const el = $("legend");
    const stops = mode === "err" ? ERR_STOPS : VALUE_STOPS, ticks = mode === "err" ? ["Model too high", "Exact", "Model too low"] : ["$50k", "$275k", "$500k+"];
    el.innerHTML = `<div class="col2"><div class="ramp" style="background:${cssGradient(stops)}"></div><div class="ticks"><span>${ticks[0]}</span><span>${ticks[1]}</span><span>${ticks[2]}</span></div></div>`;
  }

  /* ───────── nearest districts ───────── */
  function nearest(n) {
    const out = [], P = window.Predictor;
    const d = new Float64Array(D.lon.length);
    for (let i = 0; i < d.length; i++) d[i] = P.haversine(S.lat, S.lon, D.lat[i], D.lon[i]);
    const used = new Set();
    for (let k = 0; k < n; k++) { let b = -1; for (let i = 0; i < d.length; i++) if (!used.has(i) && (b < 0 || d[i] < d[b])) b = i; used.add(b); out.push({ i: b, km: d[b] }); }
    return out;
  }

  /* ───────── prediction ───────── */
  function compute() {
    const raw = { longitude: S.lon, latitude: S.lat, housing_median_age: S.age, total_rooms: S.rpp * S.hh, total_bedrooms: S.bed * S.rpp * S.hh, population: S.pph * S.hh, households: S.hh, median_income: S.income, ocean_proximity: S.ocean };
    if (Object.values(raw).some((v) => typeof v === "number" && !isFinite(v))) return;
    const p = model.predict(raw), m = results.medape / 100;
    $("price").textContent = usd1k(p);
    $("range").textContent = `Half of the districts the model had never seen were estimated within ${results.medape.toFixed(1)} % of their true value, so expect roughly ${usd1k(p * (1 - m))} to ${usd1k(p * (1 + m))}.`;
    $("caveat").textContent = `1990 dollars. About ${results.within20.toFixed(0)} % of unseen districts were estimated within 20 %.` + (p >= 0.9 * results.cap ? " Values in the data stop at $500,001, so estimates near the top are conservative." : "");
    $("derived").textContent = `That is about ${Math.round(raw.total_rooms).toLocaleString("en-US")} rooms, ${Math.round(raw.total_bedrooms).toLocaleString("en-US")} bedrooms and ${Math.round(raw.population).toLocaleString("en-US")} people across ${Math.round(raw.households).toLocaleString("en-US")} households.`;
    const near = nearest(5); lastNear = near;
    $("neartable").innerHTML = near.map(({ i, km }) => `<tr><td>${km < 10 ? km.toFixed(1) : Math.round(km)} km</td><td>${usd(D.income[i] * 10000)}</td><td>${usd(D.actual[i])}</td><td>${usd(D.pred[i])}</td></tr>`).join("");
    const notes = [];
    if (near[0].km > 40) notes.push(`The nearest district in the data is ${Math.round(near[0].km)} km away, so this pin is outside the areas the census file covers and the estimate is unreliable.`);
    const rng = model.meta.train_range, chk = { "household income": ["median_income", raw.median_income], "house age": ["housing_median_age", raw.housing_median_age], "total rooms": ["total_rooms", raw.total_rooms], "total bedrooms": ["total_bedrooms", raw.total_bedrooms], "population": ["population", raw.population], "households": ["households", raw.households] };
    const outside = Object.entries(chk).filter(([, [k, v]]) => v < rng[k][0] || v > rng[k][1]).map(([n]) => n);
    if (outside.length) notes.push(`These values lie outside what the model was trained on: ${outside.join(", ")}.`);
    const warn = $("warn"); warn.hidden = !notes.length; warn.textContent = notes.join(" ");
    $("mapstatus").textContent = `Pin at ${S.lat.toFixed(2)} degrees north, ${(-S.lon).toFixed(2)} degrees west. Estimated median house value ${usd1k(p)}.`;
    return p;
  }

  function syncInputs() {
    SLIDERS.forEach((k) => { $(k).value = S[k]; $(k + "_o").textContent = FMT[k](S[k]); });
    $("lat").value = S.lat.toFixed(2); $("lon").value = S.lon.toFixed(2); $("ocean").value = S.ocean;
  }
  function setPin(lon, lat, keepOcean) {
    S.lon = clamp(lon, B.lonMin, B.lonMax); S.lat = clamp(lat, B.latMin, B.latMax);
    if (!keepOcean) { const n = nearest(1)[0]; S.ocean = D.categories[D.ocean[n.i]]; }
    syncInputs(); compute(); draw();
  }

  function showTip(e) {
    const tip = $("tip"), rect = cv.getBoundingClientRect(), x = e.clientX - rect.left, y = e.clientY - rect.top;
    let best = -1, bd = 100;
    for (let i = 0; i < D.lon.length; i++) { const dx = px(D.lon[i]) - x, dy = py(D.lat[i]) - y, dd = dx * dx + dy * dy; if (dd < bd) { bd = dd; best = i; } }
    if (best < 0) { tip.hidden = true; return; }
    const err = D.actual[best] - D.pred[best];
    tip.textContent = `Actual ${usd(D.actual[best])}, model ${usd(D.pred[best])} (${err >= 0 ? "+" : "−"}${usd(Math.abs(err))}), income ${usd(D.income[best] * 10000)}`;
    tip.hidden = false; tip.style.left = clamp(x + 14, 4, rect.width - 300) + "px"; tip.style.top = clamp(y + 14, 4, rect.height - 34) + "px";
  }

  function wire() {
    SLIDERS.forEach((k) => $(k).addEventListener("input", () => { S[k] = +$(k).value; $(k + "_o").textContent = FMT[k](S[k]); compute(); }));
    const fromNumbers = () => { const la = parseFloat($("lat").value), lo = parseFloat($("lon").value); if (isFinite(la) && isFinite(lo)) setPin(lo, la); };
    $("lat").addEventListener("change", fromNumbers); $("lon").addEventListener("change", fromNumbers);
    $("ocean").addEventListener("change", () => { S.ocean = $("ocean").value; compute(); });
    document.querySelectorAll("[data-preset]").forEach((b) => b.addEventListener("click", () => { const p = PRESETS[b.dataset.preset]; Object.assign(S, p); setPin(p.lon, p.lat); }));
    document.querySelectorAll("input[name=mode]").forEach((r) => r.addEventListener("change", () => { mode = r.value; renderLegend(); draw(); }));
    if (!cv) return;
    const fromEvent = (e) => { const rect = cv.getBoundingClientRect(); setPin(invLon(e.clientX - rect.left), invLat(e.clientY - rect.top)); };
    cv.addEventListener("pointerdown", (e) => { dragging = true; try { cv.setPointerCapture(e.pointerId); } catch (_) {} fromEvent(e); $("tip").hidden = true; });
    cv.addEventListener("pointermove", (e) => { if (dragging) fromEvent(e); else showTip(e); });
    cv.addEventListener("pointerup", () => { dragging = false; });
    cv.addEventListener("pointerleave", () => { $("tip").hidden = true; });
    cv.addEventListener("keydown", (e) => {
      const step = e.shiftKey ? 0.25 : 0.05, k = e.key;
      if (k === "ArrowLeft") setPin(S.lon - step, S.lat); else if (k === "ArrowRight") setPin(S.lon + step, S.lat);
      else if (k === "ArrowUp") setPin(S.lon, S.lat + step); else if (k === "ArrowDown") setPin(S.lon, S.lat - step); else return;
      e.preventDefault();
    });
    if (window.ResizeObserver) new ResizeObserver(() => { layout(); draw(); }).observe(cv.parentElement); else window.addEventListener("resize", () => { layout(); draw(); });
  }

  async function init() {
    cv = $("map"); ctx = cv.getContext && cv.getContext("2d");
    try {
      const [res, dist, m] = await Promise.all([fetch("results.json").then((r) => r.json()), fetch("districts.json").then((r) => r.json()), window.Predictor.load("")]);
      results = res; D = dist; model = m;
    } catch (err) {
      $("price").textContent = "Could not load the model";
      $("range").textContent = "The model files did not download. Check your connection and reload the page.";
      return;
    }
    const sel = $("ocean"); D.categories.forEach((c) => { const o = document.createElement("option"); o.value = c; o.textContent = OCEAN_LABEL[c] || c; sel.appendChild(o); });
    renderResults(); buildColors(); renderLegend(); wire(); layout(); syncInputs(); setPin(S.lon, S.lat, false);
    const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce || !ctx) { reveal = 1; draw(); return; }
    const t0 = performance.now(), dur = 1500;                       // the page's single entrance: California is drawn west to east
    (function frame(t) { reveal = clamp((t - t0) / dur, 0, 1); draw(); if (reveal < 1) requestAnimationFrame(frame); })(t0);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
