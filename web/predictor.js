/* Browser/Node port of the trained pipeline: feature engineering -> log/scale/one-hot -> 800 XGBoost trees.
   Loads model.json (constants) + model.bin (tree arrays). Works in browsers (window.Predictor) and Node (module.exports). */
(function (root) {
  "use strict";
  const R_EARTH = 6371.0088;
  const rad = (d) => (d * Math.PI) / 180;
  function haversine(lat1, lon1, lat2, lon2) {
    const a = Math.sin(rad(lat2 - lat1) / 2) ** 2 + Math.cos(rad(lat1)) * Math.cos(rad(lat2)) * Math.sin(rad(lon2 - lon1) / 2) ** 2;
    return 2 * R_EARTH * Math.asin(Math.sqrt(a));
  }

  function build(meta, buffer) {
    const n = meta.nodes;
    const left = new Int32Array(buffer, 0, n);
    const right = new Int32Array(buffer, 4 * n, n);
    const val = new Float32Array(buffer, 8 * n, n);
    const feat = new Int16Array(buffer, 12 * n, n);
    const idx = Object.fromEntries(meta.features.map((f, i) => [f, i]));

    /* raw = {longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, ocean_proximity} */
    function vectorize(raw) {
      const x = new Float64Array(meta.features.length);
      const f = Object.assign({}, raw);
      f.rooms_per_household = f.total_rooms / f.households;
      f.bedrooms_per_room = f.total_bedrooms / f.total_rooms;
      f.population_per_household = f.population / f.households;
      let nearest = Infinity;
      for (const [name, [la, lo]] of Object.entries(meta.hubs)) {
        const d = haversine(f.latitude, f.longitude, la, lo);
        f["dist_" + name + "_km"] = d;
        if (d < nearest) nearest = d;
      }
      f.dist_nearest_hub_km = nearest;
      meta.log_cols.forEach((c, i) => { x[idx[c]] = (Math.log1p(f[c]) - meta.log_mean[i]) / meta.log_scale[i]; });
      meta.num_cols.forEach((c, i) => { x[idx[c]] = (f[c] - meta.num_mean[i]) / meta.num_scale[i]; });
      meta.categories.forEach((c) => { x[idx["ocean_proximity_" + c]] = f.ocean_proximity === c ? 1 : 0; });
      return x;
    }

    function predictVector(x) {
      let sum = meta.base_score;
      for (let t = 0; t < meta.roots.length; t++) {
        let node = meta.roots[t];
        while (feat[node] >= 0) node = Math.fround(x[feat[node]]) < val[node] ? left[node] : right[node];
        sum += val[node];
      }
      return sum;
    }
    return { predict: (raw) => predictVector(vectorize(raw)), vectorize, predictVector, features: meta.features, meta };
  }

  async function load(base) {
    base = base || "";
    const [meta, buf] = await Promise.all([fetch(base + "model.json").then((r) => r.json()), fetch(base + "model.bin").then((r) => r.arrayBuffer())]);
    return build(meta, buf);
  }

  const api = { build, load, haversine };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.Predictor = api;
})(typeof self !== "undefined" ? self : this);
