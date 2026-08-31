import { useMemo, useRef, useState } from "react";
import { parseCSV, downloadCSV, csvTemplate } from "../../lib/report.js";
import { analyzeInput } from "../../lib/inputAnalysis.js";
import { fmtNum, fmtPctPlain, fmtPrice } from "../../lib/format.js";
import { fetchLiveProducts } from "../../lib/api.js";

// Mirrors parseCSV's validation in lib/report.js so an inline edit can't put
// data into a worse state than a CSV import would ever be allowed to:
// unconstrained fields only need to be finite, the rest match the same
// domain rule parseCSV enforces.
const NUMERIC_FIELD_RULES = {
  currentPrice: { test: (n) => n > 0, message: "must be a positive number" },
  unitCost: { test: (n) => n >= 0, message: "must be zero or positive" },
  baselineQuantity: { test: (n) => n >= 0, message: "must be zero or positive" },
  elasticity: { test: (n) => n < 0, message: "must be negative (elastic demand)" },
};
const DEFAULT_NUMERIC_RULE = { test: () => true, message: "must be a number" };

export default function DataView({ items, setItems }) {
  const fileRef = useRef(null);
  const [importMessage, setImportMessage] = useState("");
  const [drafts, setDrafts] = useState({});
  const [invalidCells, setInvalidCells] = useState(new Set());
  const [liveLoading, setLiveLoading] = useState(false);
  const analysis = useMemo(() => analyzeInput(items), [items]);

  const loadLiveData = async () => {
    setLiveLoading(true);
    try {
      const products = await fetchLiveProducts();
      setItems(products);
      setImportMessage(`Loaded ${products.length} products live from the pricing API.`);
    } catch (err) {
      setImportMessage("");
      alert("Could not load live data: " + err.message);
    } finally {
      setLiveLoading(false);
    }
  };

  const cellKey = (idx, key) => idx + ":" + key;

  const update = (idx, key, val) => {
    if (key === "category" || key === "itemId") {
      setItems(items.map((it, i) => (i === idx ? { ...it, [key]: val } : it)));
      return;
    }

    const ck = cellKey(idx, key);
    const num = parseFloat(val);
    const rule = NUMERIC_FIELD_RULES[key] || DEFAULT_NUMERIC_RULE;
    const isValid = val.trim() !== "" && Number.isFinite(num) && rule.test(num);

    if (!isValid) {
      // Keep exactly what the user typed visible, including a mid-edit state
      // like "-" or "", but never commit it -- the last valid value stays in
      // `items` so NaN can never reach the optimizer or downstream views.
      setDrafts((d) => ({ ...d, [ck]: val }));
      setInvalidCells((s) => new Set(s).add(ck));
      return;
    }

    setDrafts((d) => {
      if (!(ck in d)) return d;
      const next = { ...d };
      delete next[ck];
      return next;
    });
    setInvalidCells((s) => {
      if (!s.has(ck)) return s;
      const next = new Set(s);
      next.delete(ck);
      return next;
    });
    setItems(items.map((it, i) => (i === idx ? { ...it, [key]: num } : it)));
  };

  const addRow = () => {
    const n = items.length + 1;
    setItems([
      ...items,
      { itemId: "NEW" + n, productId: "NEW" + n, category: "Beverages", currentPrice: 5, unitCost: 2.5, competitorPrice: 5.2, inventory: 50, baselineQuantity: 40, baselineSource: "manual", elasticity: -1.5, elasticitySource: "manual", elasticityIsCausal: false, promotionUpliftRate: 0, promotionModelReliable: false, confidence: 0.7, ragSignal: 0, promotionFlag: 0 },
    ]);
  };

  const del = (idx) => setItems(items.filter((_, i) => i !== idx));

  const onFile = (e) => {
    const f = e.target.files[0];
    if (!f) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const parsed = parseCSV(reader.result);
        if (parsed.length) {
          setItems(parsed);
          const causal = parsed.filter((row) => row.elasticityIsCausal).length;
          setImportMessage(`Imported ${parsed.length} rows; ${causal} have causal elasticity and ${parsed.length - causal} are held for review.`);
        }
        else alert("No rows parsed. Check that headers match the template.");
      } catch (err) {
        setImportMessage("");
        alert("Could not import CSV: " + err.message);
      }
    };
    reader.readAsText(f);
  };

  const cols = [
    ["itemId", "Item", "text"], ["category", "Category", "text"], ["currentPrice", "Price", "n"], ["unitCost", "Cost", "n"],
    ["competitorPrice", "Comp.", "n"], ["inventory", "Inv.", "n"], ["baselineQuantity", "Base qty", "n"], ["elasticity", "Elas.", "n"],
    ["promotionUpliftRate", "Uplift", "n"], ["confidence", "Conf.", "n"], ["ragSignal", "RAG", "n"],
  ];

  return (
    <div className="stack">
      <div className="input-analysis-hero">
        <div><span>Initial input analysis</span><h2>{analysis.rows.toLocaleString()} decision units are loaded</h2><p>Before optimization, ELASTIQ profiles coverage, commercial structure, demand, inventory pressure, elasticity evidence and promotion readiness. Editing or importing data updates this analysis immediately.</p></div>
        <div className={analysis.invalidRows || analysis.duplicateIds ? "input-score warn" : "input-score"}><b>{fmtPctPlain(analysis.completenessRate)}</b><span>numeric completeness</span><small>{analysis.invalidRows} invalid / {analysis.duplicateIds} duplicate IDs</small></div>
      </div>

      <div className="input-profile-grid standalone">
        <div><span>Coverage</span><b>{analysis.categoryCount} categories</b><small>{analysis.productCount || "—"} products / {analysis.storeCount || "—"} stores</small></div>
        <div><span>Price structure</span><b>{fmtPrice(analysis.avgPrice)} average</b><small>{fmtPrice(analysis.priceRange[0])} to {fmtPrice(analysis.priceRange[1])}</small></div>
        <div><span>Current margin</span><b>{fmtPctPlain(analysis.avgMargin, 1)}</b><small>Average before recommendations</small></div>
        <div><span>Competitor position</span><b>{fmtPctPlain(analysis.avgCompetitorPosition, 1)}</b><small>Average price premium / discount</small></div>
        <div><span>Baseline demand</span><b>{fmtNum(analysis.totalDemand)}</b><small>Median {fmtNum(analysis.demandP50, 1)} per decision unit</small></div>
        <div><span>Inventory pressure</span><b>{fmtPctPlain(analysis.inventoryRiskRate)}</b><small>{"Stock <= 105% of baseline demand"}</small></div>
        <div><span>Causal readiness</span><b>{fmtPctPlain(analysis.causalRate)}</b><small>Rows with product or pooled IV evidence</small></div>
        <div><span>Promotion readiness</span><b>{fmtPctPlain(analysis.promotionReadyRate)}</b><small>Rows eligible to use modelled uplift</small></div>
      </div>

      <div className="panel">
        <div className="panel-head"><div><h3>Category profile</h3><p className="panel-sub">Commercial and evidence structure before any price is optimized.</p></div></div>
        <div className="tbl-scroll"><table className="tbl compact"><thead><tr><th>Category</th><th className="r">Rows</th><th className="r">Avg price</th><th className="r">Avg margin</th><th className="r">Forecast demand</th><th className="r">Inventory risk</th><th className="r">Causal coverage</th><th className="r">Promo ready</th></tr></thead><tbody>{analysis.categories.map((row) => <tr key={row.category}><td className="strong">{row.category}</td><td className="r mono">{row.items}</td><td className="r mono">{fmtPrice(row.avgPrice)}</td><td className="r mono">{fmtPctPlain(row.avgMargin, 1)}</td><td className="r mono">{fmtNum(row.totalDemand)}</td><td className="r mono">{fmtPctPlain(row.inventoryRiskRate)}</td><td className="r mono">{fmtPctPlain(row.causalRate)}</td><td className="r mono">{fmtPctPlain(row.promotionReadyRate)}</td></tr>)}</tbody></table></div>
      </div>

      <div className="panel">
        <div className="panel-head"><div><h3>What the input fields mean</h3><p className="panel-sub">The optimizer needs commercial baselines, model evidence and operating constraints for each decision unit.</p></div></div>
        <div className="input-dictionary">
          <div><b>Identity</b><p><code>itemId</code>, <code>productId</code>, store, category and region define what is being priced and support aggregation and market matching.</p></div>
          <div><b>Current economics</b><p><code>currentPrice</code>, <code>unitCost</code> and <code>competitorPrice</code> define margin and permitted market position.</p></div>
          <div><b>Future baseline</b><p><code>baselineQuantity</code> is the next-period forecast at the current price; <code>inventory</code> limits realized sales.</p></div>
          <div><b>Price response</b><p><code>elasticity</code>, source, causal flag and confidence describe how demand changes with price and whether repricing is allowed.</p></div>
          <div><b>Promotion evidence</b><p>Uplift, reliability and current promotion flag determine whether incremental demand can enter the candidate scenarios.</p></div>
          <div><b>Market evidence</b><p><code>ragSignal</code> is a bounded demand adjustment from time-safe notes; evidence counts and filters provide provenance.</p></div>
        </div>
      </div>

      <div className="panel">
        <div className="panel-head">
          <div><h3>Catalog data</h3><p className="panel-sub">Edit any cell, add units, or import a CSV. Everything recomputes live.</p></div>
          <div className="btn-group">
            <button className="btn ghost" onClick={loadLiveData} disabled={liveLoading} title="Fetch the live catalog from the ELASTIQ pricing API (service-pricing-optimization)">{liveLoading ? "Loading…" : "Load live data"}</button>
            <button className="btn ghost" onClick={() => fileRef.current.click()}>Import CSV</button>
            <button className="btn ghost" onClick={() => downloadCSV(csvTemplate(items), "pricing_input_template.csv")}>Download template</button>
            <button className="btn" onClick={addRow}>Add unit</button>
            <input ref={fileRef} type="file" accept=".csv" onChange={onFile} style={{ display: "none" }} />
          </div>
        </div>
        <p className="panel-sub" style={{ padding: "0 18px 12px" }}>
          {importMessage || `${items.filter((row) => row.elasticityIsCausal || ["product_iv", "pooled_iv"].includes(row.elasticitySource)).length}/${items.length} rows have causal elasticity. Manual or correlational rows remain visible but are not repriced while the causal-evidence gate is enabled.`}
        </p>
        {invalidCells.size > 0 ? (
          <p className="panel-sub cell-error-note" style={{ padding: "0 18px 12px" }}>
            {invalidCells.size} cell{invalidCells.size === 1 ? "" : "s"} highlighted in red {invalidCells.size === 1 ? "has" : "have"} an invalid value and {invalidCells.size === 1 ? "is" : "are"} still using the last valid number — hover a cell for the specific rule.
          </p>
        ) : null}
        <div className="tbl-scroll">
          <table className="tbl editable">
            <thead>
              <tr>
                {cols.map((c) => <th key={c[0]} className={c[2] === "n" ? "r" : ""}>{c[1]}</th>)}
                <th></th>
              </tr>
            </thead>
            <tbody>
              {items.map((it, idx) => (
                <tr key={idx}>
                  {cols.map((c) => {
                    const ck = cellKey(idx, c[0]);
                    const invalid = invalidCells.has(ck);
                    const rule = NUMERIC_FIELD_RULES[c[0]] || DEFAULT_NUMERIC_RULE;
                    return (
                      <td key={c[0]} className={c[2] === "n" ? "r" : ""}>
                        <input
                          className={"cell-input" + (c[2] === "n" ? " mono r" : "") + (invalid ? " invalid" : "")}
                          value={invalid ? drafts[ck] : it[c[0]]}
                          onChange={(e) => update(idx, c[0], e.target.value)}
                          title={invalid ? `${c[1]} ${rule.message} — keeping the previous value until this is fixed` : undefined}
                        />
                      </td>
                    );
                  })}
                  <td><button className="row-del" onClick={() => del(idx)} title="Remove">×</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
