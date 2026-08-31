import { useState, useMemo, useEffect } from "react";
import { fmtMoney, fmtNum, fmtPrice, fmtPct, fmtPctPlain } from "../../lib/format.js";
import { Toggle, DeltaPill, ActionTag } from "../ui.jsx";
import { ResponseCurve } from "../charts.jsx";
import { fetchLiveProductFeatures } from "../../lib/api.js";

// `result` is App.jsx's optimizePortfolio(items, cfg, tech) output, reused
// here rather than recomputed per item, so this view's recommendations
// always match Portfolio Overview, Validation, Report and the exported PDF
// -- including the portfolio-level capacity-shadow-price and cannibalization
// passes that a standalone per-item optimize call would skip.
export default function SkuDetailView({ items, cfg, result, selectedId, setSelectedId }) {
  const [sortKey, setSortKey] = useState("profitChange");
  const [dir, setDir] = useState(-1);
  const [promoView, setPromoView] = useState(false);
  const [whatIf, setWhatIf] = useState(null);
  const rows = result.rows;
  const sorted = useMemo(() => [...rows].sort((a, b) => (a[sortKey] > b[sortKey] ? dir : -dir)), [rows, sortKey, dir]);
  const selected = items.find((i) => i.itemId === selectedId) || items[0];
  const selRes = selected ? rows.find((r) => r.itemId === selected.itemId) : null;
  useEffect(() => { setWhatIf(null); }, [selectedId]);

  // Fetched once (not per selected item) from service-feature-generation,
  // same "fetch the whole product-level rollup, look up by itemId" pattern
  // App.jsx uses for live pricing recommendations. Silently unavailable
  // (no alert) if the service isn't configured/reachable -- this panel is
  // best-effort market context, not part of the price decision itself.
  const [liveFeatures, setLiveFeatures] = useState(null);
  const [liveFeaturesError, setLiveFeaturesError] = useState("");
  useEffect(() => {
    let cancelled = false;
    fetchLiveProductFeatures()
      .then((data) => { if (!cancelled) setLiveFeatures(new Map(data.map((f) => [f.itemId, f]))); })
      .catch((err) => { if (!cancelled) setLiveFeaturesError(err.message); });
    return () => { cancelled = true; };
  }, []);
  const evidence = selected && liveFeatures ? liveFeatures.get(selected.itemId) : null;

  const th = (key, label, right) => (
    <th
      className={right ? "r sortable" : "sortable"}
      onClick={() => { if (sortKey === key) setDir(-dir); else { setSortKey(key); setDir(-1); } }}
    >
      {label}
      {sortKey === key ? <span className="sort-arrow">{dir < 0 ? " ↓" : " ↑"}</span> : ""}
    </th>
  );

  return (
    <div className="stack">
      {selected && selRes ? (
        <div className="panel">
          <div className="panel-head">
            <div>
              <h3>{selected.itemId} <span className="cat-chip">{selected.category}</span></h3>
              <p className="panel-sub">Elasticity {selected.elasticity.toFixed(2)} · {selRes.elasticitySource} · confidence {fmtPctPlain(selected.confidence)}</p>
            </div>
            <Toggle label="Model promotion" checked={promoView} onChange={setPromoView} />
          </div>
          <div className="detail-grid">
            <ResponseCurve item={selected} cfg={cfg} res={selRes} promoOn={promoView} whatIf={whatIf} setWhatIf={setWhatIf} />
            <div className="scenario-cards">
              {[
                { label: "Current", price: selected.currentPrice, s: selRes._scenarios.current, cls: "sc-current" },
                { label: "Revenue-max", price: selRes.revenueMaxPrice, s: selRes._scenarios.revenueMax, cls: "sc-rev" },
                { label: "Profit-max", price: selRes.profitMaxPrice, s: selRes._scenarios.profitMax, cls: "sc-pro" },
                { label: "Recommended", price: selRes.recommendedPrice, s: selRes._scenarios.recommended, cls: "sc-rec" },
              ].map((c) => (
                <div className={"scard " + c.cls} key={c.label}>
                  <div className="scard-top"><span className="scard-label">{c.label}</span><span className="scard-price mono">{fmtPrice(c.price)}</span></div>
                  <div className="scard-rows">
                    <div><span>Demand</span><b className="mono">{fmtNum(c.s.demand)}</b></div>
                    <div><span>Revenue</span><b className="mono">{fmtMoney(c.s.revenue)}</b></div>
                    <div><span>Profit</span><b className="mono">{fmtMoney(c.s.profit)}</b></div>
                    <div><span>Margin</span><b className="mono">{fmtPctPlain(c.s.marginRate)}</b></div>
                  </div>
                </div>
              ))}
            </div>
          </div>
          <div className="guardrail-strip">
            <span className="gr">Margin floor <b className="mono">{fmtPrice(selRes._guardrails.marginFloor)}</b></span>
            <span className="gr">Competitor band <b className="mono">{fmtPrice(selRes._guardrails.compLo)}–{fmtPrice(selRes._guardrails.compHi)}</b></span>
            <span className="gr">Step band <b className="mono">{fmtPrice(selRes._guardrails.stepLo)}–{fmtPrice(selRes._guardrails.stepHi)}</b></span>
            <span className="gr">Recommend promo <b>{selRes.recommendedPromotionFlag ? "Yes" : "No"}</b></span>
            {selRes.inventoryBinding ? <span className="gr warn">Inventory-bound</span> : null}
            {selRes.competitorBinding ? <span className="gr warn">Competitor-capped</span> : null}
          </div>
          {evidence ? (
            <div className="guardrail-strip">
              <span className="gr">Market evidence <b className="mono">{evidence.ragEvidenceCount}</b></span>
              <span className="gr">Weighted impact <b className="mono">{fmtNum(evidence.ragWeightedImpactScore, 3)}</b></span>
              <span className="gr">Net demand signal <b className="mono">{evidence.netDemandSignal}</b></span>
              <span className="gr">Stores <b className="mono">{evidence.storeCount}</b></span>
            </div>
          ) : liveFeaturesError ? (
            <p className="panel-sub" style={{ padding: "8px 18px 0" }}>Live market evidence unavailable: {liveFeaturesError}</p>
          ) : null}
        </div>
      ) : null}
      <div className="panel">
        <div className="panel-head"><div><h3>Decision units</h3><p className="panel-sub">Select a row to inspect its response curve.</p></div></div>
        <div className="tbl-scroll">
          <table className="tbl">
            <thead>
              <tr>
                {th("itemId", "Item")}<th>Cat</th>{th("elasticity", "Elas", true)}{th("currentPrice", "Current", true)}{th("recommendedPrice", "Rec.", true)}{th("priceChangeRate", "Δ Price", true)}<th className="r">Action</th>{th("expectedProfit", "Exp. profit", true)}{th("profitChange", "Δ Profit", true)}
              </tr>
            </thead>
            <tbody>
              {sorted.map((r) => (
                <tr key={r.itemId} className={r.itemId === selected.itemId ? "row-active" : ""} onClick={() => setSelectedId(r.itemId)}>
                  <td className="mono">{r.itemId}</td>
                  <td><span className="cat-chip sm">{r.category}</span></td>
                  <td className="r mono">{r.elasticity.toFixed(2)}</td>
                  <td className="r mono">{fmtPrice(r.currentPrice)}</td>
                  <td className="r mono strong">{fmtPrice(r.recommendedPrice)}</td>
                  <td className="r mono">{fmtPct(r.priceChangeRate)}</td>
                  <td className="r"><ActionTag action={r.recommendationAction} /></td>
                  <td className="r mono">{fmtMoney(r.expectedProfit)}</td>
                  <td className="r"><DeltaPill value={r.profitChangeRate} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
