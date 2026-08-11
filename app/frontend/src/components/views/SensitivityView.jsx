import { useState, useMemo } from "react";
import { sensitivityAnalysis } from "../../lib/engine.js";
import { fmtMoney, fmtPct } from "../../lib/format.js";
import { Segmented, DeltaPill } from "../ui.jsx";
import { SensitivityHeatmap } from "../charts.jsx";

export default function SensitivityView({ items, cfg, tech }) {
  const [metric,setMetric]=useState("profitChangeRate");
  const grid=useMemo(()=>sensitivityAnalysis(items,cfg,tech),[items,cfg,tech]);
  const base=grid.find(g=>g.elasticityMult===1&&g.costMult===1);
  const worst=[...grid].sort((a,b)=>a[metric]-b[metric])[0];
  const best=[...grid].sort((a,b)=>b[metric]-a[metric])[0];
  const positive=grid.filter(g=>g.profitChangeRate>=0).length;
  return <div className="stack">
    <div className="metric-grid three"><article className="metric-card"><span>Base-case profit</span><strong>{fmtMoney(base.expectedProfit)}</strong><DeltaPill value={base.profitChangeRate}/><small>elasticity ×1.00 · cost ×1.00</small></article><article className="metric-card"><span>Downside case</span><strong>{fmtPct(worst[metric])}</strong><small>elasticity ×{worst.elasticityMult.toFixed(2)} · cost ×{worst.costMult.toFixed(2)}</small></article><article className="metric-card"><span>Robustness coverage</span><strong>{positive}/9</strong><div className="metric-inline">scenarios remain profitable</div><small>{best?`Best case ${fmtPct(best[metric])}`:""}</small></article></div>
    <section className="panel"><div className="panel-head"><div><span className="section-kicker">Assumption stress test</span><h3>Does the recommendation survive estimation error?</h3><p className="panel-sub">Each cell re-optimizes the full portfolio after changing elasticity and unit cost assumptions.</p></div><Segmented options={[{value:"profitChangeRate",label:"Profit uplift"},{value:"revenueChangeRate",label:"Revenue uplift"},{value:"avgPriceChangeRate",label:"Average move"}]} value={metric} onChange={setMetric}/></div><SensitivityHeatmap grid={grid} metric={metric}/></section>
    <section className="panel"><div className="panel-head"><div><span className="section-kicker">Scenario ledger</span><h3>Detailed stress-test results</h3><p className="panel-sub">Use the table to isolate whether cost or elasticity creates the greatest risk.</p></div></div><div className="tbl-scroll"><table className="tbl"><thead><tr><th>Elasticity factor</th><th>Cost factor</th><th className="r">Expected profit</th><th className="r">Δ Profit</th><th className="r">Expected revenue</th><th className="r">Δ Revenue</th><th className="r">Avg price move</th></tr></thead><tbody>{grid.map((g,i)=><tr key={i} className={g.elasticityMult===1&&g.costMult===1?"row-active":""}><td className="mono">×{g.elasticityMult.toFixed(2)}</td><td className="mono">×{g.costMult.toFixed(2)}</td><td className="r mono">{fmtMoney(g.expectedProfit)}</td><td className="r"><DeltaPill value={g.profitChangeRate}/></td><td className="r mono">{fmtMoney(g.expectedRevenue)}</td><td className="r"><DeltaPill value={g.revenueChangeRate}/></td><td className="r mono">{fmtPct(g.avgPriceChangeRate)}</td></tr>)}</tbody></table></div></section>
  </div>;
}
