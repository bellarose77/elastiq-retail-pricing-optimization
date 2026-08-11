import { useState } from "react";
import { fmtMoney, fmtNum, fmtPct, fmtPrice } from "../../lib/format.js";
import { Segmented, DeltaPill, ActionTag } from "../ui.jsx";
import { ScenarioBars, CategoryBars } from "../charts.jsx";
import Icon from "../Icon.jsx";

export default function PortfolioView({ result, onSelectItem, onNavigate }) {
  const [metric, setMetric] = useState("profit");
  const e = result.exec;
  const all = result.validation.find(v => v.check === "All checks passed");
  const top = [...result.rows].sort((a,b)=>b.profitChange-a.profitChange).slice(0,5);
  const strongest = [...result.categories].sort((a,b)=>b.profitChange-a.profitChange)[0];
  return <div className="stack">
    <section className="decision-hero">
      <div className="hero-copy"><span className="hero-label">Recommended portfolio plan</span><h2>Capture <strong>{fmtMoney(e.totalProfitChange)}</strong> in expected profit uplift while keeping every recommendation within policy.</h2><p>{e.increase} increases, {e.decrease} decreases and {e.hold} holds across {e.totalItems} decision units. The plan is ready for commercial review, not automatic execution.</p><div className="hero-actions"><button className="btn" onClick={()=>onNavigate("sku")}>Review recommendations <Icon name="arrow" size={16}/></button><button className="btn ghost" onClick={()=>onNavigate("sensitivity")}>Stress-test plan</button></div></div>
      <div className="hero-score"><div className="score-ring" style={{"--score":`${Math.round(all.passRate*100)*3.6}deg`}}><span>{Math.round(all.passRate*100)}%</span></div><b>Control pass rate</b><small>{all.passed}/{all.total} units pass all checks</small></div>
    </section>

    <div className="metric-grid">
      <article className="metric-card"><span>Expected profit</span><strong>{fmtMoney(e.totalExpectedProfit)}</strong><DeltaPill value={e.profitChangeRate}/><small>vs. {fmtMoney(e.totalCurrentProfit)} current</small></article>
      <article className="metric-card"><span>Expected revenue</span><strong>{fmtMoney(e.totalExpectedRevenue)}</strong><DeltaPill value={e.revenueChangeRate}/><small>vs. {fmtMoney(e.totalCurrentRevenue)} current</small></article>
      <article className="metric-card"><span>Average price move</span><strong>{fmtPct(e.avgPriceChangeRate)}</strong><div className="metric-inline"><b>{e.increase}</b> up <b>{e.decrease}</b> down</div><small>{e.promoRecommended} promotions proposed</small></article>
      <article className="metric-card"><span>Leading category</span><strong>{strongest?.category || "—"}</strong><div className="metric-inline positive">{fmtMoney(strongest?.profitChange || 0)}</div><small>{strongest?.items || 0} decision units</small></article>
    </div>

    <div className="dashboard-grid">
      <section className="panel span-7"><div className="panel-head"><div><span className="section-kicker">Portfolio scenarios</span><h3>What changes under each strategy?</h3><p className="panel-sub">Recommended is the implementable plan after commercial controls.</p></div><Segmented options={[{value:"profit",label:"Profit"},{value:"revenue",label:"Revenue"}]} value={metric} onChange={setMetric}/></div><div className="panel-body chart-panel"><ScenarioBars portfolio={result.portfolio} metric={metric}/></div></section>
      <section className="panel span-5"><div className="panel-head"><div><span className="section-kicker">Action allocation</span><h3>Recommended action mix</h3><p className="panel-sub">How the catalog will change in the next execution cycle.</p></div></div><div className="action-donut-wrap"><div className="action-donut" style={{background:`conic-gradient(var(--blue) 0 ${e.increase/e.totalItems*100}%, var(--rose) ${e.increase/e.totalItems*100}% ${(e.increase+e.decrease)/e.totalItems*100}%, var(--slate-300) ${(e.increase+e.decrease)/e.totalItems*100}% 100%)`}}><div><strong>{e.totalItems}</strong><span>units</span></div></div><div className="action-legend"><div><i className="dot blue"/><span>Increase</span><b>{e.increase}</b></div><div><i className="dot rose"/><span>Decrease</span><b>{e.decrease}</b></div><div><i className="dot gray"/><span>Hold</span><b>{e.hold}</b></div></div></div></section>
    </div>

    <section className="panel"><div className="panel-head"><div><span className="section-kicker">Execution queue</span><h3>Highest-value recommendations</h3><p className="panel-sub">Start review with decisions that create the largest expected profit contribution.</p></div><button className="link-btn" onClick={()=>onNavigate("sku")}>Open full queue <Icon name="arrow" size={14}/></button></div><div className="tbl-scroll"><table className="tbl"><thead><tr><th>Priority</th><th>Decision unit</th><th>Category</th><th className="r">Current</th><th className="r">Recommended</th><th className="r">Move</th><th>Action</th><th className="r">Profit impact</th></tr></thead><tbody>{top.map((r,i)=><tr key={r.itemId} onClick={()=>{onSelectItem(r.itemId);onNavigate("sku")}}><td><span className="rank-badge">{i+1}</span></td><td className="strong mono">{r.itemId}</td><td>{r.category}</td><td className="r mono">{fmtPrice(r.currentPrice)}</td><td className="r mono strong">{fmtPrice(r.recommendedPrice)}</td><td className="r mono">{fmtPct(r.priceChangeRate)}</td><td><ActionTag action={r.recommendationAction}/></td><td className="r"><span className="money-positive">{fmtMoney(r.profitChange)}</span></td></tr>)}</tbody></table></div></section>

    <div className="dashboard-grid"><section className="panel span-6"><div className="panel-head"><div><span className="section-kicker">Contribution</span><h3>Profit impact by category</h3></div></div><div className="panel-body"><CategoryBars categories={result.categories}/></div></section><section className="panel span-6"><div className="panel-head"><div><span className="section-kicker">Scenario ledger</span><h3>Portfolio totals</h3></div></div><div className="tbl-scroll"><table className="tbl compact"><thead><tr><th>Strategy</th><th className="r">Demand</th><th className="r">Revenue</th><th className="r">Profit</th><th className="r">Δ Profit</th></tr></thead><tbody>{result.portfolio.map(p=><tr key={p.key} className={p.key==="recommended"?"row-rec":""}><td>{p.name}</td><td className="r mono">{fmtNum(p.demand)}</td><td className="r mono">{fmtMoney(p.revenue)}</td><td className="r mono">{fmtMoney(p.profit)}</td><td className="r"><DeltaPill value={p.profitChangeRate}/></td></tr>)}</tbody></table></div></section></div>
  </div>;
}
