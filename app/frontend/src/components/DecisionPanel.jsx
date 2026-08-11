import Icon from "./Icon.jsx";
import { explain } from "../lib/explain.js";
import { fmtMoney, fmtPct, fmtPctPlain, fmtPrice } from "../lib/format.js";

export default function DecisionPanel({ tab, result, cfg, tech, meta, selectedItem, onSelectItem, onNavigate, onOpenSettings, onPdf }) {
  const all = result.validation.find(v => v.check === "All checks passed");
  const readiness = all ? Math.round(all.passRate * 100) : 100;
  const selected = result.rows.find(r => r.itemId === selectedItem?.itemId) || result.rows[0];
  const context = explain(tab, { result, cfg, techId: tech.id, selected: selectedItem });
  const top = [...result.rows].sort((a,b) => b.profitChange - a.profitChange).slice(0,3);
  const binding = {
    inventory: result.rows.filter(r => r.inventoryBinding).length,
    competitor: result.rows.filter(r => r.competitorBinding).length,
    step: result.rows.filter(r => r.stepBinding).length,
  };
  return <aside className="decision-panel">
    <div className="decision-scroll">
      <section className="decision-status">
        <div className="dp-eyebrow"><span className="status-dot"/> Decision readiness</div>
        <div className="readiness-row"><strong>{readiness}%</strong><span>{readiness === 100 ? "Ready for review" : "Needs attention"}</span></div>
        <div className="readiness-track"><i style={{width:`${readiness}%`}}/></div>
        <p>{all.passed} of {all.total} recommendations pass every configured control.</p>
      </section>

      <section className="dp-section outcome-card">
        <div className="dp-title">Expected portfolio outcome</div>
        <div className="dp-metric"><span>Profit impact</span><strong>{fmtPct(result.exec.profitChangeRate)}</strong><small>{fmtMoney(result.exec.totalProfitChange)}</small></div>
        <div className="dp-mini-grid"><div><span>Revenue</span><b>{fmtPct(result.exec.revenueChangeRate)}</b></div><div><span>Avg move</span><b>{fmtPct(result.exec.avgPriceChangeRate)}</b></div></div>
      </section>

      {tab === "sku" && selected ? <section className="dp-section selected-card">
        <div className="dp-title">Selected recommendation</div>
        <div className="selected-price"><span>{selected.itemId}</span><strong>{fmtPrice(selected.currentPrice)} <i>→</i> {fmtPrice(selected.recommendedPrice)}</strong></div>
        <div className="dp-mini-grid"><div><span>Profit</span><b>{fmtPct(selected.profitChangeRate)}</b></div><div><span>Margin</span><b>{fmtPctPlain(selected.optimizedMarginRate)}</b></div></div>
      </section> : null}

      <section className="dp-section">
        <div className="dp-title-row"><div className="dp-title">Context for this page</div><Icon name="spark" size={16}/></div>
        <p className="dp-summary">{context.summary}</p>
        <div className="insight-list">{(context.readings || []).slice(0,3).map((r,i)=><div className={`insight ${r.tone || "neutral"}`} key={i}><span>{r.label}</span><p>{r.text}</p></div>)}</div>
      </section>

      <section className="dp-section">
        <div className="dp-title-row"><div className="dp-title">Priority opportunities</div><button className="link-btn" onClick={()=>onNavigate("sku")}>View all</button></div>
        <div className="priority-list">{top.map((r,i)=><button key={r.itemId} onClick={()=>{onSelectItem(r.itemId);onNavigate("sku")}}><span className="priority-rank">{i+1}</span><span><b>{r.itemId}</b><small>{r.category}</small></span><strong>{fmtMoney(r.profitChange)}</strong></button>)}</div>
      </section>

      <section className="dp-section">
        <div className="dp-title">Active controls</div>
        <div className="control-counts"><div><b>{binding.step}</b><span>step capped</span></div><div><b>{binding.competitor}</b><span>competitor capped</span></div><div><b>{binding.inventory}</b><span>inventory bound</span></div></div>
        <button className="secondary-wide" onClick={onOpenSettings}><Icon name="settings" size={16}/> Review run settings</button>
      </section>
    </div>
    <div className="decision-actions"><button className="secondary-wide" onClick={()=>onNavigate("report")}>Open report</button><button className="primary-wide" onClick={onPdf}><Icon name="download" size={16}/> Export PDF</button></div>
  </aside>;
}
