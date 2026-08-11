import { fmtPctPlain } from "../lib/format.js";
import { TECHNIQUE_LIST, TECHNIQUE_MAP } from "../lib/techniques.js";
import { Segmented, Slider, Toggle, Select } from "./ui.jsx";
import Icon from "./Icon.jsx";

export default function ConfigDrawer({ open, onClose, cfg, setCfg, tech, setTech, onReset, onResetData }) {
  if (!open) return null;
  const set = (k,v)=>setCfg({...cfg,[k]:v});
  const setParam=(k,v)=>setTech({...tech,params:{...tech.params,[k]:v}});
  const meta=TECHNIQUE_MAP[tech.id] || TECHNIQUE_LIST[0];
  return <div className="config-scrim" onClick={onClose}><aside className="config-drawer" onClick={e=>e.stopPropagation()}>
    <div className="config-head"><div><span>Optimization setup</span><h2>Configure decision run</h2><p>Choose the method, objective and commercial limits used for every recommendation.</p></div><button className="icon-btn" onClick={onClose}><Icon name="close"/></button></div>
    <div className="config-body">
      <section className="config-section"><h3>Method</h3><Select value={tech.id} onChange={id=>setTech({...tech,id})} options={TECHNIQUE_LIST.map(t=>({value:t.id,label:t.name}))}/><p className="config-help">{meta.tagline}</p><div className="method-tags"><span>{meta.family}</span>{meta.portfolioCoupled?<span>Portfolio coupled</span>:null}</div></section>
      {meta.objectiveAware?<section className="config-section"><h3>Commercial objective</h3><Segmented options={[{value:"profit",label:"Profit"},{value:"revenue",label:"Revenue"},{value:"quantity",label:"Units"}]} value={cfg.objective} onChange={v=>set("objective",v)}/></section>:null}
      {meta.params?.length?<section className="config-section"><h3>Method parameters</h3>{meta.params.map(p=><Slider key={p.key} label={p.label} value={tech.params[p.key]} min={p.min} max={p.max} step={p.step} onChange={v=>setParam(p.key,v)} format={p.fmt}/>)}</section>:null}
      <section className="config-section"><h3>Price movement</h3><Slider label="Minimum gross margin" value={cfg.minMarginRate} min={0} max={.6} step={.01} onChange={v=>set("minMarginRate",v)} format={fmtPctPlain}/><Slider label="Maximum increase" value={cfg.maxPriceChangeRate} min={.02} max={.5} step={.01} onChange={v=>set("maxPriceChangeRate",v)} format={fmtPctPlain}/><Slider label="Maximum decrease" value={-cfg.minPriceChangeRate} min={.02} max={.5} step={.01} onChange={v=>set("minPriceChangeRate",-v)} format={fmtPctPlain}/><Slider label="Implementation step" value={cfg.maxImplementationStep} min={.02} max={.25} step={.01} onChange={v=>set("maxImplementationStep",v)} format={fmtPctPlain} hint="Caps the change proposed for one execution cycle."/></section>
      <section className="config-section"><h3>Market and promotion</h3><Slider label="Competitor tolerance" value={cfg.competitorPriceTolerance} min={.05} max={.5} step={.01} onChange={v=>set("competitorPriceTolerance",v)} format={x=>"±"+fmtPctPlain(x)}/><Slider label="Promotion uplift threshold" value={cfg.promotionUpliftThreshold} min={0} max={1} step={.05} onChange={v=>set("promotionUpliftThreshold",v)} format={fmtPctPlain}/><Slider label="Promotion cost rate" value={cfg.promotionCostRate} min={0} max={.15} step={.005} onChange={v=>set("promotionCostRate",v)} format={x=>fmtPctPlain(x,1)}/><Slider label="Market-signal adjustment" value={cfg.ragDemandAdjustmentLimit} min={0} max={.3} step={.01} onChange={v=>set("ragDemandAdjustmentLimit",v)} format={fmtPctPlain}/></section>
      <section className="config-section"><h3>Advanced pricing</h3><Toggle label="Round to psychological ending ($x.99)" checked={cfg.priceEnding!=null} onChange={v=>set("priceEnding",v?0.99:null)} hint="Rounds the recommended price, then re-checks it against every guardrail above."/><Slider label="Category cannibalization" value={cfg.crossPriceElasticity} min={0} max={.5} step={.01} onChange={v=>set("crossPriceElasticity",v)} format={fmtPctPlain} hint="How much a category sibling's price move shifts this item's expected demand."/></section>
      <section className="config-section"><Toggle label="Require causal elasticity" checked={cfg.requireCausalElasticity} onChange={v=>set("requireCausalElasticity",v)} hint="Hold manual, default, or correlational elasticities for review instead of proposing a price move."/><Toggle label="Enforce non-negative profit" checked={cfg.enforceNonnegativeProfit} onChange={v=>set("enforceNonnegativeProfit",v)} hint="Reject candidates with negative expected unit profit."/></section>
    </div>
    <div className="config-foot"><div><button className="text-btn" onClick={onReset}>Reset settings</button><button className="text-btn" onClick={onResetData}>Reset demo data</button></div><button className="btn" onClick={onClose}>Apply and close</button></div>
  </aside></div>;
}
