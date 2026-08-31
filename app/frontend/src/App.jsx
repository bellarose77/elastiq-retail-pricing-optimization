import { useState, useMemo, useEffect } from "react";
import { DEFAULT_CONFIG, DEFAULT_TECHNIQUE, optimizePortfolio, summarizePortfolio } from "./lib/engine.js";
import { applyLiveRecommendations } from "./lib/liveRecommendations.js";
import { fetchLiveProductRecommendations } from "./lib/api.js";
import { DEMO_DATA } from "./lib/demoData.js";
import { TECHNIQUE_MAP } from "./lib/techniques.js";
import { downloadCSV, recommendationsCSV, generatePDF } from "./lib/report.js";
import Sidebar from "./components/Sidebar.jsx";
import DecisionPanel from "./components/DecisionPanel.jsx";
import ConfigDrawer from "./components/ConfigDrawer.jsx";
import Icon from "./components/Icon.jsx";
import PortfolioView from "./components/views/PortfolioView.jsx";
import SkuDetailView from "./components/views/SkuDetailView.jsx";
import SensitivityView from "./components/views/SensitivityView.jsx";
import ValidationView from "./components/views/ValidationView.jsx";
import TechniquesView from "./components/views/TechniquesView.jsx";
import DataView from "./components/views/DataView.jsx";
import ReportView from "./components/views/ReportView.jsx";
import HelpView from "./components/views/HelpView.jsx";
import LiveLabView from "./components/views/LiveLabView.jsx";

const PAGE = {
  portfolio: ["Decision overview", "Understand the financial impact, action mix and most valuable opportunities."],
  sku: ["Recommendations", "Review every price decision and inspect the demand response for one unit."],
  sensitivity: ["Stress test", "Challenge the plan under alternative elasticity and cost assumptions."],
  validation: ["Controls and readiness", "Confirm that each recommendation respects the configured commercial limits."],
  techniques: ["Optimization methods", "Compare alternative pricing logics before choosing the active run."],
  live: ["Test case runner", "Generate Small, Medium or Large data, run all six approaches, and follow the complete pipeline live."],
  data: ["Input data", "Prepare, validate and manage the decision-unit catalog used by the optimizer."],
  report: ["Decision report", "Package the recommendation, expected impact and controls for stakeholder review."],
  help: ["User guide", "Learn the workflow, model assumptions, methods and interpretation rules."],
};

export default function App() {
  const [cfg, setCfg] = useState(DEFAULT_CONFIG);
  const [tech, setTech] = useState(DEFAULT_TECHNIQUE);
  const [items, setItems] = useState(DEMO_DATA);
  const [tab, setTab] = useState("live");
  const [selectedId, setSelectedId] = useState(DEMO_DATA[0].itemId);
  const [settingsOpen, setSettingsOpen] = useState(false);
  // "grid" is the one technique the pricing service also implements (see
  // src/optimization/pricing.py's optimize_price_portfolio) -- for it,
  // the workspace calls the live service instead of running the
  // client-side engine, falling back to the client computation (with a
  // visible note, see the "Live" chip below) if the service is
  // unreachable. The other five techniques have no server equivalent and
  // always run client-side -- see lib/liveRecommendations.js.
  const [liveStatus, setLiveStatus] = useState("idle"); // idle | loading | live | error
  const [liveRecommendations, setLiveRecommendations] = useState(null);
  const [liveError, setLiveError] = useState("");

  useEffect(() => {
    if (tech.id !== "grid") return;
    let cancelled = false;
    setLiveStatus("loading");
    fetchLiveProductRecommendations()
      .then((data) => {
        if (cancelled) return;
        setLiveRecommendations(data);
        setLiveStatus("live");
        setLiveError("");
      })
      .catch((err) => {
        if (cancelled) return;
        setLiveRecommendations(null);
        setLiveStatus("error");
        setLiveError(err.message);
      });
    return () => { cancelled = true; };
  }, [tech.id]);

  const clientResult = useMemo(() => optimizePortfolio(items, cfg, tech), [items, cfg, tech]);
  const result = useMemo(() => {
    if (tech.id !== "grid" || !liveRecommendations) return clientResult;
    const liveByItemId = new Map(liveRecommendations.map((r) => [r.itemId, r]));
    const rows = applyLiveRecommendations(clientResult.rows, liveByItemId, cfg);
    return summarizePortfolio(rows, cfg, tech, clientResult.portfolioInfo);
  }, [clientResult, liveRecommendations, tech, cfg]);
  const meta = TECHNIQUE_MAP[tech.id];
  useEffect(() => { if (!items.find(i => i.itemId === selectedId) && items[0]) setSelectedId(items[0].itemId); }, [items, selectedId]);
  useEffect(() => { const close = e => e.key === "Escape" && setSettingsOpen(false); window.addEventListener("keydown", close); return () => window.removeEventListener("keydown", close); }, []);

  const onPdf = () => generatePDF(items, result, cfg, meta.name).catch((error) => {
    console.error(error);
    alert("Could not generate the PDF report. Please try again.");
  });
  const onCsv = () => downloadCSV(recommendationsCSV(result), "elastiq-price-recommendations.csv");
  const selectedItem = items.find(i => i.itemId === selectedId) || items[0];
  const [title, subtitle] = PAGE[tab];

  return <div className={`app-shell ${tab === "live" ? "lab-shell" : ""}`}>
    <Sidebar tab={tab} setTab={setTab} meta={meta} itemCount={items.length} onOpenSettings={() => setSettingsOpen(true)}/>
    <main className="workspace">
      <header className="workspace-head">
        <div className="page-heading"><div className="page-kicker">Pricing decision workspace</div><h1>{title}</h1><p>{subtitle}</p></div>
        <div className="head-actions">{tab === "live" ? <><div className="run-chip"><span>Execution</span><b>Real browser worker</b></div><button className="btn ghost" onClick={() => setSettingsOpen(true)}><Icon name="settings" size={16}/> Pricing policy</button></> : <><div className="run-chip"><span>Method</span><b>{meta.short}</b></div>{tech.id === "grid" ? <div className={`run-chip${liveStatus === "error" ? " warn" : ""}`} title={liveStatus === "error" ? liveError : "Recommendations for this technique are scored by the pricing service, not computed in the browser."}><span>Source</span><b>{liveStatus === "live" ? "Live pricing service" : liveStatus === "loading" ? "Loading…" : liveStatus === "error" ? "Local fallback" : "Local"}</b></div> : null}<button className="btn ghost" onClick={() => setSettingsOpen(true)}><Icon name="settings" size={16}/> Edit run</button><button className="btn" onClick={onPdf}><Icon name="download" size={16}/> Export</button></>}</div>
      </header>
      <div className="workspace-body">
        {tab === "portfolio" && <PortfolioView result={result} onSelectItem={setSelectedId} onNavigate={setTab}/>} 
        {tab === "sku" && <SkuDetailView items={items} cfg={cfg} result={result} selectedId={selectedId} setSelectedId={setSelectedId}/>}
        {tab === "sensitivity" && <SensitivityView items={items} cfg={cfg} tech={tech}/>} 
        {tab === "validation" && <ValidationView result={result}/>} 
        {tab === "techniques" && <TechniquesView items={items} cfg={cfg} tech={tech} setTech={setTech}/>} 
        {tab === "live" && <LiveLabView cfg={cfg} tech={tech} workspaceItems={items} onUseScenario={(scenario) => { setItems(scenario); setSelectedId(scenario[0]?.itemId); setTab("portfolio"); }}/>} 
        {tab === "data" && <DataView items={items} setItems={setItems}/>} 
        {tab === "report" && <ReportView items={items} result={result} cfg={cfg} techName={meta.name} onPdf={onPdf} onCsv={onCsv}/>} 
        {tab === "help" && <HelpView/>}
      </div>
    </main>
    {tab !== "live" ? <DecisionPanel tab={tab} result={result} cfg={cfg} tech={tech} meta={meta} selectedItem={selectedItem} onSelectItem={setSelectedId} onNavigate={setTab} onOpenSettings={() => setSettingsOpen(true)} onPdf={onPdf}/> : null}
    <ConfigDrawer open={settingsOpen} onClose={() => setSettingsOpen(false)} cfg={cfg} setCfg={setCfg} tech={tech} setTech={setTech} onReset={() => setCfg(DEFAULT_CONFIG)} onResetData={() => {setItems(DEMO_DATA); setSelectedId(DEMO_DATA[0].itemId);}}/>
  </div>;
}
