import { useEffect, useMemo, useRef, useState } from "react";
import { generateScenario, PROBLEM_SIZES, randomSeed, SCENARIO_PROFILES } from "../../lib/scenarioGenerator.js";
import { analyzeInput } from "../../lib/inputAnalysis.js";
import { TECHNIQUE_LIST, TECHNIQUE_MAP } from "../../lib/techniques.js";
import { ANALYSIS_DEPTHS, analysisPlan } from "../../lib/deepAnalysis.js";
import { fmtMoney, fmtNum, fmtPct, fmtPctPlain, fmtPrice, ACTION_LABEL } from "../../lib/format.js";
import { Segmented } from "../ui.jsx";
import Icon from "../Icon.jsx";
import { generateDeepAnalysisPDF } from "../../lib/report.js";

const ALL_METHODS = TECHNIQUE_LIST.map((method) => method.id);
const PRESET_SIZES = PROBLEM_SIZES.filter((item) => item.size);
const nowLabel = () => new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
const fmtDuration = (ms = 0) => ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${ms.toFixed(1)} ms`;
const fmtWork = (value = 0) => Number(value).toLocaleString();
const makeDataset = ({ size, profile, seed = randomSeed(), sizeId }) => {
  const started = performance.now();
  const rows = generateScenario({ size, seed, profile });
  return { rows, seed, profile, sizeId, generationMs: performance.now() - started, createdAt: new Date().toLocaleTimeString() };
};

function Pipeline({ stage, completedMethods, totalMethods, activeMethod, result, runPlan }) {
  const order = ["data", "refine", "stress", "simulate", "validate", "finish"];
  const activeIndex = stage === "refinement" ? 1 : stage === "stress" ? 2 : stage === "simulation" ? 3 : stage === "validation" || stage === "hybrid" ? 4 : stage === "complete" ? 5 : 1;
  const steps = [
    ["data", "Start + profile", "Generate, inspect and freeze input"],
    ["refine", "Optimize + refine", totalMethods ? `${completedMethods}/${totalMethods} methods · ${runPlan?.refinementPasses || "–"} resolutions` : "Waiting to execute"],
    ["stress", "Stress test", runPlan ? `${runPlan.stressScenarios} re-optimization scenarios` : "Demand, cost and competitor shocks"],
    ["simulate", "Simulate risk", runPlan ? `${fmtWork(runPlan.monteCarloDraws)} Monte Carlo draws` : "Uncertainty distribution"],
    ["validate", "Validate", "Consensus, controls and hybrid routing"],
    ["finish", "Finish", result ? `${result.finalSummary.itemCount.toLocaleString()} decisions ready` : "Charts and report"],
  ];
  return <div className="pipeline-visual">
    {steps.map(([id, label, note], index) => {
      const state = stage === "ready" ? (index < 1 ? "done" : "pending") : index < activeIndex ? "done" : index === activeIndex ? (stage === "complete" ? "done" : "active") : "pending";
      return <div className={`pipeline-step ${state}`} key={id}>
        <div className="pipeline-node">{state === "done" ? "✓" : index + 1}</div>
        <div><b>{label}</b><span>{id === "refine" && activeMethod ? `Running ${TECHNIQUE_MAP[activeMethod]?.short || activeMethod}` : note}</span></div>
        {index < order.length - 1 ? <i/> : null}
      </div>;
    })}
  </div>;
}

export default function LiveLabView({ cfg, tech, workspaceItems = [], onUseScenario }) {
  const [sizeId, setSizeId] = useState("small");
  const [customSize, setCustomSize] = useState(750);
  const [profile, setProfile] = useState("balanced");
  const [source, setSource] = useState("generated");
  const [freshEachRun, setFreshEachRun] = useState(true);
  const [approach, setApproach] = useState("compare");
  const [depth, setDepth] = useState("deep");
  const [methods, setMethods] = useState(ALL_METHODS);
  const [dataset, setDataset] = useState(() => makeDataset({ size: 50, profile: "balanced", sizeId: "small" }));
  const [status, setStatus] = useState("ready");
  const [stage, setStage] = useState("ready");
  const [progress, setProgress] = useState(0);
  const [activeMethod, setActiveMethod] = useState("");
  const [completedMethods, setCompletedMethods] = useState(0);
  const [logs, setLogs] = useState([`${nowLabel()}  Small random test case is ready. Choose Run all six approaches.`]);
  const [result, setResult] = useState(null);
  const [liveSummaries, setLiveSummaries] = useState([]);
  const [runContext, setRunContext] = useState(null);
  const [liveMetrics, setLiveMetrics] = useState({});
  const [history, setHistory] = useState([]);
  const workerRef = useRef(null);
  const runIdRef = useRef(0);
  const runStartedRef = useRef(0);

  const chosenSize = sizeId === "custom" ? Math.max(5, Math.min(2000, Number(customSize) || 5)) : PRESET_SIZES.find((item) => item.id === sizeId)?.size || 50;
  const activeRows = source === "workspace" ? workspaceItems : dataset.rows;
  const input = useMemo(() => analyzeInput(activeRows), [activeRows]);

  useEffect(() => () => workerRef.current?.terminate(), []);

  const addLog = (message) => setLogs((current) => [...current.slice(-19), `${nowLabel()}  ${message}`]);
  const resetResults = () => {
    setResult(null); setLiveSummaries([]); setProgress(0); setCompletedMethods(0); setActiveMethod(""); setLiveMetrics({}); setStatus("ready"); setStage("ready");
  };
  const regenerate = ({ nextSizeId = sizeId, nextProfile = profile, nextSize } = {}) => {
    const size = nextSize ?? (nextSizeId === "custom" ? Math.max(5, Math.min(2000, Number(customSize) || 5)) : PRESET_SIZES.find((item) => item.id === nextSizeId)?.size || 50);
    const next = makeDataset({ size, profile: nextProfile, sizeId: nextSizeId });
    setDataset(next); setSource("generated"); resetResults();
    setLogs([`${nowLabel()}  Generated a new random ${nextSizeId} test: ${size.toLocaleString()} rows, ${nextProfile} profile, seed ${next.seed}.`]);
    return next;
  };
  const chooseSize = (id) => {
    setSizeId(id);
    if (id !== "custom") regenerate({ nextSizeId: id, nextProfile: profile });
  };
  const chooseProfile = (id) => { setProfile(id); regenerate({ nextSizeId: sizeId, nextProfile: id }); };
  const toggleMethod = (id) => setMethods((current) => current.includes(id) ? (current.length > 1 ? current.filter((item) => item !== id) : current) : [...current, id]);

  const execute = (requestedApproach = approach, requestedMethods = methods) => {
    const endToEndStarted = performance.now();
    workerRef.current?.terminate();
    let runDataset = source === "generated" ? dataset : { rows: workspaceItems, seed: null, profile: "workspace", sizeId: "workspace", generationMs: 0, createdAt: new Date().toLocaleTimeString() };
    if (source === "generated" && freshEachRun) {
      runDataset = makeDataset({ size: chosenSize, profile, sizeId });
      setDataset(runDataset);
    }
    if (!runDataset.rows.length) { setLogs([`${nowLabel()}  No input rows are available.`]); return; }
    const techniqueIds = requestedApproach === "single" ? [requestedMethods[0] || tech.id] : requestedMethods;
    const worker = new Worker(new URL("../../workers/experimentWorker.js", import.meta.url), { type: "module" });
    workerRef.current = worker;
    const runId = ++runIdRef.current;
    runStartedRef.current = endToEndStarted;
    const frozenInput = analyzeInput(runDataset.rows);
    const plannedAnalysis = analysisPlan(depth, runDataset.rows.length, techniqueIds.length);
    const context = {
      source, size: runDataset.rows.length, sizeId: runDataset.sizeId, profile: runDataset.profile, seed: runDataset.seed,
      generationMs: runDataset.generationMs, createdAt: runDataset.createdAt, approach: requestedApproach, techniqueIds,
      objective: cfg.objective, minMarginRate: cfg.minMarginRate, minPriceChangeRate: cfg.minPriceChangeRate,
      maxPriceChangeRate: cfg.maxPriceChangeRate, maxImplementationStep: cfg.maxImplementationStep,
      competitorPriceTolerance: cfg.competitorPriceTolerance, promotionUpliftThreshold: cfg.promotionUpliftThreshold,
      requireCausalElasticity: cfg.requireCausalElasticity, inputAnalysis: frozenInput, depth, analysisPlan: plannedAnalysis,
    };
    setRunContext(context); setStatus("running"); setStage("refinement"); setProgress(8); setResult(null); setLiveSummaries([]); setCompletedMethods(0); setActiveMethod(""); setLiveMetrics({});
    setLogs([
      `${nowLabel()}  START — ${context.size.toLocaleString()} ${context.sizeId} rows; ${context.profile} profile${context.seed ? `; seed ${context.seed}` : ""}.`,
      `${nowLabel()}  Input generated in ${context.generationMs.toFixed(2)} ms and profiled before execution.`,
      `${nowLabel()}  ${plannedAnalysis.label}: ${plannedAnalysis.refinementPasses} search resolutions, ${plannedAnalysis.stressScenarios} stress scenarios and ${fmtWork(plannedAnalysis.monteCarloDraws)} uncertainty draws.`,
    ]);
    worker.onmessage = ({ data }) => {
      if (data.runId !== runIdRef.current) return;
      if (data.metrics) setLiveMetrics(data.metrics);
      if (data.type === "started") addLog(`Execution worker accepted ${data.itemCount.toLocaleString()} units and planned ${fmtWork(data.plan.plannedOptimizerRuns)} optimizer runs.`);
      if (data.type === "refinementStarted") addLog(`Multi-resolution optimization started at ${data.plan.refinementSteps.map((step) => `${(step * 100).toFixed(3)}%`).join(", ")} price steps.`);
      if (data.type === "refinementProgress") {
        const unit = data.passIndex * data.totalMethods + data.methodIndex;
        const totalUnits = data.passes * data.totalMethods;
        setStage("refinement"); setActiveMethod(data.techniqueId); setProgress(10 + (unit / totalUnits) * 25);
        if (data.methodIndex === 0) addLog(`Refinement pass ${data.passIndex + 1}/${data.passes} started with ${(data.step * 100).toFixed(3)}% price increments.`);
      }
      if (data.type === "methodCompleted") {
        setCompletedMethods(data.index + 1); setProgress(35);
        setLiveSummaries((current) => [...current.filter((summary) => summary.id !== data.summary.id), data.summary]);
        addLog(`${data.summary.name} final refined solution ready; ${fmtPctPlain(data.summary.passRate)} controls passed.`);
      }
      if (data.type === "refinementCompleted") { setStage("stress"); setActiveMethod(""); setProgress(37); addLog(`Refinement completed after ${data.history.length} full resolution passes.`); }
      if (data.type === "stressStarted") addLog(`Stress re-optimization started across ${data.total} independently shocked market states.`);
      if (data.type === "stressProgress") {
        setStage("stress"); setProgress(37 + (data.completed / data.total) * 28);
        if (data.completed === 1 || data.completed % Math.max(1, Math.floor(data.total / 10)) === 0 || data.completed === data.total) addLog(`Stress scenario ${data.completed}/${data.total} completed.`);
      }
      if (data.type === "stressCompleted") { setStage("simulation"); setProgress(67); addLog("Stress distributions complete; uncertainty simulation is next."); }
      if (data.type === "monteCarloStarted") addLog(`Monte Carlo policy simulation started: ${fmtWork(data.total)} draws per method across all units.`);
      if (data.type === "monteCarloProgress") {
        setStage("simulation"); setProgress(67 + (data.completed / data.total) * 18);
        if (data.completed === data.total || data.completed % Math.max(1, Math.floor(data.total / 5)) === 0) addLog(`Uncertainty simulation ${fmtWork(data.completed)}/${fmtWork(data.total)} draws.`);
      }
      if (data.type === "monteCarloCompleted") { setProgress(86); addLog("Risk percentiles, volatility and probability of positive uplift computed."); }
      if (data.type === "hybridStarted") { setStage("hybrid"); setActiveMethod("hybrid"); setProgress(88); addLog("Hybrid routing started across all feasible item-level method results."); }
      if (data.type === "hybridCompleted") {
        setLiveSummaries((current) => [...current.filter((summary) => summary.id !== "hybrid"), data.summary]);
        addLog(`Hybrid routing finished with ${Object.keys(data.summary.methodMix).length} contributing methods.`);
      }
      if (data.type === "validationStarted") { setStage("validation"); setActiveMethod(""); setProgress(92); addLog("Independent controls, method consensus and result validation started."); }
      if (data.type === "validationCompleted") { setProgress(97); addLog(`Validation finished: ${data.passed}/${data.total} decisions passed; ${fmtPctPlain(data.consensus.unanimousRate)} action consensus.`); }
      if (data.type === "completed") {
        const wallElapsedMs = performance.now() - runStartedRef.current;
        const completedResult = { ...data.result, inputAnalysis: context.inputAnalysis, inputRows: runDataset.rows, wallElapsedMs };
        setResult(completedResult); setStatus("complete"); setStage("complete"); setProgress(100); setActiveMethod("");
        addLog(`FINISH — ${fmtWork(data.result.analysis.metrics.candidateEvaluations)} candidate evaluations and ${fmtWork(data.result.analysis.metrics.portfolioOptimizations)} optimizer runs completed in ${fmtDuration(wallElapsedMs)}.`);
        const final = data.result.finalSummary;
        const record = { id: runId, at: new Date().toLocaleString(), size: context.size, sizeId: context.sizeId, profile: context.profile, approach: context.approach, methods: techniqueIds.length, elapsedMs: wallElapsedMs, profitChangeRate: final.profitChangeRate, passRate: final.passRate, result: completedResult, context };
        setHistory((current) => [record, ...current].slice(0, 10));
        worker.terminate(); workerRef.current = null;
      }
      if (data.type === "failed") { setStatus("failed"); setStage("validation"); setActiveMethod(""); addLog(`Run failed: ${data.error}`); worker.terminate(); workerRef.current = null; }
    };
    worker.onerror = (error) => { setStatus("failed"); addLog(`Worker error: ${error.message}`); worker.terminate(); workerRef.current = null; };
    worker.postMessage({ type: "run", runId, items: runDataset.rows, cfg, techniqueIds, params: tech.params, mode: requestedApproach, depth, seed: runDataset.seed || Date.now() });
  };

  const cancel = () => {
    workerRef.current?.terminate(); workerRef.current = null; runIdRef.current += 1;
    setStatus("cancelled"); setActiveMethod(""); addLog("Run cancelled by the user.");
  };
  const final = result?.finalSummary;
  const shownSummaries = result ? [...result.methodSummaries, ...(result.hybridSummary ? [result.hybridSummary] : [])] : liveSummaries;
  const maxUplift = Math.max(...shownSummaries.map((summary) => Math.abs(summary.profitChangeRate)), .001);
  const maxRuntime = Math.max(...shownSummaries.map((summary) => summary.elapsedMs), 1);
  const maxMix = final ? Math.max(...Object.values(final.methodMix), 1) : 1;
  const best = shownSummaries.length ? [...shownSummaries].sort((a, b) => b.profitChangeRate - a.profitChangeRate)[0] : null;
  const fastest = shownSummaries.length ? [...shownSummaries].filter((item) => item.id !== "hybrid").sort((a, b) => a.elapsedMs - b.elapsedMs)[0] : null;
  const currentPlan = runContext?.analysisPlan || analysisPlan(depth, activeRows.length, methods.length);
  const analysis = result?.analysis;

  const openHistory = (record) => {
    if (record.context.source === "generated" && record.result.inputRows) {
      setSource("generated"); setSizeId(record.sizeId); setProfile(record.profile);
      setDataset({ rows: record.result.inputRows, seed: record.context.seed, profile: record.profile, sizeId: record.sizeId, generationMs: record.context.generationMs, createdAt: record.context.createdAt });
    }
    setResult(record.result); setRunContext(record.context); setStatus("complete"); setStage("complete"); setProgress(100); setCompletedMethods(record.methods);
    setDepth(record.context.depth || "deep"); setLiveMetrics(record.result.analysis?.metrics || {});
    setLiveSummaries([...(record.result.methodSummaries || []), ...(record.result.hybridSummary ? [record.result.hybridSummary] : [])]);
    setLogs([`${nowLabel()}  Reopened run #${record.id}: ${record.sizeId}, ${record.profile}, ${record.approach}.`]);
  };
  const downloadRun = () => {
    if (!result) return;
    const payload = JSON.stringify({ generatedAt: new Date().toISOString(), configuration: runContext, inputAnalysis: result.inputAnalysis, result }, null, 2);
    const url = URL.createObjectURL(new Blob([payload], { type: "application/json" }));
    const link = document.createElement("a"); link.href = url; link.download = `elastiq-${runContext.sizeId}-${Date.now()}.json`; link.click(); URL.revokeObjectURL(url);
  };
  const downloadRunPDF = () => generateDeepAnalysisPDF(result, runContext).catch((error) => addLog(`PDF export failed: ${error.message}`));

  return <div className="stack test-runner">
    <section className="runner-hero">
      <div><span className="runner-eyebrow">DEEP ANALYSIS RUNNER · BUILD 1.2.0</span><h2>Run a full pricing experiment—not a one-pass demo</h2><p>Every run performs multi-resolution optimization, market-shock re-optimization, Monte Carlo policy simulation, method consensus and independent control validation.</p></div>
      <div className={`runner-state ${status}`}><span className="live-pulse"/><b>{status === "running" ? "RUNNING NOW" : status.toUpperCase()}</b><small>{status === "complete" ? "Results and charts ready" : `${activeRows.length.toLocaleString()} input rows selected`}</small></div>
    </section>

    <section className="panel pipeline-panel">
      <div className="panel-head"><div><h3>Run pipeline</h3><p className="panel-sub">The highlighted step shows exactly where the current run starts, progresses, and finishes.</p></div><span className="mono lab-progress-label">{Math.round(progress)}%</span></div>
      <Pipeline stage={stage} completedMethods={completedMethods} totalMethods={runContext?.techniqueIds.length || methods.length} activeMethod={activeMethod} result={result} runPlan={currentPlan}/>
      <div className="lab-progress"><i style={{ width: `${progress}%` }}/></div>
      <div className="live-workload"><div><span>Optimizer runs</span><b>{fmtWork(liveMetrics.portfolioOptimizations || 0)}</b><small>{fmtWork(currentPlan.plannedOptimizerRuns)} planned</small></div><div><span>Candidate evaluations</span><b>{fmtWork(liveMetrics.candidateEvaluations || 0)}</b><small>Exact engine counter</small></div><div><span>Demand / profit evaluations</span><b>{fmtWork((liveMetrics.demandEvaluations || 0) + (liveMetrics.profitEvaluations || 0))}</b><small>Uncertainty-aware scoring</small></div><div><span>Policy simulations</span><b>{status === "complete" ? fmtWork(currentPlan.plannedPolicySimulations) : "Live"}</b><small>{fmtWork(currentPlan.monteCarloDraws)} draws per method</small></div></div>
    </section>

    <div className="runner-grid">
      <section className="panel runner-control-panel">
        <div className="panel-head"><div><span className="step-badge">STEP 1</span><h3>Select input data</h3><p className="panel-sub">Pick a generated test size or use the rows already loaded in the workspace.</p></div></div>
        <div className="source-switch"><Segmented value={source} onChange={(value) => { setSource(value); resetResults(); }} options={[{ value: "generated", label: "Random test data" }, { value: "workspace", label: `Current data (${workspaceItems.length})` }]}/></div>
        {source === "generated" ? <>
          <div className="test-size-cards">{PRESET_SIZES.map((item) => <button key={item.id} className={sizeId === item.id ? "selected" : ""} onClick={() => chooseSize(item.id)} disabled={status === "running"}><span>{item.label}</span><strong>{item.size.toLocaleString()}</strong><small>decision units</small><em>{item.note}</em></button>)}</div>
          <div className="custom-row"><button className={sizeId === "custom" ? "custom-selected" : ""} onClick={() => setSizeId("custom")}>Custom size</button><input type="number" min="5" max="2000" value={customSize} onChange={(event) => setCustomSize(event.target.value)}/><button className="btn ghost sm" onClick={() => regenerate({ nextSizeId: "custom", nextSize: Math.max(5, Math.min(2000, Number(customSize) || 5)) })}>Generate</button></div>
          <div className="profile-label">Test data pattern</div>
          <div className="profile-cards">{SCENARIO_PROFILES.map((item) => <button key={item.id} className={profile === item.id ? "selected" : ""} onClick={() => chooseProfile(item.id)} disabled={status === "running"}><b>{item.label}</b><small>{item.note}</small></button>)}</div>
          <div className="dataset-actions"><button className="btn ghost" onClick={() => regenerate()} disabled={status === "running"}><Icon name="spark" size={15}/> Generate new random data</button><label><input type="checkbox" checked={freshEachRun} onChange={(event) => setFreshEachRun(event.target.checked)}/> Fresh random data for every run</label></div>
        </> : <div className="workspace-source-note"><b>{workspaceItems.length.toLocaleString()} current decision units</b><p>This option runs the exact dataset from the Data page. Return to Random test data to generate Small, Medium, or Large cases.</p></div>}
      </section>

      <section className="panel runner-data-panel">
        <div className="panel-head"><div><span className="step-badge">STEP 2</span><h3>Inspect generated input</h3><p className="panel-sub">This is the exact dataset that the next run will use.</p></div><span className="dataset-seed">{source === "generated" ? `Seed ${dataset.seed}` : "Workspace data"}</span></div>
        <div className="input-profile-grid runner-profile">
          <div><span>Rows</span><b>{fmtNum(input.rows)}</b><small>{input.categoryCount} categories / {input.storeCount} stores</small></div>
          <div><span>Price</span><b>{fmtPrice(input.avgPrice)}</b><small>{fmtPrice(input.priceRange[0])}–{fmtPrice(input.priceRange[1])}</small></div>
          <div><span>Demand</span><b>{fmtNum(input.totalDemand)}</b><small>Median {fmtNum(input.demandP50, 1)}</small></div>
          <div><span>Margin</span><b>{fmtPctPlain(input.avgMargin, 1)}</b><small>Average current margin</small></div>
          <div><span>Causal evidence</span><b>{fmtPctPlain(input.causalRate)}</b><small>Rows ready for pricing</small></div>
          <div><span>Inventory pressure</span><b>{fmtPctPlain(input.inventoryRiskRate)}</b><small>Potentially capacity-bound</small></div>
          <div><span>Completeness</span><b>{fmtPctPlain(input.completenessRate)}</b><small>{input.invalidRows} invalid / {input.duplicateIds} duplicate</small></div>
          <div><span>Elasticity</span><b>{input.elasticityP50.toFixed(2)}</b><small>Median demand response</small></div>
        </div>
        <div className="tbl-scroll preview-table"><table className="tbl compact"><thead><tr><th>Unit</th><th>Category</th><th className="r">Price</th><th className="r">Cost</th><th className="r">Demand</th><th className="r">Inventory</th><th className="r">Elasticity</th></tr></thead><tbody>{activeRows.slice(0, 6).map((row) => <tr key={row.itemId}><td className="mono strong">{row.itemId}</td><td>{row.category}</td><td className="r mono">{fmtPrice(row.currentPrice)}</td><td className="r mono">{fmtPrice(row.unitCost)}</td><td className="r mono">{fmtNum(row.baselineQuantity, 1)}</td><td className="r mono">{fmtNum(row.inventory, 1)}</td><td className="r mono">{Number(row.elasticity).toFixed(2)}</td></tr>)}</tbody></table></div>
      </section>
    </div>

    <section className="panel approach-panel">
      <div className="panel-head"><div><span className="step-badge">STEP 3</span><h3>Run and compare approaches</h3><p className="panel-sub">All six approaches are selected by default and receive the same input rows and pricing policy.</p></div><div className="method-tools"><button onClick={() => setMethods(ALL_METHODS)}>Select all</button><span>{methods.length}/6 selected</span></div></div>
      <div className="approach-body">
        <div className="method-select-grid">{TECHNIQUE_LIST.map((method, index) => <button key={method.id} className={methods.includes(method.id) ? "selected" : ""} onClick={() => toggleMethod(method.id)} disabled={status === "running"}><span>{index + 1}</span><div><b>{method.short}</b><small>{method.family}</small></div><i>{methods.includes(method.id) ? "✓" : ""}</i></button>)}</div>
        <div className="depth-selector"><div><b>Analysis depth</b><span>Choose actual computational work. Deep analysis is the default.</span></div>{ANALYSIS_DEPTHS.map((item) => <button key={item.id} className={depth === item.id ? "selected" : ""} onClick={() => setDepth(item.id)} disabled={status === "running"}><b>{item.label}</b><small>{item.note}</small></button>)}</div>
        <div className="run-options"><Segmented value={approach} onChange={setApproach} options={[{ value: "compare", label: "Compare methods" }, { value: "hybrid", label: "Hybrid portfolio" }, { value: "single", label: "Single method" }]}/><p>{approach === "hybrid" ? "Runs every selected method, then routes each feasible item using inventory, uncertainty, promotion and objective context." : approach === "single" ? "Runs only the first selected method." : "Runs selected methods independently and chooses a portfolio champion."}</p></div>
        <div className="primary-run-actions"><button className="btn run-all" onClick={() => execute("compare", ALL_METHODS)} disabled={status === "running"}><Icon name="play" size={17}/> Run all six approaches</button><button className="btn hybrid-run" onClick={() => execute("hybrid", ALL_METHODS)} disabled={status === "running"}><Icon name="spark" size={16}/> Run all + hybrid</button><button className="btn ghost" onClick={() => execute(approach, methods)} disabled={status === "running"}>Run selected</button>{status === "running" ? <button className="btn ghost cancel-run" onClick={cancel}>Cancel</button> : null}</div>
      </div>
    </section>

    <div className="runner-grid results-monitor-grid">
      <section className="panel">
        <div className="panel-head"><div><span className="step-badge">STEP 4</span><h3>Live execution process</h3><p className="panel-sub">Watch every real refinement, stress, simulation and validation stage as it executes.</p></div><span className={`live-status ${status}`}>{status}</span></div>
        <div className="live-log runner-log">{logs.map((line, index) => <div key={`${index}-${line}`}>{line}</div>)}</div>
      </section>
      <section className="panel">
        <div className="panel-head"><div><h3>Live comparison chart</h3><p className="panel-sub">Profit uplift and measured computation time update after every method.</p></div><span className="live-counter">{shownSummaries.length} complete</span></div>
        {shownSummaries.length ? <div className="dual-live-chart">{shownSummaries.map((summary) => <div className="dual-chart-row" key={summary.id}><div><b>{summary.name}</b><small>{fmtPct(summary.profitChangeRate)} profit</small></div><div className="dual-bars"><span><i className="uplift" style={{ width: `${Math.max(3, Math.abs(summary.profitChangeRate) / maxUplift * 100)}%` }}/><em>{fmtPct(summary.profitChangeRate)}</em></span><span><i className="runtime" style={{ width: `${Math.max(3, summary.elapsedMs / maxRuntime * 100)}%` }}/><em>{summary.elapsedMs.toFixed(1)} ms</em></span></div></div>)}</div> : <div className="empty-chart"><Icon name="chart" size={28}/><b>Results appear here during the run</b><span>Start all six approaches to compare normalized uplift and runtime.</span></div>}
      </section>
    </div>

    {final ? <>
      <section className="results-heading"><div><span className="step-badge complete">STEP 5 — FINISH</span><h2>Run results and deep-analysis evidence</h2><p>The full path, workload counters, uncertainty results and validated decisions are retained below.</p></div><div className="btn-group"><button className="btn ghost" onClick={downloadRun}><Icon name="download" size={15}/> Download JSON</button><button className="btn" onClick={downloadRunPDF}><Icon name="download" size={15}/> Download analysis PDF</button></div></section>
      <div className="metric-grid">
        <div className="metric-card"><span>Test case</span><strong>{runContext.sizeId}</strong><small>{runContext.size.toLocaleString()} {runContext.source === "generated" ? "random" : "workspace"} decision units</small></div>
        <div className="metric-card"><span>End-to-end time</span><strong>{fmtDuration(result.wallElapsedMs)}</strong><small>{analysis.depth.label}; real computation, no delay</small></div>
        <div className="metric-card"><span>Candidate evaluations</span><strong>{fmtWork(analysis.metrics.candidateEvaluations)}</strong><small>{fmtWork(analysis.metrics.portfolioOptimizations)} portfolio optimizer runs</small></div>
        <div className="metric-card"><span>Policy simulations</span><strong>{fmtWork(analysis.depth.plannedPolicySimulations)}</strong><small>{fmtWork(analysis.depth.monteCarloDraws)} draws × {result.methodSummaries.length} methods × {runContext.size} units</small></div>
      </div>

      <section className="panel workload-evidence"><div className="panel-head"><div><h3>What the processor actually completed</h3><p className="panel-sub">Exact counters are instrumented inside the optimization engine; scenario and simulation counts come from the frozen run plan.</p></div><span className="tag">No artificial wait</span></div><div className="workload-grid"><div><span>Search resolutions</span><b>{analysis.depth.refinementPasses}</b><small>{analysis.depth.refinementSteps.map((step) => `${(step * 100).toFixed(3)}%`).join(" → ")}</small></div><div><span>Stress re-optimizations</span><b>{analysis.depth.stressScenarios}</b><small>Demand, cost, elasticity, inventory and competitor shocks</small></div><div><span>Candidate evaluations</span><b>{fmtWork(analysis.metrics.candidateEvaluations)}</b><small>Prices and promotions scored inside the engine</small></div><div><span>Demand evaluations</span><b>{fmtWork(analysis.metrics.demandEvaluations)}</b><small>Robust and Bayesian uncertainty scoring</small></div><div><span>Capacity iterations</span><b>{fmtWork(analysis.metrics.capacitySolverIterations)}</b><small>Lagrangian bid-price solver iterations</small></div><div><span>Validation checks</span><b>{fmtWork(analysis.validationChecks)}</b><small>Seven independent controls per selected decision</small></div></div></section>

      <div className="runner-grid analysis-summary-grid">
        <section className="panel analysis-narrative"><div className="panel-head"><div><h3>Automatic result analysis</h3><p className="panel-sub">A plain-language interpretation of the deterministic and uncertainty results.</p></div></div><div className="analysis-copy"><p><b>{best.name}</b> produced the strongest expected profit rate at <b>{fmtPct(best.profitChangeRate)}</b>. The complete deep-analysis pipeline took <b>{fmtDuration(result.wallElapsedMs)}</b>.</p><p>The run completed <b>{fmtWork(analysis.depth.stressScenarios)}</b> stressed market re-optimizations and <b>{fmtWork(analysis.depth.monteCarloDraws)}</b> uncertainty draws per method. The champion's simulated probability of non-negative profit uplift is <b>{fmtPctPlain(analysis.risk[result.championId]?.positiveRate || 0)}</b>.</p><p>Methods agreed unanimously on the action for <b>{fmtPctPlain(analysis.consensus.unanimousRate)}</b> of units; <b>{analysis.consensus.disputedItems}</b> units deserve closer review. {result.hybridSummary ? `The hybrid combined ${Object.keys(final.methodMix).length} methods at item level.` : `The selected portfolio champion is ${TECHNIQUE_MAP[result.championId]?.short}.`}</p></div></section>
        <section className="panel"><div className="panel-head"><div><h3>Frozen run configuration</h3><p className="panel-sub">What started this run and what finished it.</p></div></div><div className="frozen-config"><div><span>Input</span><b>{runContext.sizeId} / {runContext.profile}</b><small>{runContext.seed ? `Random seed ${runContext.seed}` : "Current workspace data"}</small></div><div><span>Methods</span><b>{runContext.techniqueIds.length} approaches</b><small>{runContext.techniqueIds.map((id) => TECHNIQUE_MAP[id]?.short).join(", ")}</small></div><div><span>Analysis depth</span><b>{analysis.depth.label}</b><small>{analysis.depth.refinementPasses} refinements / {analysis.depth.stressScenarios} stress states</small></div><div><span>Outcome</span><b>{final.itemCount} decisions</b><small>Finished in {fmtDuration(result.wallElapsedMs)}</small></div></div></section>
      </div>

      <section className="panel"><div className="panel-head"><div><h3>Uncertainty and stress results by method</h3><p className="panel-sub">P05 is the downside simulation result; P50 is the median; positive probability is the share of draws with non-negative profit uplift.</p></div><span className="tag">{fmtWork(analysis.depth.monteCarloDraws)} draws / method</span></div><div className="tbl-scroll"><table className="tbl"><thead><tr><th>Approach</th><th className="r">Stress mean</th><th className="r">Stress downside</th><th className="r">Simulation P05</th><th className="r">Simulation P50</th><th className="r">Simulation P95</th><th className="r">Positive probability</th></tr></thead><tbody>{result.methodSummaries.map((summary) => { const stress = analysis.stress.summary[summary.id]; const risk = analysis.risk[summary.id]; return <tr key={summary.id}><td className="strong">{summary.name}</td><td className="r mono">{fmtPct(stress.mean)}</td><td className="r mono">{fmtPct(stress.p05)}</td><td className="r mono">{fmtPct(risk.p05)}</td><td className="r mono">{fmtPct(risk.median)}</td><td className="r mono">{fmtPct(risk.p95)}</td><td className="r mono">{fmtPctPlain(risk.positiveRate)}</td></tr>; })}</tbody></table></div></section>

      <section className="panel"><div className="panel-head"><div><h3>Full method comparison</h3><p className="panel-sub">Normalized business result, speed, movement and validation for the same test case.</p></div><span className="tag">Champion: {TECHNIQUE_MAP[result.championId]?.short}</span></div><div className="tbl-scroll"><table className="tbl"><thead><tr><th>Approach</th><th className="r">Runtime</th><th className="r">Profit uplift</th><th className="r">Revenue uplift</th><th className="r">Avg price move</th><th className="r">Controls</th><th className="r">Decisions</th></tr></thead><tbody>{shownSummaries.map((summary) => <tr key={summary.id} className={summary.id === "hybrid" ? "row-rec" : ""}><td className="strong">{summary.name}{summary.id === "hybrid" ? <small className="block-muted">Context-aware item routing</small> : null}</td><td className="r mono">{summary.elapsedMs.toFixed(2)} ms</td><td className="r mono money-positive">{fmtPct(summary.profitChangeRate)}</td><td className="r mono">{fmtPct(summary.revenueChangeRate)}</td><td className="r mono">{fmtPct(summary.avgPriceChangeRate)}</td><td className="r mono">{fmtPctPlain(summary.passRate)}</td><td className="r mono">{summary.itemCount}</td></tr>)}</tbody></table></div></section>

      {result.hybridSummary ? <section className="panel"><div className="panel-head"><div><h3>Hybrid method contribution</h3><p className="panel-sub">How the final portfolio used different approaches across operating conditions.</p></div></div><div className="mix-bars">{Object.entries(final.methodMix).sort((a, b) => b[1] - a[1]).map(([id, count]) => <div key={id}><span>{TECHNIQUE_MAP[id]?.short || id}</span><div><i style={{ width: `${count / maxMix * 100}%` }}/></div><b>{count} <small>{fmtPctPlain(count / final.itemCount)}</small></b></div>)}</div></section> : null}

      <section className="panel"><div className="panel-head"><div><h3>Top decisions from this run</h3><p className="panel-sub">Highest expected profit contributions, including the method and hybrid routing reason.</p></div><button className="btn ghost sm" onClick={() => onUseScenario(result.inputRows)}>Open dataset in decision workspace</button></div><div className="tbl-scroll"><table className="tbl compact"><thead><tr><th>Unit</th><th>Category</th><th>Method / reason</th><th className="r">Current</th><th className="r">Recommended</th><th>Action</th><th className="r">Profit delta</th></tr></thead><tbody>{result.topRows.slice(0, 15).map((row) => <tr key={row.itemId}><td className="strong mono">{row.itemId}</td><td>{row.category}</td><td>{TECHNIQUE_MAP[row.selectedTechnique]?.short || row.selectedTechnique}<small className="block-muted">{row.selectionReason}</small></td><td className="r mono">{fmtPrice(row.currentPrice)}</td><td className="r mono">{fmtPrice(row.recommendedPrice)}</td><td>{ACTION_LABEL[row.recommendationAction]}</td><td className="r mono money-positive">{fmtMoney(row.profitChange)}</td></tr>)}</tbody></table></div></section>
    </> : null}

    {history.length ? <section className="panel"><div className="panel-head"><div><h3>Cross-test run history</h3><p className="panel-sub">Compare Small, Medium, Large, profiles and approaches. Open any row to restore its complete result.</p></div></div><div className="tbl-scroll"><table className="tbl history-table"><thead><tr><th>Run</th><th>Test case</th><th>Profile</th><th>Approach</th><th className="r">Methods</th><th className="r">Wall time</th><th className="r">Profit uplift</th><th className="r">Controls</th><th></th></tr></thead><tbody>{history.map((record) => <tr key={record.id}><td className="strong">#{record.id}<small className="block-muted">{record.at}</small></td><td>{record.sizeId}<small className="block-muted">{record.size.toLocaleString()} units</small></td><td>{record.profile}</td><td>{record.approach}</td><td className="r mono">{record.methods}</td><td className="r mono">{record.elapsedMs.toFixed(1)} ms</td><td className="r mono money-positive">{fmtPct(record.profitChangeRate)}</td><td className="r mono">{fmtPctPlain(record.passRate)}</td><td><button className="text-btn" onClick={() => openHistory(record)}>Open report →</button></td></tr>)}</tbody></table></div></section> : null}
  </div>;
}
