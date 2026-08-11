import { isFin } from "./engine.js";
import { fmtMoney, fmtPrice, fmtPct, fmtPctPlain, fmtNum, ACTION_LABEL } from "./format.js";

async function loadReportFonts(doc) {
  try {
    const addFont = async (url, fileName, style) => {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`Font request failed: ${response.status}`);
      const bytes = new Uint8Array(await response.arrayBuffer());
      let binary = "";
      for (let offset = 0; offset < bytes.length; offset += 0x8000) binary += String.fromCharCode(...bytes.subarray(offset, offset + 0x8000));
      doc.addFileToVFS(fileName, btoa(binary));
      doc.addFont(fileName, "DejaVuSans", style);
    };
    await Promise.all([
      addFont("/fonts/DejaVuSans.ttf", "DejaVuSans.ttf", "normal"),
      addFont("/fonts/DejaVuSans-Bold.ttf", "DejaVuSans-Bold.ttf", "bold"),
    ]);
    return "DejaVuSans";
  } catch (error) {
    console.warn("Report font could not be embedded; using the PDF fallback font.", error);
    return "helvetica";
  }
}

export const CSV_COLS = [
  "itemId", "category", "currentPrice", "unitCost", "competitorPrice", "inventory",
  "baselineQuantity", "baselineSource", "elasticity", "elasticitySource",
  "elasticityIsCausal", "promotionUpliftRate", "promotionModelReliable",
  "confidence", "ragSignal", "dataAsOfDate",
];

const csvCell = (value) => {
  const text = value == null ? "" : String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
};

function parseCSVRows(text) {
  const rows = [];
  let row = [], cell = "", quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') { cell += '"'; i += 1; }
      else if (ch === '"') quoted = false;
      else cell += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ",") { row.push(cell); cell = ""; }
    else if (ch === "\n") { row.push(cell); rows.push(row); row = []; cell = ""; }
    else if (ch !== "\r") cell += ch;
  }
  if (quoted) throw new Error("CSV contains an unclosed quoted field.");
  if (cell.length || row.length) { row.push(cell); rows.push(row); }
  return rows.filter((r) => r.some((v) => v.trim() !== ""));
}

const firstValue = (record, aliases) => {
  for (const alias of aliases) {
    if (record[alias] !== undefined && record[alias] !== "") return record[alias];
  }
  return "";
};

const asNumber = (value) => value === "" ? NaN : Number(value);
const asBool = (value, fallback = false) => {
  if (value === "" || value == null) return fallback;
  return ["true", "1", "yes", "y"].includes(String(value).trim().toLowerCase());
};

export function csvTemplate(items) {
  const head = CSV_COLS.join(",");
  const body = items.map((it) => CSV_COLS.map((c) => csvCell(it[c])).join(",")).join("\n");
  return head + "\n" + body;
}

export function parseCSV(text) {
  const rows = parseCSVRows(text);
  if (rows.length < 2) throw new Error("CSV must include a header and at least one data row.");
  const headers = rows[0].map((h) => h.trim());
  const parsed = rows.slice(1).map((values, i) => {
    const o = {};
    headers.forEach((h, j) => { o[h] = values[j] !== undefined ? values[j].trim() : ""; });
    const itemId = firstValue(o, ["itemId", "item_id", "productId", "product_id"]) || "ITEM" + (i + 1);
    const elasticitySource = firstValue(o, ["elasticitySource", "elasticity_source"]) || "imported";
    const explicitCausal = firstValue(o, ["elasticityIsCausal", "elasticity_is_causal"]);
    return {
      itemId, productId: firstValue(o, ["productId", "product_id"]) || itemId,
      category: o.category || "Uncategorized",
      currentPrice: asNumber(firstValue(o, ["currentPrice", "current_price", "selling_price"])),
      unitCost: asNumber(firstValue(o, ["unitCost", "unit_cost"])),
      competitorPrice: asNumber(firstValue(o, ["competitorPrice", "competitor_price"])),
      inventory: asNumber(firstValue(o, ["inventory", "available_inventory", "available_inventory_for_pricing", "inventory_level"])),
      baselineQuantity: asNumber(firstValue(o, ["baselineQuantity", "baseline_quantity_for_pricing", "forecast_quantity", "baseline_quantity", "units_sold"])),
      baselineSource: firstValue(o, ["baselineSource", "baseline_quantity_source"]) || "imported",
      elasticity: asNumber(firstValue(o, ["elasticity", "elasticity_for_pricing", "iv_elasticity"])),
      elasticitySource,
      elasticityIsCausal: explicitCausal !== ""
        ? asBool(explicitCausal)
        : ["product_iv", "pooled_iv"].includes(elasticitySource),
      promotionUpliftRate: asNumber(firstValue(o, ["promotionUpliftRate", "promotion_uplift_rate_for_pricing", "promotion_uplift_rate"])) || 0,
      promotionModelReliable: asBool(firstValue(o, ["promotionModelReliable", "promotion_model_is_reliable"]), false),
      confidence: asNumber(o.confidence) || 0.7,
      ragSignal: asNumber(firstValue(o, ["ragSignal", "rag_weighted_impact_score", "market_demand_adjustment_rate"])) || 0,
      dataAsOfDate: firstValue(o, ["dataAsOfDate", "date"]), promotionFlag: 0,
    };
  });

  const errors = [];
  const seen = new Set();
  parsed.forEach((row, i) => {
    const label = `row ${i + 2} (${row.itemId})`;
    if (seen.has(row.itemId)) errors.push(`${label}: duplicate item ID`);
    seen.add(row.itemId);
    if (!isFin(row.currentPrice) || row.currentPrice <= 0) errors.push(`${label}: current price must be positive`);
    if (!isFin(row.unitCost) || row.unitCost < 0) errors.push(`${label}: unit cost must be zero or positive`);
    if (!isFin(row.baselineQuantity) || row.baselineQuantity < 0) errors.push(`${label}: baseline quantity must be zero or positive`);
    if (!isFin(row.elasticity) || row.elasticity >= 0) errors.push(`${label}: elasticity must be a finite negative value`);
  });
  if (errors.length) throw new Error(errors.slice(0, 6).join("; ") + (errors.length > 6 ? `; plus ${errors.length - 6} more` : ""));
  return parsed;
}

export function downloadCSV(text, name) {
  const blob = new Blob([text], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a"); a.href = url; a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function recommendationsCSV(result) {
  const cols = ["itemId", "category", "status", "holdReason", "currentPrice", "recommendedPrice", "priceChangeRate", "recommendationAction", "recommendedPromotionFlag", "elasticity", "elasticitySource", "elasticityIsCausal", "baselineQuantity", "baselineSource", "promotionModelReliable", "currentDemand", "expectedDemand", "currentRevenue", "expectedRevenue", "revenueChange", "currentProfit", "expectedProfit", "profitChange", "optimizedMarginRate", "competitorBinding", "stepBinding", "inventoryBinding"];
  const head = cols.join(",");
  const body = result.rows.map((r) => cols.map((c) => csvCell(isFin(r[c]) ? +Number(r[c]).toFixed(4) : r[c])).join(",")).join("\n");
  return head + "\n" + body;
}

export async function generatePDF(items, result, cfg, techName) {
  const [{ jsPDF }, autoTableImport] = await Promise.all([
    import("jspdf"),
    import("jspdf-autotable"),
  ]);
  const autoTableModule = autoTableImport.default || autoTableImport;
  const autoTable = typeof autoTableModule === "function"
    ? autoTableModule
    : autoTableModule.default;
  const doc = new jsPDF({ unit: "pt", format: "a4" });
  const fontFamily = await loadReportFonts(doc);
  const W = doc.internal.pageSize.getWidth(), H = doc.internal.pageSize.getHeight(), M = 40;
  const navy = [23, 32, 51], blue = [41, 98, 255], green = [22, 130, 93], ink = [23, 32, 51], muted = [103, 116, 137], line = [226, 232, 240], pale = [248, 250, 252];
  const all = result.validation.find(v => v.check === "All checks passed");
  const top = [...result.rows].sort((a,b) => b.profitChange - a.profitChange).slice(0, 8);
  const binding = { step: result.rows.filter(r => r.stepBinding).length, competitor: result.rows.filter(r => r.competitorBinding).length, inventory: result.rows.filter(r => r.inventoryBinding).length };
  const date = new Date().toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" });

  const header = (section) => {
    doc.setFillColor(...navy); doc.rect(0, 0, W, 78, "F");
    doc.setTextColor(255); doc.setFont(fontFamily, "bold"); doc.setFontSize(17); doc.text("ELASTIQ Pricing Decision Report", M, 31);
    doc.setFont(fontFamily, "normal"); doc.setFontSize(8.5); doc.setTextColor(188, 201, 224); doc.text(`${section}  |  ${techName || "Grid search"}  |  ${date}`, M, 50);
    doc.setTextColor(115, 148, 255); doc.setFont(fontFamily, "bold"); doc.setFontSize(10); doc.text("DECISION SUPPORT", W - M, 42, { align: "right" });
  };
  const sectionTitle = (title, y) => { doc.setFont(fontFamily, "bold"); doc.setFontSize(11); doc.setTextColor(...ink); doc.text(title, M, y); doc.setDrawColor(...line); doc.line(M, y + 6, W - M, y + 6); return y + 18; };
  const tableStyle = { theme: "plain", headStyles: { fillColor: navy, textColor: 255, font: fontFamily, fontSize: 7.7, fontStyle: "bold" }, bodyStyles: { font: fontFamily, fontSize: 7.8, textColor: ink }, alternateRowStyles: { fillColor: pale }, styles: { font: fontFamily, cellPadding: 4.5, lineColor: line, lineWidth: .35 }, margin: { left: M, right: M } };

  header("Executive recommendation");
  let y = 103;
  doc.setTextColor(...blue); doc.setFont(fontFamily, "bold"); doc.setFontSize(9); doc.text("RECOMMENDED PORTFOLIO PLAN", M, y); y += 20;
  doc.setTextColor(...ink); doc.setFontSize(19);
  const headline = `Capture ${fmtMoney(result.exec.totalProfitChange)} in expected profit uplift with ${Math.round(all.passRate * 100)}% control readiness.`;
  const hLines = doc.splitTextToSize(headline, W - M * 2 - 120); doc.text(hLines, M, y); y += hLines.length * 21 + 6;
  doc.setFont(fontFamily, "normal"); doc.setFontSize(9.2); doc.setTextColor(...muted);
  const summary = `The plan covers ${result.exec.totalItems} decision units and proposes ${result.exec.increase} price increases, ${result.exec.decrease} decreases, and ${result.exec.hold} holds. Expected revenue changes by ${fmtPct(result.exec.revenueChangeRate)} and expected profit by ${fmtPct(result.exec.profitChangeRate)} versus current pricing. Recommendations remain subject to commercial review before execution.`;
  const sLines = doc.splitTextToSize(summary, W - M * 2); doc.text(sLines, M, y); y += sLines.length * 13 + 18;

  const cards = [
    ["EXPECTED PROFIT", fmtMoney(result.exec.totalExpectedProfit), fmtPct(result.exec.profitChangeRate)],
    ["EXPECTED REVENUE", fmtMoney(result.exec.totalExpectedRevenue), fmtPct(result.exec.revenueChangeRate)],
    ["AVERAGE MOVE", fmtPct(result.exec.avgPriceChangeRate), `${result.exec.increase} up / ${result.exec.decrease} down`],
    ["CONTROL READINESS", fmtPctPlain(all.passRate), `${all.passed}/${all.total} pass`],
  ];
  const cw = (W - M * 2 - 30) / 4;
  cards.forEach((c,i) => { const x = M + i * (cw + 10); doc.setFillColor(...pale); doc.setDrawColor(...line); doc.roundedRect(x, y, cw, 57, 5, 5, "FD"); doc.setFont(fontFamily, "bold"); doc.setFontSize(6.8); doc.setTextColor(...muted); doc.text(c[0], x + 9, y + 15); doc.setFontSize(13); doc.setTextColor(...(i===0||i===3?green:ink)); doc.text(c[1], x + 9, y + 35); doc.setFont(fontFamily, "normal"); doc.setFontSize(7); doc.setTextColor(...muted); doc.text(c[2], x + 9, y + 49); });
  y += 79;

  y = sectionTitle("Priority action queue", y);
  autoTable(doc, { startY: y, head: [["Priority", "Unit", "Category", "Current", "Recommended", "Action", "Profit impact"]], body: top.map((r,i) => [i+1, r.itemId, r.category, fmtPrice(r.currentPrice), fmtPrice(r.recommendedPrice), ACTION_LABEL[r.recommendationAction], fmtMoney(r.profitChange)]), columnStyles: {0:{halign:"center"},3:{halign:"right"},4:{halign:"right"},6:{halign:"right"}}, ...tableStyle });
  y = doc.lastAutoTable.finalY + 18;
  y = sectionTitle("Control and execution summary", y);
  autoTable(doc, { startY: y, head: [["Control indicator", "Result", "Interpretation"]], body: [
    ["All checks passed", `${all.passed}/${all.total}`, all.passRate === 1 ? "Plan is ready for commercial review" : "Resolve failed recommendations before approval"],
    ["Implementation-step capped", binding.step, "Model optimum was pulled toward the current price"],
    ["Competitor capped", binding.competitor, "Recommendation was bounded by the competitor range"],
    ["Inventory bound", binding.inventory, "Expected demand reached available inventory"],
    ["Promotions recommended", result.exec.promoRecommended, "Promotion met the configured uplift threshold"],
  ], columnStyles:{1:{halign:"center"}}, ...tableStyle });

  doc.addPage(); header("Evidence and assumptions"); y = 103;
  y = sectionTitle("Scenario comparison", y);
  autoTable(doc, { startY:y, head:[["Strategy","Demand","Revenue","Profit","Revenue delta","Profit delta"]], body:result.portfolio.map(p=>[p.name,fmtNum(p.demand),fmtMoney(p.revenue),fmtMoney(p.profit),fmtPct(p.revenueChangeRate),fmtPct(p.profitChangeRate)]), columnStyles:{1:{halign:"right"},2:{halign:"right"},3:{halign:"right"},4:{halign:"right"},5:{halign:"right"}}, ...tableStyle });
  y = doc.lastAutoTable.finalY + 18;
  y = sectionTitle("Category contribution", y);
  autoTable(doc, { startY:y, head:[["Category","Units","Average move","Revenue impact","Profit impact"]], body:result.categories.map(c=>[c.category,c.items,fmtPct(c.avgPriceChangeRate),fmtMoney(c.revenueChange),fmtMoney(c.profitChange)]), columnStyles:{1:{halign:"right"},2:{halign:"right"},3:{halign:"right"},4:{halign:"right"}}, ...tableStyle });
  y = doc.lastAutoTable.finalY + 18;
  y = sectionTitle("Run configuration", y);
  autoTable(doc, { startY:y, head:[["Parameter","Value","Parameter","Value"]], body:[
    ["Optimization method", techName || "Grid search", "Objective", cfg.objective],
    ["Minimum margin", fmtPctPlain(cfg.minMarginRate), "Implementation step", fmtPctPlain(cfg.maxImplementationStep)],
    ["Maximum increase", fmtPctPlain(cfg.maxPriceChangeRate), "Maximum decrease", fmtPctPlain(-cfg.minPriceChangeRate)],
    ["Competitor tolerance", "±"+fmtPctPlain(cfg.competitorPriceTolerance), "Promotion threshold", fmtPctPlain(cfg.promotionUpliftThreshold)],
    ["Promotion cost", fmtPctPlain(cfg.promotionCostRate,1), "Market adjustment cap", fmtPctPlain(cfg.ragDemandAdjustmentLimit)],
  ], ...tableStyle });
  y = doc.lastAutoTable.finalY + 17;
  doc.setFillColor(237, 242, 255); doc.setDrawColor(199, 212, 255); doc.roundedRect(M, y, W-M*2, 58, 5, 5, "FD");
  doc.setFont(fontFamily, "bold"); doc.setFontSize(8); doc.setTextColor(...blue); doc.text("MODEL AND GOVERNANCE NOTE", M+10, y+16);
  doc.setFont(fontFamily, "normal"); doc.setFontSize(7.7); doc.setTextColor(...ink);
  const note = "Demand uses a constant-elasticity response q(P)=q0*(P/P0)^e. Revenue is price times demand; profit is unit margin times demand less promotion cost. Recommendations are bounded by margin, competitor, inventory and implementation-step controls. They support human pricing decisions and are not automatic execution instructions.";
  doc.text(doc.splitTextToSize(note, W-M*2-20), M+10, y+31);

  const pages = doc.internal.getNumberOfPages();
  for (let i=1;i<=pages;i++) { doc.setPage(i); doc.setDrawColor(...line); doc.line(M,H-31,W-M,H-31); doc.setFont(fontFamily,"normal"); doc.setFontSize(7); doc.setTextColor(...muted); doc.text("ELASTIQ | Confidential decision support",M,H-18); doc.text(`Page ${i} of ${pages}`,W-M,H-18,{align:"right"}); }
  doc.save("ELASTIQ-Pricing-Decision-Report.pdf");
}

export async function generateDeepAnalysisPDF(result, context) {
  if (!result?.analysis) throw new Error("Complete a deep-analysis run before exporting its report.");
  const [{ jsPDF }, autoTableImport] = await Promise.all([import("jspdf"), import("jspdf-autotable")]);
  const autoTableModule = autoTableImport.default || autoTableImport;
  const autoTable = typeof autoTableModule === "function" ? autoTableModule : autoTableModule.default;
  const doc = new jsPDF({ unit: "pt", format: "a4" });
  const fontFamily = await loadReportFonts(doc);
  const W = doc.internal.pageSize.getWidth(), H = doc.internal.pageSize.getHeight(), M = 40;
  const navy = [23, 32, 51], blue = [41, 98, 255], green = [22, 130, 93], ink = [23, 32, 51], muted = [103, 116, 137], line = [226, 232, 240], pale = [248, 250, 252];
  const analysis = result.analysis, metrics = analysis.metrics, plan = analysis.depth;
  const date = new Date().toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" });
  const tableStyle = { theme: "plain", headStyles: { fillColor: navy, textColor: 255, font: fontFamily, fontSize: 7.4, fontStyle: "bold" }, bodyStyles: { font: fontFamily, fontSize: 7.5, textColor: ink }, alternateRowStyles: { fillColor: pale }, styles: { font: fontFamily, cellPadding: 4.2, lineColor: line, lineWidth: .35 }, margin: { left: M, right: M } };
  const header = (section) => {
    doc.setFillColor(...navy); doc.rect(0, 0, W, 78, "F");
    doc.setTextColor(255); doc.setFont(fontFamily, "bold"); doc.setFontSize(17); doc.text("ELASTIQ Deep Analysis Run Report", M, 31);
    doc.setFont(fontFamily, "normal"); doc.setFontSize(8.5); doc.setTextColor(188, 201, 224); doc.text(`${section}  |  ${plan.label}  |  ${date}`, M, 50);
    doc.setTextColor(115, 148, 255); doc.setFont(fontFamily, "bold"); doc.setFontSize(10); doc.text("COMPUTATION EVIDENCE", W - M, 42, { align: "right" });
  };
  const sectionTitle = (title, y) => { doc.setFont(fontFamily, "bold"); doc.setFontSize(11); doc.setTextColor(...ink); doc.text(title, M, y); doc.setDrawColor(...line); doc.line(M, y + 6, W - M, y + 6); return y + 18; };
  const duration = result.wallElapsedMs >= 1000 ? `${(result.wallElapsedMs / 1000).toFixed(2)} s` : `${result.wallElapsedMs.toFixed(1)} ms`;
  const champion = result.methodSummaries.find((item) => item.id === result.championId) || result.methodSummaries[0];

  header("Run scope and executed workload");
  let y = 105;
  doc.setTextColor(...blue); doc.setFont(fontFamily, "bold"); doc.setFontSize(9); doc.text("COMPLETED LIVE COMPUTATION", M, y); y += 21;
  doc.setTextColor(...ink); doc.setFontSize(19);
  const headline = `${metrics.candidateEvaluations.toLocaleString()} candidate evaluations completed in ${duration}.`;
  doc.text(doc.splitTextToSize(headline, W - M * 2), M, y); y += 32;
  doc.setFont(fontFamily, "normal"); doc.setFontSize(9); doc.setTextColor(...muted);
  const description = `The run analyzed ${context.size.toLocaleString()} decision units with ${result.methodSummaries.length} methods. It used ${plan.refinementPasses} price-grid resolutions, ${plan.stressScenarios} shocked-market re-optimizations, and ${plan.monteCarloDraws.toLocaleString()} Monte Carlo draws per method. No artificial waiting is included in the measured runtime.`;
  const lines = doc.splitTextToSize(description, W - M * 2); doc.text(lines, M, y); y += lines.length * 13 + 17;

  const cards = [
    ["OPTIMIZER RUNS", metrics.portfolioOptimizations.toLocaleString(), `${plan.plannedOptimizerRuns.toLocaleString()} planned`],
    ["CANDIDATE EVALUATIONS", metrics.candidateEvaluations.toLocaleString(), "Exact engine counter"],
    ["POLICY SIMULATIONS", plan.plannedPolicySimulations.toLocaleString(), `${plan.monteCarloDraws.toLocaleString()} draws / method`],
    ["CONTROL READINESS", fmtPctPlain(result.finalSummary.passRate), `${result.finalSummary.passed}/${result.finalSummary.itemCount} pass`],
  ];
  const cw = (W - M * 2 - 30) / 4;
  cards.forEach((card, index) => { const x = M + index * (cw + 10); doc.setFillColor(...pale); doc.setDrawColor(...line); doc.roundedRect(x, y, cw, 58, 5, 5, "FD"); doc.setFont(fontFamily, "bold"); doc.setFontSize(6.7); doc.setTextColor(...muted); doc.text(card[0], x + 8, y + 15); doc.setFontSize(12); doc.setTextColor(...(index === 3 ? green : ink)); doc.text(card[1], x + 8, y + 35); doc.setFont(fontFamily, "normal"); doc.setFontSize(6.8); doc.setTextColor(...muted); doc.text(card[2], x + 8, y + 49); });
  y += 80;

  y = sectionTitle("Executed operations", y);
  autoTable(doc, { startY: y, head: [["Operation", "Completed", "Technical meaning"]], body: [
    ["Price-grid refinement", `${plan.refinementPasses} passes`, plan.refinementSteps.map((step) => `${(step * 100).toFixed(3)}%`).join(" -> ")],
    ["Stress re-optimization", plan.stressScenarios.toLocaleString(), "Full method reruns after demand, cost, elasticity, inventory and competitor shocks"],
    ["Candidate evaluation", metrics.candidateEvaluations.toLocaleString(), "Price and promotion alternatives scored by the engine"],
    ["Demand evaluation", metrics.demandEvaluations.toLocaleString(), "Uncertainty-aware demand calculations"],
    ["Profit evaluation", metrics.profitEvaluations.toLocaleString(), "Robust/Bayesian profit calculations"],
    ["Capacity iteration", metrics.capacitySolverIterations.toLocaleString(), "Lagrangian bid-price solver iterations"],
    ["Independent validation", analysis.validationChecks.toLocaleString(), "Seven controls per selected recommendation"],
  ], columnStyles: { 1: { halign: "right" } }, ...tableStyle });
  y = doc.lastAutoTable.finalY + 18;
  y = sectionTitle("Deterministic method comparison", y);
  autoTable(doc, { startY: y, head: [["Method", "Profit uplift", "Revenue uplift", "Average move", "Controls"]], body: result.methodSummaries.map((item) => [item.name, fmtPct(item.profitChangeRate), fmtPct(item.revenueChangeRate), fmtPct(item.avgPriceChangeRate), fmtPctPlain(item.passRate)]), columnStyles: { 1: { halign: "right" }, 2: { halign: "right" }, 3: { halign: "right" }, 4: { halign: "right" } }, ...tableStyle });

  doc.addPage(); header("Uncertainty, stability and decisions"); y = 105;
  y = sectionTitle("Stress and Monte Carlo result distribution", y);
  autoTable(doc, { startY: y, head: [["Method", "Stress P05", "Stress mean", "Sim P05", "Sim P50", "Sim P95", "Positive"]], body: result.methodSummaries.map((item) => { const stress = analysis.stress.summary[item.id], risk = analysis.risk[item.id]; return [item.name, fmtPct(stress.p05), fmtPct(stress.mean), fmtPct(risk.p05), fmtPct(risk.median), fmtPct(risk.p95), fmtPctPlain(risk.positiveRate)]; }), columnStyles: { 1: { halign: "right" }, 2: { halign: "right" }, 3: { halign: "right" }, 4: { halign: "right" }, 5: { halign: "right" }, 6: { halign: "right" } }, ...tableStyle });
  y = doc.lastAutoTable.finalY + 18;
  y = sectionTitle("Decision and consensus summary", y);
  autoTable(doc, { startY: y, head: [["Indicator", "Result", "Interpretation"]], body: [
    ["Portfolio champion", champion.name, `${fmtPct(champion.profitChangeRate)} deterministic profit uplift`],
    ["Unanimous action agreement", fmtPctPlain(analysis.consensus.unanimousRate), `${analysis.consensus.disputedItems} units have method disagreement`],
    ["Average method agreement", fmtPctPlain(analysis.consensus.meanAgreementRate), "Majority action agreement across methods and units"],
    ["Champion positive probability", fmtPctPlain(analysis.risk[result.championId].positiveRate), "Share of simulated outcomes with non-negative profit uplift"],
    ["Final control readiness", fmtPctPlain(result.finalSummary.passRate), `${result.finalSummary.passed}/${result.finalSummary.itemCount} recommendations passed`],
  ], columnStyles: { 1: { halign: "right" } }, ...tableStyle });
  y = doc.lastAutoTable.finalY + 18;
  y = sectionTitle("Priority decisions", y);
  autoTable(doc, { startY: y, head: [["Unit", "Category", "Method", "Current", "Recommended", "Action", "Profit delta"]], body: result.topRows.slice(0, 10).map((row) => [row.itemId, row.category, row.selectedTechnique, fmtPrice(row.currentPrice), fmtPrice(row.recommendedPrice), ACTION_LABEL[row.recommendationAction], fmtMoney(row.profitChange)]), columnStyles: { 3: { halign: "right" }, 4: { halign: "right" }, 6: { halign: "right" } }, ...tableStyle });
  y = doc.lastAutoTable.finalY + 16;
  doc.setFillColor(237, 242, 255); doc.setDrawColor(199, 212, 255); doc.roundedRect(M, y, W - M * 2, 54, 5, 5, "FD");
  doc.setFont(fontFamily, "bold"); doc.setFontSize(8); doc.setTextColor(...blue); doc.text("GOVERNANCE NOTE", M + 10, y + 16);
  doc.setFont(fontFamily, "normal"); doc.setFontSize(7.5); doc.setTextColor(...ink); doc.text(doc.splitTextToSize("Stress and Monte Carlo results quantify modelled uncertainty under stated assumptions; they do not guarantee realized performance. Recommendations require commercial review and controlled measurement before publication.", W - M * 2 - 20), M + 10, y + 31);

  const pages = doc.internal.getNumberOfPages();
  for (let page = 1; page <= pages; page += 1) { doc.setPage(page); doc.setDrawColor(...line); doc.line(M, H - 31, W - M, H - 31); doc.setFont(fontFamily, "normal"); doc.setFontSize(7); doc.setTextColor(...muted); doc.text("ELASTIQ | Confidential decision support", M, H - 18); doc.text(`Page ${page} of ${pages}`, W - M, H - 18, { align: "right" }); }
  doc.save(`ELASTIQ-Deep-Analysis-${context.sizeId || "run"}.pdf`);
}
