import { TECHNIQUE_LIST } from "../../lib/techniques.js";

export default function HelpView() {
  return (
    <div className="stack help">
      <div className="panel">
        <div className="panel-head"><div><h3>What this app does</h3><p className="panel-sub">A working reference for the model, the methods, and the guardrails.</p></div></div>
        <div className="help-body">
          <p>
            ELASTIQ recommends prices for a catalog of products. For each unit it estimates how demand
            responds to price, then searches for the price — optionally with a promotion — that best meets
            your goal, while staying inside a set of commercial safety limits. It reports the expected
            impact on demand, revenue, profit, and margin, validates every recommendation against the
            guardrails, and exports a board-ready report.
          </p>
          <p>
            The same demand model underpins every method. What changes between methods is the objective and
            the way uncertainty and shared constraints are handled — so you can match the pricing logic to
            your actual situation and see how the recommendation and its impact change.
          </p>
        </div>
      </div>

      <div className="panel">
        <div className="panel-head"><div><h3>The demand model</h3><p className="panel-sub">Shared by all methods.</p></div></div>
        <div className="help-body">
          <div className="help-eq">q(P) = q<sub>0</sub> · (P / P<sub>0</sub>)<sup>ε</sup></div>
          <p>
            Demand follows a constant price elasticity ε: a 1% price change moves demand by about ε%.
            The baseline quantity q<sub>0</sub> is adjusted by a bounded market (RAG) signal. Revenue is
            price × demand; profit is (price − unit cost) × demand, less promotion cost. A promotion
            multiplies demand by (1 + uplift) and costs a small fraction of revenue. Realized demand is
            capped by inventory. Elastic units (|ε| &gt; 1) lose demand faster than price rises; inelastic
            units (|ε| &lt; 1) tolerate increases more easily.
          </p>
        </div>
      </div>

      <div className="panel">
        <div className="panel-head"><div><h3>Optimization methods</h3><p className="panel-sub">Choose one in the rail; the data and report adapt to it.</p></div></div>
        <div className="help-methods">
          {TECHNIQUE_LIST.map((t) => (
            <div className="method" key={t.id}>
              <div className="method-head">
                <div className="method-name">{t.name}</div>
                <div className="method-family">{t.family}</div>
              </div>
              <p className="method-summary">{t.summary}</p>
              <div className="method-eq mono">{t.math}</div>
              <div className="method-cols">
                <div>
                  <div className="method-col-title">Best for</div>
                  <p>{t.when}</p>
                </div>
                <div>
                  <div className="method-col-title">Strengths</div>
                  <ul>{t.strengths.map((s, i) => <li key={i}>{s}</li>)}</ul>
                </div>
                <div>
                  <div className="method-col-title">Watch-outs</div>
                  <ul>{t.limits.map((s, i) => <li key={i}>{s}</li>)}</ul>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="panel">
        <div className="panel-head"><div><h3>Guardrails</h3><p className="panel-sub">Applied to every recommendation, regardless of method.</p></div></div>
        <div className="help-body">
          <ul className="help-list">
            <li><b>Minimum margin &amp; cost floor</b> — no price below the margin floor or unit cost.</li>
            <li><b>Competitor band</b> — price kept within a tolerance around the competitor price.</li>
            <li><b>Implementation step</b> — a conservative cap on how far a single move can go, tightened automatically for low-confidence elasticities.</li>
            <li><b>Inventory cap</b> — expected demand never exceeds available stock.</li>
            <li><b>Non-negative profit</b> — optional hard floor on unit profitability.</li>
          </ul>
          <p>
            A guardrail that actively shapes a recommendation is shown as a binding flag on the SKU and
            Validation pages. Binding is not an error — it is the safety system working. Relax a guardrail
            in the rail only when the business deliberately allows it.
          </p>
        </div>
      </div>

      <div className="panel">
        <div className="panel-head"><div><h3>Run live experiments</h3><p className="panel-sub">Each run is computed in your browser and remains reviewable.</p></div></div>
        <div className="help-body">
          <p>
            The application opens on <b>Test Case Runner</b>. Choose Small (50), Medium (250),
            Large (1,000), or a custom number of items. Choose balanced, inventory-constrained,
            promotion-intensive, or volatile conditions, then run all six methods, one selected method, or build a hybrid
            portfolio that selects the strongest feasible method for each item.
          </p>
          <p>
            Choose Standard, Deep analysis, or Research depth. Deep analysis is the default and performs
            three progressively finer optimizer passes, size-adjusted stress re-optimizations, 1,000 Monte
            Carlo draws per method, consensus analysis, and independent validation. Research increases the
            search resolution, shocked scenarios, and uncertainty draws; Standard is intentionally lighter.
          </p>
          <p>
            A visible six-stage pipeline follows Start and input profiling, optimization and refinement,
            stress testing, risk simulation, validation, and Finish. Exact engine counters show portfolio
            optimizations, candidate price/promotion evaluations, demand and profit evaluations, and capacity
            iterations. There is no artificial wait. Completed runs retain the stress distributions, risk
            percentiles, consensus, workload evidence, charts, top decisions, and downloadable JSON record.
          </p>
        </div>
      </div>

      <div className="panel">
        <div className="panel-head"><div><h3>Working with your own data</h3><p className="panel-sub">Everything recomputes live.</p></div></div>
        <div className="help-body">
          <p>
            On the Data page you can edit any value inline, add or remove units, or import a CSV. Before
            any decision, the initial analysis summarizes completeness, invalid and duplicate rows,
            price and demand ranges, margin, competitor position, inventory pressure, causal coverage,
            promotion readiness, and category-level patterns. The importer and downloadable template
            use these columns:
          </p>
          <div className="help-eq mono small">
            itemId, category, currentPrice, unitCost, competitorPrice, inventory, baselineQuantity,
            elasticity, promotionUpliftRate, confidence, ragSignal
          </div>
          <p>
            Use the <b>Explain</b> button on any analytical page for a live read-out of what you are seeing,
            and the <b>Techniques</b> page to compare every method on your data before committing to one.
            Recommendations are conservative, guardrail-bounded implementations intended for decision
            support, not automatic execution.
          </p>
        </div>
      </div>
    </div>
  );
}
