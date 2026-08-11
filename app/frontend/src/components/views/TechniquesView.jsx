import { useMemo } from "react";
import { optimizePortfolio } from "../../lib/engine.js";
import { TECHNIQUE_LIST } from "../../lib/techniques.js";
import { fmtMoney, fmtPrice, fmtPct } from "../../lib/format.js";
import { DeltaPill } from "../ui.jsx";

export default function TechniquesView({ items, cfg, tech, setTech }) {
  const runs = useMemo(
    () =>
      TECHNIQUE_LIST.map((t) => {
        const res = optimizePortfolio(items, cfg, { id: t.id, params: tech.params });
        const allPass = res.validation.find((v) => v.check === "All checks passed");
        return { meta: t, res, allPass };
      }),
    [items, cfg, tech.params]
  );

  const maxProfit = Math.max(...runs.map((r) => r.res.exec.profitChangeRate), 0.0001);
  const active = tech.id;

  return (
    <div className="stack">
      <div className="panel">
        <div className="panel-head">
          <div>
            <h3>Optimization method comparison</h3>
            <p className="panel-sub">Every method is run on the current catalog and settings. Compare the uplift each produces, then apply one to drive the whole app.</p>
          </div>
        </div>
        <div className="tech-compare">
          {runs.map(({ meta, res, allPass }) => {
            const e = res.exec;
            const isActive = meta.id === active;
            return (
              <div className={"tcard" + (isActive ? " active" : "")} key={meta.id}>
                <div className="tcard-top">
                  <div>
                    <div className="tcard-name">{meta.name}</div>
                    <div className="tcard-tag">{meta.family}</div>
                  </div>
                  {isActive ? <span className="tcard-badge">Active</span> : <button className="btn ghost sm" onClick={() => setTech({ ...tech, id: meta.id })}>Apply</button>}
                </div>
                <div className="tcard-bar">
                  <div className="tcard-bar-fill" style={{ width: Math.max(4, (Math.max(e.profitChangeRate, 0) / maxProfit) * 100) + "%" }} />
                </div>
                <div className="tcard-metrics">
                  <div><span>Profit</span><b className="mono pro">{fmtPct(e.profitChangeRate)}</b></div>
                  <div><span>Revenue</span><b className="mono rev">{fmtPct(e.revenueChangeRate)}</b></div>
                  <div><span>Avg move</span><b className="mono">{fmtPct(e.avgPriceChangeRate)}</b></div>
                  <div><span>Checks</span><b className="mono">{allPass.passed}/{allPass.total}</b></div>
                </div>
                {meta.id === "lagrangian" && res.portfolioInfo ? (
                  <div className="tcard-note">Bid price {fmtPrice(res.portfolioInfo.lambda)} / unit · capacity {res.portfolioInfo.binding ? "binding" : "slack"}</div>
                ) : (
                  <div className="tcard-note">{meta.tagline}</div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      <div className="panel">
        <div className="panel-head"><div><h3>Side-by-side ledger</h3><p className="panel-sub">Portfolio totals under each method versus current pricing.</p></div></div>
        <div className="tbl-scroll">
          <table className="tbl">
            <thead>
              <tr><th>Method</th><th>Family</th><th className="r">Exp. profit</th><th className="r">Δ Profit</th><th className="r">Exp. revenue</th><th className="r">Δ Revenue</th><th className="r">Avg move</th><th className="c">Checks</th></tr>
            </thead>
            <tbody>
              {runs.map(({ meta, res, allPass }) => (
                <tr key={meta.id} className={meta.id === active ? "row-active" : ""} onClick={() => setTech({ ...tech, id: meta.id })}>
                  <td className="strong">{meta.short}</td>
                  <td>{meta.family}</td>
                  <td className="r mono">{fmtMoney(res.exec.totalExpectedProfit)}</td>
                  <td className="r"><DeltaPill value={res.exec.profitChangeRate} /></td>
                  <td className="r mono">{fmtMoney(res.exec.totalExpectedRevenue)}</td>
                  <td className="r"><DeltaPill value={res.exec.revenueChangeRate} /></td>
                  <td className="r mono">{fmtPct(res.exec.avgPriceChangeRate)}</td>
                  <td className="c mono">{allPass.passed}/{allPass.total}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
