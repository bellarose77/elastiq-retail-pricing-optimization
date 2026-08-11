import React, { useMemo, useRef, useEffect, useCallback } from "react";
import { responseCurve, optimizeItem, evaluateCandidate, DEFAULT_CONFIG, clamp, isFin } from "../lib/engine.js";
import { fmtMoney, fmtNum, fmtPrice, fmtPct, fmtPctPlain } from "../lib/format.js";

/* ==========================================================================
   Response curve — draggable what-if
   ========================================================================== */
export function ResponseCurve({ item, cfg, tech, promoOn, whatIf, setWhatIf }) {
  const W = 720, H = 340, padL = 58, padR = 54, padT = 24, padB = 42;
  const svgRef = useRef(null);
  const curve = useMemo(() => responseCurve(item, cfg, promoOn), [item, cfg, promoOn]);
  const res = useMemo(() => optimizeItem(item, cfg, tech), [item, cfg, tech]);

  const prices = curve.map((d) => d.price);
  const xMin = Math.min(...prices), xMax = Math.max(...prices);
  const curMax = Math.max(...curve.map((d) => Math.max(d.revenue, d.profit)), 1);
  const demMax = Math.max(...curve.map((d) => d.demand), 1);

  const x = (p) => padL + ((p - xMin) / (xMax - xMin)) * (W - padL - padR);
  const yC = (v) => H - padB - (v / curMax) * (H - padT - padB);
  const yD = (v) => H - padB - (v / demMax) * (H - padT - padB);
  const path = (key, sc) => curve.map((d, i) => (i ? "L" : "M") + x(d.price).toFixed(1) + "," + sc(d[key]).toFixed(1)).join(" ");

  const g = res._guardrails;
  const bandX0 = x(Math.max(g.compLo, xMin)), bandX1 = x(Math.min(g.compHi, xMax));
  const floorX = x(clamp(g.marginFloor, xMin, xMax));

  const markers = [
    { p: item.currentPrice, key: "current", label: "Current", cls: "m-current" },
    { p: res.revenueMaxPrice, key: "rev", label: "Rev-max", cls: "m-rev" },
    { p: res.profitMaxPrice, key: "pro", label: "Profit-max", cls: "m-pro" },
    { p: res.recommendedPrice, key: "rec", label: "Recommended", cls: "m-rec" },
  ];
  const profitAt = (p) => evaluateCandidate(item, p, promoOn, { ...DEFAULT_CONFIG, ...cfg }).profit;

  const wi = whatIf ?? item.currentPrice;
  const wiCand = evaluateCandidate(item, clamp(wi, xMin, xMax), promoOn, { ...DEFAULT_CONFIG, ...cfg });

  const onMove = useCallback(
    (e) => {
      const rect = svgRef.current.getBoundingClientRect();
      const cx = ((e.touches ? e.touches[0].clientX : e.clientX) - rect.left) * (W / rect.width);
      const p = xMin + ((clamp(cx, padL, W - padR) - padL) / (W - padL - padR)) * (xMax - xMin);
      setWhatIf(+p.toFixed(2));
    },
    [xMin, xMax, setWhatIf]
  );

  const dragging = useRef(false);
  const start = (e) => { dragging.current = true; onMove(e); };
  const move = (e) => { if (dragging.current) { e.preventDefault(); onMove(e); } };
  const end = () => { dragging.current = false; };
  useEffect(() => {
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", end);
    return () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", end);
    };
  });

  const yTicks = 4;
  return (
    <div className="chart-wrap">
      <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} className="resp-svg" onPointerDown={start} onTouchStart={start} onTouchMove={move}>
        {isFin(g.compLo) && bandX1 > bandX0 ? <rect x={bandX0} y={padT} width={bandX1 - bandX0} height={H - padT - padB} className="band-comp" /> : null}
        {floorX > padL ? <rect x={padL} y={padT} width={floorX - padL} height={H - padT - padB} className="band-floor" /> : null}
        {Array.from({ length: yTicks + 1 }).map((_, i) => {
          const v = (curMax / yTicks) * i;
          return (
            <g key={i}>
              <line x1={padL} x2={W - padR} y1={yC(v)} y2={yC(v)} className="grid" />
              <text x={padL - 8} y={yC(v) + 3} className="axis-lbl" textAnchor="end">{"$" + Math.round(v).toLocaleString()}</text>
            </g>
          );
        })}
        {[xMin, item.currentPrice, xMax].map((p, i) => (
          <text key={i} x={x(p)} y={H - padB + 16} className="axis-lbl" textAnchor="middle">{"$" + p.toFixed(2)}</text>
        ))}
        <path d={path("demand", yD)} className="line-demand" fill="none" />
        <path d={path("revenue", yC)} className="line-rev" fill="none" />
        <path d={path("profit", yC)} className="line-pro" fill="none" />
        <line x1={x(clamp(wi, xMin, xMax))} x2={x(clamp(wi, xMin, xMax))} y1={padT} y2={H - padB} className="handle-line" />
        <circle cx={x(clamp(wi, xMin, xMax))} cy={yC(wiCand.profit)} r="5" className="handle-dot" />
        {markers.map((m) => (
          <g key={m.key}>
            <line x1={x(m.p)} x2={x(m.p)} y1={yC(profitAt(m.p))} y2={H - padB} className={"mk-stem " + m.cls} />
            <circle cx={x(m.p)} cy={yC(profitAt(m.p))} r="4.5" className={"mk " + m.cls} />
          </g>
        ))}
        <text x={padL} y={14} className="axis-title">Revenue &amp; profit ($)</text>
        <text x={W - padR} y={14} className="axis-title" textAnchor="end">demand (units, faint)</text>
      </svg>
      <div className="whatif">
        <div className="whatif-head">Drag the chart — what-if at <span className="mono">{fmtPrice(clamp(wi, xMin, xMax))}</span></div>
        <div className="whatif-grid">
          <div><span className="k">Demand</span><span className="v mono">{fmtNum(wiCand.demand, 0)}</span></div>
          <div><span className="k">Revenue</span><span className="v mono rev">{fmtMoney(wiCand.revenue)}</span></div>
          <div><span className="k">Profit</span><span className="v mono pro">{fmtMoney(wiCand.profit)}</span></div>
          <div><span className="k">Margin</span><span className="v mono">{fmtPctPlain(wiCand.marginRate)}</span></div>
        </div>
      </div>
    </div>
  );
}

/* ==========================================================================
   Scenario bars
   ========================================================================== */
export function ScenarioBars({ portfolio, metric }) {
  const W = 720, H = 260, padL = 64, padR = 16, padT = 16, padB = 46;
  const max = Math.max(...portfolio.map((p) => p[metric]), 1);
  const bw = (W - padL - padR) / portfolio.length;
  const y = (v) => H - padB - (v / max) * (H - padT - padB);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="bars-svg">
      {[0, 0.25, 0.5, 0.75, 1].map((f, i) => (
        <g key={i}>
          <line x1={padL} x2={W - padR} y1={y(max * f)} y2={y(max * f)} className="grid" />
          <text x={padL - 8} y={y(max * f) + 3} className="axis-lbl" textAnchor="end">{"$" + Math.round(max * f).toLocaleString()}</text>
        </g>
      ))}
      {portfolio.map((p, i) => {
        const cx = padL + i * bw + bw / 2;
        const cls = p.key === "recommended" ? "bar-rec" : metric === "profit" ? "bar-pro" : "bar-rev";
        return (
          <g key={p.key}>
            <rect x={cx - bw * 0.3} y={y(p[metric])} width={bw * 0.6} height={H - padB - y(p[metric])} className={cls} rx="2" />
            <text x={cx} y={y(p[metric]) - 6} className="bar-val mono" textAnchor="middle">{fmtMoney(p[metric])}</text>
            <text x={cx} y={H - padB + 16} className="axis-lbl" textAnchor="middle">{p.name.replace(" Strategy", "").replace(" Maximizing", "-max")}</text>
            <text x={cx} y={H - padB + 30} className={"axis-delta " + (p[metric === "profit" ? "profitChangeRate" : "revenueChangeRate"] >= 0 ? "up" : "down")} textAnchor="middle">
              {i === 0 ? "baseline" : fmtPct(p[metric === "profit" ? "profitChangeRate" : "revenueChangeRate"])}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

/* ==========================================================================
   Category bars
   ========================================================================== */
export function CategoryBars({ categories }) {
  const max = Math.max(...categories.map((c) => Math.abs(c.profitChange)), 1);
  return (
    <div className="catbars">
      {categories.map((c) => (
        <div className="catbar-row" key={c.category}>
          <div className="catbar-name">{c.category}<span className="catbar-n">{c.items}</span></div>
          <div className="catbar-track">
            <div className={"catbar-fill " + (c.profitChange >= 0 ? "pos" : "neg")} style={{ width: (Math.abs(c.profitChange) / max) * 100 + "%" }} />
          </div>
          <div className="catbar-val mono">
            {(c.profitChange >= 0 ? "+" : "") + fmtMoney(c.profitChange)}
            <span className="catbar-pct">{fmtPct(c.profitChangeRate)}</span>
          </div>
        </div>
      ))}
    </div>
  );
}

/* ==========================================================================
   Sensitivity heatmap
   ========================================================================== */
export function SensitivityHeatmap({ grid, metric }) {
  const elas = [...new Set(grid.map((g) => g.elasticityMult))];
  const cost = [...new Set(grid.map((g) => g.costMult))];
  const vals = grid.map((g) => g[metric]);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const color = (v) => {
    if (v >= 0) { const t = hi > 0 ? v / hi : 0; return `rgba(14,124,91,${0.12 + t * 0.7})`; }
    const t = lo < 0 ? v / lo : 0;
    return `rgba(169,80,106,${0.12 + t * 0.7})`;
  };
  const cell = (e, c) => grid.find((g) => g.elasticityMult === e && g.costMult === c);
  return (
    <div className="heat">
      <div className="heat-grid" style={{ gridTemplateColumns: `auto repeat(${cost.length}, 1fr)` }}>
        <div className="heat-corner">elas ↓ / cost →</div>
        {cost.map((c) => <div key={c} className="heat-colhead mono">×{c.toFixed(2)}</div>)}
        {elas.map((e) => (
          <React.Fragment key={e}>
            <div className="heat-rowhead mono">×{e.toFixed(2)}</div>
            {cost.map((c) => {
              const g = cell(e, c);
              return <div key={c} className="heat-cell mono" style={{ background: color(g[metric]) }}>{fmtPct(g[metric])}</div>;
            })}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}
