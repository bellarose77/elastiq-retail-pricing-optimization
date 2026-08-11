import { isFin } from "../lib/engine.js";
import { fmtPct, ACTION_LABEL } from "../lib/format.js";

export function Segmented({ options, value, onChange }) {
  return (
    <div className="seg">
      {options.map((o) => (
        <button key={o.value} className={"seg-btn" + (value === o.value ? " on" : "")} onClick={() => onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Slider({ label, value, min, max, step, onChange, format, hint }) {
  return (
    <div className="ctl">
      <div className="ctl-head">
        <span className="ctl-label">{label}</span>
        <span className="ctl-val mono">{format ? format(value) : value}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(parseFloat(e.target.value))} />
      {hint ? <div className="ctl-hint">{hint}</div> : null}
    </div>
  );
}

export function Toggle({ label, checked, onChange, hint }) {
  return (
    <div className="ctl toggle-row" onClick={() => onChange(!checked)}>
      <div>
        <div className="ctl-label">{label}</div>
        {hint ? <div className="ctl-hint">{hint}</div> : null}
      </div>
      <div className={"toggle" + (checked ? " on" : "")}>
        <span />
      </div>
    </div>
  );
}

export function DeltaPill({ value, invert }) {
  if (!isFin(value)) return <span className="pill neutral">—</span>;
  const good = invert ? value < 0 : value >= 0;
  return <span className={"pill " + (Math.abs(value) < 1e-9 ? "neutral" : good ? "up" : "down")}>{fmtPct(value)}</span>;
}

export function ActionTag({ action }) {
  return <span className={"tag tag-" + action}>{ACTION_LABEL[action] || action}</span>;
}

export function Select({ label, value, options, onChange }) {
  return (
    <label className="sel">
      {label ? <span className="sel-label">{label}</span> : null}
      <span className="sel-wrap">
        <select value={value} onChange={(e) => onChange(e.target.value)}>
          {options.map((o) => (
            <option key={o.value} value={o.value}>{o.label}</option>
          ))}
        </select>
        <span className="sel-caret">▾</span>
      </span>
    </label>
  );
}
