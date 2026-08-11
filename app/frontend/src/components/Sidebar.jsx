import Icon from "./Icon.jsx";

const NAV = [
  { id: "live", label: "Test case runner", icon: "live", group: "Test & compare" },
  { id: "portfolio", label: "Decision overview", icon: "overview", group: "Optimize" },
  { id: "sku", label: "Recommendations", icon: "decisions" },
  { id: "sensitivity", label: "Stress test", icon: "sensitivity" },
  { id: "validation", label: "Controls", icon: "validation", group: "Review" },
  { id: "techniques", label: "Methods", icon: "methods" },
  { id: "data", label: "Input data", icon: "data", group: "Manage" },
  { id: "report", label: "Decision report", icon: "report" },
  { id: "help", label: "User guide", icon: "guide" },
];

export default function Sidebar({ tab, setTab, meta, itemCount, onOpenSettings }) {
  return <aside className="sidebar">
    <div className="brand"><div className="brand-mark"><span>E</span></div><div><div className="brand-name">ELASTIQ</div><div className="brand-sub">DEEP ANALYSIS · BUILD 1.2.0</div></div></div>
    <nav className="nav-list">
      {NAV.map((n) => <div key={n.id}>{n.group ? <div className="nav-group">{n.group}</div> : null}<button className={`nav-item ${tab === n.id ? "active" : ""}`} onClick={() => setTab(n.id)}><Icon name={n.icon}/><span>{n.label}</span></button></div>)}
    </nav>
    <div className="sidebar-foot">
      <button className="run-card" onClick={onOpenSettings}>
        <div className="run-card-top"><span className="run-dot"/><span>Active run</span><Icon name="settings" size={16}/></div>
        <strong>{meta.short}</strong><small>{itemCount} decision units</small>
      </button>
      <div className="privacy-note">Client-side optimization<br/>Data stays in your browser</div>
    </div>
  </aside>;
}
