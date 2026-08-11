import { Component } from "react";
import Icon from "./Icon.jsx";

// React error boundaries must be class components -- there is no hook
// equivalent. Without this, any exception thrown while rendering (e.g. from
// optimizePortfolio inside App.jsx's useMemo) unmounts the whole React tree
// and leaves a blank white page with no way back except a manual refresh.
export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    console.error("ELASTIQ crashed:", error, info.componentStack);
  }

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;

    return (
      <div className="error-boundary">
        <div className="panel error-panel">
          <div className="panel-head">
            <div>
              <h3>Something went wrong</h3>
              <p className="panel-sub">
                The pricing workbench hit an unexpected error and stopped rendering. Your
                data never left this browser tab, so reloading is safe -- nothing was sent
                anywhere and nothing was saved that a reload would lose.
              </p>
            </div>
          </div>
          <div className="panel-body">
            <pre className="error-detail mono">{error.message || String(error)}</pre>
            <div className="btn-group">
              <button className="btn" onClick={() => window.location.reload()}>
                <Icon name="refresh" size={16} /> Reload
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }
}
