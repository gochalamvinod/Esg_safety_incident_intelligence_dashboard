import React from "react";

export default function Overview({ analytics, setActiveTab }) {
  if (!analytics) return null;

  const total = analytics.total_incidents || 6;
  const hc = analytics.high_critical_count || 0;
  const catAcc = (analytics.model_health?.category_accuracy * 100 || 100).toFixed(1);
  const sevAcc = (analytics.model_health?.severity_accuracy * 100 || 100).toFixed(1);

  const severities = analytics.severities || {};
  const categories = analytics.categories || {};

  return (
    <div className="tab-pane animate-fade">
      {/* 4 Top KPI Cards */}
      <div className="kpi-grid">
        <div className="kpi-card">
          <div className="kpi-label">Benchmark Incidents</div>
          <div className="kpi-value text-emerald">{total}</div>
          <div className="kpi-desc">Verified benchmark records (S001-S006)</div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">High & Critical Risks</div>
          <div className="kpi-value text-amber">{hc}</div>
          <div className="kpi-desc">Incidents requiring prioritized containment</div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">Category Accuracy</div>
          <div className="kpi-value text-blue">{catAcc}%</div>
          <div className="kpi-desc">TF-IDF + Balanced Logistic Regression</div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">Severity Accuracy</div>
          <div className="kpi-value text-purple">{sevAcc}%</div>
          <div className="kpi-desc">Domain Lexicon Augmented Classifier</div>
        </div>
      </div>

      {/* Breakdown Panels */}
      <div className="panels-grid">
        <div className="panel-card">
          <h3 className="panel-title">Severity Distribution</h3>
          <div className="distribution-list">
            {["critical", "high", "medium", "low"].map((s) => {
              const count = severities[s] || 0;
              const pct = total > 0 ? Math.round((count / total) * 100) : 0;
              return (
                <div key={s} className="dist-row">
                  <div className="dist-header">
                    <span className={`badge-pill badge-${s}`}>{s.toUpperCase()}</span>
                    <span className="dist-count">{count} records ({pct}%)</span>
                  </div>
                  <div className="progress-bar">
                    <div className={`progress-fill fill-${s}`} style={{ width: `${pct}%` }}></div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div className="panel-card">
          <h3 className="panel-title">ESG & Safety Categories</h3>
          <div className="distribution-list">
            {Object.entries(categories).map(([cat, count]) => {
              const pct = total > 0 ? Math.round((count / total) * 100) : 0;
              return (
                <div key={cat} className="dist-row">
                  <div className="dist-header">
                    <span className="cat-name">{cat.replace("_", " ").toUpperCase()}</span>
                    <span className="dist-count">{count} records ({pct}%)</span>
                  </div>
                  <div className="progress-bar">
                    <div className="progress-fill fill-emerald" style={{ width: `${pct}%` }}></div>
                  </div>
                </div>
              );
            })}
          </div>
          <div className="panel-cta">
            <button className="btn-primary" onClick={() => setActiveTab("classifier")}>
              Run AI Incident Classifier &rarr;
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
