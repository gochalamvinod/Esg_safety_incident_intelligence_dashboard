import React, { useState } from "react";

const PRESETS = [
  { label: "Solar Micro-Crack", site: "Solar PV", text: "Micro-cracks detected on tracker string 4B during thermographic drone survey; inverter trip risk." },
  { label: "Wind Gearbox Leak", site: "Wind Farm", text: "Oil seepage observed near wind turbine #14 gearbox bay, contained with secondary spill kit." },
  { label: "Hydro Valve Jam", site: "Hydro Power", text: "Debris blockage caused penstock intake control valve to jam midway during peak load adjustment." },
  { label: "Substation Arc Flash", site: "Substation", text: "Electrical arc flash incident near 33kV switchyard breaker during scheduled relay maintenance." },
  { label: "Turbine Blade Crack", site: "Wind Farm", text: "Severe composite blade delamination and leading-edge cracking observed on WTG-08 after high-wind storm." },
  { label: "BESS Thermal Surge", site: "BESS", text: "Thermal runaway warning triggered in battery container rack 03 with rapid temperature escalation." }
];

export default function Classifier() {
  const [narrative, setNarrative] = useState(PRESETS[0].text);
  const [siteType, setSiteType] = useState(PRESETS[0].site);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleRun = async () => {
    if (!narrative.trim()) return;
    setLoading(true);
    try {
      const res = await fetch("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: jsonBody({ text: narrative }),
      });
      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error("Classification error:", err);
    } finally {
      setLoading(false);
    }
  };

  function jsonBody(obj) {
    return JSON.stringify(obj);
  }

  return (
    <div className="tab-pane animate-fade">
      <div className="classifier-layout">
        {/* Input Column */}
        <div className="card input-card">
          <h3 className="card-title">Live Incident Assessment Engine</h3>
          <p className="card-subtitle">Select a scenario or enter field narrative to trigger Classical ML + GenAI synthesis.</p>

          <div className="preset-container">
            <span className="preset-label">Preset Scenarios:</span>
            <div className="preset-chips">
              {PRESETS.map((p, idx) => (
                <button
                  key={idx}
                  className={`chip ${narrative === p.text ? "active" : ""}`}
                  onClick={() => {
                    setNarrative(p.text);
                    setSiteType(p.site);
                  }}
                >
                  {p.label}
                </button>
              ))}
            </div>
          </div>

          <div className="form-group">
            <label className="form-label">Asset / Facility Type</label>
            <select className="form-select" value={siteType} onChange={(e) => setSiteType(e.target.value)}>
              <option value="Solar PV">Solar PV Plant</option>
              <option value="Wind Farm">Wind Farm Site</option>
              <option value="Hydro Power">Hydroelectric Facility</option>
              <option value="Substation">33kV / 132kV Substation</option>
              <option value="BESS">Battery Energy Storage (BESS)</option>
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Incident Narrative</label>
            <textarea
              className="form-textarea"
              rows="4"
              value={narrative}
              onChange={(e) => setNarrative(e.target.value)}
              placeholder="Describe incident narrative, hazards observed, affected equipment..."
            />
          </div>

          <button className="btn-primary w-full" onClick={handleRun} disabled={loading}>
            {loading ? "Running AI Pipeline..." : "Run AI Assessment & ISO CAPA"}
          </button>
        </div>

        {/* Results Column */}
        <div className="card result-card">
          <h3 className="card-title">Assessment Dossier</h3>
          {!result ? (
            <div className="empty-state">
              <p>Click "Run AI Assessment" to generate classification, risk scores, ISO CAPA actions, and Gemini 3.1 Flash Lite synthesis.</p>
            </div>
          ) : (
            <div className="result-content">
              {/* Top Prediction Bar */}
              <div className="prediction-header">
                <div className="pred-item">
                  <span className="pred-label">Category</span>
                  <span className="pred-val text-emerald">{result.result?.predicted_category?.toUpperCase()}</span>
                  <span className="pred-conf">{(result.result?.confidence * 100).toFixed(1)}% confidence</span>
                </div>
                <div className="pred-item">
                  <span className="pred-label">Severity</span>
                  <span className={`badge-pill badge-${result.result?.predicted_severity}`}>
                    {result.result?.predicted_severity?.toUpperCase()}
                  </span>
                </div>
                <div className="pred-item">
                  <span className="pred-label">Composite Risk</span>
                  <span className="risk-score-pill">{result.result?.composite_risk_score} / 100</span>
                </div>
              </div>

              {/* Extractive Summary */}
              <div className="result-section">
                <h4 className="section-subtitle">Extractive NLP Summary</h4>
                <p className="summary-box">{result.result?.summary}</p>
              </div>

              {/* CAPA Recommendations */}
              <div className="result-section">
                <h4 className="section-subtitle">ISO 45001 / ISO 14001 Recommended CAPA</h4>
                <ul className="capa-list">
                  {result.result?.preventive_actions?.map((act, idx) => (
                    <li key={idx} className="capa-item">
                      <span className="capa-num">{idx + 1}</span>
                      <span>{act}</span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Gemini 3.1 Flash Lite Card */}
              {result.gemini_insights && (
                <div className="result-section gemini-card">
                  <div className="gemini-header">
                    <span className="gemini-badge">Gemini 3.1 Flash Lite</span>
                    <span className="gemini-model">{result.gemini_insights.model_used}</span>
                  </div>
                  <p className="gemini-summary">{result.gemini_insights.ai_executive_summary}</p>
                  <div className="gemini-sub">
                    <strong>Root Cause:</strong> {result.gemini_insights.root_cause_analysis}
                  </div>
                  <div className="gemini-sub">
                    <strong>BRSR / GRI 403 Reporting Impact:</strong> {result.gemini_insights.esg_disclosure_impact}
                  </div>
                </div>
              )}

              {/* Nearest Benchmark Precedents */}
              {result.retrieval_evidence?.length > 0 && (
                <div className="result-section">
                  <h4 className="section-subtitle">Cosine Similarity Historical Precedents</h4>
                  <div className="evidence-list">
                    {result.retrieval_evidence.map((ev, i) => (
                      <div key={i} className="evidence-item">
                        <div className="ev-header">
                          <span className="ev-id">{ev.incident_id}</span>
                          <span className="ev-score">Similarity: {(ev.similarity * 100).toFixed(1)}%</span>
                        </div>
                        <div className="ev-text">{ev.narrative}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
