import React from "react";

export default function Governance({ metrics }) {
  if (!metrics) return null;

  const cat = metrics.category_model || {};
  const sev = metrics.severity_model || {};

  return (
    <div className="tab-pane animate-fade">
      <div className="card">
        <h3 className="card-title">Model Governance & Baseline Transparency</h3>
        <p className="card-subtitle">
          Evaluates ML classification models against Zero-R (Majority-Class) baselines to demonstrate genuine learning.
        </p>

        <div className="governance-grid">
          <div className="gov-card">
            <h4 className="gov-title">Category Classifier</h4>
            <div className="gov-stat-list">
              <div className="gov-stat">
                <span>Model Accuracy:</span>
                <span className="font-mono text-emerald">{(cat.accuracy * 100).toFixed(1)}%</span>
              </div>
              <div className="gov-stat">
                <span>Weighted F1-Score:</span>
                <span className="font-mono text-blue">{(cat.f1_weighted * 100).toFixed(1)}%</span>
              </div>
              <div className="gov-stat">
                <span>Majority Baseline:</span>
                <span className="font-mono text-muted">{(cat.baseline_accuracy * 100).toFixed(1)}%</span>
              </div>
              <div className="gov-stat highlight">
                <span>Gain Over Baseline:</span>
                <span className="font-mono text-emerald">+{(cat.gain * 100).toFixed(1)}%</span>
              </div>
            </div>
          </div>

          <div className="gov-card">
            <h4 className="gov-title">Severity Classifier</h4>
            <div className="gov-stat-list">
              <div className="gov-stat">
                <span>Model Accuracy:</span>
                <span className="font-mono text-purple">{(sev.accuracy * 100).toFixed(1)}%</span>
              </div>
              <div className="gov-stat">
                <span>Weighted F1-Score:</span>
                <span className="font-mono text-blue">{(sev.f1_weighted * 100).toFixed(1)}%</span>
              </div>
              <div className="gov-stat">
                <span>Majority Baseline:</span>
                <span className="font-mono text-muted">{(sev.baseline_accuracy * 100).toFixed(1)}%</span>
              </div>
              <div className="gov-stat highlight">
                <span>Gain Over Baseline:</span>
                <span className="font-mono text-purple">+{(sev.gain * 100).toFixed(1)}%</span>
              </div>
            </div>
          </div>
        </div>

        <div className="audit-note">
          <strong>ISO 45001 & ISO 14001 Audit Trail:</strong> All incoming classifications are logged with timestamp, confidence scores, and root-cause hazard tags for regulatory compliance and safety reporting.
        </div>
      </div>
    </div>
  );
}
