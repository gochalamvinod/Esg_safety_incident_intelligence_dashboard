import React from "react";

export default function Clusters({ clusters }) {
  return (
    <div className="tab-pane animate-fade">
      <div className="card">
        <h3 className="card-title">Recurring Incident Pattern Clusters</h3>
        <p className="card-subtitle">
          Unsupervised K-Means clustering discovers latent hazard clusters across incident narratives.
        </p>

        <div className="clusters-grid">
          {clusters.map((c) => (
            <div key={c.cluster_id} className="cluster-card">
              <div className="cluster-header">
                <span className="cluster-badge">Cluster #{c.cluster_id + 1}</span>
                <span className="cluster-count">{c.count} Incidents</span>
              </div>
              <h4 className="cluster-title">{c.cluster_name}</h4>
              <div className="cluster-body">
                <div className="cluster-field">
                  <span className="field-label">Dominant Category:</span>
                  <span className="cat-tag">{c.category?.toUpperCase()}</span>
                </div>
                <div className="cluster-field">
                  <span className="field-label">Top Contributing NLP Keywords:</span>
                  <div className="keyword-chips">
                    {c.top_keywords?.map((kw, i) => (
                      <span key={i} className="kw-chip">
                        {kw}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
