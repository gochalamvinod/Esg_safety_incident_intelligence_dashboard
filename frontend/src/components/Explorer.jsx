import React, { useState } from "react";

export default function Explorer({ incidents }) {
  const [search, setSearch] = useState("");
  const [catFilter, setCatFilter] = useState("all");
  const [sevFilter, setSevFilter] = useState("all");

  const filtered = incidents.filter((inc) => {
    const matchSearch = inc.narrative.toLowerCase().includes(search.toLowerCase()) ||
                        inc.incident_id.toLowerCase().includes(search.toLowerCase());
    const matchCat = catFilter === "all" || inc.target_category.toLowerCase() === catFilter.toLowerCase();
    const matchSev = sevFilter === "all" || inc.target_severity.toLowerCase() === sevFilter.toLowerCase();
    return matchSearch && matchCat && matchSev;
  });

  return (
    <div className="tab-pane animate-fade">
      <div className="card">
        <div className="explorer-header">
          <div>
            <h3 className="card-title">Benchmark Incident Database</h3>
            <p className="card-subtitle">Verified benchmark records for training, validation, and historical grounding.</p>
          </div>
          <div className="explorer-filters">
            <input
              type="text"
              className="filter-input"
              placeholder="Search narrative or ID..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
            <select className="filter-select" value={catFilter} onChange={(e) => setCatFilter(e.target.value)}>
              <option value="all">All Categories</option>
              <option value="safety">Safety</option>
              <option value="environment">Environment</option>
              <option value="esg_compliance">ESG Compliance</option>
            </select>
            <select className="filter-select" value={sevFilter} onChange={(e) => setSevFilter(e.target.value)}>
              <option value="all">All Severities</option>
              <option value="critical">Critical</option>
              <option value="high">High</option>
              <option value="medium">Medium</option>
              <option value="low">Low</option>
            </select>
          </div>
        </div>

        <div className="table-responsive">
          <table className="data-table">
            <thead>
              <tr>
                <th>Incident ID</th>
                <th>Facility / Site</th>
                <th>Incident Narrative</th>
                <th>Category</th>
                <th>Severity</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((inc) => (
                <tr key={inc.incident_id}>
                  <td className="font-mono text-emerald">{inc.incident_id}</td>
                  <td>{inc.site_type || "Renewable Facility"}</td>
                  <td className="narrative-cell">{inc.narrative}</td>
                  <td>
                    <span className="cat-tag">{inc.target_category?.replace("_", " ").toUpperCase()}</span>
                  </td>
                  <td>
                    <span className={`badge-pill badge-${inc.target_severity?.toLowerCase()}`}>
                      {inc.target_severity?.toUpperCase()}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
