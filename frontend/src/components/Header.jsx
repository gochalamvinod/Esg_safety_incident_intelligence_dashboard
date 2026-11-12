import React from "react";

export default function Header({ activeTab, setActiveTab, theme, toggleTheme, totalIncidents }) {
  const tabs = [
    { id: "overview", label: "Executive Overview" },
    { id: "classifier", label: "AI Classifier" },
    { id: "explorer", label: "Benchmark Explorer" },
    { id: "clusters", label: "Pattern Clusters" },
    { id: "governance", label: "Model Governance" },
  ];

  return (
    <header className="app-header">
      <div className="header-brand">
        <div className="brand-logo">G</div>
        <div className="brand-text">
          <div className="brand-title">GREENKO ESG & SAFETY</div>
          <div className="brand-sub">Incident Intelligence Dashboard</div>
        </div>
      </div>

      <nav className="header-nav">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            className={`nav-item ${activeTab === tab.id ? "active" : ""}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      <div className="header-actions">
        <div className="status-badge">
          <span className="dot pulse"></span>
          <span>{totalIncidents} Incidents Monitored</span>
        </div>
        <button className="theme-toggle" onClick={toggleTheme} title="Toggle Dark/Light Mode">
          {theme === "dark" ? "☀️ Light" : "🌙 Dark"}
        </button>
      </div>
    </header>
  );
}
