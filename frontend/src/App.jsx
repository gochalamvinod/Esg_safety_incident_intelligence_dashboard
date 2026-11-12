import React, { useState, useEffect } from "react";
import Header from "./components/Header";
import Overview from "./components/Overview";
import Classifier from "./components/Classifier";
import Explorer from "./components/Explorer";
import Clusters from "./components/Clusters";
import Governance from "./components/Governance";

export default function App() {
  const [activeTab, setActiveTab] = useState("overview");
  const [theme, setTheme] = useState("dark");
  const [analytics, setAnalytics] = useState(null);
  const [incidents, setIncidents] = useState([]);
  const [clusters, setClusters] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
      try {
        const [anRes, incRes, clRes, mRes] = await Promise.all([
          fetch("/api/analytics").then((r) => r.json()),
          fetch("/api/incidents").then((r) => r.json()),
          fetch("/api/clusters").then((r) => r.json()),
          fetch("/api/metrics").then((r) => r.json()),
        ]);
        setAnalytics(anRes);
        setIncidents(incRes.incidents || []);
        setClusters(clRes.clusters || []);
        setMetrics(mRes);
      } catch (err) {
        console.error("Failed to load initial data:", err);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  const toggleTheme = () => {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    document.documentElement.setAttribute("data-theme", next);
  };

  return (
    <div className={`app-container ${theme}`}>
      <Header
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        theme={theme}
        toggleTheme={toggleTheme}
        totalIncidents={analytics?.total_incidents || incidents.length || 6}
      />
      <main className="main-content">
        {loading ? (
          <div className="loading-spinner">Loading Incident Intelligence Engine...</div>
        ) : (
          <>
            {activeTab === "overview" && (
              <Overview analytics={analytics} setActiveTab={setActiveTab} />
            )}
            {activeTab === "classifier" && (
              <Classifier onNewIncident={() => {}} />
            )}
            {activeTab === "explorer" && (
              <Explorer incidents={incidents} />
            )}
            {activeTab === "clusters" && (
              <Clusters clusters={clusters} />
            )}
            {activeTab === "governance" && (
              <Governance metrics={metrics} />
            )}
          </>
        )}
      </main>
    </div>
  );
}
