"""
FastAPI Backend Application for Greenko ESG & Safety Incident Intelligence Dashboard.
"""

from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, Query as FastQuery, status
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from backend.ml_engine import get_pipeline

app = FastAPI(title="Greenko ESG & Safety Intelligence", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

BASE_DIR = Path(__file__).resolve().parents[1]
FRONTEND_FILE = BASE_DIR / "frontend" / "index.html"
CASE_STUDY_FILE = BASE_DIR / "case_study.html"

class Query(BaseModel):
    text: str = Field(..., min_length=3)

class IncidentInput(BaseModel):
    narrative: str = Field(..., min_length=5)
    site_type: Optional[str] = "Solar PV"

@app.get("/", response_class=FileResponse, tags=["Web Interface"])
def serve_dashboard():
    if not FRONTEND_FILE.exists(): raise HTTPException(404, "Dashboard index.html not found.")
    return FileResponse(FRONTEND_FILE)

@app.get("/case-study", response_class=FileResponse, tags=["Web Interface"])
def serve_case_study():
    if not CASE_STUDY_FILE.exists(): raise HTTPException(404, "case_study.html not found.")
    return FileResponse(CASE_STUDY_FILE)

@app.get("/api/demo", tags=["Case Study Starter API"])
def demo():
    pipe = get_pipeline()
    an = pipe.get_dashboard_analytics()
    return {
        "case_study": "ESG / Safety Incident Intelligence Dashboard",
        "status": "production ready",
        "expected_outcome": "A dashboard that classifies incident type, severity, recurring patterns, and recommended preventive actions using NLP and summarization logic.",
        "dataset_records_count": an["total_incidents"],
        "model_performance": an["model_health"]
    }

@app.post("/api/run", tags=["Case Study Starter API"])
def run_case(query: Query):
    pipe = get_pipeline()
    analysis = pipe.classify_and_analyze(query.text)
    m = pipe.evaluation_results
    return {
        "input": query.text,
        "result": {
            "predicted_category": analysis["prediction"]["category"],
            "predicted_severity": analysis["prediction"]["severity"],
            "confidence": analysis["prediction"]["category_confidence"],
            "composite_risk_score": analysis["prediction"]["composite_risk_score"],
            "summary": analysis["nlp_insights"]["extractive_summary"],
            "preventive_actions": analysis["recommended_preventive_actions"]
        },
        "predictions": analysis["prediction"],
        "nlp_insights": analysis["nlp_insights"],
        "retrieval_evidence": analysis["retrieval_evidence"],
        "gemini_insights": analysis.get("gemini_insights"),
        "reasoning_steps": analysis["reasoning_steps"],
        "evaluation_metrics": {
            "category_accuracy": m.get("category_model", {}).get("accuracy"),
            "category_f1_weighted": m.get("category_model", {}).get("f1_weighted"),
            "severity_accuracy": m.get("severity_model", {}).get("accuracy"),
            "severity_f1_weighted": m.get("severity_model", {}).get("f1_weighted"),
            "baseline_gain": m.get("category_model", {}).get("accuracy_gain_over_baseline")
        },
        "limitations": analysis["limitations"]
    }

@app.post("/api/classify", tags=["Incident Intelligence"])
def classify_narrative(payload: IncidentInput):
    return get_pipeline().classify_and_analyze(payload.narrative, site_type=payload.site_type or "Solar PV")

@app.get("/api/incidents", tags=["Incident Management"])
def list_incidents(category: Optional[str] = None, severity: Optional[str] = None, search: Optional[str] = None, limit: int = 100):
    pipe = get_pipeline()
    df = pipe.df.copy()
    if category: df = df[df["target_category"].str.lower() == category.lower()]
    if severity: df = df[df["target_severity"].str.lower() == severity.lower()]
    if search: df = df[df["narrative"].str.contains(search, case=False, na=False)]
    records = df.head(limit).to_dict(orient="records")
    return {"count": len(records), "total_in_database": len(pipe.df), "incidents": records}

@app.post("/api/incidents", tags=["Incident Management"], status_code=status.HTTP_201_CREATED)
def create_incident(payload: IncidentInput):
    return get_pipeline().add_incident(payload.narrative, site_type=payload.site_type or "Solar PV")

@app.get("/api/metrics", tags=["Model Governance"])
def get_metrics():
    return get_pipeline().evaluation_results

@app.get("/api/analytics", tags=["Dashboard Analytics"])
def get_analytics():
    return get_pipeline().get_dashboard_analytics()

@app.get("/api/clusters", tags=["Pattern Extraction"])
def get_clusters():
    return {"clusters": get_pipeline().cluster_info}
