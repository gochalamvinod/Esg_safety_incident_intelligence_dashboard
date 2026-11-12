# ESG & Safety Incident Intelligence Platform

**Client:** Greenko Group — Renewable Energy Operations  
**Operational Scope:** Utility-Scale Solar PV, Wind Fleets, Hydro & Pumped Storage, High-Voltage Substations, and Battery Energy Storage Systems (BESS)  
**Deliverable:** End-to-End Incident Intelligence Pipeline, Decision-Support Dashboard & REST API  

---

## 1. Project Overview

The **ESG & Safety Incident Intelligence Platform** automates incident triage, severity assessment, and compliance reporting across Greenko's renewable energy operations.

### Operational Challenges & Solutions
* **Manual Triage Delays:** Field logs from solar arrays and wind nacelles are unstructured. The platform provides **sub-10ms automated classification** across Safety, Environment, and ESG Compliance.
* **Subjective Severity Tagging:** Inconsistent human risk rating is solved through **deterministic severity scoring** augmented with domain hazard guardrails.
* **Compliance Exposure:** Critical incidents trigger immediate **ISO 45001 (Safety)** and **ISO 14001 (Environment)** Corrective and Preventive Actions (CAPA).
* **GenAI Executive Synthesis:** **Google Gemini 3.1 Flash Lite** delivers dual-perspective root-cause analysis (Technical vs. Organizational) and evaluates statutory **SEBI BRSR (Principles 3 & 6)** and **GRI 403-9 / GRI 306** disclosure impacts.

---

## 2. Technical Stack & Whitelisted Dependencies

Strictly compliant with client requirements, using **only the 7 whitelisted packages** in `requirements.txt`:

| Package | Version | Architectural Responsibility |
| :--- | :--- | :--- |
| **fastapi** | 0.115.0 | Asynchronous REST API routing, CORS middleware, and static asset delivery |
| **uvicorn** | 0.30.6 | High-throughput ASGI production server |
| **pandas** | 2.2.2 | Dataframe manipulation, CSV persistence, and tabular filtering |
| **numpy** | 1.26.4 | Numerical matrix operations, array indexing, and vector similarity mathematics |
| **scikit-learn** | 1.5.1 | TF-IDF feature extraction, Logistic Regression, K-Means clustering, and metrics |
| **pydantic** | 2.8.2 | Schema validation, type parsing, and response serialization |
| **python-multipart** | 0.0.9 | Form handling and multipart payload support |

> **Zero-Dependency GenAI:** Gemini 3.1 Flash Lite is queried natively via Python's standard library (`urllib.request`, `json`, `os`), eliminating the need for `google-generativeai`, `requests`, or other non-whitelisted packages.

---

## 3. System Architecture & Intelligence Pipeline

```
+----------------------------------------------------------------------------------------------------+
|                                    INPUT: INCIDENT NARRATIVE                                       |
|      "Contractor slipped near wet walkway at solar block, minor injury, housekeeping issue"        |
+----------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                                   PHASE 1: NLP PREPROCESSING                                       |
|  - Text Normalization & Whitespace Harmonization                                                   |
|  - Domain Hazard Lexicon Injection: [__cue_medium__, __cue_slip__]                                 |
|  - Sublinear TF-IDF (1, 2) N-gram Vectorization                                                    |
+----------------------------------------------------------------------------------------------------+
                                                  |
                     +----------------------------+----------------------------+
                     |                                                         |
                     v                                                         v
+-------------------------------------------+             +------------------------------------------+
|      PHASE 2: DETERMINISTIC ML CORE       |             |      PHASE 3: EVIDENCE & SUMMARIZATION   |
| - Category: Logistic Regression (Balanced)|             | - RAG Precedent Search: Cosine Similarity|
|   Output: Safety (Confidence: 89.4%)      |             |   Output: Top-3 Past S001 Precedents     |
| - Severity: Lexicon-Augmented Classifier  |             | - Extractive Summarizer: TF-IDF Salience |
|   Output: Medium                          |             |   Output: "Contractor slipped on walkway"|
+-------------------------------------------+             +------------------------------------------+
                     |                                                         |
                     +----------------------------+----------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                               PHASE 4: ISO 45001 / ISO 14001 CAPA ENGINE                           |
| - Standardized Preventive Actions: Tool-box talk, non-slip coatings, contractor notice             |
| - Operational Risk Score Calculation (0 - 100)                                                     |
+----------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                    PHASE 5: GOOGLE GEMINI 3.1 FLASH LITE (BUILT-IN HTTP CLIENT)                    |
| - Dual Root Cause Analysis: Technical (Surface drainage) vs. Organizational (Risk assessments)     |
| - Regulatory ESG Impact: SEBI BRSR Principle 3 & GRI 403-9 Work-Related Injury Metric              |
+----------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                               PHASE 6: FASTAPI REST API & ENTERPRISE GUI                           |
| - Endpoints: /api/demo, /api/run, /api/classify, /api/incidents, /api/analytics, /api/metrics      |
| - Responsive Web Interface: Dark/Light Mode, SVG Vector Icons, Scenario Studio Chips               |
+----------------------------------------------------------------------------------------------------+
```

* **Deterministic ML Layer:** Fast, offline, zero-hallucination classification and severity scoring.
* **Generative AI Layer:** Deep contextual reasoning for systemic root causes and ESG disclosures.

---

## 4. Benchmark Dataset Specification

The platform runs directly on the bundled benchmark dataset (`data/safety_esg_incidents.csv`) containing the 6 client-supplied operational records:

| Incident ID | Category | Severity | Operational Narrative |
| :---: | :---: | :---: | :--- |
| **S001** | Safety | Medium | Contractor slipped near wet walkway at solar block, minor injury, housekeeping issue |
| **S002** | Environment | High | Oil seepage observed near wind gearbox maintenance bay, contained with spill kit |
| **S003** | Safety | Medium | PPE non-compliance observed during panel cleaning activity |
| **S004** | Safety | High | Near miss: vehicle movement close to cable trench during night shift |
| **S005** | Environment | Medium | Water usage spike during module cleaning due to valve leakage |
| **S006** | ESG Compliance | Low | Repeated waste segregation deviation at contractor storage area |

* **Adaptive Small-Sample Handling:** Automatically handles small corpora ($N \le 10$) without attempting invalid stratified splits. Asset types are dynamically inferred from text when metadata is omitted.

---

## 5. ISO & Regulatory Standards Decision Matrix

| Category | Severity | Standard CAPA Protocol | ESG & Regulatory Impact |
| :--- | :--- | :--- | :--- |
| **Safety** | Critical | Facility E-stop, perimeter evacuation, industrial EMT dispatch | ISO 45001 Cl. 10.2, SEBI BRSR P3, GRI 403-9 |
| **Safety** | High | Stop Work Authority (SWA), LOTO audit, fall-protection lifeline inspection | ISO 45001 Cl. 8.1.2, TRIFR reporting |
| **Safety** | Medium | Mandatory pre-shift toolbox talk, physical hazard remediation in 24h | Near-miss / First-aid register tracking |
| **Safety** | Low | Site EHS register logging, on-shift PPE replacement | EHS monthly committee review |
| **Environment** | High | Secondary containment berms, absorbent booms, stormwater runoff sampling | ISO 14001 Cl. 8.2, SEBI BRSR P6, GRI 306 |
| **Environment** | Medium | Leaking valve isolation, portable drip tray capture, fluid loss accounting | ISO 14001 water/effluent accounting |
| **ESG Compliance** | Medium / Low | Waste segregation re-labeling, statutory PCB manifest verification | SEBI BRSR waste diverted from disposal |

---

## 6. Project Architecture & File Organization

* **backend/app.py:** FastAPI application, REST endpoints, CORS policies, and web serving.
* **backend/ml_engine.py:** NLP pipeline, scikit-learn models, CAPA decision matrix, and standard-library Gemini client.
* **backend/__init__.py:** Python package initializer.
* **data/safety_esg_incidents.csv:** Bundled client benchmark records (S001–S006).
* **frontend/index.html:** Single-page enterprise dashboard (Dark/Light mode, SVG icons, Gemini dossier).
* **.env:** Environment credentials (`GEMINI_API_KEY`, `GEMINI_MODEL`).
* **.gitignore:** Security rule preventing credential leaks.
* **Procfile:** Process declaration for Railway cloud deployment.
* **requirements.txt:** Exact list of the 7 approved Python packages.
* **README.md:** Complete technical documentation and execution guide.

---

## 7. Step-by-Step Setup & Execution Guide

### Step 1: Check Prerequisites
Ensure Python 3.10+ is installed:
```bash
python --version
```

### Step 2: Navigate to Directory
```bash
cd 05_esg_safety_incident_intelligence_dashboard
```

### Step 3: Set Up Virtual Environment (Recommended)
* **Windows (PowerShell):**
  ```bash
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
* **Linux / macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 5: Configure Environment
Verify `.env` exists in the project root:
```ini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.1-flash-lite
```

### Step 6: Start the Server
```bash
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

### Step 7: Open in Browser
* **Interactive Dashboard:** `http://127.0.0.1:8000/`
* **Swagger API Documentation:** `http://127.0.0.1:8000/docs`
* **Case Study Overview:** `http://127.0.0.1:8000/case-study`

---

## 8. Dashboard Feature Walkthrough

* **Dashboard Module:** Real-time KPI counters (Total Incidents, High/Critical alerts), severity distribution bars, category donut, and renewable site breakdowns. Toggle Dark/Light mode via top-right icon.
* **Incident Analyzer Studio:** Enter custom narratives or click quick scenario presets (*HV Transformer Arc Flash*, *Contractor Fall*, *Effluent Turbidity*, *BRSR Diesel Leak*). Generates live confidence meters, extractive summaries, ISO CAPA action checklists, and the Gemini GenAI synthesis card.
* **Incident Register:** Full tabular view of past incidents with live keyword search and dropdown filtering by category and severity.
* **Operational Clusters:** Visualizes unsupervised K-Means clusters and recurring hazard themes mined from past records.
* **Governance & Metrics:** Displays holdout accuracy scores, precision/recall/F1 metrics, and multi-class confusion matrices.

---

## 9. REST API Reference

| Endpoint | Method | Input Parameters | Key Response Payload |
| :--- | :---: | :--- | :--- |
| **/api/demo** | GET | None | Case study title, active status, total records (6), and model health |
| **/api/run** | POST | JSON body with `text` | Predictions, extractive summary, ISO CAPA, top-3 RAG evidence precedents, Gemini GenAI root cause & BRSR impact, audit reasoning steps, limitations |
| **/api/classify** | POST | JSON body with `narrative` and optional `site_type` | Real-time classification, detected asset & hazard entities, and specific CAPA protocols |
| **/api/incidents** | GET | Query filters: `category`, `severity`, `search`, `limit` | Filtered list of incident records from persistent register with total database count |
| **/api/incidents** | POST | JSON body with `narrative` and optional `site_type` | Ingestion confirmation, assigned incident ID, and full analytical outputs |
| **/api/analytics** | GET | None | Total count, category distribution, severity breakdown, asset site distribution, and high/critical alert count |
| **/api/clusters** | GET | None | Recurring thematic operational clusters, sample incidents, and top discriminative keywords |
| **/api/metrics** | GET | None | Model validation scores, baseline accuracy comparison gain, and confusion matrices |

---

## 10. Railway Cloud Deployment Guide

The application is pre-configured for one-click deployment on **Railway**:

1. **Procfile:** Located in the project root:
   ```
   web: uvicorn backend.app:app --host 0.0.0.0 --port $PORT
   ```
2. **Deployment Steps:**
   * Log in to [Railway.app](https://railway.app/) and select **New Project** &rarr; **Deploy from GitHub repo**.
   * Select your repository: `Esg_safety_incident_intelligence_dashboard`.
   * Under the **Variables** tab in Railway, define:
     * `GEMINI_API_KEY`: Your Gemini API key
     * `GEMINI_MODEL`: `gemini-3.1-flash-lite`
   * Railway automatically installs `requirements.txt`, executes `Procfile`, and provisions an HTTPS URL.

---

## 11. Assumptions, Known Limitations & Roadmap

* **Operational Assumptions:** Inputs represent primary field event reports logged by technicians or safety supervisors.
* **Known Limitations:** Terse descriptions (<5 words) provide limited lexical signals. Deep GenAI synthesis requires outbound network access to Google's API (deterministic ML and ISO CAPA remain 100% offline).
* **Future Roadmap:** SCADA sensor telemetry fusion (inverter voltages, transformer gas analysis), multilingual vernacular voice-logging (Telugu, Kannada, Hindi), and automated SAP/EAM work-order dispatch.
