# UCSD Clinical Query & Evidence Concierge — Build & Architecture Log
**Lead Architect:** Troy Melander (UCSD Health / ACTRI)  
**Repository:** `query-intelligence-hub`

---

## Changelog & Iteration History

### v0.3.0 — Zero-Friction Clinician Concierge Pivot (2026-11-09)
- **Problem Statement Alignment:** Shifted UI from a multi-step analyst tool to a zero-friction single screen for busy clinicians and PIs.
- **UI/UX Refinements:**
  - Removed M365 Copilot bottom-right quick link button to reduce clutter.
  - Styled primary system launch buttons in high-contrast green (`#2e7d32`) to clarify the primary call-to-action.
  - Collapsed architect settings sidebar by default.
  - Added `[ ] Also generate DIY system instructions` checkbox so step-by-step guides only render when explicitly requested.
- **Evidence Brief & Target-Trial Output:**
  - Added JAMA-style publication abstract container (`Objective`, `Design`, `Exposures`, `Main Outcomes`, `Results`, `Conclusions`).
  - Integrated `matplotlib` cumulative incidence step-plot generator with 30-day grace period shading and 95% CI annotations (modeled on Medeloop HealthVerity output).
  - Added downloadable `.html` Evidence Brief export and Epic Hyperspace URL parameter listener (`?email=`).

### v0.2.0 — AI Workflow Generator & UCSD Portal Deep-Links (2026-11-09)
- Added one-click clipboard copy (`st.code`) for system-specific execution steps.
- Mapped direct UCSD launch URLs:
  - **Epic Cosmos:** UCSD Pulse Intranet portal.
  - **ENACT SHRINE:** SHRINE Network - Find Patients portal.
  - **Medeloop:** `https://analytics.medeloop.ai/` (HealthVerity closed-claims + linked pharmacy).
- Added dual-mode AI provider scaffolding (`AzureOpenAI` for HIPAA-compliant UCSD tenant + `google-genai` for POC testing).

### v0.1.0 — Initial Tri-Platform Showcase POC (2026-11-09)
- Created initial Streamlit layout comparing **Epic Cosmos**, **ENACT Shrine**, and **Medeloop**.
- Loaded initial benchmark query: *Sulfonylureas (glimepiride, glipizide, glyburide) vs. DPP4 inhibitors cardiovascular (MACE) risk in Type 2 Diabetes*.

---

## Current Architecture Status (MVP vs. Production Roadmap)

### 1. Currently Functional (Live in v0.3.0)
- Single-screen clinician query intake and optional `@health.ucsd.edu` email capture.
- Epic Radar Web Component URL parameter ingestion (`?email=`).
- Dynamic keyword triage routing across Epic Cosmos, ENACT Shrine, and Medeloop (+ OpenEvidence).
- Direct URL payload handoff to OpenEvidence (`?q=`) and Medeloop (`?query=`).
- Downloadable HTML Evidence Brief and searchable Validated Query Catalog.

### 2. Simulated for Stakeholder Showcase (Phase 2 Backlog)
- **Live LLM Dynamic Triage & DIY Generation:** Currently uses deterministic rules + fallback templates when no API key is injected. *Next Step: Connect UCSD Azure OpenAI / Gemini API key via `.streamlit/secrets.toml`.*
- **Medeloop / OpenEvidence Execution:** Currently synthesizes target-trial emulation statistics and Aalen-Johansen plots locally for demo speed. *Next Step: Implement persistent query intake logging (`submitted_queries.csv` / RDSG SQL table) to queue asynchronous Medeloop runs.*
- **Automated Email / Epic InBasket Dispatch:** Currently confirms queuing in UI and provides direct HTML download. *Next Step: Wire `smtplib` to `smtp.ucsd.edu` inside UCSD firewall.*
- **V3.2 — One-Time Enterprise Tenant Onboarding & Custom Routing Config:**
  - Allow hospital/CTSA enterprise customers leasing the tool to configure a persistent `tenant_config.json` (one-time admin onboarding screen) storing their institution's specific intranet portal URLs, custom local data tools, and site-specific routing rules.
- **V3.3 — "Request Analytics Team Execution" Concierge Ticket Handoff:**
  - For busy clinicians who do not want to execute the DIY steps themselves, add a 1-click **"Submit Request to Analytics Team"** button that packages the clinical query, AI-triaged system, and pre-drafted DIY logic into a structured ticket routed directly to the institution's clinical informatics / analytics intake team (via ServiceNow, Jira, Jotform, or email).
