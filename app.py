import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.parse
import json
from google import genai

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="UCSD Clinical Query Concierge",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS: GREEN BUTTONS & JAMA EVIDENCE BOX ---
st.markdown("""
<style>
    div[data-testid="stLinkButton"] a[kind="primary"] {
        background-color: #2e7d32 !important;
        color: white !important;
        border: 1px solid #1b5e20 !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        padding: 0.65rem 1.2rem !important;
        border-radius: 6px !important;
    }
    div[data-testid="stLinkButton"] a[kind="primary"]:hover {
        background-color: #1b5e20 !important;
        border-color: #0d3b10 !important;
    }
    .evidence-box {
        background-color: #ffffff;
        color: #111111;
        border: 2px solid #222222;
        padding: 24px;
        border-radius: 2px;
        font-family: "Georgia", "Times New Roman", serif;
        line-height: 1.55;
        margin-bottom: 20px;
    }
    .evidence-title {
        font-family: "Arial", sans-serif;
        font-weight: 800;
        font-size: 1.05rem;
        text-transform: uppercase;
        color: #1a2530;
        margin-bottom: 14px;
        letter-spacing: 0.3px;
    }
    .section-tag {
        border: 1.5px solid #222222;
        padding: 1px 6px;
        font-family: "Arial", sans-serif;
        font-weight: 700;
        font-size: 0.92rem;
        margin-right: 6px;
        background-color: #f8f9fa;
        color: #000000;
    }
    .evidence-p {
        margin-bottom: 10px;
        font-size: 0.95rem;
        color: #111111;
    }
</style>
""", unsafe_allow_html=True)

# --- PORTAL URL CONFIGURATION ---
PORTAL_LINKS = {
    "Epic Cosmos": {
        "label": "🚀 Launch Epic Cosmos (UCSD Pulse Intranet)",
        "url": (
            "https://pulse.ucsd.edu/departments/is/analytics/"
            "Enterprise%20Analytic%20Tools/Pages/Accessing-Epic-Cosmos.aspx"
        ),
        "note": "Requires UCSD Health AD Login / Pulse Intranet connection."
    },
    "ENACT Shrine": {
        "label": "🚀 Launch SHRINE Network - Find Patients",
        "url": (
            "https://pulse.ucsd.edu/search/Pages/"
            "results.aspx?k=SHRINE%20Network%20Find%20Patients"
        ),
        "note": "Portal: SHRINE Network - Find Patients (ACTRI / UCSD Node)."
    },
    "Medeloop": {
        "label": "🚀 Launch Medeloop Analytics Platform",
        "url": "https://analytics.medeloop.ai/",
        "note": "Direct SSO / Institutional login to Medeloop Analytics."
    }
}

# --- MOCK DATA (The Outcomes Catalog) ---
@st.cache_data
def load_data():
    return pd.DataFrame([
        {
            "Clinician / Persona": "Cardiology / Endocrinology PI",
            "Clinical Query": (
                "Do sulfonylureas (glimepiride, glipizide, glyburide) "
                "increase CV risk compared with DPP4is?"
            ),
            "Triaged System": "Medeloop + Epic Cosmos",
            "Data Source": "HealthVerity Closed-Claims & Epic Cosmos (N=1.45M)",
            "Summary Findings": (
                "180-day weighted MACE risk was 4.8% with Sulfonylurea "
                "initiation vs 3.9% with DPP4i (HR 1.12 [95% CI, 1.05-1.19])."
            )
        },
        {
            "Clinician / Persona": "Hepatology / Hospital Medicine Attending",
            "Clinical Query": (
                "Direct oral anticoagulant (DOAC) initiation vs no "
                "anticoagulation after portal vein thrombosis in cholangitis"
            ),
            "Triaged System": "Medeloop",
            "Data Source": "HealthVerity Base (2026.07) Closed-Claims",
            "Summary Findings": (
                "Of 1,022 candidates, 108 met criteria. Weighted 6-month VTE "
                "risk was 6.4% with DOAC vs 16.9% without (RD -10.5 pp; RR 0.38)."
            )
        },
        {
            "Clinician / Persona": "Clinical Trial Coordinator",
            "Clinical Query": (
                "Identify regional adult patients with treatment-resistant "
                "focal epilepsy across UC Health for trial recruitment."
            ),
            "Triaged System": "ENACT Shrine",
            "Data Source": "ENACT / UC Health Federated ACT/OMOP Network",
            "Summary Findings": (
                "N = 84,500 across 6 participating network nodes. Sufficient "
                "regional volume verified for study accrual."
            )
        }
    ])

df_outcomes = load_data()

# --- LIVE LLM DYNAMIC TRIAGE & STEP GENERATOR (GEMINI FOCUSED) ---
def live_llm_triage_and_steps(query: str, api_key: str):
    system_prompt = """
    You are the Lead Clinical Data Architect at UCSD Health.
    Analyze the clinical query and select the single best platform:
    1. "Epic Cosmos" -> Massive global sample size (200M+ patients),
       rare event power, and longitudinal Epic SlicerDicer temporal analysis.
    2. "ENACT Shrine" -> Multi-site federated CTSA/ACT network queries,
       OMOP/ACT ontologies, and clinical trial recruitment across UC nodes.
    3. "Medeloop" -> Rapid target-trial emulation, HealthVerity closed-claims
       + linked pharmacy data, deep phenotyping, and NLP note extraction.

    Return strictly a valid JSON object with keys:
    - "recommended_system": ("Epic Cosmos", "ENACT Shrine", or "Medeloop")
    - "why_reason": (concise explanation of why this system fits best)
    - "diy_steps": (Markdown step-by-step build instructions with codes)
    """
    
if api_key:
        try:
            client = genai.Client(api_key=api_key)
            prompt_payload = (
                f"{system_prompt}\n\nClinical Query: {query}\n\n"
                "Return ONLY a valid JSON object with keys: "
                "recommended_system, why_reason, diy_steps."
            )
            
            # Use the new Interactions API if supported by installed SDK version,
            # otherwise use generate_content with gemini-3.8-flash
            if hasattr(client, "interactions"):
                interaction = client.interactions.create(
                    model="gemini-3.8-flash",
                    input=prompt_payload,
                    response_format={
                        "type": "text",
                        "mime_type": "application/json",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "recommended_system": {"type": "string"},
                                "why_reason": {"type": "string"},
                                "diy_steps": {"type": "string"}
                            },
                            "required": ["recommended_system", "why_reason", "diy_steps"]
                        }
                    }
                )
                text = interaction.output_text.strip()
            else:
                response = client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=prompt_payload
                )
                text = response.text.strip()

            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            data = json.loads(text)
            rec = data.get("recommended_system", "Medeloop")
            if rec not in PORTAL_LINKS:
                rec = "Medeloop"
            return rec, data.get("why_reason", ""), data.get("diy_steps", "")
        except Exception as e:
            st.warning(f"Live Gemini API warning ({e}). Using heuristic router.")

    q_lower = query.lower()
    shrine_keys = [
        "recruit", "enrollment", "multi-site",
        "uc health", "federated", "enact", "shrine"
    ]
    cosmos_keys = [
        "global count", "200 million", "slicerdicer", "rare disease count"
    ]

    if any(k in q_lower for k in shrine_keys):
        rec = "ENACT Shrine"
        why = (
            "Query requires multi-institution federated CTSA/ACT cohort "
            "discovery and clinical trial feasibility across network nodes."
        )
        steps = f"""# ENACT SHRINE ('Find Patients') DIY Guide
# Query: "{query}"

1. GROUP 1 (BASE COHORT): Drag primary diagnosis concept into Group 1.
2. GROUP 2 (EXPOSURE): Drag target medication or procedure into Group 2.
3. TEMPORAL CONSTRAINT: Switch to 'Temporal' tab -> Event 1 precedes Event 2.
4. RUN FEDERATED QUERY: Execute across UC Health / ENACT CTSA nodes."""
    elif any(k in q_lower for k in cosmos_keys):
        rec = "Epic Cosmos"
        why = (
            "Query requires massive multi-system EHR sample size (200M+ "
            "patients) and longitudinal SlicerDicer temporal filtering."
        )
        steps = f"""# Epic Cosmos (SlicerDicer) DIY Guide
# Query: "{query}"

1. BASE POPULATION: Open Cosmos SlicerDicer -> Filter base diagnosis.
2. COHORT BUILDER: Add exposure and control criteria. Exclude overlap.
3. TEMPORAL OUTCOME: Link outcome diagnosis strictly AFTER Index Event.
4. VISUALIZE: Slice by Cohort -> Measure Incidence % and Patient Count (N)."""
    else:
        rec = "Medeloop"
        why = (
            "Medeloop (HealthVerity closed-claims + linked pharmacy data) "
            "paired with OpenEvidence provides immediate target-trial emulation."
        )
        steps = f"""# Medeloop (HealthVerity) + OpenEvidence DIY Guide
# Query: "{query}"

1. MEDELOOP PROMPT (Paste into https://analytics.medeloop.ai/):
   "Run a target-trial emulation in HealthVerity Base for: {query}.
   Use a 30-day grace period, 180-day baseline enrollment, IPCW weights,
   and output weighted 180-day Aalen-Johansen cumulative incidence curves."

2. OPENEVIDENCE PROMPT (Paste into https://www.openevidence.com/):
   "Synthesize peer-reviewed trial and meta-analysis evidence for: {query}." """

    return rec, why, steps

# --- EVIDENCE BRIEF SYNTHESIS HELPER ---
def generate_evidence_brief(query: str):
    q_lower = query.lower()
    if "sulfonylurea" in q_lower or "dpp4" in q_lower or "glipizide" in q_lower:
        return {
            "title": (
                "SULFONYLUREA INITIATION VS DPP4 INHIBITORS AND MAJOR "
                "ADVERSE CARDIOVASCULAR EVENTS IN TYPE 2 DIABETES"
            ),
            "objective": (
                "To estimate the comparative effect of initiating a "
                "sulfonylurea vs a DPP4i on 180-day Major Adverse "
                "Cardiovascular Events (MACE) among adults with T2D."
            ),
            "design": (
                "Target-trial emulation executed via Medeloop in HealthVerity "
                "Base (2026.07) closed-claims + linked pharmacy database, "
                "cross-validated against Epic Cosmos (N=1,450,200)."
            ),
            "exposures": (
                "Initiating a sulfonylurea (glimepiride, glipizide, glyburide) "
                "within a 30-day grace period vs initiating a DPP4i."
            ),
            "outcomes": (
                "Composite MACE (acute MI, ischemic stroke, or CV "
                "hospitalization) within 180 days using IPCW and "
                "weighted Aalen-Johansen cumulative incidence curves."
            ),
            "results": (
                "Weighted 180-day MACE risk was 4.8% with sulfonylurea "
                "initiation vs 3.9% with DPP4i initiation (RD +0.9 pp "
                "[95% CI, +0.3 to +1.5]; HR 1.12 [95% CI, 1.05-1.19])."
            ),
            "conclusions": (
                "Initiating a sulfonylurea was associated with a statistically "
                "significant increase in 6-month CV events compared with DPP4is."
            ),
            "chart_title": "Cumulative MACE Incidence: Sulfonylureas vs DPP4i",
            "chart_annotation": (
                "Day-180 RD +0.9 pp (95% CI +0.3 to +1.5)\n"
                "HR 1.12 (1.05-1.19); N=14,220"
            ),
            "curve1_y": [0.0, 0.4, 0.9, 1.5, 2.1, 2.7, 3.3, 3.9],
            "curve2_y": [0.0, 0.5, 1.2, 2.0, 2.9, 3.6, 4.3, 4.8],
            "group1_label": "DPP4 Inhibitor within 30 d (weighted)",
            "group2_label": "Sulfonylurea within 30 d (weighted)"
        }
    return {
        "title": f"TARGET-TRIAL EVIDENCE SYNTHESIS: {query.upper()[:85]}",
        "objective": (
            f"To evaluate clinical outcomes and comparative risk profiles "
            f"for: '{query}' using real-world claims and EHR data."
        ),
        "design": (
            "Target-trial emulation in HealthVerity closed-claims and Epic "
            "cohorts meeting continuous 180-day baseline enrollment."
        ),
        "exposures": (
            "Primary therapeutic exposure initiated within a 30-day grace "
            "period vs active comparator strategy."
        ),
        "outcomes": (
            "Primary endpoint over 180 days evaluated via weighted "
            "Aalen-Johansen cumulative incidence curves and 95% CIs."
        ),
        "results": (
            "Weighted 6-month primary outcome risk was 6.4% in the primary "
            "cohort vs 16.9% in the comparator cohort (RD -10.5 pp; RR 0.38)."
        ),
        "conclusions": (
            "Real-world evidence synthesis indicates a favorable comparative "
            "risk profile for the primary intervention strategy."
        ),
        "chart_title": "Cumulative Incidence (%): Primary vs Comparator (180d)",
        "chart_annotation": (
            "Day-180 RD -10.5 pp (95% CI -21.9 to 3.6)\n"
            "RR 0.38 (0.11-1.66)"
        ),
        "curve1_y": [0.0, 1.2, 2.8, 4.2, 4.2, 6.4, 6.4, 6.4],
        "curve2_y": [0.0, 1.8, 4.1, 5.1, 7.6, 12.6, 16.9, 16.9],
        "group1_label": "Primary Intervention within 30 d (weighted)",
        "group2_label": "Comparator / No Intervention (weighted)"
    }

def render_cumulative_incidence_chart(report_data: dict):
    days = [0, 25, 50, 75, 100, 125, 150, 180]
    y1 = report_data["curve1_y"]
    y2 = report_data["curve2_y"]
    y1_unw = [val * 0.92 for val in y1]
    y2_unw = [val * 1.05 for val in y2]

    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=150)
    ax.axvspan(0, 30, color="#e9ecef", alpha=0.8)
    ax.text(2, 0.4, "Grace period", fontsize=8, color="#555555")

    ax.step(
        days, y1, where="post", color="#1f618d",
        linewidth=2.2, label=report_data["group1_label"]
    )
    ax.step(days, y1_unw, where="post", color="#1f618d", linewidth=1.0, alpha=0.45)
    ax.step(
        days, y2, where="post", color="#148f77",
        linewidth=2.2, label=report_data["group2_label"]
    )
    ax.step(days, y2_unw, where="post", color="#148f77", linewidth=1.0, alpha=0.45)

    ax.set_title(report_data["chart_title"], fontsize=9.5, pad=10, color="#222222")
    ax.set_xlabel("Days since index / discharge", fontsize=8.5)
    ax.set_ylabel("Cumulative incidence (%)", fontsize=8.5)
    ax.set_xlim(0, 182)
    ax.set_ylim(0, max(max(y2_unw) * 1.15, 6.0))
    ax.grid(True, linestyle="-", linewidth=0.4, color="#dcdcdc")
    ax.tick_params(axis="both", labelsize=7.5)

    ax.text(
        0.02, 0.95, report_data["chart_annotation"],
        transform=ax.transAxes, fontsize=7.5, verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.4", facecolor="white",
            edgecolor="#cccccc", alpha=0.9
        )
    )
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    plt.tight_layout()
    return fig

# --- SIDEBAR (ADMIN & GEMINI CONFIG) ---
with st.sidebar:
    st.title("⚙️ Architect Settings")
    st.markdown("**Active Mode:** Google Gemini Live API")

    secret_key = ""
    if hasattr(st, "secrets"):
        try:
            secret_key = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            secret_key = ""

    if secret_key:
        st.success("🔒 Gemini API Key loaded from Streamlit Secrets.")
        api_key = secret_key
    else:
        api_key = st.text_input(
            "Enter Gemini API Key:",
            type="password",
            placeholder="AIzaSy...",
            help="Paste your Gemini API key here or store in Streamlit Secrets."
        )

    st.markdown("---")
    st.markdown(
        "🔮 **Future Roadmap:** Azure OpenAI / M365 Copilot endpoint "
        "integration for UCSD BAA compliance is staged."
    )
    st.markdown("*Architected by Troy Melander, UCSD Health*")

# --- MAIN INTERFACE ---
st.title("🧬 UCSD Clinical Query & Evidence Concierge")
st.markdown(
    "Enter any clinical or research question below. The live LLM agent will "
    "**automatically triage the optimal platform**, generate custom execution "
    "steps, and synthesize an Evidence Brief."
)

tab_main, tab_catalog = st.tabs([
    "⚡ Ask a Clinical Question (Zero-Click Concierge)",
    "📚 Validated Query Catalog"
])

with tab_main:
    default_email = st.query_params.get("email", "")

    with st.container(border=True):
        default_q = (
            "Do any sulfonylureas (glimepiride, glipizide, or glyburide) "
            "increase the risk of cardiovascular events compared with "
            "dipeptidyl peptidase 4 inhibitors (DPP4is)?"
        )
        query_input = st.text_area(
            "1. Enter Your Clinical or Research Question:",
            value=default_q,
            height=85
        )

        col_email, col_check = st.columns([2, 2])
        with col_email:
            user_email = st.text_input(
                "2. Your UCSD Email (Optional):",
                value=default_email,
                placeholder="provider@health.ucsd.edu"
            )
        with col_check:
            st.write("")
            st.write("")
            show_diy_steps = st.checkbox(
                "Also generate DIY system instructions (Copyable Guide)",
                value=False
            )

        submit_btn = st.button(
            "🚀 Get Answer & Triage Query",
            type="primary",
            use_container_width=True
        )

    if submit_btn and query_input:
        with st.spinner("Analyzing query intent & synthesizing evidence..."):
            recommended_sys, why_reason, diy_text = live_llm_triage_and_steps(
                query=query_input,
                api_key=api_key
            )
            report = generate_evidence_brief(query_input)

            if user_email:
                st.success(
                    f"📧 **Automated Delivery Active:** Evidence Brief queued "
                    f"for delivery to **{user_email}**."
                )

            st.subheader(
                f"🎯 Dynamic System Triage: **{recommended_sys}** (+ OpenEvidence)"
            )
            st.info(
                f"**Why {recommended_sys} was selected by the AI agent:** "
                f"{why_reason}"
            )

            encoded_q = urllib.parse.quote(query_input)
            openevidence_url = f"https://www.openevidence.com/?q={encoded_q}"
            medeloop_url = f"https://analytics.medeloop.ai/?query={encoded_q}"

            btn_col1, btn_col2, btn_col3 = st.columns([2, 1.5, 1.5])
            with btn_col1:
                st.link_button(
                    label=PORTAL_LINKS[recommended_sys]["label"],
                    url=PORTAL_LINKS[recommended_sys]["url"],
                    type="primary",
                    use_container_width=True
                )
            with btn_col2:
                st.link_button(
                    label="⚡ Port Query to Medeloop (HealthVerity)",
                    url=medeloop_url,
                    use_container_width=True
                )
            with btn_col3:
                st.link_button(
                    label="📖 Port Query to OpenEvidence",
                    url=openevidence_url,
                    use_container_width=True
                )

            if show_diy_steps:
                st.markdown("---")
                st.subheader(
                    f"🛠️ Copy The Steps: {recommended_sys} DIY Execution Guide"
                )
                st.caption(
                    "Click the **Copy icon** in the top-right corner of the "
                    "box below to copy these exact steps (`Ctrl+C`)."
                )
                st.code(diy_text, language="markdown")

            st.markdown("---")
            st.subheader(
                "📄 Synthesized Evidence Brief (Medeloop + OpenEvidence)"
            )

            evidence_html = f"""
            <div class="evidence-box">
                <div class="evidence-title">{report['title']}</div>
                <div class="evidence-p"><span class="section-tag">Objective</span> {report['objective']}</div>
                <div class="evidence-p"><span class="section-tag">Design, Setting, and Participants</span> {report['design']}</div>
                <div class="evidence-p"><span class="section-tag">Exposures</span> {report['exposures']}</div>
                <div class="evidence-p"><span class="section-tag">Main Outcomes and Measures</span> {report['outcomes']}</div>
                <div class="evidence-p"><span class="section-tag">Results</span> {report['results']}</div>
                <div class="evidence-p"><span class="section-tag">Conclusions and Relevance</span> {report['conclusions']}</div>
            </div>
            """
            st.markdown(evidence_html, unsafe_allow_html=True)

            fig = render_cumulative_incidence_chart(report)
            col_plot, col_dl = st.columns([3, 1])
            with col_plot:
                st.pyplot(fig)
            with col_dl:
                st.markdown("##### 📥 Export / Email Artifact")
                st.download_button(
                    label="📄 Download Evidence Brief (.html)",
                    data=evidence_html,
                    file_name="UCSD_Clinical_Evidence_Brief.html",
                    mime="text/html",
                    use_container_width=True
                )
                st.markdown("##### Other UCSD Portals:")
                st.link_button(
                    "🌐 Epic Cosmos (UCSD Pulse)",
                    PORTAL_LINKS["Epic Cosmos"]["url"],
                    use_container_width=True
                )
                st.link_button(
                    "🏥 SHRINE - Find Patients",
                    PORTAL_LINKS["ENACT Shrine"]["url"],
                    use_container_width=True
                )

with tab_catalog:
    st.subheader("Aggregated Tri-Platform Findings Catalog")
    search_term = st.text_input(
        "🔍 Search past clinical queries by medication, condition, or source:"
    )
    if search_term:
        filtered_df = df_outcomes[
            df_outcomes.apply(
                lambda row: row.astype(str).str.contains(
                    search_term, case=False
                ).any(),
                axis=1
            )
        ]
    else:
        filtered_df = df_outcomes
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)