import streamlit as st
import pandas as pd

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Tri-Platform Query Showcase",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MOCK DATA (The Outcomes) ---
# Pre-loading the dataframe with your exact use case + a couple of examples
@st.cache_data
def load_data():
    return pd.DataFrame([
        {
            "Face (Persona)": "Principal Investigator",
            "Clinical Query": "Do sulfonylureas increase CV risk compared to DPP4is?",
            "System Used": "Epic Cosmos",
            "Logic / Build": "Base: T2D. Cohort A: Sulfonylureas. Cohort B: DPP4i. Outcome: MACE after Index.",
            "Outcome / Findings": "N=2.4M. Slight statistically significant increase in MACE within 3 years for Cohort A."
        },
        {
            "Face (Persona)": "Clinical Trial Coordinator",
            "Clinical Query": "Find regional adult patients with treatment-resistant focal epilepsy for trial enrollment.",
            "System Used": "ENACT Shrine",
            "Logic / Build": "OMOP Concept: Focal Epilepsy AND > 2 anti-seizure meds.",
            "Outcome / Findings": "N=450 across 3 regional UC Health network nodes. Cohort pinged for recruitment."
        },
        {
            "Face (Persona)": "Research Fellow",
            "Clinical Query": "Extract phenotypic markers from clinical notes for patients with Long COVID.",
            "System Used": "Medeloop",
            "Logic / Build": "Natural language prompt targeting progress notes and discharge summaries.",
            "Outcome / Findings": "Identified 4 dominant symptom clusters. Drafted preliminary background for R01 grant."
        }
    ])

df_outcomes = load_data()

# --- SIDEBAR (The Face) ---
with st.sidebar:
    st.title("🧬 Query Intelligence")
    st.markdown("### 1. Select Your Persona")
    persona = st.selectbox(
        "Who are you?", 
        ["Principal Investigator", "Doctor / Clinician", "Data Analyst", "Clinical Trial Coordinator"]
    )
    
    st.markdown("---")
    st.markdown("**Infrastructure Managed By:**")
    st.markdown("*Troy Melander, Lead Clinical Data Architect*")

# --- MAIN APP ---
st.title("Clinical Query Routing & Knowledge Base")
st.markdown("Submit a research question to see the recommended data system, or search past findings to accelerate your research.")

# TAB SETUP
tab1, tab2 = st.tabs(["🔍 Triage a New Query", "📚 The Outcome Catalog"])

# --- TAB 1: TRIAGE (The Query & The System) ---
with tab1:
    st.subheader("Query Routing Engine")
    query_input = st.text_area("Enter your clinical or research question:", placeholder="e.g., Do any sulfonylureas increase the risk of cardiovascular events compared with DPP4is?")
    
    if st.button("Analyze & Route Query"):
        if query_input:
            with st.spinner("Analyzing query intent..."):
                # Mock routing logic for POC presentation
                st.success("Analysis Complete!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.info("**Primary Recommendation**\n### Epic Cosmos")
                    st.markdown("**Why:** Massive longitudinal N-counts needed for rare event statistical power.")
                with col2:
                    st.warning("**Secondary Option**\n### ENACT Shrine")
                    st.markdown("**Why:** Useful if you need to build a multi-site prospective cohort for this.")
                with col3:
                    st.error("**Alternative Option**\n### Medeloop")
                    st.markdown("**Why:** Best used if you need unstructured note extraction, less relevant for purely structured Rx/Dx queries.")
        else:
            st.warning("Please enter a query to route.")

# --- TAB 2: CATALOG (The Outcome) ---
with tab2:
    st.subheader("Aggregated Findings Catalog")
    st.markdown("Search past queries executed across Cosmos, Shrine, and Medeloop.")
    
    # Search bar for dataframe
    search_term = st.text_input("Search catalog by keyword, disease, or medication:")
    
    # Filter logic
    if search_term:
        filtered_df = df_outcomes[
            df_outcomes.apply(lambda row: row.astype(str).str.contains(search_term, case=False).any(), axis=1)
        ]
    else:
        filtered_df = df_outcomes

    # Display dataframe
    st.dataframe(
        filtered_df, 
        use_container_width=True,
        hide_index=True
    )
    
    # Detail Expander for the presentation
    st.markdown("### Deep Dive: Featured Benchmark Query")
    with st.expander("Sulfonylureas vs. DPP4is CV Risk (Epic Cosmos)"):
        st.markdown("""
        * **Platform:** Epic Cosmos (SlicerDicer)
        * **Base:** Patients with Type 2 Diabetes
        * **Exposure:** Glimepiride, Glipizide, or Glyburide
        * **Comparison:** DPP4 inhibitors (Sitagliptin, etc.)
        * **Outcome:** MACE (Myocardial infarction, Ischemic stroke) strictly *after* index medication date.
        
        **Conclusion:** [Insert your actual N-counts and findings here for the demo]
        """)
