import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="OncoOmics | Subtype Explorer", page_icon="🧬", layout="wide")

CLASS_NAMES = {0: "Basal", 1: "HER2-enriched", 2: "Luminal A", 3: "Luminal B"}

# Styling
st.markdown("""
<style>
    .stApp {background: #F5F7FB;}
    .block-container {max-width: 1220px; padding-top: 2rem; padding-bottom: 3rem;}
    [data-testid="stSidebar"] {background: #002366;}
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] li::marker,
    [data-testid="stSidebar"] strong {color: #FFFFFF !important;}
    [data-testid="stSidebar"] hr {border-color: #4169A8 !important; opacity: 0.8;}
    .hero {background: linear-gradient(110deg, #001747, #002366, #174A91);
           padding: 2.2rem 2.5rem; border-radius: 18px; margin-bottom: 1.4rem;}
    .hero h1 {color: #FFFFFF; font-size: 2.15rem; line-height: 1.2; margin: .1rem 0 .65rem;}
    .hero p {color: #E4ECF9; max-width: 730px; margin: 0; font-size: 1rem;}
    .eyebrow {font-size: .74rem; font-weight: 750; letter-spacing: .15em;
              color: #B9D0F5; margin-bottom: .55rem;}
    .section-title {font-size: 1.27rem; font-weight: 750; color: #002366;
                    margin: 1.1rem 0 .4rem;}
    .subtype-pill {display: inline-block; padding: .35rem .8rem;
                   background: #DCE7F8; border-radius: 99px;
                   color: #002366; font-weight: 750;}
    div[data-testid="stMetric"] {background: #FFFFFF; border: 1px solid #DFE6F1;
        border-radius: 13px; padding: 1rem 1.15rem;}
    div[data-testid="stMetricLabel"] {color: #63748E;}
    div[data-testid="stMetricValue"] {color: #002366;}
    div[data-testid="stFileUploader"] {background: #FFFFFF; border-radius: 13px;
        padding: .8rem; border: 1px solid #DFE6F1;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load("breast_cancer_stacking.pkl")

with st.sidebar:
    st.markdown("## 🧬 OncoOmics")
    st.caption("MOLECULAR SUBTYPE EXPLORER")
    st.divider()
    st.markdown("""
    **Model Architecture**
    Stacking Ensemble:
    - HistGradientBoosting
    - L1 Logistic Regression
    - Logistic Regression meta-model
    """)
    st.divider()
    st.markdown("""
    **Molecular Inputs**
    - Gene Expression (EXPR)
    - Copy Number Variation (CNV)
    - Somatic Mutation Status (SNP)
    """)
    st.divider()
    st.caption("Trained on METABRIC | Externally evaluated on TCGA-BRCA")
    st.divider()
    st.caption("Research demonstration only. Not for clinical diagnosis or treatment decisions.")

st.markdown("""
<div class="hero">
  <div class="eyebrow">MULTI-OMICS · MACHINE LEARNING</div>
  <h1>Breast Cancer Molecular Subtype Explorer</h1>
  <p>Explore molecular subtype classifications from gene expression, copy-number variation,
  and somatic mutation profiles using a trained stacking ensemble.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-title">01 · Upload molecular data</div>', unsafe_allow_html=True)
st.markdown('<p class="muted">Upload a CSV containing the Multi-omics data (EXPR, CNV, SNP). The model automatically applies its fitted preprocessing and feature selection.</p>', unsafe_allow_html=True)
file = st.file_uploader("Multi-omics CSV", type="csv", label_visibility="collapsed")

if file is None:
    st.info("Upload a dataset to explore subtype predictions and probabilities.")
    st.stop()

try:
    df = pd.read_csv(file)
except Exception as exc:
    st.error(f"Unable to read CSV: {exc}")
    st.stop()

if df.empty:
    st.error("The uploaded CSV is empty.")
    st.stop()

model = load_model()
features = list(model.feature_names_in_)
missing = [feature for feature in features if feature not in df.columns]
if missing:
    st.error(f"Missing {len(missing):,} of {len(features):,} required molecular features.")
    with st.expander("View missing feature names"):
        st.write(missing[:100])
    st.stop()

X = df.loc[:, features]
if X.isna().any().any():
    st.error("Required molecular features contain missing values. Please correct the CSV before prediction.")
    st.stop()

with st.spinner("Generating subtype predictions…"):
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

subtypes = [CLASS_NAMES[c] for c in model.classes_]
probs = pd.DataFrame(probabilities, columns=subtypes)
results = pd.DataFrame({"Predicted subtype": [CLASS_NAMES[p] for p in predictions]})
for name in subtypes:
    results[f"{name} (%)"] = (probs[name] * 100).round(2)

# Preserve an uploaded sample identifier when one is supplied.
for identifier in ("Sample_ID", "sample_id", "SampleID"):
    if identifier in df.columns:
        sample_ids = df[identifier].values
        break
else:
    sample_ids = [f"Sample_{i + 1}" for i in range(len(df))]
results.insert(0, "Sample_ID", sample_ids)

st.markdown('<div class="section-title">02 · Cohort overview</div>', unsafe_allow_html=True)
counts = results["Predicted subtype"].value_counts().reindex(subtypes, fill_value=0)
most_common = counts.idxmax()
col1, col2, col3 = st.columns(3)
col1.metric("Samples analysed", f"{len(results):,}")
col2.metric("Subtypes detected", f"{(counts > 0).sum()} / 4")
col3.metric("Most frequent prediction", most_common)

left, right = st.columns([1.7, 1], gap="large")
with left:
    st.markdown("**Predicted subtype distribution**")
    chart_df = counts.rename_axis("Subtype").reset_index(name="Samples")
    st.bar_chart(chart_df, x="Subtype", y="Samples", color="Subtype", height=310)
with right:
    st.markdown("**Cohort composition**")
    for subtype in subtypes:
        count = int(counts[subtype])
        percent = count / len(results)
        st.write(f"**{subtype}** · {count:,} samples ({percent:.1%})")
        st.progress(percent)

st.markdown('<div class="section-title">03 · Explore an individual sample</div>', unsafe_allow_html=True)
selector_col, summary_col = st.columns([1, 2], gap="large")
with selector_col:
    sample_idx = st.selectbox("Select a sample", range(len(results)), format_func=lambda i: f"Sample {i + 1}")
    subtype = results.iloc[sample_idx]["Predicted subtype"]
    st.markdown("**Predicted molecular subtype**")
    st.markdown(f'<span class="subtype-pill">{subtype}</span>', unsafe_allow_html=True)
    st.caption("Predicted class is not a final clinical diagnosis.")
with summary_col:
    st.markdown("**Class probabilities**")
    sample_probs = probs.iloc[sample_idx]
    for subtype in subtypes:
        probability = float(sample_probs[subtype])
        a, b = st.columns([3, 1])
        a.write(f"{subtype}")
        b.markdown(f"**{probability:.1%}**")
        st.progress(probability)
    st.caption("Model probabilities are not validated clinical risk estimates.")

st.markdown('<div class="section-title">04 · Prediction results</div>', unsafe_allow_html=True)
st.dataframe(results, use_container_width=True, hide_index=True)
st.download_button(
    "⬇ Download prediction results (CSV)",
    data=results.to_csv(index=False),
    file_name="breast_cancer_predictions.csv",
    mime="text/csv",
    type="primary"
)
st.divider()
st.markdown('<p class="footnote">Research demonstration · Input measurements must follow the same upstream normalisation and encoding conventions as the training data. Not for clinical use.</p>', unsafe_allow_html=True)
