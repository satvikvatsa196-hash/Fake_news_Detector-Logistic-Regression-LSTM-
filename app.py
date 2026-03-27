"""
Fake News Detection System — Streamlit App
==========================================
Run with:  streamlit run app.py

Pages:
  1. 🏠 Overview        — project intro, dataset stats
  2. 🔬 Train Model     — train LR pipeline, live metrics
  3. 🔍 Live Detector   — paste article → real-time prediction
  4. 📊 Model Evaluation— confusion matrix, ROC, PR curve, CV
  5. 🧠 Embeddings      — TF-IDF top features, word importance
  6. 📚 LSTM Theory     — conceptual LSTM walkthrough
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from data.generate_dataset import generate_dataset
from utils.train import train_and_evaluate, load_model, predict_article
from utils.features import LinguisticFeatureExtractor, CREDIBILITY_TERMS, SENSATIONALISM_TERMS

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
        margin: 4px;
    }
    .metric-value { font-size: 28px; font-weight: 600; margin: 0; }
    .metric-label { font-size: 13px; color: #6c757d; margin: 4px 0 0; }
    .fake-badge {
        background: #FAECE7; color: #D85A30;
        padding: 4px 14px; border-radius: 20px;
        font-weight: 600; font-size: 15px;
    }
    .real-badge {
        background: #E1F5EE; color: #1D9E75;
        padding: 4px 14px; border-radius: 20px;
        font-weight: 600; font-size: 15px;
    }
    .stProgress > div > div > div { border-radius: 4px; }
    .feature-row { padding: 6px 0; border-bottom: 1px solid #f0f0f0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state helpers
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_dataset(n_real: int, n_fake: int) -> pd.DataFrame:
    return generate_dataset(n_real, n_fake)


def get_results() -> dict | None:
    return st.session_state.get("results", None)


def get_model():
    if "model" in st.session_state:
        return st.session_state["model"]
    m = load_model()
    if m:
        st.session_state["model"] = m
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 Fake News Detector")
    st.markdown("*NLP-powered misinformation classifier*")
    st.divider()

    page = st.radio(
        "Navigate",
        ["🏠 Overview", "🔬 Train Model", "🔍 Live Detector",
         "📊 Model Evaluation", "🧠 Feature Analysis", "📚 LSTM Theory"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**Tech stack**")
    st.markdown("""
- `scikit-learn` — ML pipeline
- `TF-IDF` — text embeddings
- `Logistic Regression` — classifier
- `Streamlit` — UI
    """)
    st.markdown("---")
    st.caption("Built as an educational ML project")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Overview
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🔍 Fake News Detection System")
    st.markdown("### NLP + Machine Learning Pipeline for Misinformation Classification")
    st.markdown("""
    This project demonstrates a complete machine learning pipeline for detecting fake news,
    covering text preprocessing, TF-IDF embeddings, Logistic Regression classification,
    and thorough model evaluation (precision, recall, ROC, confusion matrix).
    """)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        #### 📐 Pipeline Steps
        1. **Data Generation** — synthetic real/fake corpus
        2. **Text Cleaning** — lowercase, remove noise
        3. **Feature Extraction** — TF-IDF + linguistic
        4. **Model Training** — Logistic Regression
        5. **Evaluation** — precision/recall/AUC
        6. **Live Inference** — real-time prediction
        """)
    with col2:
        st.markdown("""
        #### 🧰 Models & Techniques
        - **TF-IDF** (word & char n-grams)
        - **Logistic Regression** with L2 regularization
        - **FeatureUnion** — combine multiple feature sets
        - **Cross-validation** — k-fold CV on training set
        - **LSTM** (conceptual walkthrough in tab 6)
        """)
    with col3:
        st.markdown("""
        #### 📈 Evaluation Metrics
        - **Accuracy** — overall correct predictions
        - **Precision** — of predicted fakes, how many real?
        - **Recall** — of real fakes, how many caught?
        - **F1 Score** — harmonic mean of P & R
        - **AUC-ROC** — discrimination ability
        - **Confusion Matrix** — error breakdown
        """)

    st.markdown("---")
    st.subheader("📦 Dataset Preview")

    n_samples = st.slider("Samples to generate (real + fake each)", 100, 1000, 300, 100)

    with st.spinner("Generating dataset…"):
        df = get_dataset(n_samples, n_samples)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Articles", len(df))
    col2.metric("Real News", (df["label"] == 0).sum())
    col3.metric("Fake News", (df["label"] == 1).sum())
    col4.metric("Avg Text Length", f"{int(df['full_text'].str.len().mean())} chars")

    st.dataframe(
        df[["title", "label_name", "text"]].rename(columns={"label_name": "label"}),
        use_container_width=True,
        height=300,
    )

    # Label distribution
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    # Pie
    counts = df["label_name"].value_counts()
    axes[0].pie(counts, labels=counts.index, autopct="%1.1f%%",
                colors=["#1D9E75", "#D85A30"], startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[0].set_title("Label Distribution", fontweight="bold")

    # Text length distribution
    for lbl, color, name in [(0, "#1D9E75", "Real"), (1, "#D85A30", "Fake")]:
        subset = df[df["label"] == lbl]["full_text"].str.len()
        axes[1].hist(subset, bins=30, alpha=0.6, color=color, label=name)
    axes[1].set_xlabel("Text Length (chars)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Text Length by Label", fontweight="bold")
    axes[1].legend()

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Train Model
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Train Model":
    st.title("🔬 Train the Model")
    st.markdown("Configure the dataset and pipeline, then click **Train** to build the classifier.")

    col1, col2 = st.columns(2)
    with col1:
        n_samples = st.slider("Articles per class", 100, 1000, 400, 100)
        test_size = st.slider("Test set fraction", 0.1, 0.4, 0.2, 0.05)
    with col2:
        st.markdown("""
        **Pipeline architecture:**
        ```
        Input text
          ├── TF-IDF (word n-grams, 1-2)  ─┐
          ├── TF-IDF (char n-grams, 2-4)   ├── FeatureUnion
          └── Linguistic features (14)    ─┘
                         │
              Logistic Regression (C=1.0, L2)
                         │
              P(fake) ∈ [0, 1]
        ```
        """)

    if st.button("🚀 Train Model", type="primary", use_container_width=True):
        with st.spinner("Generating dataset…"):
            df = get_dataset(n_samples, n_samples)

        progress = st.progress(0, text="Preparing features…")

        with st.spinner("Training pipeline (TF-IDF + Logistic Regression)…"):
            progress.progress(30, text="Extracting TF-IDF features…")
            results = train_and_evaluate(df, test_size=test_size)
            progress.progress(80, text="Evaluating model…")
            st.session_state["results"] = results
            st.session_state["model"] = results["model"]
            st.session_state["df"] = df
            progress.progress(100, text="Done!")

        st.success("✅ Model trained and saved!")

        m = results["metrics"]
        c1, c2, c3, c4, c5 = st.columns(5)
        for col, name, val, color in zip(
            [c1, c2, c3, c4, c5],
            ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"],
            [m["accuracy"], m["precision"], m["recall"], m["f1"], m["auc"]],
            ["#378ADD", "#1D9E75", "#EF9F27", "#D85A30", "#7F77DD"],
        ):
            col.markdown(f"""
            <div class="metric-card">
              <p class="metric-value" style="color:{color}">{val*100:.1f}%</p>
              <p class="metric-label">{name}</p>
            </div>""", unsafe_allow_html=True)

        # CV scores
        cv = results["cv_scores"]
        st.markdown(f"""
        **5-fold Cross-Validation F1:** {cv.mean()*100:.1f}% ± {cv.std()*100:.1f}%
        — fold scores: {', '.join(f'{s*100:.1f}%' for s in cv)}
        """)

        st.markdown("#### Full Classification Report")
        st.code(results["classification_report"])

    else:
        if get_results():
            st.info("✅ A model is already trained. Navigate to **Live Detector** or **Model Evaluation** to use it.")
        else:
            st.info("👆 Click **Train Model** to start. Training takes ~5 seconds.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Live Detector
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Live Detector":
    st.title("🔍 Live Fake News Detector")

    model = get_model()
    if model is None:
        st.warning("⚠️ No trained model found. Please go to **Train Model** first.")
        st.stop()

    SAMPLES = {
        "Real — NASA Mars study": (
            "Scientists at NASA confirm water ice presence below Martian surface",
            "NASA researchers confirmed the presence of water ice deposits beneath the Martian "
            "surface using the SHARAD radar instrument aboard the Mars Reconnaissance Orbiter. "
            "The peer-reviewed findings, published in the journal Science, indicate significant "
            "subsurface ice sheets at latitudes above 55 degrees. Dr. Williams, lead researcher, "
            "stated the data was collected over a 12-year period and controlled for seasonal variation. "
            "The study recommends further analysis before extrapolating to potential resource utilization.",
        ),
        "Fake — 5G vaccines conspiracy": (
            "BREAKING: Government secretly puts 5G chips in vaccines!!!",
            "The TRUTH they don't want you to know!!! A brave WHISTLEBLOWER has EXPOSED the "
            "shocking plot by the deep state to track citizens through mandatory injections. "
            "The mainstream media is SILENT on this story — ask yourself WHY. "
            "Anonymous insiders confirm the cover-up goes all the way to the TOP. "
            "Share this IMMEDIATELY before it gets DELETED!!! Wake up, sheeple!!! "
            "Big Pharma is making BILLIONS from your suffering. They are SILENCING anyone who speaks out.",
        ),
        "Real — Central bank rate decision": (
            "Federal Reserve raises interest rates amid persistent inflation",
            "The Federal Reserve voted 10-2 to raise the benchmark interest rate by 0.25 percentage points, "
            "bringing it to its highest level in 15 years. Chair Williams stated the committee remains "
            "data-dependent and will assess incoming economic data at future meetings. "
            "Markets responded with a 0.3% rise in Treasury yields. "
            "Economists surveyed by Reuters had largely anticipated the decision. "
            "The next policy meeting is scheduled for March.",
        ),
        "Fake — Celebrity cover-up": (
            "Famous celebrity found dead — MEDIA BLACKOUT ordered by FBI!!!",
            "You will NOT see this on CNN or Fox News. A well-known celebrity has been found dead "
            "under MYSTERIOUS circumstances that authorities are desperately covering up. "
            "Our anonymous source inside the FBI says this goes DEEPER than anyone knows. "
            "The globalists DON'T WANT YOU to connect the dots. "
            "Share this with everyone you trust before they CENSOR the truth. "
            "Government documents PROVE the cover-up. This is NOT a conspiracy — it's FACT!!!",
        ),
    }

    sample_choice = st.selectbox("Load a sample article:", ["(write your own)"] + list(SAMPLES.keys()))

    if sample_choice != "(write your own)":
        default_title, default_text = SAMPLES[sample_choice]
    else:
        default_title, default_text = "", ""

    title = st.text_input("Headline / Title", value=default_title)
    text = st.text_area("Article Body", value=default_text, height=160)

    if st.button("🔍 Analyze Article", type="primary", use_container_width=True):
        full_text = (title + " " + text).strip()
        if not full_text:
            st.warning("Please enter some text.")
        else:
            with st.spinner("Analyzing…"):
                result = predict_article(full_text, model)

            # ── Verdict ────────────────────────────────────────────────────
            st.markdown("---")
            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("#### Verdict")
                badge = "fake-badge" if result["label"] == 1 else "real-badge"
                label_text = "🚨 FAKE NEWS" if result["label"] == 1 else "✅ LIKELY REAL"
                st.markdown(f'<span class="{badge}">{label_text}</span>', unsafe_allow_html=True)

                st.metric("Confidence", f"{result['confidence']*100:.1f}%")
                st.metric("P(Fake)", f"{result['prob_fake']*100:.1f}%")
                st.metric("P(Real)", f"{result['prob_real']*100:.1f}%")

            with col2:
                st.markdown("#### Probability breakdown")
                fig, ax = plt.subplots(figsize=(5, 2))
                colors = ["#1D9E75", "#D85A30"]
                bars = ax.barh(["Real", "Fake"], [result["prob_real"], result["prob_fake"]],
                               color=colors, height=0.4)
                for bar, val in zip(bars, [result["prob_real"], result["prob_fake"]]):
                    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                            f"{val*100:.1f}%", va="center", fontsize=11, fontweight="bold")
                ax.set_xlim(0, 1.15)
                ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
                ax.set_xlabel("Probability")
                ax.spines[["top", "right"]].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            # ── Linguistic feature breakdown ────────────────────────────────
            st.markdown("#### Linguistic Feature Signals")
            feats = result["features"]
            feat_cols = st.columns(3)
            items = list(feats.items())
            for i, (fname, fval) in enumerate(items[:12]):
                with feat_cols[i % 3]:
                    display = f"{fval:.2f}" if isinstance(fval, float) and fval != int(fval) else str(int(fval))
                    risk_color = "#D85A30" if fname in ["exclamation_count", "caps_ratio", "sensationalism_score", "all_caps_words"] and float(fval) > 0 else "#1D9E75"
                    st.markdown(f"""
                    <div class="feature-row">
                      <span style="font-size:12px;color:#888">{fname.replace('_',' ')}</span><br/>
                      <span style="font-size:16px;font-weight:600;color:{risk_color}">{display}</span>
                    </div>""", unsafe_allow_html=True)

            # ── Keyword highlights ─────────────────────────────────────────
            lower_text = full_text.lower()
            found_fake = [t for t in SENSATIONALISM_TERMS if t in lower_text]
            found_real = [t for t in CREDIBILITY_TERMS if t in lower_text]

            if found_fake:
                st.markdown(f"🚨 **Sensationalism markers found:** {', '.join(f'`{t}`' for t in found_fake[:10])}")
            if found_real:
                st.markdown(f"✅ **Credibility markers found:** {', '.join(f'`{t}`' for t in found_real[:10])}")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Model Evaluation
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Evaluation":
    st.title("📊 Model Evaluation")

    results = get_results()
    if results is None:
        st.warning("⚠️ No results found. Please go to **Train Model** first.")
        st.stop()

    m = results["metrics"]
    cv = results["cv_scores"]
    cm = results["confusion_matrix"]

    # ── Summary metrics ────────────────────────────────────────────────────
    st.subheader("Performance Summary")
    cols = st.columns(5)
    for col, name, val, color in zip(
        cols,
        ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"],
        [m["accuracy"], m["precision"], m["recall"], m["f1"], m["auc"]],
        ["#378ADD", "#1D9E75", "#EF9F27", "#D85A30", "#7F77DD"],
    ):
        col.markdown(f"""
        <div class="metric-card">
          <p class="metric-value" style="color:{color}">{val*100:.1f}%</p>
          <p class="metric-label">{name}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"**5-fold CV F1:** {cv.mean()*100:.1f}% ± {cv.std()*100:.1f}%")

    st.markdown("---")
    col1, col2 = st.columns(2)

    # ── Confusion Matrix ───────────────────────────────────────────────────
    with col1:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Predicted Real", "Predicted Fake"],
            yticklabels=["Actual Real", "Actual Fake"],
            ax=ax, linewidths=0.5, cbar=False,
            annot_kws={"size": 14, "weight": "bold"},
        )
        ax.set_title("Confusion Matrix", fontweight="bold")
        tn, fp, fn, tp = cm.ravel()
        ax.set_xlabel(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}", fontsize=9, color="gray")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        with st.expander("What does each cell mean?"):
            st.markdown(f"""
| Cell | Count | Meaning |
|------|-------|---------|
| True Negative (TN) | {tn} | Real article correctly identified as real |
| False Positive (FP) | {fp} | Real article incorrectly flagged as fake |
| False Negative (FN) | {fn} | Fake article missed (classified as real) |
| True Positive (TP) | {tp} | Fake article correctly detected |

**Precision** = TP / (TP + FP) = {tp}/{tp+fp} = **{tp/(tp+fp)*100:.1f}%**  
**Recall** = TP / (TP + FN) = {tp}/{tp+fn} = **{tp/(tp+fn)*100:.1f}%**
            """)

    # ── ROC Curve ─────────────────────────────────────────────────────────
    with col2:
        st.subheader("ROC Curve")
        roc = results["roc"]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(roc["fpr"], roc["tpr"], color="#378ADD", lw=2,
                label=f"LR (AUC = {m['auc']:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Random classifier")
        ax.fill_between(roc["fpr"], roc["tpr"], alpha=0.1, color="#378ADD")
        ax.set_xlabel("False Positive Rate (FPR)")
        ax.set_ylabel("True Positive Rate (Recall)")
        ax.set_title("ROC Curve", fontweight="bold")
        ax.legend(loc="lower right")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Precision-Recall curve ─────────────────────────────────────────────
    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Precision–Recall Curve")
        pr = results["pr_curve"]
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(pr["recall"], pr["precision"], color="#D85A30", lw=2)
        ax.fill_between(pr["recall"], pr["precision"], alpha=0.1, color="#D85A30")
        ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision–Recall Curve", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col4:
        st.subheader("Cross-Validation Scores")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        folds = [f"Fold {i+1}" for i in range(len(cv))]
        colors = ["#1D9E75" if s >= cv.mean() else "#D85A30" for s in cv]
        bars = ax.bar(folds, cv * 100, color=colors, edgecolor="white", linewidth=1.5)
        ax.axhline(cv.mean() * 100, color="#378ADD", linestyle="--", lw=1.5,
                   label=f"Mean = {cv.mean()*100:.1f}%")
        for bar, val in zip(bars, cv):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{val*100:.1f}%", ha="center", fontsize=9, fontweight="bold")
        ax.set_ylabel("F1 Score (%)")
        ax.set_title("5-Fold Cross-Validation", fontweight="bold")
        ax.set_ylim(0, 115)
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Classification report ──────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Full Classification Report")
    st.code(results["classification_report"])


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Feature Analysis
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🧠 Feature Analysis":
    st.title("🧠 Feature Analysis & TF-IDF Embeddings")

    results = get_results()
    if results is None:
        st.warning("⚠️ No results found. Please go to **Train Model** first.")
        st.stop()

    top = results["top_features"]
    df = st.session_state.get("df", get_dataset(300, 300))

    # ── Top LR features ────────────────────────────────────────────────────
    st.subheader("Logistic Regression — Top Weighted Features")
    st.markdown("Features with the **highest positive coefficients** push predictions toward **Fake**; negative toward **Real**.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🚨 Top features → FAKE")
        if top["fake_features"]:
            names, vals = zip(*top["fake_features"][:15])
            fig, ax = plt.subplots(figsize=(5, 4.5))
            y_pos = np.arange(len(names))
            ax.barh(y_pos, vals, color="#D85A30", alpha=0.85)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([n[:35] for n in names], fontsize=9)
            ax.set_xlabel("Coefficient weight")
            ax.set_title("Fake-leaning features", fontweight="bold")
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with col2:
        st.markdown("#### ✅ Top features → REAL")
        if top["real_features"]:
            names_r, vals_r = zip(*top["real_features"][:15])
            fig, ax = plt.subplots(figsize=(5, 4.5))
            y_pos = np.arange(len(names_r))
            ax.barh(y_pos, np.abs(vals_r), color="#1D9E75", alpha=0.85)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([n[:35] for n in names_r], fontsize=9)
            ax.set_xlabel("Coefficient weight (absolute)")
            ax.set_title("Real-leaning features", fontweight="bold")
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # ── Linguistic feature distributions ──────────────────────────────────
    st.markdown("---")
    st.subheader("Linguistic Feature Distributions by Label")

    from utils.features import describe_features
    with st.spinner("Computing features…"):
        feat_df = describe_features(df)

    features_to_plot = ["exclamation_count", "caps_ratio", "credibility_score",
                        "sensationalism_score", "vocab_diversity", "avg_word_len"]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.flatten()

    for i, feat in enumerate(features_to_plot):
        ax = axes[i]
        for lbl, color, name in [(0, "#1D9E75", "Real"), (1, "#D85A30", "Fake")]:
            data = feat_df[feat_df["label"] == lbl][feat]
            ax.hist(data, bins=25, alpha=0.6, color=color, label=name,
                    density=True, edgecolor="white", linewidth=0.5)
        ax.set_title(feat.replace("_", " ").title(), fontweight="bold", fontsize=11)
        ax.set_ylabel("Density")
        ax.spines[["top", "right"]].set_visible(False)
        if i == 0:
            ax.legend()

    plt.suptitle("Feature Distributions: Real vs Fake", fontweight="bold", y=1.02)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── TF-IDF explanation ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("How TF-IDF Embeddings Work")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        **Term Frequency (TF)**  
        How often does word *t* appear in document *d*?  
        ```
        TF(t, d) = count(t in d) / total_words(d)
        ```

        **Inverse Document Frequency (IDF)**  
        How rare is word *t* across all documents?  
        ```
        IDF(t) = log(N / df(t)) + 1
        ```

        **TF-IDF Score**  
        ```
        TFIDF(t, d) = TF(t, d) × IDF(t)
        ```

        **Why it works for fake news:**  
        - Rare sensational words (*whistleblower*, *sheeple*) get high scores in fake docs  
        - Common scientific terms (*published*, *peer-reviewed*) score highly in real docs  
        - Stop words (the, is, a) have near-zero weight  
        """)

    with col2:
        # Demo TF-IDF computation
        sample_texts = [
            ("Fake", "SHOCKING truth exposed deep state silenced whistleblower!!!"),
            ("Real", "Scientists published peer-reviewed study confirming findings"),
        ]
        st.markdown("**Example TF-IDF scores:**")
        demo_words = ["shocking", "truth", "published", "study", "scientists", "exposed", "whistleblower", "findings"]
        data = []
        for label, text in sample_texts:
            words = text.lower().split()
            n = len(words)
            for word in demo_words:
                tf = words.count(word) / n
                data.append({"Document": label, "Word": word, "TF": round(tf, 4)})
        demo_df = pd.DataFrame(data).pivot(index="Word", columns="Document", values="TF").fillna(0)
        st.dataframe(demo_df.style.background_gradient(cmap="RdYlGn", axis=None), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 6 — LSTM Theory
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📚 LSTM Theory":
    st.title("📚 LSTM for Text Classification")
    st.markdown("A conceptual walkthrough of how Long Short-Term Memory networks would be applied to fake news detection.")

    # Architecture diagram (matplotlib)
    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4)
    ax.axis("off")

    boxes = [
        (0.5, "Raw\nText", "#E6F1FB", "#378ADD"),
        (2.2, "Tokenize\n+ Embed", "#EAF3DE", "#1D9E75"),
        (4.2, "LSTM\nLayer 1", "#FAEEDA", "#BA7517"),
        (6.2, "LSTM\nLayer 2", "#FAECE7", "#D85A30"),
        (8.2, "Dropout\n0.3", "#F1EFE8", "#888780"),
        (10.2, "Dense\nSigmoid", "#E1F5EE", "#1D9E75"),
        (12.0, "P(fake)\n∈ [0,1]", "#FAECE7", "#D85A30"),
    ]

    for x, label, bg, border in boxes:
        rect = mpatches.FancyBboxPatch((x, 1.2), 1.4, 1.6, boxstyle="round,pad=0.1",
                                        facecolor=bg, edgecolor=border, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.7, 2.0, label, ha="center", va="center",
                fontsize=8.5, color=border, fontweight="bold")

    for i in range(len(boxes) - 1):
        x_start = boxes[i][0] + 1.4
        x_end = boxes[i+1][0]
        ax.annotate("", xy=(x_end, 2.0), xytext=(x_start, 2.0),
                    arrowprops=dict(arrowstyle="->", color="#888780", lw=1.2))

    ax.set_title("LSTM Architecture for Fake News Detection", fontweight="bold", fontsize=13, pad=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Why LSTM over Logistic Regression?

        | Property | LR + TF-IDF | LSTM |
        |----------|-------------|------|
        | Captures word order | ❌ | ✅ |
        | Long-range context | ❌ | ✅ |
        | Training speed | ~2 sec | ~30 min |
        | Interpretability | High | Low |
        | Data required | Low (~500) | High (~10k+) |
        | Typical accuracy | 84–87% | 88–93% |

        **When to use LSTM:**  
        - Large labeled dataset available (10,000+ articles)
        - Sequential patterns matter ("was not true" vs "was true")
        - Resources for GPU training available
        """)

    with col2:
        st.markdown("""
        ### LSTM Cell Equations

        The LSTM cell controls information flow via three gates:

        **Forget gate** — what to discard from memory:
        ```
        f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
        ```

        **Input gate** — what new info to store:
        ```
        i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
        c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)
        ```

        **Cell state update:**
        ```
        c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
        ```

        **Output gate:**
        ```
        o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
        h_t = o_t ⊙ tanh(c_t)
        ```
        """)

    st.markdown("---")
    st.subheader("Step-by-step: How LSTM reads a fake news article")

    example_tokens = ["SHOCKING", "truth", "EXPOSED", "deep", "state", "whistleblower", "silenced", "!!!"]
    cols = st.columns(len(example_tokens))
    gate_vals = [0.9, 0.3, 0.85, 0.7, 0.6, 0.8, 0.75, 0.95]

    for i, (col, token, gate) in enumerate(zip(cols, example_tokens, gate_vals)):
        color = f"rgb({int(gate*220)}, {int((1-gate)*180)}, 60)"
        col.markdown(f"""
        <div style="text-align:center; padding:8px; background:#f8f9fa; border-radius:8px; border:1px solid #dee2e6; margin:2px">
          <div style="font-size:11px; font-weight:bold; word-break:break-word">{token}</div>
          <div style="font-size:10px; color:#888; margin-top:4px">step {i+1}</div>
          <div style="height:4px; border-radius:2px; background:{color}; margin-top:4px; width:{int(gate*100)}%"></div>
          <div style="font-size:9px; color:#888">{gate:.2f}</div>
        </div>""", unsafe_allow_html=True)

    st.caption("The colored bar represents the approximate input gate activation — how strongly each token influences the LSTM hidden state. Sensational all-caps words activate strongly.")

    st.markdown("---")
    st.subheader("Implementation with Keras (code reference)")
    st.code("""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Tokenize
tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=300)
X_test_seq  = pad_sequences(tokenizer.texts_to_sequences(X_test),  maxlen=300)

# 2. Build model
model = Sequential([
    Embedding(input_dim=20000, output_dim=128, input_length=300),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid"),         # P(fake) ∈ [0, 1]
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# 3. Train
history = model.fit(
    X_train_seq, y_train,
    epochs=10, batch_size=64,
    validation_split=0.1,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)],
)

# 4. Evaluate
y_proba = model.predict(X_test_seq).flatten()
y_pred  = (y_proba > 0.5).astype(int)
    """, language="python")
