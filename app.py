# this is for streamlit page
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(
    page_title="Fused Dataset – Suicide Risk Analysis",
    layout="wide"
)

st.title("🧠 Suicide Risk Analysis (Fused Dataset)")
st.markdown("This dashboard presents **classification, regression and Clustering results** using the fused dataset.")

st.sidebar.title("Analysis Type")

section = st.sidebar.selectbox(
    "Select Analysis Type",
    ["Introduction", "Classification", "Regression", "Clustering"]
)

st.title("🌍 Global Mental Health Dashboard")

# ----------
# Classification
# ----------

if section == "Classification":
    st.header("📊 Classification Results (Fused Dataset)")

    # Baseline Results
    st.subheader("1️⃣ Baseline Models")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**No Scaling (80–20)**")
        df_ns_80 = pd.read_csv("outputs/nonscaled_fused_cls_baseline_80_20.csv")
        st.dataframe(df_ns_80)

    with col2:
        st.markdown("**No Scaling (70–30)**")
        df_ns_70 = pd.read_csv("outputs/nonscaled_fused_cls_baseline_70_30.csv")
        st.dataframe(df_ns_70)

    #Comparision Graphs
    st.subheader("📊 Scaling vs No Scaling (Fused Dataset)")

    labels_fused = [
        "80-20 Scaled (SVM)",
        "80-20 Non-Scaled (SVM)",
        "70-30 Scaled (ANN)",
        "70-30 Non-Scaled (Naive Bayes)"
    ]

    f1_fused = [0.658262, 0.642985, 0.658356, 0.643151]

    fig, ax = plt.subplots()
    ax.bar(labels_fused, f1_fused)
    ax.set_ylabel("F1 Score")
    ax.set_title("Fused Dataset: F1 Comparison (80–20 vs 70–30)")
    ax.set_ylim(0.6, 0.7)
    plt.xticks(rotation=20)

    st.pyplot(fig)

    
    # Hyperparameter Tuning
    st.subheader("2️⃣ Hyperparameter Tuning Results")
    tuned_df = pd.read_csv("outputs/tuned_cls.csv")
    st.dataframe(tuned_df)

    #Compare GridSearch and RandomSearch
    st.subheader("📊 Grid Search vs Random Search (Fused Dataset)")

    models_fused = ["SVM", "ANN", "XGBoost"]

    grid_f1_fused = [0.6723, 0.6722, 0.6815]
    random_f1_fused = [0.6723, 0.6531, 0.6737]

    x = np.arange(len(models_fused))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width/2, grid_f1_fused, width, label="Grid Search")
    ax.bar(x + width/2, random_f1_fused, width, label="Random Search")

    ax.set_xticks(x)
    ax.set_xticklabels(models_fused)
    ax.set_ylabel("F1 Score")
    ax.set_title("Scaled Fused Dataset: Grid vs Random Search")
    ax.legend()
    ax.set_ylim(0.64, 0.70)

    st.pyplot(fig)


    # AutoML Comparison
    st.subheader("3️⃣ AutoML vs Manual Modeling")
    fused_automl_df = pd.read_csv("outputs/fused_cls_automl.csv")
    st.dataframe(fused_automl_df)

    automl_df = pd.read_csv("outputs/cls_manual_automl_compare.csv")
    st.dataframe(automl_df)

    #Compare Manual Modeling and AutoML
    st.subheader("📊 Manual Modeling vs AutoML (Fused Dataset)")

    labels_fused = ["Grid Search (XGBoost)", "AutoML (LightGBM)"]
    f1_fused = [0.6815, 0.679750]

    fig, ax = plt.subplots()
    ax.bar(labels_fused, f1_fused)
    ax.set_ylabel("F1 Score")
    ax.set_title("Scaled Fused Dataset: Tuned Model vs AutoML")
    ax.set_ylim(0.65, 0.72)

    st.pyplot(fig)



# ----------
# Regression
# ----------

# ----------
# Clustering
# ----------