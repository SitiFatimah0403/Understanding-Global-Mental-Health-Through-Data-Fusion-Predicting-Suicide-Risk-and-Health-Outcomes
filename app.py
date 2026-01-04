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

st.markdown(
    """
    <style>
    /* Make link buttons blue */
    div[data-testid="stLinkButton"] > a {
        background-color: #1f77b4 !important;
        color: white !important;
        border-radius: 8px;
        padding: 0.6em 1em;
        font-weight: 600;
        text-decoration: none;
    }

    /* Sidebar radio selected circle */
    section[data-testid="stSidebar"] div[role="radiogroup"] label span:first-child {
        border-color: #1f77b4 !important;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] input:checked + div span:first-child {
        background-color: #1f77b4 !important;
        border-color: #1f77b4 !important;
    }

    /* Hover effect */
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        color: #1f77b4 !important;
    }

    /* Make st.info text brighter blue */
        div[data-testid="stInfo"] {
            border-left: 4px solid #1f77b4 !important;
            background-color: rgba(31, 119, 180, 0.08) !important;
        }

        div[data-testid="stInfo"] * {
            color: #a9cee9ff !important;
            font-weight: 500;
        }

    </style>
    """,
    unsafe_allow_html=True
)



# -----------------
# Sidebar
# -----------------
st.sidebar.markdown(
    """
    # Suicide Risk Dashboard
    _Fused Dataset Analysis_
    """
)

st.sidebar.divider()

section = st.sidebar.radio(
    "Go to:",
    ["Introduction", "Classification", "Regression", "Clustering"],
    label_visibility="collapsed"
)

st.sidebar.divider()

st.sidebar.caption("DebugDivas . HealthVerse")

# -----------------
# Main Page Title
# -----------------
st.title("🌍 Global Mental Health Dashboard")

# -----------------
# Pages
# -----------------
if section == "Introduction":
    st.header("Suicide Risk Analysis")

    col1, col2, col3 = st.columns([2,1, 1])

# -----------------
# LEFT COLUMN
# -----------------
    with col1:
        st.markdown(
            """
            This dashboard presents **Classification, Regression, and Clustering**
            results using a **fused global mental health dataset**.

            ### Objectives
            - Predict suicide risk using machine learning  
            - Compare baseline, tuned, and AutoML models  
            - Explain model decisions using XAI (SHAP)  
            """
        )

    # -----------------
    # MIDDLE COLUMN
    # -----------------
    with col2:
        st.info(
            """
            ### **Classification & Regression Methods**
            - Manual ML (Scaled & Non-Scaled)
            - Hyperparameter Tuning
            - AutoML
            - XAI (SHAP)

            ### **Clustering Methods**
            - K-Means
            - DBSCAN
            - Hierarchical Clustering
            """
        )

    # -----------------
    # RIGHT COLUMN
    # -----------------
    with col3:
        st.info("### **Fused Dataset consists of:**")

        st.link_button(
            "Suicide Statistics Dataset",
            "https://www.kaggle.com/datasets/omkargowda/suicide-rates-overview-1985-to-2021"
        )

        st.link_button(
            "Life Expectancy Dataset",
            "https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who"
        )


# ----------
# Classification
# ----------

elif section == "Classification":
    st.header("Classification Results")

    tabs = st.tabs([
        "Baseline",
        "Tuning",
        "AutoML",
        "XAI"
    ])

    # -------------------
    # TAB 1: BASELINE
    # -------------------
    with tabs[0]:
        st.subheader("Baseline Models - Scaling vs No Scaling")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**No Scaling (80–20)**")
            df_ns_80 = pd.read_csv("outputs/nonscaled_fused_cls_baseline_80_20.csv")
            st.dataframe(df_ns_80, use_container_width=True)

        with col2:
            st.markdown("**No Scaling (70–30)**")
            df_ns_70 = pd.read_csv("outputs/nonscaled_fused_cls_baseline_70_30.csv")
            st.dataframe(df_ns_70, use_container_width=True)


        labels_fused = [
            "80-20 Scaled (SVM)",
            "80-20 Non-Scaled (SVM)",
            "70-30 Scaled (ANN)",
            "70-30 Non-Scaled (Naive Bayes)"
        ]
        f1_fused = [0.658262, 0.642985, 0.658356, 0.643151]

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.bar(labels_fused, f1_fused)
            ax.set_ylabel("F1 Score")
            ax.set_title("Scaling vs No Scaling (The best Models)")
            ax.set_ylim(0.6, 0.7)
            plt.xticks(rotation=20)
            st.pyplot(fig, use_container_width=False)

        st.markdown(
        """
        ### What the graph shows
        This graph compares the **best F1 score** achieved under different **train–test splits**
        and **scaling settings** for the fused dataset.

        ---

        ### Interpretation
        - For both **80–20** and **70–30** splits, **scaled data consistently achieves higher F1 scores**
        than non-scaled data.
        - The **80–20 scaled (SVM)** and **70–30 scaled (ANN)** settings achieved the **highest performance**
        (≈ **0.658**).
        - In contrast, **non-scaled models** (SVM and Naive Bayes) show noticeably lower F1 scores
        (≈ **0.643**).

        ---

        ### Conclusion
        Scaling is beneficial for the fused dataset because it contains **heterogeneous features**
        (health, demographic, and socio-economic indicators) with different value ranges.

      
        """
        )


    # -------------------
    # TAB 2: TUNING
    # -------------------
    with tabs[1]:
        st.subheader("Hyperparameter Tuning - gridSearch vs randomSearch")

        tuned_df = pd.read_csv("outputs/tuned_cls.csv")
        st.dataframe(tuned_df, use_container_width=True)

        models_fused = ["SVM", "ANN", "XGBoost"]
        grid_f1_fused = [0.6723, 0.6722, 0.6815]
        random_f1_fused = [0.6723, 0.6531, 0.6737]

        x = np.arange(len(models_fused))
        width = 0.35

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.bar(x - width/2, grid_f1_fused, width, label="Grid Search")
            ax.bar(x + width/2, random_f1_fused, width, label="Random Search")
            ax.set_xticks(x)
            ax.set_xticklabels(models_fused)
            ax.set_ylabel("F1 Score")
            ax.set_title("Grid vs Random Search")
            ax.legend()
            ax.set_ylim(0.64, 0.70)

            st.pyplot(fig, use_container_width=False)

        st.markdown(
        """

        ### What the graph shows
        This graph compares **GridSearchCV** and **RandomizedSearchCV** for the top three
        classification models (**SVM, ANN, and XGBoost**) using the **scaled fused dataset**.

        ---

        ### Interpretation
        - **SVM** shows identical performance under both Grid Search and Random Search  
        (F1 ≈ **0.6723**), indicating **stable hyperparameter sensitivity**.
        - **ANN** performs better with **Grid Search** (F1 ≈ **0.6722**) than with
        Random Search (≈ **0.6531**), suggesting ANN benefits from **systematic parameter exploration**.
        - **XGBoost** achieves the **highest F1 score overall**, where  
        **Grid Search (0.6815)** outperforms Random Search (0.6737).

        ---

        ### Conclusion
        Grid Search is more effective for the fused dataset, especially for
        **complex models such as ANN and XGBoost**.

        As a result, **XGBoost tuned using Grid Search** is selected as the
        **best-performing manual classification model**.
        """
        )

    # -------------------
    # TAB 3: AUTOML
    # -------------------
    with tabs[2]:
        st.subheader("AutoML vs Manual Modeling")

        fused_automl_df = pd.read_csv("outputs/fused_cls_automl.csv")
        st.dataframe(fused_automl_df, use_container_width=True)

        fused_automl_df = pd.read_csv("outputs/cls_manual_automl_compare.csv")
        st.dataframe(fused_automl_df, use_container_width=True)

        labels = ["Grid Search (XGBoost)", "AutoML (LightGBM)"]
        f1_vals = [0.6815,0.67975]

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.bar(labels, f1_vals)
            ax.set_ylabel("F1 Score")
            ax.set_title("Tuned Model vs AutoML")
            ax.set_ylim(0.65, 0.72)


            st.pyplot(fig, use_container_width=False)

        st.markdown(
        """

        ### What the graph shows
        This graph compares the **best manually tuned model**
        (**XGBoost with Grid Search**) against an **AutoML-generated model**
        (**LightGBM**) using the scaled fused dataset.

        ---

        ### Interpretation
        - **Tuned XGBoost** achieves an F1 score of **0.6815**.
        - **AutoML (LightGBM)** achieves a slightly lower F1 score of **0.6798**.
        - The performance gap is **very small**, indicating that AutoML is
        **highly competitive**, though it does not surpass expert-guided tuning.

        ---

        ### Conclusion
        Manual hyperparameter tuning **slightly outperforms AutoML** for the fused dataset.

        However, **AutoML remains a strong alternative** due to:
        - faster setup,
        - automated model selection,
        - and performance close to the optimal tuned model.
        """
        )

    
    # -------------------
    # TAB 4: XAI
    # -------------------
    with tabs[3]:
        st.subheader("XAI for The Best Model")

        st.markdown(
            """
            This section explains **why** the best-performing classification model
            makes its predictions, using **XGBoost + SHAP**.
            """
        )

        # Feature Importance
        st.markdown("### Feature Importance (XGBoost)")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(
                "outputs/xai/scaled_xgb_feature_importance.png",
                width=600
            )

        st.markdown(
            """ 
            This figure presents the global feature importance from the best tuned XGBoost model. The results show that 
            development status is the most dominant predictor of suicide risk, indicating that whether a country is developing or developed strongly influences 
            the model’s classification. Health and social indicators such as thinness among children aged 5–9 years, schooling, HIV/AIDS prevalence, income composition of resources, and adult mortality also contribute notably, 
            while economic indicators and immunisation variables play a relatively minor role. However, this plot only reflects feature importance and does not indicate whether a feature increases or decreases suicide risk.
                """
        )

        # SHAP Summary
        st.markdown("### SHAP Summary Plot")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(
                "outputs/xai/scaled_shap_summary.png",
                width=600
            )
        st.markdown(
            """ 
            This figure  shows the SHAP summary plot for the high suicide risk class, providing directional interpretability.
            Higher levels of child thinness, developing-country status, higher adult mortality, and lower life expectancy consistently push predictions toward high suicide risk. 
            In contrast, higher schooling levels and longer life expectancy reduce predicted risk, highlighting the protective role of education and overall health. 
            These findings indicate that health vulnerability, malnutrition, and lower education are key drivers of high suicide risk.
                """
        )

        # SHAP Dependence
        st.markdown("### SHAP Dependence Plot")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(
                "outputs/xai/scaled_shap_dependence_life_expectancy.png",
                width=600
            )
        st.markdown(
            """ 
            This figure illustrates the SHAP dependence plot for life expectancy with adult mortality as an interacting feature. Lower life expectancy strongly increases 
            suicide risk, and this effect is amplified when adult mortality is high, demonstrating an interaction between these two health indicators. Overall, the SHAP 
            analysis confirms that suicide risk is primarily driven by health and social conditions rather than economic factors.
                """
        )




# ----------
# Regression
# ----------

# ----------
# Clustering
# ----------