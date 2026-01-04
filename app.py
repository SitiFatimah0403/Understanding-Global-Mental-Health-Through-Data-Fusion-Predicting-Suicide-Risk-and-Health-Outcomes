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
            - Compare clustering evaluation metrics across different algorithms
            - Examine how clusters are formed and structured under each clustering approach

            ### Data Preparation and Justification
            During the preprocessing stage of the fused dataset, non-predictive and high-cardinality variables such as country,
            year, age, sex, suicide_no, population, gdp_for_year and generation were removed. These attributes were excluded because they do not 
            contribute directly to model learning, may introduce unnecessary dimensionality and could lead to biased or unstable model behaviour.

            ### Features Used
            """
        ) 

        fcol1, fcol2, fcol3 = st.columns(3)

        with fcol1:
            st.markdown(
                """
                - Status  
                - Life expectancy  
                - Adult Mortality  
                - Infant deaths  
                - Alcohol  
                - Percentage expenditure  
                - Hepatitis B  
                """
            )

        with fcol2:
            st.markdown(
                """
                - Measles  
                - BMI  
                - Under-five deaths  
                - Polio  
                - Total expenditure  
                - Diphtheria  
                - HIV/AIDS  
                """
            )

        with fcol3:
            st.markdown(
                """
                - GDP  
                - Thinness 5–9 years
                - Income composition of resources  
                - Schooling  
                - GDP per capita ($)  
                - HDI for year  
                - Suicides/100k population  
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
elif section == "Clustering":
    st.header("Clustering Analysis")

    tabs = st.tabs([
        "Overview & Comparison",
        "K-Means",
        "Hierarchical",
        "DBSCAN"
    ])

  
    # -------------------
    # CLUSTERING ALGORITHM COMPARISON
    # -------------------
    with tabs[0]:
        st.subheader("Clustering Algorithm Comparison")

        st.markdown(
            """
            Here we compare clustering evaluation metrics across different algorithms
            to provide context before analysing how the number of clusters is determined
            for each method.
            """
        )

        # Load comparison table
        df = pd.read_csv("outputs/clustering_evaluation.csv")

        comparison_df = df[
            [
                "Algorithm",
                "No. of Clusters",
                "Cluster Selection Method",
                "Silhouette Score",
                "Davies-Bouldin Index (DBI)"
            ]
        ]

        st.dataframe(comparison_df, use_container_width=True)

        # Metric Comparison Graphs
        st.markdown("### Metric Comparison")

        algorithms = ["k-Means", "Hierarchical"]
        silhouette_scores = [0.24, 0.17]
        dbi_scores = [1.52, 1.84]

        col1, col2 = st.columns(2)

        # Silhouette Score graph
        with col1:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.bar(algorithms, silhouette_scores)
            ax.set_ylabel("Silhouette Score")
            ax.set_title("Silhouette Score Comparison")
            ax.set_ylim(0, 0.3)
            st.pyplot(fig, use_container_width=True)

        # DBI graph
        with col2:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.bar(algorithms, dbi_scores)
            ax.set_ylabel("Davies-Bouldin Index")
            ax.set_title("Davies-Bouldin Index Comparison")
            ax.set_ylim(1.4, 2.0)
            st.pyplot(fig, use_container_width=True)

        # Interpretation
        st.markdown(
            """
            ### Interpretation

            **Silhouette Score**  
            The silhouette score comparison shows that k-Means achieves slightly stronger
            cluster separation than hierarchical clustering. This indicates that k-Means
            forms more compact and well-defined clusters, while hierarchical clustering
            emphasises broader groupings. This behaviour aligns with the role of hierarchical
            clustering in capturing high-level structural patterns rather than maximising
            internal cluster separation.

            **Davies-Bouldin Index (DBI)**  
            The Davies-Bouldin Index comparison indicates that k-Means produces clusters
            with lower overlap and better compactness compared to hierarchical clustering.
            The higher DBI value observed for hierarchical clustering reflects its tendency
            to form more general groupings instead of tightly compact clusters, which is
            consistent with its use for high-level wellbeing structure analysis.

            **DBSCAN**  
            DBSCAN is not directly comparable using these metrics, as its results are highly
            sensitive to noise and parameter settings, making silhouette score and DBI values
            unstable or less meaningful for this dataset.
            """
        )

    # -------------------
    # K-MEANS 
    # -------------------
    with tabs[1]:
        st.subheader("K-Means Cluster Selection")

        st.markdown(
         """
         The optimal number of clusters was selected using a combination of the
         Elbow Method, Silhouette Score and Davies–Bouldin Index.
         """
        )

         # Display validation plots side by side
        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(
                "outputs/metric/elbow.png",
                caption="Elbow Method",
                use_container_width=True
            )

        with col2:
            st.image(
                "outputs/metric/silhouette.png",
                caption="Silhouette Score",
                use_container_width=True
            )

        with col3:
            st.image(
                "outputs/metric/dbi.png",
                caption="Davies–Bouldin Index",
                use_container_width=True
            )

        # Explanation
        st.markdown(
            """
            **Elbow Method**  
            The Elbow Method shows a clear reduction in inertia up to K = 4, after which
            the rate of improvement becomes more gradual. This indicates diminishing returns
            from adding additional clusters.

            **Silhouette Score**  
            Silhouette analysis further supports this choice, with the highest silhouette
            score observed at K = 4 (0.2353), suggesting the best balance between cluster
            cohesion and separation.

            **Davies–Bouldin Index (DBI)**  
            Although the Davies–Bouldin Index reaches its minimum at K = 7 (1.3727),
            this reduction is primarily due to increased cluster fragmentation, where clusters
            become smaller and less interpretable.

            **Final Selection**  
            Therefore, K = 4 was selected as it provides the most meaningful and stable
            clustering structure, balancing quantitative validation metrics with
            interpretability at the country level.
            """
        )

        st.markdown("### K-Means Cluster Evaluation")

        # Load cluster evaluation table
        kmeans_eval_df = pd.read_csv("outputs/kmeans_evaluation.csv")

        st.dataframe(kmeans_eval_df, use_container_width=True)

        # Interpretation
        st.markdown(
            """
            ### Cluster Interpretation

            **Cluster 0 : Low Development Countries**  
            It exhibits the lowest average life expectancy, GDP and schooling levels, indicating weak performance
            across key development indicators. This cluster also records the lowest average suicide rate, suggesting that
            lower reported suicide rates may be associated with underdeveloped health systems, reporting practices or
            demographic factors rather than improved mental wellbeing.

            **Cluster 1 : High Development Countries**  
            It records the highest average life expectancy, GDP and schooling levels, reflecting strong health,
            economic, and educational outcomes. Despite high development status, this cluster shows a relatively high
            average suicide rate, highlighting that higher socioeconomic development does not necessarily correspond
            to lower suicide risk.

            **Cluster 2 : Medium Development Countries**  
            It shows the second highest average life expectancy, GDP and schooling, indicating medium
            developing countries that are approaching high development status. The average suicide rate in this cluster
            is also relatively high, suggesting increasing mental health challenges as countries transition toward higher
            development levels.

            **Cluster 3 : Medium Development Countries**  
            It ranks below Cluster 2 in life expectancy, GDP and schooling, representing medium developing
            countries with weaker development outcomes. The average suicide rate in this cluster remains elevated,
            though slightly lower than in Cluster 2, indicating that suicide risk varies across different stages of
            development.
            """
        )


    # -------------------
    # HIERARCHICAL 
    # -------------------
    with tabs[2]:
        st.subheader("Hierarchical Clustering Analysis")

        st.markdown(
            """
            Hierarchical clustering was analysed using a dendrogram to examine how
            countries are grouped at different linkage distances. The number of clusters
            was determined by inspecting the dendrogram structure.
            """
        )

        # Display dendrogram
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(
                "outputs/metric/hierarchical.png",
                caption="Hierarchical Clustering Dendrogram",
                use_container_width=True
            )

        # Interpretation
        st.markdown(
            """
            - The dendrogram illustrates how countries are progressively merged based on
            similarity across health, socioeconomic and demographic indicators.
            - A higher-level cut of the dendrogram was selected, resulting in
            3 main clusters. This cut captures broad and interpretable groupings
            while avoiding excessive fragmentation into small clusters.
            - Cutting the dendrogram at lower levels would produce a larger number of
            clusters with finer distinctions, but these clusters become less stable
            and harder to interpret at the country level.
            - Therefore, K = 3 was chosen as it provides a meaningful balance between
            structural simplicity and interpretability, making it suitable for
            high-level wellbeing analysis.
            """
        )

        st.markdown("### Hierarchical Cluster Evaluation")

        # Load hierarchical cluster evaluation table
        hierarchical_eval_df = pd.read_csv("outputs/hierarchical_evaluation.csv")

        # Display table
        st.dataframe(hierarchical_eval_df, use_container_width=True)

        # Interpretation
        st.markdown(
            """
            ### Cluster Interpretation

            **Cluster 1 : High Development Countries**  
            This cluster records the highest average life expectancy, GDP and schooling levels,
            indicating strong overall development outcomes. Despite high socioeconomic status,
            the average suicide rate is relatively high, suggesting that higher development does
            not necessarily correspond to lower suicide risk.

            **Cluster 2 : Low Development Countries**  
            Cluster 2 exhibits the lowest average life expectancy, GDP and schooling levels.
            This cluster also shows the lowest average suicide rate, which may reflect differences
            in population structure, underreporting, or limited mental health surveillance rather
            than improved mental wellbeing.

            **Cluster 3 : Medium Development Countries**  
            This cluster represents countries with moderate levels of life expectancy, GDP and
            schooling. The average suicide rate in this group is higher than in low development
            countries but lower than in high development countries, indicating a transitional
            pattern of suicide risk across development stages.
            """
        )

    
    # -------------------
    # DBSCAN
    # -------------------
    with tabs[3]:
        st.subheader("DBSCAN Clustering Analysis")

        st.markdown(
            """
            DBSCAN was applied to the scaled fused dataset using all numerical features,
            including suicides/100k population, to allow the algorithm to identify
            atypical records across both health and socioeconomic dimensions. Unlike
            k-Means and hierarchical, DBSCAN is a density-based clustering algorithm and does not require
            pre-specifying the number of clusters.
            """
        )

        # DBSCAN visualization
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(
                "outputs/metric/dbscan.png",
                caption="DBSCAN Clustering Results (PCA-reduced)",
                use_container_width=True
            )

        st.markdown(
            """
            ### DBSCAN Graph Interpretation

            The PCA-based visualization shows that the scaled fused dataset exhibits a
            heterogeneous structure, with several localized dense regions and a
            substantial number of noise points. This suggests that the data does not form
            clearly separated global clusters.

            Instead, the dataset displays gradually changing and overlapping feature
            patterns. Records with unusual combinations of socioeconomic and health-related
            features are captured as noise points, highlighting atypical or extreme
            country profiles.
            """
        )

        st.markdown(
            """
            ### DBSCAN Configuration and Outcome

            DBSCAN was applied using eps = 1.0 and min_samples = 10. With these
            parameters, the algorithm identified 579 clusters and classified
            1,727 data points as noise.

            The tuned parameters resulted in fewer fragmented clusters compared to more
            restrictive settings, merging smaller dense regions into larger groupings.
            However, the presence of many clusters and noise points reflects the complex
            and non-uniform structure of the fused dataset.
            """
        )

        st.markdown(
            """
            ### Interpretation and Limitation

            DBSCAN highlights the existence of local similarity patterns within the
            dataset rather than globally separable clusters. While this is useful for
            identifying anomalous or atypical records, it limits the interpretability of
            DBSCAN for country-level clustering.

            As a result, DBSCAN is not used for further cluster profiling or comparison
            and the clustering analysis focuses on k-Means and hierarchical clustering,
            which provide more stable and interpretable global groupings.
            """
        )
