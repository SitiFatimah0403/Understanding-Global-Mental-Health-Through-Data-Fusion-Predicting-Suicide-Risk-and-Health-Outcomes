
"""# **Prepare Data**

"""

import pandas as pd

#load first
df = pd.read_csv('new_fused_dataset.csv')

#display first 10 row
df.head(10)

#check if there are missing values
df.isnull().sum()

#check the data frame info
df.info()

#calculate the descriptive statistics
df.describe()

"""# **One-hot encoding for 'status' column**


*   Status_developing = 1 (developing)
*   Status_developing = 0 (developed)


"""

df = pd.get_dummies(df, columns=['Status'], drop_first=True, dtype=int)
#check the first 10 rows
print(df.head(10))

#check the data frame after one hot encoding
print(df.info())

"""# **Select Feature**

So our target feature is 'suicides/100k pop' = “Is this country high suicide risk or low suicide risk?”

"""

y = df['suicides/100k pop']
x = df.drop('suicides/100k pop', axis=1)

# x (target)
print('First 5 rows of X:')
print(x.head())

# Check the shape of the x
print('\nShape of X:', x.shape)

# y
print('\nFirst 5 rows of y:')
print(y.head())

# Check the shape of the y
print('\nShape of y:', y.shape)

"""# **Discretise Target**
convert 'suicides/100k pop' into categorical 'Low' and 'High'
"""

# We descretised based on median of 'suicides/100k pop'

#calculate the median first
median = y.median()
y_dis = y.apply(lambda x: 'Low' if x < median else 'High') #less then median = low , more or equal = high
print(f"Median 'suicides/100k pop': {median}")
print('\nValue counts of discretized target variable:')
print(y_dis.value_counts())

"""# **Train-test split**

We split into :

*   80-20
*   70-30

# A.   **80-20 train-test settting**
"""

from sklearn.model_selection import train_test_split

# Split the data into 80-20 train test setting with random state of 42
x_train_80, x_test_80, y_train_80, y_test_80 = train_test_split(x, y_dis, test_size=0.2, random_state=42)

print('Shape of X_train:', x_train_80.shape)
print('Shape of X_test:', x_test_80.shape)
print('Shape of y_train:', y_train_80.shape)
print('Shape of y_test:', y_test_80.shape)

"""# Scaling for 80-20
We scale it with StandardScaler from sklearn.preprocessing
"""

from sklearn.preprocessing import StandardScaler

# Initialise the StandardScaler
scaler = StandardScaler()

#fit on x_train & transform both x_train & x_test
x_train_80_scaled = scaler.fit_transform(x_train_80)
x_test_80_scaled = scaler.transform(x_test_80)

#convert to dataframe
x_train_80_scaled = pd.DataFrame(x_train_80_scaled, columns=x_train_80.columns)
x_test_80_scaled = pd.DataFrame(x_test_80_scaled, columns=x_test_80.columns)

#check the first 5 rows
print('First 5 rows of x_train_scaled:')
print(x_train_80_scaled.head())
print('\nFirst 5 rows of x_test_scaled:')
print(x_test_80_scaled.head())

"""# Train Models for 80-20

For classification we will use :


*  Decision Tree
* Random Forest
* Logistic Regression
* Naive Bayes
* KNN
* SVM
* XGBoost
* MLPClassifier (ANN)




"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

#encode y_train & y_test
le = LabelEncoder()
y_train_80_encoded = le.fit_transform(y_train_80)
y_test_80_encoded = le.transform(y_test_80)

print(f"Original y_train unique values: {y_train_80.unique()}")
print(f"Encoded y_train unique values: {le.classes_}")


# Instantiate and train decision tree classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(x_train_80_scaled, y_train_80_encoded)
print('Decision Tree Model Trained')

# Instantiate and train random forest classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(x_train_80_scaled, y_train_80_encoded)
print('Random Forest Model Trained')

# Instantiate and train logistic regression
lr_model = LogisticRegression(random_state=42, solver='liblinear') # solver='liblinear' for small datasets and 'l1' regularization
lr_model.fit(x_train_80_scaled, y_train_80_encoded)
print('Logistic Regression Model Trained')

# Instantiate and train naive bayes
nb_model = GaussianNB()
nb_model.fit(x_train_80_scaled, y_train_80_encoded)
print('Naive Bayes Model Trained')

# Instantiate and train k-nearest neighbors (KNN)
knn_model = KNeighborsClassifier()
knn_model.fit(x_train_80_scaled, y_train_80_encoded)
print('K-Nearest Neighbors Model Trained')

# Instantiate and train SVM classifier
svm_model = SVC(random_state=42, probability=True)
svm_model.fit(x_train_80_scaled, y_train_80_encoded)
print('SVM Model Trained')

# Instantiate and train XGBoost classifier
xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(x_train_80_scaled, y_train_80_encoded)
print('XGBoost Model Trained')

# Instantiate and train Artificial Neural Network (ANN) - MLPClassifier
ann_model = MLPClassifier(random_state=42, max_iter=500) # Increased max_iter for convergence
ann_model.fit(x_train_80_scaled, y_train_80_encoded)
print('ANN Model Trained')

"""# Evaluate
We evalute each model by its accuracy and F1 score
"""

from sklearn.metrics import accuracy_score, f1_score

models = {
    'Desision Tree' : dt_model,
    'Random Forest' : rf_model,
    'Logistic Regression' : lr_model,
    'Naive Bayes' : nb_model,
    'KNN' : knn_model,
    'SVM' : svm_model,
    'XGBoost' : xgb_model,
    'ANN' : ann_model
}

all_results = []

for name, model in models.items():
    y_pred_80 = model.predict(x_test_80_scaled)
    accuracy = accuracy_score(y_test_80_encoded, y_pred_80)
    f1 = f1_score(y_test_80_encoded, y_pred_80)
    all_results.append({'Model': name, 'Accuracy': accuracy, 'F1-Score': f1})

results = pd.DataFrame(all_results) # Create DataFrame from the list of dictionaries

print('Model Evualuation Results: ')
print(results)

"""
# **B.   70-30 train test setting**

"""

from sklearn.model_selection import train_test_split

# Split the data into 70-30 train test setting with random state of 42
x_train_70, x_test_70, y_train_70, y_test_70 = train_test_split(x,y_dis, test_size=0.3, random_state=42)

print('Shape of X_train:', x_train_70.shape)
print('Shape of X_test:', x_test_70.shape)
print('Shape of y_train:', y_train_70.shape)
print('Shape of y_test:', y_test_70.shape)

"""# Scaling for 70-30"""

from sklearn.preprocessing import StandardScaler

#Initialise Scaler
scaler = StandardScaler()

#fit on x_train_70 & transform both x_train_70 & x_test_70
x_train_70_scaled = scaler.fit_transform(x_train_70)
x_test_70_scaled = scaler.transform(x_test_70)

#convert to dataframe
x_train_70_scaled = pd.DataFrame(x_train_70_scaled, columns=x_train_70.columns)
x_test_70_scaled = pd.DataFrame(x_test_70_scaled, columns=x_test_70.columns)

#check the first 5 rows
print('First 5 rows of x_train_scaled:')
print(x_train_70_scaled.head())
print('\nFirst 5 rows of x_test_scaled:')
print(x_test_70_scaled.head())

"""# Train Model for 70-30"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

#encode y_train_70 & y_test_70
le = LabelEncoder()
y_train_70_encoded = le.fit_transform(y_train_70)
y_test_70_encoded = le.transform(y_test_70)

print(f"Original y_train_70 unique values: {y_train_70.unique()}")
print(f"Encoded y_train_70 unique values: {le.classes_}")

#Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(x_train_70_scaled, y_train_70_encoded)
print('Decision Tree Model Trained')

#Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(x_train_70_scaled, y_train_70_encoded)
print('Random Forest Model Trained')

#Logistic Regression
lr_model = LogisticRegression(random_state=42, solver='liblinear')
lr_model.fit(x_train_70_scaled, y_train_70_encoded)
print('Logistic Regression Model Trained')

#Naive Bayes
nb_model = GaussianNB()
nb_model.fit(x_train_70_scaled, y_train_70_encoded)
print('Naive Bayes Model Trained')

#KNN
knn_model = KNeighborsClassifier()
knn_model.fit(x_train_70_scaled, y_train_70_encoded)
print('K-Nearest Neighbors Model Trained')

#SVM
svm_model = SVC(random_state=42, probability=True)
svm_model.fit(x_train_70_scaled, y_train_70_encoded)
print('SVM Model Trained')

#XGBoost
xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(x_train_70_scaled, y_train_70_encoded) # y_train_70 should be y_train_70_encoded
print('XGBoost Model Trained')

#ANN
ann_model = MLPClassifier(random_state=42, max_iter=500)
ann_model.fit(x_train_70_scaled, y_train_70_encoded)
print('ANN Model Trained')

"""# Evaluate"""

from sklearn.metrics import accuracy_score, f1_score

models = {
    'Desision Tree' : dt_model,
    'Random Forest' : rf_model,
    'Logistic Regression' : lr_model,
    'Naive Bayes' : nb_model,
    'KNN' : knn_model,
    'SVM' : svm_model,
    'XGBoost' : xgb_model,
    'ANN' : ann_model

}

all_results_70 = []

for name, model in models.items():
    y_pred_70 = model.predict(x_test_70_scaled)
    accuracy = accuracy_score(y_test_70_encoded, y_pred_70)
    f1 = f1_score(y_test_70_encoded, y_pred_70)
    all_results_70.append({'Model': name, 'Accuracy': accuracy, 'F1-Score': f1})

results_70 = pd.DataFrame(all_results_70) # Create DataFrame from the list of dictionaries

print('Model Evaluation Results for 70-30 Split:')
print(results_70)

"""# **Hyperparamter Tuning for 80-20 train test top 3 models**

Top 3 model of 80-20 :

1.   SVM
2.   ANN
3.   XGBoost

## Define Parameter Grids/Distributions

Define parameter grids for GridSearchCV and parameter distributions for RandomizedSearchCV for ANN (MLPClassifier), XGBoost, and SVM.

1.   **SVM**
"""

import scipy.stats as stats

# SVM (SVC) Parameter Grids/Distributions
param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf']
}
param_dist_svm = {
    'C': [0.01, 0.1, 1, 10],
    'kernel': ['rbf']
}

print("Defined param_grid_svm and param_dist_svm.")

"""

2.   **ANN**


"""

import numpy as np

# ANN (MLPClassifier) Parameter Grids/Distributions
param_grid_ann = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001],
    'max_iter': [300]
}

param_dist_ann = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01],
    'max_iter': [200, 300, 400]
}

print("Defined param_grid_ann and param_dist_ann.")

"""

3.   **XGBoost**


"""

import scipy.stats as stats

# XGBoost Parameter Grids/Distributions
param_grid_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1]
}

param_dist_xgb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1]
}

print("Defined param_grid_xgb and param_dist_xgb.")

"""# GridSearch <br>
It coundn't execute here in google colab so i took the output from vscode

1. SVM
"""

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

grid_seacrh_svm = GridSearchCV(
    SVC(random_state=42),
    param_grid_svm,
    scoring='f1',
    cv=3,
    n_jobs=-1

)

grid_seacrh_svm.fit(x_train_80_scaled, y_train_80_encoded)



from sklearn.model_selection import GridSearchCV

grid_search_ann =  GridSearchCV(
    MLPClassifier(random_state=42),
    param_grid_ann,
    scoring='f1',
    cv=3,
    n_jobs=-1
)

grid_search_ann.fit(x_train_80_scaled, y_train_80_encoded)



from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

grid_search_xgb = GridSearchCV(
    XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    param_grid_xgb,
    scoring='f1',
    cv=3,
    n_jobs=-1
)

grid_search_xgb.fit(x_train_80_scaled, y_train_80_encoded)



random_search_svm = RandomizedSearchCV(
    SVC(random_state=42),
    param_dist_svm,
    n_iter=10,
    scoring='f1',
    cv=3,
    random_state=42,
    n_jobs=-1
)

random_search_svm = random_search_svm.fit(x_train_80_scaled, y_train_80_encoded)

"""2. ANN"""

from sklearn.model_selection import RandomizedSearchCV

random_search_ann = RandomizedSearchCV(
    MLPClassifier(random_state=42),
    param_dist_ann,
    n_iter=10,
    scoring='f1',
    cv=3,
    random_state=42,
    n_jobs=-1
)

random_search_ann = random_search_ann.fit(x_train_80_scaled, y_train_80_encoded)

"""3. XGBoost"""

random_search_xgb = RandomizedSearchCV(
    XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    param_dist_xgb,
    n_iter=10,
    scoring='f1',
    cv=3,
    random_state=42,
    n_jobs=-1
)

random_search_xgb = random_search_xgb.fit(x_train_80_scaled, y_train_80_encoded)

"""# Evaluate"""

models = {
    "ANN Grid": grid_search_ann.best_estimator_,
    "ANN Random": random_search_ann.best_estimator_,
    "XGB Grid": grid_search_xgb.best_estimator_,
    "XGB Random": random_search_xgb.best_estimator_,
    "SVM Grid": grid_search_svm.best_estimator_,
    "SVM Random": random_search_svm.best_estimator_,
}

for name, model in models.items():
    y_pred = model.predict(x_test_80_scaled)
    acc = accuracy_score(y_test_80_encoded, y_pred)
    f1 = f1_score(y_test_80_encoded, y_pred)
    print(f"{name} → Accuracy: {acc:.4f}, F1: {f1:.4f}")


#install AutoGluon 
# import sys
# !{sys.executable} -m pip install autogluon

"""# Split into scaled 80-20"""

#Split into scaled 80-20
from autogluon.tabular import TabularPredictor

train_data_80 = x_train_80_scaled.copy()
train_data_80['Suicide_Risk'] = y_train_80.values

test_data_80 = x_test_80_scaled.copy()
test_data_80['Suicide_Risk'] = y_test_80.values

"""# Train using autoGluon"""

#train using autoGluon

predictor = TabularPredictor(
    label = 'Suicide_Risk',
    eval_metric = 'f1'
).fit(
    train_data_80,
    time_limit = 600, # Equalvalance to 10 minutes
    presets = 'medium_quality' # for quick and accuracy
)

"""# Evaluate The BEST Model"""

performance = predictor.evaluate(test_data_80)
print(performance) #Ouput = F1, accuracy, precision and recall

"""# Best Model"""

predictor.leaderboard(test_data_80, silent=False)

"""# **XAI - Best Tuned Model**

### XAI - Final Corrected Code
"""

# CELL 1: Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# CELL 2: Get best XGBoost model
best_xgb_model = grid_search_xgb.best_estimator_

# CELL 3: Convert SCALED data back to DataFrames
# Recover feature names from original training data
feature_names = x.columns  # Assuming 'x' still holds the original feature names before scaling and splitting

X_train_df = pd.DataFrame(
    x_train_80_scaled,
    columns=feature_names
)

X_test_df = pd.DataFrame(
    x_test_80_scaled,
    columns=feature_names
)

print("X_train_df.columns:")
print(X_train_df.columns)
print("\nX_test_df.columns:")
print(X_test_df.columns)

# CELL 4: Feature Importance
feature_importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': best_xgb_model.feature_importances_
})

feature_importances = feature_importances.sort_values(
    by='Importance',
    ascending=False
)

plt.figure(figsize=(12, 8))
sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_importances
)
plt.title('XGBoost Feature Importances (Best Tuned Model)')
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()


# CELL 5: SHAP Values
explainer = shap.TreeExplainer(best_xgb_model)
shap_values = explainer.shap_values(X_test_df)

# Binary classification:
# shap_values[0] -> High suicide risk
# shap_values[1] -> Low suicide risk

# CELL 6: SHAP Summary Plot (High Risk)
shap.summary_plot(
    shap_values,  # Changed from shap_values[0] to shap_values
    X_test_df,
    show=False
)

plt.title('SHAP Summary Plot for XGBoost (High Suicide Risk)')
plt.tight_layout()
plt.show()


# CELL 7: SHAP Dependence Plot (FIXED)
plt.figure(figsize=(10, 6))
shap.dependence_plot(
    'Life expectancy',
    shap_values,  # Changed from shap_values[0] to shap_values
    X_test_df,
    interaction_index='Adult Mortality',
    show=False
)
plt.title(
    'SHAP Dependence Plot: Life Expectancy vs Adult Mortality (High Suicide Risk)'
)
plt.tight_layout()
plt.show()


