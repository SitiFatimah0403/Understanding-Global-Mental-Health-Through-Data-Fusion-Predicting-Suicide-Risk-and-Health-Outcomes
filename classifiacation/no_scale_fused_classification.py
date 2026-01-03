
"""# **Prepare Data**

"""

import pandas as pd

#load first
df = pd.read_csv('new_fused_dataset.csv')

#display first 5 rows
df.head()

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

So our target feature is 'suicides/100k pop'

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

#Discretised based on median of 'suicides/100k pop'

#Calculate the median first
median = y.median()
y_dis = y.apply(lambda x: 'Low' if x < median else 'High') #less then median = low , more or equal = high
print(f"Median 'suicides/100k pop': {median}")
print('\nValue counts of discretized target variable:')
print(y_dis.value_counts())

"""# **Train-test split**

We split into :

*   80-20
*   70-30

# 1.   **80-20 train-test settting**
"""

from sklearn.model_selection import train_test_split

# Split the data into 80-20 train test setting with random state of 42
x_train_80, x_test_80, y_train_80, y_test_80 = train_test_split(x, y_dis, test_size=0.2, random_state=42)

print('Shape of X_train:', x_train_80.shape)
print('Shape of X_test:', x_test_80.shape)
print('Shape of y_train:', y_train_80.shape)
print('Shape of y_test:', y_test_80.shape)

"""# No Scaling for 80-20

# **Train Models for 80-20**

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
dt_model.fit(x_train_80, y_train_80_encoded)
print('Decision Tree Model Trained')

# Instantiate and train random forest classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(x_train_80, y_train_80_encoded)
print('Random Forest Model Trained')

# Instantiate and train logistic regression
lr_model = LogisticRegression(random_state=42, solver='liblinear') # solver='liblinear' for small datasets and 'l1' regularization
lr_model.fit(x_train_80, y_train_80_encoded)
print('Logistic Regression Model Trained')

# Instantiate and train naive bayes
nb_model = GaussianNB()
nb_model.fit(x_train_80, y_train_80_encoded)
print('Naive Bayes Model Trained')

# Instantiate and train k-nearest neighbors (KNN)
knn_model = KNeighborsClassifier()
knn_model.fit(x_train_80, y_train_80_encoded)
print('K-Nearest Neighbors Model Trained')

# Instantiate and train SVM classifier
svm_model = SVC(random_state=42, probability=True)
svm_model.fit(x_train_80, y_train_80_encoded)
print('SVM Model Trained')

# Instantiate and train XGBoost classifier
xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(x_train_80, y_train_80_encoded)
print('XGBoost Model Trained')

# Instantiate and train Artificial Neural Network (ANN) - MLPClassifier
ann_model = MLPClassifier(random_state=42, max_iter=500) # Increased max_iter for convergence
ann_model.fit(x_train_80, y_train_80_encoded)
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
    y_pred_80 = model.predict(x_test_80)
    accuracy = accuracy_score(y_test_80_encoded, y_pred_80)
    f1 = f1_score(y_test_80_encoded, y_pred_80)
    all_results.append({'Model': name, 'Accuracy': accuracy, 'F1-Score': f1})

results = pd.DataFrame(all_results) # Create DataFrame from the list of dictionaries

print('Model Evualuation Results: ')
print(results)

#to save results to csv
import os
os.makedirs("outputs", exist_ok=True)

results.to_csv(
    "outputs/fused_cls_baseline_80_20_noscale.csv",
    index=False
)


"""

---


# **2.   70-30 train test setting**

"""

from sklearn.model_selection import train_test_split

# Split the data into 70-30 train test setting with random state of 42
x_train_70, x_test_70, y_train_70, y_test_70 = train_test_split(x,y_dis, test_size=0.3, random_state=42)

print('Shape of X_train:', x_train_70.shape)
print('Shape of X_test:', x_test_70.shape)
print('Shape of y_train:', y_train_70.shape)
print('Shape of y_test:', y_test_70.shape)

"""# No Scaling for 70-30

# Train Model for 70-30
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

#encode y_train_70 & y_test_70
le = LabelEncoder()
y_train_70_encoded = le.fit_transform(y_train_70)
y_test_70_encoded = le.transform(y_test_70)

print(f"Original y_train_70 unique values: {y_train_70.unique()}")
print(f"Encoded y_train_70 unique values: {le.classes_}")

#Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(x_train_70, y_train_70_encoded)
print('Decision Tree Model Trained')

#Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(x_train_70, y_train_70_encoded)
print('Random Forest Model Trained')

#Logistic Regression
lr_model = LogisticRegression(random_state=42, solver='liblinear')
lr_model.fit(x_train_70, y_train_70_encoded)
print('Logistic Regression Model Trained')

#Naive Bayes
nb_model = GaussianNB()
nb_model.fit(x_train_70, y_train_70_encoded)
print('Naive Bayes Model Trained')

#KNN
knn_model = KNeighborsClassifier()
knn_model.fit(x_train_70, y_train_70_encoded)
print('K-Nearest Neighbors Model Trained')

#SVM
svm_model = SVC(random_state=42, probability=True)
svm_model.fit(x_train_70, y_train_70_encoded)
print('SVM Model Trained')

#XGBoost
xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(x_train_70, y_train_70_encoded) # y_train_70 should be y_train_70_encoded
print('XGBoost Model Trained')

#ANN
ann_model = MLPClassifier(random_state=42, max_iter=500)
ann_model.fit(x_train_70, y_train_70_encoded)
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
    y_pred_70 = model.predict(x_test_70)
    accuracy = accuracy_score(y_test_70_encoded, y_pred_70)
    f1 = f1_score(y_test_70_encoded, y_pred_70)
    all_results_70.append({'Model': name, 'Accuracy': accuracy, 'F1-Score': f1})

results_70 = pd.DataFrame(all_results_70) # Create DataFrame from the list of dictionaries

print('Model Evaluation Results for 70-30 Split:')
print(results_70)


results_70.to_csv(
    "outputs/fused_cls_baseline_70_30_noscale.csv",
    index=False
)
