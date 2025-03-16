import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn_features.transformers import DataFrameSelector

# Load dataset
df = pd.read_csv("dataset.csv")

# Data Preprocessing
df = df.iloc[:, 1:-2]
df["Attrition_Flag"] = df["Attrition_Flag"].map({"Existing Customer": 0, "Attrited Customer": 1})

# Handle Customer_Age
df["Age"] = 0
df.loc[(df["Customer_Age"] > 25) & (df["Customer_Age"] <= 35), "Age"] = 0
df.loc[(df["Customer_Age"] > 35) & (df["Customer_Age"] <= 45), "Age"] = 1
df.loc[(df["Customer_Age"] > 45) & (df["Customer_Age"] <= 55), "Age"] = 2
df.loc[df["Customer_Age"] > 55, "Age"] = 3
df.drop(["Customer_Age"], axis=1, inplace=True)

# Handle Bank_Relationship_Period
df["Bank_Relationship_Period"] = 0
df.loc[(df["Months_on_book"] > 10) & (df["Months_on_book"] <= 20), "Bank_Relationship_Period"] = 1
df.loc[(df["Months_on_book"] > 20) & (df["Months_on_book"] <= 30), "Bank_Relationship_Period"] = 2
df.loc[(df["Months_on_book"] > 30) & (df["Months_on_book"] <= 40), "Bank_Relationship_Period"] = 3
df.loc[df["Months_on_book"] > 50, "Bank_Relationship_Period"] = 4
df.drop(["Months_on_book"], axis=1, inplace=True)

# Split data
X = df.drop(columns=["Attrition_Flag"], axis=1)
y = df["Attrition_Flag"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=41)

# Define column types
num_cols = [
    "Credit_Limit", "Total_Revolving_Bal", "Total_Amt_Chng_Q4_Q1",
    "Total_Trans_Amt", "Total_Trans_Ct", "Total_Ct_Chng_Q4_Q1", "Avg_Utilization_Ratio"
]
categ_cols = ["Education_Level", "Income_Category", "Marital_Status", "Card_Category", "Gender"]
ready_cols = list(set(X_train.columns.tolist()) - set(num_cols) - set(categ_cols))

# Pipelines
num_pipeline = Pipeline([
    ("selector", DataFrameSelector(num_cols)),
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categ_pipeline = Pipeline([
    ("selector", DataFrameSelector(categ_cols)),
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(drop="first", sparse_output=False)),
])

ready_pipeline = Pipeline([
    ("selector", DataFrameSelector(ready_cols)),
    ("imputer", SimpleImputer(strategy="most_frequent")),
])

all_pipeline = FeatureUnion([
    ("numerical", num_pipeline),
    ("categorical", categ_pipeline),
    ("ready", ready_pipeline),
])

# Transform Data
X_train_final = all_pipeline.fit_transform(X_train)
X_test_final = all_pipeline.transform(X_test)

# Handling Imbalance
dict_weights = {i: v for i, v in enumerate((1 - (np.bincount(y_train) / len(y_train))) / np.sum(1 - (np.bincount(y_train) / len(y_train))))}
over = SMOTE(sampling_strategy=0.7)
X_train_resampled, y_train_resampled = over.fit_resample(X_train_final, y_train)

# Function to Train Model
def train_model(X_train, y_train, plot_name='', class_weight=None):
    clf = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=45, class_weight=class_weight)
    clf.fit(X_train, y_train)
    y_pred_test = clf.predict(X_test_final)
    
    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred_test), annot=True, cbar=False, fmt='.2f', cmap='Blues')
    plt.title(f'{plot_name}')
    plt.xticks(ticks=np.arange(2) + 0.5, labels=[False, True])
    plt.yticks(ticks=np.arange(2) + 0.5, labels=[False, True])
    plt.savefig(f'{plot_name}.png', bbox_inches='tight', dpi=300)
    plt.close()

# Train Models
train_model(X_train_final, y_train, 'without-imbalance', None)
train_model(X_train_final, y_train, 'with-class-weights', dict_weights)
train_model(X_train_resampled, y_train_resampled, 'with-SMOTE', None)

# Combine Confusion Matrices
confusion_matrix_paths = ['./without-imbalance.png', './with-class-weights.png', './with-SMOTE.png']
plt.figure(figsize=(15, 5))
for i, path in enumerate(confusion_matrix_paths, 1):
    img = Image.open(path)
    plt.subplot(1, len(confusion_matrix_paths), i)
    plt.imshow(img)
    plt.axis('off')
plt.suptitle("RandomForestClassifier", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('conf_matrix.png', bbox_inches='tight', dpi=300)

# Cleanup
for path in confusion_matrix_paths:
    os.remove(path)


