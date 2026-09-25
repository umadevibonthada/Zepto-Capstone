import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Titanic dataset - ONLY ONCE
df = sns.load_dataset("titanic")

# Save offline copy
df.to_csv("titanic.csv", index=False)

print(df.shape)
print(df.head())
print("\n===== INFO =====")
print(df.info())

print("\n===== DESCRIBE =====")
print(df.describe())

print("\n===== SHAPE =====")
print(df.shape)

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== MISSING PERCENTAGE =====")
missing_percent = df.isnull().mean() * 100
print(missing_percent[missing_percent > 0])

#======CLEANING=====
# 1. Drop column with very high missing values
df=df.drop(columns=["deck"])

# 2. Drop rows where missing percentage is below 5%
df=df.dropna(subset=["embarked","embark_town"])
             
# 3. Impute age using median
df["age"]=df["age"].fillna(df["age"].median())  

print("\n===== AFTER CLEANING =====")
print(df.shape)
print("\n===== MISSING VALUES AFTER CLEANING =====")
print(df.isnull().sum())

# ===== TASK 3: UNIVARIATE ANALYSIS =====

# Histogram for Age
plt.figure(figsize=(8,5))
sns.histplot(df["age"],kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.savefig("age_histogram.png")
plt.show()
#Box plot for Fare

plt.figure(figsize=(8,5))
sns.boxplot(x=df["fare"])
plt.title("Fare Box Plot")
plt.savefig("fare_boxplot.png")
plt.show()

# ===== IQR OUTLIERS =====
def count_iqr_outliers(column):
    Q1=df[column].quantile(0.25)
    Q3=df[column].quantile(0.75)

    IQR=Q3-Q1

    lower= Q1-1.5 * IQR
    upper=Q3+1.5 * IQR

    outliers= df[(df[column]<lower) | (df[column]> upper)]

    return Q1,Q3,IQR,lower,upper,len(outliers)
age_result = count_iqr_outliers("age")
fare_result = count_iqr_outliers("fare")

print("\n===== AGE IQR =====")
print(age_result)

print("\n===== FARE IQR =====")
print(fare_result)

# ===== FARE STATISTICS =====

fare_mean = df["fare"].mean()
fare_median = df["fare"].median()
fare_mode = df["fare"].mode()[0]

print("\n===== FARE STATISTICS =====")
print("Mean:", fare_mean)
print("Median:", fare_median)
print("Mode:", fare_mode)

# ===== TASK 4: BIVARIATE ANALYSIS =====

# Survival rate by sex

female = df[df["sex"] == "female"]
male = df[df["sex"] == "male"]

female_survival_rate = female["survived"].mean()
male_survival_rate = male["survived"].mean()

print("\n===== SURVIVAL RATE BY SEX =====")
print("Female:", female_survival_rate)
print("Male:", male_survival_rate)

# Survival rate by passenger class
first_class=df[df["class"]==1]
second_class=df[df["class"]==2]
third_class = df[df["pclass"] == 3]

first_survival_rate = first_class["survived"].mean()
second_survival_rate = second_class["survived"].mean()
third_survival_rate = third_class["survived"].mean()

print("\n===== SURVIVAL RATE BY PCLASS =====")
print("1st Class:", first_survival_rate)
print("2nd Class:", second_survival_rate)
print("3rd Class:", third_survival_rate)

print("\n===== PCLASS VALUES =====")
print(df["pclass"].value_counts())
print(df["pclass"].unique())

print("\n===== SURVIVED BY PCLASS =====")
print(df.groupby("pclass")["survived"].agg(["count", "sum", "mean"]))

# Female passengers in 1st class

female_first = df[(df["sex"] == "female") & (df["pclass"] == 1)]

female_first_survival = female_first["survived"].mean()

print("\n===== FEMALE 1ST CLASS =====")
print("Survival rate:", female_first_survival)

# Survival rate by sex and passenger class

sex_pclass_survival = df.groupby(["sex", "pclass"])["survived"].mean()

print("\n===== SURVIVAL RATE BY SEX AND PCLASS =====")
print(sex_pclass_survival)
# ===== CORRELATION MATRIX =====
corr_columns=["survived","pclass","age","sibsp","parch","fare"]
correlation=df[corr_columns].corr()
print("\n===== CORRELATION MATRIX =====")
print(correlation.to_string())

# ===== CORRELATION HEATMAP =====

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Titanic Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()

# Age Box Plot

plt.figure(figsize=(8, 5))
sns.boxplot(x=df["age"])
plt.title("Age Box Plot")
plt.xlabel("Age")
plt.savefig("age_boxplot.png")
plt.show()

#fare histogram
plt.figure(figsize=(8, 5))
sns.histplot(df["fare"], kde=True)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Count")
plt.savefig("fare_histogram.png")
plt.show()

# ===== SAVE CLEANED DATA =====

df.to_csv("titanic.csv", index=False)

print("\n===== TITANIC CSV SAVED =====")
print("Saved rows and columns:", df.shape)

# ===== MULTIVARIATE CHART 1 =====

survival_by_sex_class = df.groupby(
    ["sex", "pclass"]
)["survived"].mean().reset_index()

plt.figure(figsize=(8, 5))

sns.barplot(
    data=survival_by_sex_class,
    x="pclass",
    y="survived",
    hue="sex"
)

plt.title("Survival Rate by Sex and Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("survival_by_sex_class.png")
plt.show()

# ===== MULTIVARIATE CHART 2 =====

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="age",
    y="fare",
    hue="survived"
)

plt.title("Age vs Fare by Survival")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.tight_layout()
plt.savefig("age_vs_fare_survival.png")
plt.show()


# ===== MULTIVARIATE CHART 3 =====

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="survived",
    y="age",
    hue="sex"
)

plt.title("Age Distribution by Survival and Sex")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Age")
plt.tight_layout()
plt.savefig("age_survival_sex.png")
plt.show()

# ===== MULTIVARIATE CHART 4 =====

embarked_class = df.groupby(
    ["embarked", "pclass"]
)["survived"].mean().reset_index()

plt.figure(figsize=(8, 5))

sns.barplot(
    data=embarked_class,
    x="embarked",
    y="survived",
    hue="pclass"
)

plt.title("Survival Rate by Embarked Port and Passenger Class")
plt.xlabel("Embarked Port")
plt.ylabel("Survival Rate")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("survival_by_embarked_class.png")
plt.show()

# ===== EXPLORATORY STANDARDIZATION =====
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df_standardized = df.copy()

df_standardized[["age", "fare"]] = scaler.fit_transform(
    df_standardized[["age", "fare"]]
)

print("\n===== BEFORE STANDARDIZATION =====")
print(df[["age", "fare"]].mean())
print(df[["age", "fare"]].std())

print("\n===== AFTER STANDARDIZATION =====")
print(df_standardized[["age", "fare"]].mean())
print(df_standardized[["age", "fare"]].std())


# ===== TRAIN TEST SPLIT =====

from sklearn.model_selection import train_test_split

X = df.drop("survived", axis=1)
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n===== TRAIN TEST SPLIT =====")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

print("\n===== SURVIVAL DISTRIBUTION =====")
print("Full data:")
print(y.value_counts(normalize=True))

print("\nTraining data:")
print(y_train.value_counts(normalize=True))

print("\nTest data:")
print(y_test.value_counts(normalize=True))

# ===== TRAIN TEST SPLIT =====

from sklearn.model_selection import train_test_split

X = df.drop("survived", axis=1)
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n===== TRAIN TEST SPLIT =====")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

print("\n===== SURVIVAL DISTRIBUTION =====")
print("Full data:")
print(y.value_counts(normalize=True))

print("\nTraining data:")
print(y_train.value_counts(normalize=True))

print("\nTest data:")
print(y_test.value_counts(normalize=True))


# ===== PREPROCESSING =====

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Numerical columns
numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

# Categorical columns
categorical_features = [
    "sex",
    "embarked"
]

# Numerical preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine both
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

print("\n===== PREPROCESSOR CREATED =====")
print(preprocessor)


# ===== LOGISTIC REGRESSION =====

from sklearn.linear_model import LogisticRegression

logistic_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

# Train
logistic_pipeline.fit(X_train, y_train)

# Predict
y_pred_logistic = logistic_pipeline.predict(X_test)

print("\n===== LOGISTIC REGRESSION =====")
print("Predictions:", y_pred_logistic[:10])

# ===== DECISION TREE =====

from sklearn.tree import DecisionTreeClassifier

decision_tree_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", DecisionTreeClassifier(
        random_state=42,
        max_depth=5
    ))
])

# Train
decision_tree_pipeline.fit(X_train, y_train)

# Predict
y_pred_tree = decision_tree_pipeline.predict(X_test)

print("\n===== DECISION TREE =====")
print("Predictions:", y_pred_tree[:10])


# ===== RANDOM FOREST =====

from sklearn.ensemble import RandomForestClassifier

random_forest_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

# Train
random_forest_pipeline.fit(X_train, y_train)

# Predict
y_pred_rf = random_forest_pipeline.predict(X_test)

print("\n===== RANDOM FOREST =====")
print("Predictions:", y_pred_rf[:10])


# ===== MODEL EVALUATION =====

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

models = {
    "Logistic Regression": (logistic_pipeline, y_pred_logistic),
    "Decision Tree": (decision_tree_pipeline, y_pred_tree),
    "Random Forest": (random_forest_pipeline, y_pred_rf)
}

results = []

for name, (model, y_pred) in models.items():

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n===== {name} =====")
    print("Confusion Matrix:")
    print(cm)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Probability for ROC-AUC
    y_probability = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_probability)

    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)
    print("ROC-AUC  :", roc_auc)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc
    })

# Comparison table
comparison_df = pd.DataFrame(results)

print("\n===== MODEL COMPARISON =====")
print(comparison_df.to_string(index=False))

# ===== CONFUSION MATRICES =====

from sklearn.metrics import ConfusionMatrixDisplay

for name, (model, y_pred) in models.items():

    plt.figure(figsize=(5, 4))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        cmap="Blues"
    )

    plt.title(f"{name} - Confusion Matrix")
    plt.tight_layout()

    filename = name.lower().replace(" ", "_") + "_confusion_matrix.png"
    plt.savefig(filename)
    plt.show()


    # ===== CLASS IMBALANCE =====

from imblearn.over_sampling import SMOTE

# Baseline
baseline_rf = random_forest_pipeline

# Balanced Random Forest
balanced_rf = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    ))
])

balanced_rf.fit(X_train, y_train)
y_pred_balanced = balanced_rf.predict(X_test)

# SMOTE
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_processed,
    y_train
)

smote_rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

smote_rf.fit(X_train_smote, y_train_smote)

y_pred_smote = smote_rf.predict(X_test_processed)

print("\n===== CLASS IMBALANCE COMPARISON =====")

for name, y_pred in [
    ("Baseline", y_pred_rf),
    ("Class Weight Balanced", y_pred_balanced),
    ("SMOTE", y_pred_smote)
]:
    print(f"\n{name}")
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1 Score :", f1_score(y_test, y_pred))


# ===== RANDOM FOREST GRID SEARCH + OOB =====

from sklearn.model_selection import GridSearchCV

rf_grid_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        random_state=42,
        oob_score=True,
        n_jobs=-1
    ))
])

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 5, 10],
    "model__max_features": ["sqrt", "log2"]
}

grid_search = GridSearchCV(
    rf_grid_pipeline,
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\n===== GRID SEARCH RESULTS =====")
print("Best Parameters:")
print(grid_search.best_params_)

best_rf_pipeline = grid_search.best_estimator_

print("Best CV F1 Score:", grid_search.best_score_)

print(
    "OOB Score:",
    best_rf_pipeline.named_steps["model"].oob_score_
)

# ===== DECISION TREE VISUALIZATION =====

from sklearn.tree import plot_tree

tree_model = decision_tree_pipeline.named_steps["model"]
tree_preprocessor = decision_tree_pipeline.named_steps["preprocessor"]

feature_names = tree_preprocessor.get_feature_names_out()

plt.figure(figsize=(20, 10))

plot_tree(
    tree_model,
    feature_names=feature_names,
    class_names=["Not Survived", "Survived"],
    filled=True,
    max_depth=3
)

plt.title("Decision Tree")
plt.tight_layout()

plt.savefig("decision_tree.png")
plt.show()

# ===== REGRESSION: PREDICT FARE =====

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

reg_numeric = ["pclass", "age", "sibsp", "parch"]
reg_categorical = ["sex", "embarked"]

reg_preprocessor = ColumnTransformer([
    ("num",
     Pipeline([
         ("imputer", SimpleImputer(strategy="median")),
         ("scaler", StandardScaler())
     ]),
     reg_numeric),

    ("cat",
     Pipeline([
         ("imputer", SimpleImputer(strategy="most_frequent")),
         ("encoder", OneHotEncoder(handle_unknown="ignore"))
     ]),
     reg_categorical)
])

X_reg = df[reg_numeric + reg_categorical]
y_reg = df["fare"]

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg,
    y_reg,
    test_size=0.2,
    random_state=42
)

regression_pipeline = Pipeline([
    ("preprocessor", reg_preprocessor),
    ("model", LinearRegression())
])

regression_pipeline.fit(X_train_reg, y_train_reg)

y_pred_reg = regression_pipeline.predict(X_test_reg)

mae = mean_absolute_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
r2 = r2_score(y_test_reg, y_pred_reg)

n = len(y_test_reg)
p = len(
    regression_pipeline
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

print("\n===== REGRESSION RESULTS =====")
print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)
print("Adjusted R²:", adjusted_r2)


# ===== RESIDUAL PLOT =====

residuals = y_test_reg - y_pred_reg

plt.figure(figsize=(8, 5))

sns.scatterplot(
    x=y_pred_reg,
    y=residuals
)

plt.axhline(y=0, linestyle="--")

plt.xlabel("Predicted Fare")
plt.ylabel("Residual")
plt.title("Residual Plot - Fare Regression")

plt.tight_layout()
plt.savefig("fare_residual_plot.png")
plt.show()


# ===== SAVE FINAL FITTED PIPELINE =====

import joblib

final_pipeline = best_rf_pipeline

joblib.dump(
    final_pipeline,
    "titanic_final_pipeline.joblib"
)

print("Final pipeline saved successfully.")


# ===== RELOAD AND TEST =====

loaded_pipeline = joblib.load(
    "titanic_final_pipeline.joblib"
)

sample = X_test.iloc[[0]]

prediction = loaded_pipeline.predict(sample)

print("Prediction from reloaded pipeline:", prediction)