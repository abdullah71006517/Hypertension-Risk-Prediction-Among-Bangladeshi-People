

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold
)

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier

# DATA LOADING & PREPROCESSING


def load_and_preprocess(yes_path, no_path):

    df_yes = pd.read_csv(yes_path)
    df_no = pd.read_csv(no_path)

    df = pd.concat([df_yes, df_no], ignore_index=True)

    df = df.dropna(subset=['hypertension'])

    # Numeric columns
    numeric_cols = [
        'age',
        'systolic_bp',
        'diastolic_bp'
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

    # Ordinal Encoding
    mapping = {
        'Low': 0,
        'Medium': 1,
        'High': 2
    }

    df['salt_intake'] = df['salt_intake'].map(mapping).fillna(1)
    df['stress_level'] = df['stress_level'].map(mapping).fillna(1)

    # Binary Encoding
    df['gender'] = df['gender'].map({
        'Male': 1,
        'Female': 0
    }).fillna(0)

    df['smoking'] = df['smoking'].map({
        'Yes': 1,
        'No': 0
    }).fillna(0)

    df['family_hypertension'] = df['family_hypertension'].map({
        'Yes': 1,
        'No': 0
    }).fillna(0)

    df['hypertension'] = df['hypertension'].map({
        'Yes': 1,
        'No': 0
    })

    features = [
        'age',
        'gender',
        'height_cm',
        'weight_kg',
        'BMI',
        'systolic_bp',
        'diastolic_bp',
        'smoking',
        'salt_intake',
        'sleep_hours',
        'stress_level',
        'family_hypertension'
    ]

    return df[features], df['hypertension']

# MAIN

if __name__ == "__main__":

    os.makedirs("results", exist_ok=True)

    path_yes = "Hypertension Risk Factor Among Bangladeshi People/Data/Exist Hypertension Hypertension Risk Factor in Bangladesh .xlsx - Hypertension Yes.csv"

    path_no = "Hypertension Risk Factor Among Bangladeshi People/Data/No Hypertension Hypertension Risk Factor in Bangladesh .xlsx - Hypertension No.csv"

    X, y = load_and_preprocess(path_yes, path_no)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # MODELS
    
    models = {

        "Logistic Regression":
            LogisticRegression(max_iter=1000),

        "Decision Tree":
            DecisionTreeClassifier(random_state=42),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            ),

        "KNN":
            KNeighborsClassifier(n_neighbors=5),

        "SVM":
            SVC(
                probability=True,
                kernel='rbf',
                random_state=42
            ),

        "Gradient Boosting":
            GradientBoostingClassifier(
                random_state=42
            ),

        "XGBoost":
            XGBClassifier(
                eval_metric='logloss',
                random_state=42
            )
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    results = []
    cv_scores_dict = {}

    print("\n==============================")
    print("MODEL PERFORMANCE")
    print("==============================")

    # TRAINING & EVALUATION
    
    for name, model in models.items():

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        cv_scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring='accuracy'
        )

        cv_scores_dict[name] = cv_scores

        results.append([
            name,
            acc,
            auc,
            cv_scores.mean()
        ])

        print(f"\n{name}")
        print(f"Accuracy : {acc:.4f}")
        print(f"ROC-AUC  : {auc:.4f}")

      # RESULTS TABLE
    results_df = pd.DataFrame(
        results,
        columns=[
            "Model",
            "Accuracy",
            "ROC_AUC",
            "CV_Accuracy"
        ]
    )

    results_df = results_df.sort_values(
        by="ROC_AUC",
        ascending=False
    )

    print("\n")
    print(results_df)

    results_df.to_csv(
        "results/model_results.csv",
        index=False
    )

    
    # GRAPH 1 - ACCURACY COMPARISON
    

    plt.figure(figsize=(10,6))

    sns.barplot(
        data=results_df,
        x="Accuracy",
        y="Model",
        palette="Blues_r"
    )

    plt.title("Model Accuracy Comparison")
    plt.tight_layout()

    plt.savefig(
        "results/accuracy_comparison.png",
        dpi=300
    )

    plt.show()

    
    # GRAPH 2 - ROC AUC COMPARISON
    
    plt.figure(figsize=(10,6))

    sns.barplot(
        data=results_df,
        x="ROC_AUC",
        y="Model",
        palette="viridis"
    )

    plt.title("Model ROC-AUC Comparison")
    plt.tight_layout()

    plt.savefig(
        "results/roc_auc_comparison.png",
        dpi=300
    )

    plt.show()

    
    # GRAPH 3 - CROSS VALIDATION BOXPLOT
    

    cv_df = pd.DataFrame(cv_scores_dict)

    plt.figure(figsize=(12,6))

    sns.boxplot(data=cv_df)

    plt.xticks(rotation=30)

    plt.title("Cross Validation Accuracy")

    plt.tight_layout()

    plt.savefig(
        "results/cross_validation_boxplot.png",
        dpi=300
    )

    plt.show()

    
    # GRAPH 4 - CORRELATION HEATMAP
    

    plt.figure(figsize=(12,10))

    sns.heatmap(
        X.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Feature Correlation Heatmap")

    plt.tight_layout()

    plt.savefig(
        "results/correlation_heatmap.png",
        dpi=300
    )

    plt.show()

    
    # GRAPH 5 - ROC CURVE COMPARISON
    

    plt.figure(figsize=(10,8))

    best_model_name = None
    best_auc = 0
    best_model = None

    for name, model in models.items():

        model.fit(X_train, y_train)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:,1]
        else:
            y_prob = model.decision_function(X_test)

        fpr, tpr, _ = roc_curve(y_test, y_prob)

        auc = roc_auc_score(y_test, y_prob)

        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUC={auc:.3f})"
        )

        if auc > best_auc:
            best_auc = auc
            best_model = model
            best_model_name = name

    plt.plot(
        [0,1],
        [0,1],
        'k--'
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "results/roc_curve_comparison.png",
        dpi=300
    )

    plt.show()

    
    # GRAPH 6 - BEST MODEL CONFUSION MATRIX
    

    y_pred = best_model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues'
    )

    plt.title(
        f"{best_model_name} Confusion Matrix"
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()

    plt.savefig(
        "results/confusion_matrix_best_model.png",
        dpi=300
    )

    plt.show()

    
    # GRAPH 7 - FEATURE IMPORTANCE
    

    if hasattr(best_model, "feature_importances_"):

        importance = best_model.feature_importances_

        indices = np.argsort(importance)[::-1]

        plt.figure(figsize=(10,6))

        sns.barplot(
            x=importance[indices],
            y=np.array(X.columns)[indices],
            palette="magma"
        )

        plt.title(
            f"{best_model_name} Feature Importance"
        )

        plt.tight_layout()

        plt.savefig(
            "results/feature_importance.png",
            dpi=300
        )

        plt.show()

    print("\n==============================")
    print("ALL RESULTS SAVED IN 'results/'")
    print("==============================")