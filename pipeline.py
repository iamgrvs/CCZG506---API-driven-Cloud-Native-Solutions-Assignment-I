import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from apscheduler.schedulers.background import BackgroundScheduler

latest_telemetry = {
    "run_count": 0,
    "last_run": None,
    "status": "INITIALIZING",
    "metrics": {}
}

def execute_dataops_mlops_pipeline():
    global latest_telemetry
    latest_telemetry["run_count"] += 1
    run_id = latest_telemetry["run_count"]
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n=======================================================")
    print(f"[DATAOPS CRON TRIGGER] Run #{run_id} started at {timestamp}")
    print(f"=======================================================")

    # 1. Ingestion (Sub-Objective 1.2)
    df = pd.read_csv("telco_churn.csv")
    print(f"[INGESTION] Total raw records loaded: {len(df)}")

    # 2. Pre-processing & Synthetic Missing Data Induction (Sub-Objective 1.3)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan))
    
    # Deliberately induce 5.0% missing values in numeric attributes
    np.random.seed(42 + run_id)
    mask_monthly = np.random.rand(len(df)) < 0.05
    mask_total = np.random.rand(len(df)) < 0.05
    df.loc[mask_monthly, 'MonthlyCharges'] = np.nan
    df.loc[mask_total, 'TotalCharges'] = np.nan
    print(f"[PREPROCESSING] Induced 5% numeric nulls. Imputing with median values...")

    # Imputation
    df['MonthlyCharges'] = df['MonthlyCharges'].fillna(df['MonthlyCharges'].median())
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # MinMax Normalization
    scaler = MinMaxScaler()
    df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(
        df[['tenure', 'MonthlyCharges', 'TotalCharges']]
    )
    print(f"[NORMALIZATION] Continuous features scaled to [0.0, 1.0]")

    # 3. Model Training & Evaluation (Sub-Objectives 2.1 - 2.4)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    X = df[['tenure', 'MonthlyCharges', 'TotalCharges']]
    y = df['Churn']

    # 70% Train / 30% Test Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # Dual Algorithm Benchmark: Logistic Regression vs Random Forest
    lr = LogisticRegression().fit(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)

    preds = rf.predict(X_test)
    probs = rf.predict_proba(X_test)[:, 1]

    # Metrics Logging (MLOps)
    acc = round(accuracy_score(y_test, preds), 4)
    prec = round(precision_score(y_test, preds), 4)
    rec = round(recall_score(y_test, preds), 4)
    f1 = round(f1_score(y_test, preds), 4)
    auc = round(roc_auc_score(y_test, probs), 4)

    latest_telemetry.update({
        "status": "SUCCESS",
        "last_run": timestamp,
        "total_records": len(df),
        "metrics": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": auc
        }
    })
    print(f"[MLOPS LOGGING] Random Forest Evaluation -> Acc: {acc}, Prec: {prec}, Rec: {rec}, F1: {f1}, AUC: {auc}")
    print(f"[PIPELINE STATUS] Next automated run in exactly 2 minutes (Cron: */2 * * * *)\n")

# Schedule workflow to run every 2 minutes (Sub-Objective 1.5)
scheduler = BackgroundScheduler()
scheduler.add_job(execute_dataops_mlops_pipeline, 'interval', minutes=2)
scheduler.start()

# Initial run on service boot
execute_dataops_mlops_pipeline()
