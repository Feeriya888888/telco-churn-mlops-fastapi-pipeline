import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_recall_curve, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def generate_telco_dataset(n_samples: int = 7000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    tenure_months = rng.integers(1, 73, size=n_samples)
    contract_type = rng.choice(
        ["Month-to-month", "One year", "Two year"], size=n_samples, p=[0.55, 0.24, 0.21]
    )
    internet_service = rng.choice(
        ["Fiber optic", "DSL", "No"], size=n_samples, p=[0.44, 0.34, 0.22]
    )
    tech_support = np.where(
        internet_service == "No",
        "No internet service",
        rng.choice(["Yes", "No"], size=n_samples, p=[0.35, 0.65]),
    )
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        size=n_samples,
        p=[0.34, 0.23, 0.22, 0.21],
    )

    base_charge = np.where(
        internet_service == "Fiber optic", 75.0, np.where(internet_service == "DSL", 50.0, 22.0)
    )
    monthly_charges = np.round(base_charge + rng.normal(12.0, 10.0, size=n_samples), 2)
    monthly_charges = np.clip(monthly_charges, 18.0, 120.0)
    total_charges = np.round(monthly_charges * tenure_months + rng.normal(0, 15, size=n_samples), 2)
    total_charges = np.maximum(total_charges, monthly_charges)

    # Формирование логита оттока на основе реальных бизнес-факторов
    logit = (
        -1.2
        - 0.045 * tenure_months
        + 0.022 * monthly_charges
        + np.where(contract_type == "Month-to-month", 1.35, np.where(contract_type == "One year", 0.3, -0.8))
        + np.where(internet_service == "Fiber optic", 0.75, 0.0)
        + np.where(tech_support == "No", 0.55, 0.0)
        + np.where(payment_method == "Electronic check", 0.45, 0.0)
        + rng.normal(0, 0.35, size=n_samples)
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    churn = (rng.random(size=n_samples) < prob).astype(int)

    return pd.DataFrame(
        {
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "contract_type": contract_type,
            "internet_service": internet_service,
            "tech_support": tech_support,
            "payment_method": payment_method,
            "churn": churn,
        }
    )


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    # Средний исторический чек и индикатор резкого роста платежа
    data["avg_historical_charge"] = data["total_charges"] / np.maximum(data["tenure_months"], 1)
    data["charge_inflation_ratio"] = data["monthly_charges"] / np.maximum(data["avg_historical_charge"], 1.0)
    data["is_new_customer"] = (data["tenure_months"] <= 6).astype(int)
    return data


def find_optimal_business_threshold(
    y_true: np.ndarray, y_proba: np.ndarray, retention_cost: float = 20.0, customer_ltv: float = 150.0
) -> float:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    best_threshold = 0.5
    best_profit = -np.inf

    for thresh in thresholds:
        preds = (y_proba >= thresh).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        # Предполагаем, что 50% верно выявленных клиентов удается удержать
        profit = tp * (0.5 * customer_ltv - retention_cost) - fp * retention_cost
        if profit > best_profit:
            best_profit = profit
            best_threshold = float(thresh)

    return round(best_threshold, 3)


def train_pipeline():
    print("1. Генерация и подготовка датасета Telco Churn...")
    raw_df = generate_telco_dataset(n_samples=7000)
    featured_df = add_domain_features(raw_df)

    target_col = "churn"
    num_features = [
        "tenure_months",
        "monthly_charges",
        "total_charges",
        "avg_historical_charge",
        "charge_inflation_ratio",
        "is_new_customer",
    ]
    cat_features = ["contract_type", "internet_service", "tech_support", "payment_method"]

    X = featured_df[num_features + cat_features]
    y = featured_df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                num_features,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_features,
            ),
        ]
    )

    clf = LGBMClassifier(
        n_estimators=250,
        learning_rate=0.04,
        num_leaves=31,
        subsample_for_bin=5000,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )

    model_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])

    print("2. Обучение модели LightGBM...")
    model_pipeline.fit(X_train, y_train)

    y_proba = model_pipeline.predict_proba(X_test)[:, 1]
    opt_threshold = find_optimal_business_threshold(y_test, y_proba)
    y_pred = (y_proba >= opt_threshold).astype(int)

    roc_auc = roc_auc_score(y_test, y_proba)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"--- Результаты на тестовой выборке ---")
    print(f"ROC-AUC:            {roc_auc:.4f}")
    print(f"Optimal Threshold:  {opt_threshold:.3f}")
    print(f"Recall (Churn):     {recall:.4f}")
    print(f"F1-Score:           {f1:.4f}")

    artifact = {
        "pipeline": model_pipeline,
        "threshold": opt_threshold,
        "num_features": num_features,
        "cat_features": cat_features,
        "metrics": {"roc_auc": round(roc_auc, 4), "recall": round(recall, 4), "f1": round(f1, 4)},
    }
    joblib.dump(artifact, "model_bundle.joblib")
    print("3. Артефакт успешно сохранен в model_bundle.joblib")


if __name__ == "__main__":
    train_pipeline()
